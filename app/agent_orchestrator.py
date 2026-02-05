import os
from dataclasses import dataclass
from typing import List

from app import db
from app import discovery
from app import extraction
from app import scoring
from app import criteria_ai
from app import risk_detector


@dataclass
class AgentResult:
    name: str
    ok: bool
    details: dict


class DiscoveryAgent:
    def run(self, conn, company_id: int, mode: str, intensity: str, allowlist: List[str], news_domains: List[str]):
        company = db.get_company(conn, company_id)
        urls = discovery.discover_urls(company, mode, allowlist, intensity, news_domains)
        return AgentResult("discovery", True, {"url_count": len(urls)})


class ExtractionAgent:
    def run(self, conn, company_id: int, mode: str, intensity: str, allowlist: List[str], news_domains: List[str]):
        saved, err = discovery.crawl_and_extract(conn, company_id, mode=mode, intensity=intensity, allowlist=allowlist, news_domains=news_domains)
        return AgentResult("extraction", err is None, {"signals_saved": saved, "error": err})


class ScoringAgent:
    def run(self, conn, company_id: int):
        version_id = db.get_active_criteria_version_id(conn)
        scoring.run_scoring_for_company(conn, version_id, company_id, actor="agent")
        return AgentResult("scoring", True, {"criteria_version_id": version_id})


class CriteriaAIAgent:
    def run(self, conn, company_id: int):
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        ai_budget = float(os.getenv("AI_BUDGET_USD", "0.2"))
        saved, err = criteria_ai.run_ai_scoring(
            conn,
            company_id,
            model=model,
            ai_budget_usd=ai_budget,
            include_website=True,
            include_founder_materials=True,
            website_url=None,
            max_pages=5,
            same_domain=True,
            top_k=8,
            search_per_signal=True,
            search_max_pages=3,
        )
        return AgentResult("criteria_ai", err is None, {"signals_saved": saved, "error": err})


class RiskAgent:
    def run(self, conn, company_id: int):
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        ai_budget = float(os.getenv("AI_BUDGET_USD", "0.2"))
        engine = risk_detector.RiskDetectionEngine(conn, model=model, ai_budget_usd=ai_budget, enable_llm=True)
        engine.run_scan(company_ids=[company_id], actor="agent")
        return AgentResult("risk", True, {"model": model})


class AgentOrchestrator:
    def __init__(self, conn, mode: str, intensity: str, allowlist: List[str], news_domains: List[str]):
        self.conn = conn
        self.mode = mode
        self.intensity = intensity
        self.allowlist = allowlist
        self.news_domains = news_domains
        self.agents = [DiscoveryAgent(), ExtractionAgent(), CriteriaAIAgent(), ScoringAgent(), RiskAgent()]

    def run(self, company_id: int):
        results = []
        for agent in self.agents:
            if isinstance(agent, ExtractionAgent):
                res = agent.run(self.conn, company_id, self.mode, self.intensity, self.allowlist, self.news_domains)
            elif isinstance(agent, CriteriaAIAgent):
                res = agent.run(self.conn, company_id)
            elif isinstance(agent, ScoringAgent):
                res = agent.run(self.conn, company_id)
            elif isinstance(agent, RiskAgent):
                res = agent.run(self.conn, company_id)
            else:
                res = agent.run(self.conn, company_id, self.mode, self.intensity, self.allowlist, self.news_domains)
            results.append(res)
            try:
                db.add_activity(self.conn, f"agent_{res.name}", "company", company_id, res.details, actor="agent")
            except Exception:
                pass
        return results
