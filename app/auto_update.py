import os

from app import db
from app import scoring
from app import risk_detector


def rescore_company(conn, company_id, actor="auto"):
    version_id = db.get_active_criteria_version_id(conn)
    scoring.run_scoring_for_company(conn, version_id, company_id, actor=actor)

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ai_budget = float(os.getenv("AI_BUDGET_USD", "0.2"))
    engine = risk_detector.RiskDetectionEngine(conn, model=model, ai_budget_usd=ai_budget, enable_llm=True)
    engine.run_scan(company_ids=[company_id], actor=actor)
    db.add_activity(conn, "rescore_company", "company", company_id, {"model": model, "ai_budget": ai_budget}, actor=actor)
