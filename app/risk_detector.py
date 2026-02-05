import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from app import db
from app import rag

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_RULES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "risk_rules.json")
MAX_LLM_CHARS = 12000

SEVERITY_WEIGHTS = {
    "low": 0.2,
    "medium": 0.4,
    "high": 0.7,
    "critical": 1.0,
}

SOURCE_CONFIDENCE = {
    "sec": 0.9,
    "opencorporates": 0.9,
    "api": 0.9,
    "gdelt": 0.9,
    "scrape": 0.7,
    "crawl": 0.7,
    "model": 0.5,
    "llm": 0.5,
    "manual": 0.6,
}


def _utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def _coerce_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        return None


def _signal_value(row):
    if row is None:
        return None
    if row.get("value_num") is not None:
        return float(row["value_num"])
    if row.get("value_bool") is not None:
        return float(row["value_bool"])
    if row.get("value_text") is not None:
        return _coerce_number(row["value_text"])
    if row.get("value_json"):
        try:
            data = json.loads(row["value_json"])
            return _coerce_number(data)
        except json.JSONDecodeError:
            return _coerce_number(row["value_json"])
    return None


def _signal_text(row):
    if row is None:
        return None
    if row.get("value_text"):
        return str(row["value_text"])
    if row.get("value_json"):
        return str(row["value_json"])
    return None


def _signal_confidence(row):
    if row is None:
        return 0.6
    if row.get("confidence") is not None:
        try:
            return max(0.0, min(1.0, float(row["confidence"])))
        except (TypeError, ValueError):
            return 0.6
    source = (row.get("source_type") or "").lower()
    return SOURCE_CONFIDENCE.get(source, 0.6)


def _severity_weight(severity):
    return SEVERITY_WEIGHTS.get((severity or "").lower(), 0.4)


class RuleBasedDetector:
    def __init__(self, rules: Dict[str, List[dict]]):
        self.rules = rules or {}

    def detect(self, signal_map: Dict[str, dict]) -> List[dict]:
        flags = []
        for category, rules in self.rules.items():
            for rule in rules:
                hit, payload = self._evaluate_rule(rule, signal_map)
                if not hit:
                    continue
                payload.update(
                    {
                        "category": category,
                        "risk_type": rule.get("type"),
                        "severity": rule.get("severity", "medium"),
                        "weight": rule.get("weight", 0.6),
                        "description": rule.get("description"),
                        "method": "rule",
                    }
                )
                flags.append(payload)
        return flags

    def _evaluate_rule(self, rule: dict, signal_map: Dict[str, dict]) -> Tuple[bool, dict]:
        signal_keys = rule.get("signal_keys") or []
        calc = rule.get("calc")
        rule_def = rule.get("rule") or {}
        op = rule_def.get("op")
        threshold = rule_def.get("threshold")

        rows = [signal_map.get(k) for k in signal_keys]
        values = [_signal_value(r) for r in rows]

        evidence_row = None
        confidences = [_signal_confidence(r) for r in rows if r is not None]
        confidence = sum(confidences) / len(confidences) if confidences else 0.6

        if calc == "tam_sam_som":
            if len(values) < 3:
                return False, {}
            tam, sam, som = values[:3]
            if tam is None or sam is None or som is None:
                return False, {}
            if tam < sam or sam < som:
                evidence_row = rows[0]
                return True, self._build_payload(confidence, evidence_row, f"TAM/SAM/SOM mismatch: {tam}, {sam}, {som}")
            return False, {}

        if calc == "any_false":
            any_false = False
            for val in values:
                if val is None:
                    any_false = True
                elif isinstance(val, (int, float)) and val <= 0:
                    any_false = True
            if any_false:
                evidence_row = rows[0]
                return True, self._build_payload(confidence, evidence_row, "Missing or false key role signals")
            return False, {}

        if calc == "missing_second":
            if len(values) < 2:
                return False, {}
            first, second = values[0], values[1]
            if first is not None and (second is None or second == 0):
                evidence_row = rows[0]
                return True, self._build_payload(confidence, evidence_row, "Primary metric present without supporting engagement metric")
            return False, {}

        if calc == "ratio":
            if len(values) < 2:
                return False, {}
            numerator, denominator = values[0], values[1]
            if numerator is None or denominator in (None, 0):
                return False, {}
            value = numerator / denominator
            hit = self._compare(op, value, threshold)
            if hit:
                evidence_row = rows[0] or rows[1]
                return True, self._build_payload(confidence, evidence_row, f"Computed ratio {value:.2f}")
            return False, {}

        if len(values) < 1:
            return False, {}
        value = values[0]
        if value is None:
            return False, {}
        hit = self._compare(op, value, threshold)
        if hit:
            evidence_row = rows[0]
            return True, self._build_payload(confidence, evidence_row, f"Value {value}")
        return False, {}

    def _compare(self, op, value, threshold):
        if op is None:
            return False
        try:
            if op == "gt":
                return value > threshold
            if op == "gte":
                return value >= threshold
            if op == "lt":
                return value < threshold
            if op == "lte":
                return value <= threshold
            if op == "eq":
                return value == threshold
            if op == "contains":
                return str(threshold).lower() in str(value).lower()
        except TypeError:
            return False
        return False

    def _build_payload(self, confidence: float, evidence_row: Optional[dict], note: str) -> dict:
        return {
            "confidence": confidence,
            "evidence_text": (evidence_row or {}).get("evidence_text"),
            "source_type": (evidence_row or {}).get("source_type"),
            "source_ref": (evidence_row or {}).get("source_ref"),
            "evidence_page": (evidence_row or {}).get("evidence_page"),
            "note": note,
        }


class StatisticalAnomalyDetector:
    def __init__(self, conn):
        self.conn = conn
        self.stats = self._load_stats()

    def _load_stats(self):
        sets = db.list_benchmark_sets(self.conn)
        if not sets:
            return []
        return db.list_benchmark_stats(self.conn, sets[0]["id"])

    def detect(self, company: dict, signal_map: Dict[str, dict]) -> List[dict]:
        flags = []
        if not self.stats:
            return flags

        sector = (company.get("industry") or "").lower()
        stage = None
        stage_row = signal_map.get("funding_stage") or signal_map.get("stage")
        if stage_row:
            stage = (stage_row.get("value_text") or "").lower()

        for key, row in signal_map.items():
            value = _signal_value(row)
            if value is None:
                continue
            stat = self._match_stat(key, sector, stage)
            if not stat:
                continue

            p10, p50, p90 = stat.get("p10"), stat.get("p50"), stat.get("p90")
            if p10 is None or p50 is None or p90 is None or p90 == p10:
                continue
            std = (p90 - p10) / 2.563 if p90 > p10 else None
            if not std:
                continue
            z = (value - p50) / std
            severity = "medium" if abs(z) >= 2 else None
            if abs(z) >= 3:
                severity = "high"
            if severity:
                flags.append(
                    {
                        "category": "anomaly",
                        "risk_type": "zscore_outlier",
                        "severity": severity,
                        "confidence": _signal_confidence(row),
                        "description": f"{key} is {z:.2f} standard deviations from median",
                        "evidence_text": row.get("evidence_text"),
                        "source_type": row.get("source_type"),
                        "source_ref": row.get("source_ref"),
                        "evidence_page": row.get("evidence_page"),
                        "method": "stat",
                        "weight": 0.6,
                    }
                )

            if value > p90 * 2 or value < p10 * 0.5:
                flags.append(
                    {
                        "category": "anomaly",
                        "risk_type": "extreme_outlier",
                        "severity": "high",
                        "confidence": _signal_confidence(row),
                        "description": f"{key} is far outside peer range",
                        "evidence_text": row.get("evidence_text"),
                        "source_type": row.get("source_type"),
                        "source_ref": row.get("source_ref"),
                        "evidence_page": row.get("evidence_page"),
                        "method": "stat",
                        "weight": 0.8,
                    }
                )
        return flags

    def _match_stat(self, metric_key: str, sector: str, stage: Optional[str]):
        candidates = [s for s in self.stats if s.get("metric_key") == metric_key]
        if not candidates:
            return None
        scored = []
        for s in candidates:
            score = 0
            if s.get("sector") and sector and s.get("sector", "").lower() == sector:
                score += 2
            if s.get("stage") and stage and s.get("stage", "").lower() == stage:
                score += 2
            if s.get("geo"):
                score += 1
            scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else None


class PatternDetector:
    def __init__(self, conn):
        self.conn = conn
        self._chunk_index = None

    def detect(self, company_id: int, signal_map: Dict[str, dict]) -> List[dict]:
        flags = []
        flags.extend(self._round_numbers(signal_map))
        flags.extend(self._vanity_metrics(signal_map))
        flags.extend(self._missing_history(company_id))
        flags.extend(self._engagement_drop(company_id))
        flags.extend(self._budget_cuts(company_id))
        flags.extend(self._plagiarism(company_id))
        return flags

    def _round_numbers(self, signal_map):
        flags = []
        key_markers = ["user", "customer", "subscriber", "download", "install", "signup", "member"]
        for key, row in signal_map.items():
            if not any(marker in key.lower() for marker in key_markers):
                continue
            value = _signal_value(row)
            if value is None:
                continue
            if value >= 1000 and value % 1000 == 0:
                flags.append(
                    {
                        "category": "traction",
                        "risk_type": "suspicious_round_number",
                        "severity": "medium",
                        "confidence": _signal_confidence(row),
                        "description": f"{key} is a suspiciously round number",
                        "evidence_text": row.get("evidence_text"),
                        "source_type": row.get("source_type"),
                        "source_ref": row.get("source_ref"),
                        "evidence_page": row.get("evidence_page"),
                        "method": "pattern",
                        "weight": 0.5,
                    }
                )
        return flags

    def _vanity_metrics(self, signal_map):
        downloads = _signal_value(signal_map.get("downloads"))
        mau = _signal_value(signal_map.get("mau"))
        dau = _signal_value(signal_map.get("dau"))
        if downloads and not mau and not dau:
            row = signal_map.get("downloads")
            return [
                {
                    "category": "traction",
                    "risk_type": "vanity_metrics",
                    "severity": "medium",
                    "confidence": _signal_confidence(row),
                    "description": "Downloads reported without MAU/DAU engagement metrics",
                    "evidence_text": row.get("evidence_text"),
                    "source_type": row.get("source_type"),
                    "source_ref": row.get("source_ref"),
                    "evidence_page": row.get("evidence_page"),
                    "method": "pattern",
                    "weight": 0.5,
                }
            ]
        return []

    def _missing_history(self, company_id):
        flags = []
        keys = ["mrr", "arr", "revenue", "burn_rate", "churn_rate"]
        for key in keys:
            history = db.list_signal_history(self.conn, company_id, key)
            if len(history) < 2:
                flags.append(
                    {
                        "category": "financial",
                        "risk_type": "missing_history",
                        "severity": "low",
                        "confidence": 0.6,
                        "description": f"Only {len(history)} data point(s) for {key}",
                        "evidence_text": None,
                        "source_type": None,
                        "source_ref": None,
                        "evidence_page": None,
                        "method": "pattern",
                        "weight": 0.3,
                    }
                )
        return flags

    def _engagement_drop(self, company_id):
        flags = []
        for key in ["dau_mau_ratio", "engagement_rate"]:
            history = db.list_signal_history(self.conn, company_id, key)
            if len(history) < 2:
                continue
            latest = _signal_value(history[0])
            prev = _signal_value(history[1])
            if latest is None or prev in (None, 0):
                continue
            drop = (prev - latest) / prev
            if drop >= 0.2:
                flags.append(
                    {
                        "category": "traction",
                        "risk_type": "declining_engagement",
                        "severity": "medium",
                        "confidence": _signal_confidence(history[0]),
                        "description": f"{key} dropped {drop:.0%} recently",
                        "evidence_text": history[0].get("evidence_text"),
                        "source_type": history[0].get("source_type"),
                        "source_ref": history[0].get("source_ref"),
                        "evidence_page": history[0].get("evidence_page"),
                        "method": "pattern",
                        "weight": 0.6,
                    }
                )
        return flags

    def _budget_cuts(self, company_id):
        flags = []
        for key in ["headcount", "team_size", "burn_rate"]:
            history = db.list_signal_history(self.conn, company_id, key)
            if len(history) < 2:
                continue
            latest = _signal_value(history[0])
            prev = _signal_value(history[1])
            if latest is None or prev in (None, 0):
                continue
            drop = (prev - latest) / prev
            if drop >= 0.2:
                flags.append(
                    {
                        "category": "financial",
                        "risk_type": "sudden_budget_cut",
                        "severity": "medium",
                        "confidence": _signal_confidence(history[0]),
                        "description": f"{key} dropped {drop:.0%} recently",
                        "evidence_text": history[0].get("evidence_text"),
                        "source_type": history[0].get("source_type"),
                        "source_ref": history[0].get("source_ref"),
                        "evidence_page": history[0].get("evidence_page"),
                        "method": "pattern",
                        "weight": 0.6,
                    }
                )
        return flags

    def _plagiarism(self, company_id):
        if self.conn is None:
            return []
        if self._chunk_index is None:
            self._chunk_index = self._build_chunk_index()

        company_chunks = db.list_document_chunks_for_company(self.conn, company_id)
        if not company_chunks:
            return []

        hashes = set()
        for ch in company_chunks:
            text = (ch.get("chunk_text") or "").strip().lower()
            if len(text) < 200:
                continue
            hashes.add(hash(text))

        shared = 0
        for h in hashes:
            companies = self._chunk_index.get(h, set())
            if len(companies) > 1 and company_id in companies:
                shared += 1

        if shared >= 3:
            return [
                {
                    "category": "pattern",
                    "risk_type": "copy_paste_content",
                    "severity": "low",
                    "confidence": 0.6,
                    "description": "Significant overlap with other pitch materials",
                    "evidence_text": None,
                    "source_type": "document",
                    "source_ref": None,
                    "evidence_page": None,
                    "method": "pattern",
                    "weight": 0.4,
                }
            ]
        return []

    def _build_chunk_index(self):
        index = {}
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT dc.chunk_text, l.company_id
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            JOIN document_company_links l ON l.document_id = d.id
            """
        )
        for text, company_id in cur.fetchall():
            if not text:
                continue
            normalized = text.strip().lower()
            if len(normalized) < 200:
                continue
            h = hash(normalized)
            index.setdefault(h, set()).add(company_id)
        return index


class CrossSourceValidator:
    def __init__(self, conn, diff_threshold: float = 0.5):
        self.conn = conn
        self.diff_threshold = diff_threshold

    def detect(self, company_id: int) -> List[dict]:
        flags = []
        keys = ["revenue", "mrr", "arr", "user_count", "customer_count", "employee_count"]
        for key in keys:
            history = db.list_signal_history(self.conn, company_id, key)
            if len(history) < 2:
                continue
            values_by_source = {}
            for row in history[:5]:
                source = row.get("source_type") or "unknown"
                value = _signal_value(row)
                if value is None:
                    continue
                values_by_source[source] = value
            if len(values_by_source) < 2:
                continue
            values = list(values_by_source.values())
            min_v = min(values)
            max_v = max(values)
            if min_v == 0:
                continue
            diff = (max_v - min_v) / min_v
            if diff >= self.diff_threshold:
                flags.append(
                    {
                        "category": "inconsistency",
                        "risk_type": "source_mismatch",
                        "severity": "high" if diff >= 1.0 else "medium",
                        "confidence": 0.7,
                        "description": f"{key} differs across sources by {diff:.0%}",
                        "evidence_text": None,
                        "source_type": "multi-source",
                        "source_ref": ", ".join(values_by_source.keys()),
                        "evidence_page": None,
                        "method": "cross_source",
                        "weight": 0.7,
                    }
                )
        return flags


class LLMRiskExtractor:
    def __init__(self, conn, model: str = DEFAULT_MODEL, ai_budget_usd: float = 0.2):
        self.conn = conn
        self.model = model
        self.ai_budget_usd = ai_budget_usd
        self.ai_cost_usd = 0.0

    def detect(self, company_id: int) -> List[dict]:
        if OpenAI is None:
            return []
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return []

        chunks = db.list_document_chunks_for_company(self.conn, company_id)
        if not chunks:
            return []

        query = (
            "lawsuit regulatory investigation breach security incident "
            "ip dispute patent troll controversy fraud layoffs churn market decline"
        )
        top_chunks = rag.retrieve_top_chunks([dict(c) for c in chunks], query, top_k=8)
        if not top_chunks:
            return []

        context = rag.build_context(top_chunks)
        if not context:
            return []

        context = context[:MAX_LLM_CHARS]
        est_tokens = max(1, len(context) // 4)
        est_cost = est_tokens / 1000 * 0.00075
        if self.ai_cost_usd + est_cost > self.ai_budget_usd:
            return []

        client = OpenAI()
        system_msg = (
            "You are an investment risk analyst. Extract explicit red flags only. "
            "Return JSON with a 'flags' array. Each flag must include: "
            "category (financial|market|team|traction|legal|inconsistency), "
            "type, severity (low|medium|high|critical), description, evidence, source, page. "
            "Only use evidence quoted from the text."
        )
        user_msg = f"Text:\n{context}\n\nReturn JSON only."

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        self.ai_cost_usd += est_cost
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []

        flags = data.get("flags") if isinstance(data, dict) else data
        if not isinstance(flags, list):
            return []

        results = []
        for f in flags:
            if not isinstance(f, dict):
                continue
            results.append(
                {
                    "category": f.get("category") or "legal",
                    "risk_type": f.get("type") or "llm_flag",
                    "severity": f.get("severity") or "medium",
                    "confidence": 0.5,
                    "description": f.get("description"),
                    "evidence_text": f.get("evidence"),
                    "source_type": "document",
                    "source_ref": f.get("source"),
                    "evidence_page": _coerce_number(f.get("page")),
                    "method": "llm",
                    "weight": 0.7,
                }
            )
        return results


class RiskDetectionEngine:
    def __init__(
        self,
        conn,
        rules_path: str = DEFAULT_RULES_PATH,
        model: str = DEFAULT_MODEL,
        ai_budget_usd: float = 0.2,
        enable_llm: bool = True,
    ):
        self.conn = conn
        self.rules_path = rules_path
        self.model = model
        self.ai_budget_usd = ai_budget_usd
        self.enable_llm = enable_llm

        self.rules = self._load_rules()
        self.rule_detector = RuleBasedDetector(self.rules)
        self.stat_detector = StatisticalAnomalyDetector(conn)
        self.pattern_detector = PatternDetector(conn)
        self.cross_validator = CrossSourceValidator(conn)
        self.llm_detector = LLMRiskExtractor(conn, model=model, ai_budget_usd=ai_budget_usd)

    def _load_rules(self):
        if not os.path.exists(self.rules_path):
            return {}
        with open(self.rules_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def compute_risk_score(flags: List[dict]) -> float:
        total = 0.0
        for flag in flags:
            severity = _severity_weight(flag.get("severity"))
            confidence = float(flag.get("confidence") or 0.6)
            weight = float(flag.get("weight") or 0.6)
            total += severity * confidence * weight
        return 1 - math.exp(-total) if total > 0 else 0.0

    @staticmethod
    def risk_level(score: float) -> str:
        if score >= 0.75:
            return "critical"
        if score >= 0.5:
            return "high"
        if score >= 0.25:
            return "medium"
        return "low"

    def run_scan(self, company_ids: Optional[List[int]] = None, actor: str = "user"):
        companies = db.list_companies(self.conn)
        if company_ids is not None:
            companies = [c for c in companies if c["id"] in company_ids]

        run_id = db.create_risk_run(self.conn, actor=actor)

        for company in companies:
            signal_rows = db.list_latest_signal_values(self.conn, company["id"])
            signal_map = {s["signal_key"]: dict(s) for s in signal_rows}

            flags = []
            flags.extend(self.rule_detector.detect(signal_map))
            flags.extend(self.stat_detector.detect(company, signal_map))
            flags.extend(self.pattern_detector.detect(company["id"], signal_map))
            flags.extend(self.cross_validator.detect(company["id"]))
            if self.enable_llm:
                flags.extend(self.llm_detector.detect(company["id"]))

            risk_score = self.compute_risk_score(flags)
            risk_level = self.risk_level(risk_score)

            run_item_id = db.create_risk_run_item(
                self.conn,
                run_id,
                company["id"],
                risk_score,
                risk_level,
            )

            for f in flags:
                f.setdefault("detected_at", _utc_now())
            db.add_risk_flags(self.conn, run_item_id, flags)

        db.complete_risk_run(self.conn, run_id)
        return run_id, None
