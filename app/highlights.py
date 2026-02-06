import json
import os
from typing import Dict, List, Optional, Tuple

from app import db
from app import secrets_store

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EST_COST_PER_1K_TOKENS = 0.00075


def _estimate_cost(text: str) -> float:
    if not text:
        return 0.0
    tokens = max(1, len(text) // 4)
    return (tokens / 1000) * EST_COST_PER_1K_TOKENS


def _build_payload(company, score_item, risk_flags, components) -> Dict[str, object]:
    payload = {
        "company": {
            "id": company.get("id"),
            "name": company.get("name"),
            "website": company.get("website"),
            "industry": company.get("industry"),
            "location": company.get("location"),
        },
        "score": {
            "normalized_total": score_item.get("normalized_total") if score_item else None,
            "raw_total": score_item.get("raw_total") if score_item else None,
            "weight_used": score_item.get("weight_used") if score_item else None,
            "scored_at": score_item.get("scored_at") if score_item else None,
        },
        "risk_flags": risk_flags,
        "score_components": components,
    }
    return payload


def generate_highlights(
    conn,
    company_id: int,
    model: str = DEFAULT_MODEL,
    ai_budget_usd: float = 0.2,
) -> Tuple[Optional[Dict[str, object]], Optional[str]]:
    company = db.get_company(conn, company_id)
    if not company:
        return None, "Company not found"

    score_item = db.list_latest_score_item(conn, company_id)
    risk_flags = [dict(r) for r in db.list_latest_risk_flags(conn, company_id)]
    components = [dict(c) for c in db.list_score_components(conn, company_id)]
    components = sorted(components, key=lambda r: (r.get("contribution") or 0), reverse=True)[:20]

    payload = _build_payload(dict(company), dict(score_item) if score_item else None, risk_flags, components)
    payload_text = json.dumps(payload, ensure_ascii=False)
    est_cost = _estimate_cost(payload_text)
    if est_cost > ai_budget_usd:
        return None, f"Estimated AI cost ${est_cost:.4f} exceeds budget ${ai_budget_usd:.2f}"

    if OpenAI is None:
        return None, "openai package is not installed"

    if not secrets_store.ensure_openai_key():
        return None, "OPENAI_API_KEY is not set"

    client = OpenAI()

    system_msg = (
        "You are a rigorous investment analyst. Use only the provided data. "
        "Return JSON with two arrays: risks and highlights. "
        "Each item: {title, rationale, evidence}. "
        "Evidence must reference the provided sources (e.g., signal key, evidence text, source_ref). "
        "Max 3 items per array. If data is insufficient, return fewer items and say why."
    )
    user_msg = (
        "Generate top 3 risks and top 3 investment highlights based on this data. "
        "Do not invent facts. JSON only.\n\nDATA:\n"
        + payload_text
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON from model: {exc}"

    if not isinstance(data, dict):
        return None, "Model did not return a JSON object"

    return data, None
