import os
from typing import Dict, List, Optional, Tuple

from app import db
from app import extraction
from app import rag

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EST_COST_PER_1K_TOKENS = 0.00075


def _estimate_cost(text: str) -> float:
    if not text:
        return 0.0
    tokens = max(1, len(text) // 4)
    return (tokens / 1000) * EST_COST_PER_1K_TOKENS


def _build_context(website_text: Optional[str], founder_text: Optional[str]) -> str:
    parts = []
    if website_text:
        parts.append("WEBSITE:\n" + website_text)
    if founder_text:
        parts.append("FOUNDER MATERIALS:\n" + founder_text)
    return "\n\n".join(parts)


def _filter_ai_defs(defs: List[dict]) -> List[dict]:
    return [
        d
        for d in defs
        if (d.get("automation_type") or "").lower() == "ai" and not d.get("disabled")
    ]


def _build_query_from_defs(defs: List[dict]) -> str:
    parts = []
    for d in defs:
        if d.get("disabled"):
            continue
        name = d.get("name") or ""
        desc = d.get("description") or ""
        parts.append(f"{name} {desc}".strip())
    base = " ".join(parts)
    return f"{base} startup traction team market finance risks".strip()


def run_ai_scoring(
    conn,
    company_id: int,
    model: str = DEFAULT_MODEL,
    ai_budget_usd: float = 0.2,
    include_website: bool = True,
    include_founder_materials: bool = True,
    website_url: Optional[str] = None,
    max_pages: int = 5,
    same_domain: bool = True,
    top_k: int = 8,
) -> Tuple[int, Optional[str]]:
    company = db.get_company(conn, company_id)
    if company is None:
        return 0, "Company not found"

    defs = db.list_signal_definitions(conn)
    ai_defs = _filter_ai_defs(defs)
    if not ai_defs:
        return 0, "No AI-automatable signals found"

    website_text = None
    pages = []
    if include_website:
        url = website_url or company["website"]
        if url:
            website_text, pages = extraction.crawl_site_text(url, max_pages=max_pages, same_domain=same_domain)

    founder_text = None
    if include_founder_materials:
        chunks = rag.get_company_chunks(conn, company_id, source_types=["founder_material"])
        if chunks:
            query = _build_query_from_defs(ai_defs)
            top_chunks = rag.retrieve_top_chunks(chunks, query, top_k=top_k)
            founder_text = rag.build_context(top_chunks)

    context = _build_context(website_text, founder_text)
    if not context:
        return 0, "No text context available for AI scoring"

    est_cost = _estimate_cost(context)
    if est_cost > ai_budget_usd:
        return 0, f"Estimated AI cost ${est_cost:.4f} exceeds budget ${ai_budget_usd:.2f}"

    try:
        signals = extraction.extract_signals_with_openai(context, ai_defs, model=model)
    except Exception as exc:
        return 0, f"AI extraction failed: {exc}"

    def_map = {d["key"]: d for d in ai_defs}
    saved = 0
    notes = f"model={model};pages={len(pages)};founder_materials={'yes' if founder_text else 'no'}"
    source_ref = website_url or company["website"] or "criteria_ai"

    for key, payload in signals.items():
        if key not in def_map:
            continue
        d = def_map[key]
        value_num, value_text, value_bool, value_json, confidence, evidence, evidence_page = extraction.normalize_extracted_signal(
            payload, d["value_type"]
        )
        raw_value = payload.get("value") if isinstance(payload, dict) else payload
        if raw_value is None:
            continue
        db.add_signal_value(
            conn,
            company_id,
            key,
            value_num,
            value_text,
            value_bool,
            value_json,
            source_type="model",
            source_ref=source_ref,
            notes=notes,
            confidence=confidence,
            evidence_text=evidence,
            evidence_page=evidence_page,
        )
        saved += 1

    return saved, None
