import os
import json
from typing import Dict, List, Optional, Tuple

from app import db
from app import extraction
from app import rag
from app import web_discovery

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


def _agent_profile_from_def(signal_def: dict) -> dict:
    profile = {}
    detail = signal_def.get("automation_detail")
    if isinstance(detail, dict):
        profile.update(detail)
    elif isinstance(detail, str) and detail.strip():
        try:
            parsed = json.loads(detail)
            if isinstance(parsed, dict):
                profile.update(parsed)
            else:
                profile["backstory"] = detail
        except json.JSONDecodeError:
            profile["backstory"] = detail
    if signal_def.get("automation_prompt"):
        profile["prompt"] = signal_def["automation_prompt"]
    tools = profile.get("tools")
    if tools and isinstance(tools, str):
        profile["tools"] = [t.strip() for t in tools.split(",") if t.strip()]
    return profile


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
    search_per_signal: bool = False,
    search_max_pages: int = 3,
    signal_keys: Optional[List[str]] = None,
    allow_web_fallback: bool = True,
) -> Tuple[int, Optional[str]]:
    company = db.get_company(conn, company_id)
    if company is None:
        return 0, "Company not found"

    defs = [dict(d) for d in db.list_signal_definitions(conn)]
    ai_defs = _filter_ai_defs(defs)
    if signal_keys:
        allowed = set(signal_keys)
        ai_defs = [d for d in ai_defs if d.get("key") in allowed]
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

    saved = 0
    total_est_cost = 0.0

    if search_per_signal:
        founder_chunks = []
        if include_founder_materials:
            founder_chunks = rag.get_company_chunks(conn, company_id, source_types=["founder_material"])
        website_text = None
        if include_website:
            url = website_url or company["website"]
            if url:
                try:
                    website_text, _ = extraction.crawl_site_text(url, max_pages=3, same_domain=True)
                except Exception:
                    website_text = None
        for d in ai_defs:
            key = d["key"]
            name = d.get("name") or key
            desc = d.get("description") or ""
            query = f"{company['name']} {name} {desc}".strip()
            agent_profile = _agent_profile_from_def(d)
            tools = set([t.lower() for t in (agent_profile.get("tools") or [])])
            allow_founder = include_founder_materials and (not tools or "founder_materials" in tools or "founder" in tools)
            allow_site = include_website and (not tools or "company_website" in tools or "website" in tools)
            allow_web = allow_web_fallback and (not tools or "web_search" in tools or "web" in tools)

            # Step 1: Try founder materials first. Only run web agents if not found.
            if founder_chunks and allow_founder:
                top_chunks = rag.retrieve_top_chunks(founder_chunks, query, top_k=top_k)
                founder_context = rag.build_context(top_chunks)
                if founder_context:
                    est_cost = _estimate_cost(founder_context)
                    if total_est_cost + est_cost > ai_budget_usd:
                        return saved, f"Estimated AI cost ${total_est_cost + est_cost:.4f} exceeds budget ${ai_budget_usd:.2f}"
                    total_est_cost += est_cost
                    try:
                        signals = extraction.extract_single_signal_with_openai(
                            founder_context,
                            d,
                            model=model,
                            company_name=company["name"],
                            agent_profile=agent_profile,
                        )
                    except Exception:
                        signals = {}
                    payload = signals.get(key)
                    if payload is not None:
                        value_num, value_text, value_bool, value_json, confidence, evidence, evidence_page = extraction.normalize_extracted_signal(
                            payload, d["value_type"]
                        )
                        raw_value = payload.get("value") if isinstance(payload, dict) else payload
                        if raw_value is not None:
                            db.add_signal_value(
                                conn,
                                company_id,
                                key,
                                value_num,
                                value_text,
                                value_bool,
                                value_json,
                                source_type="founder_material",
                                source_ref="founder_material",
                                notes=f"model={model};search_per_signal=yes;source=founder_material",
                                confidence=confidence,
                                evidence_text=evidence,
                                evidence_page=evidence_page,
                            )
                            try:
                                db.add_activity(
                                    conn,
                                    "agent_signal",
                                    "company",
                                    company_id,
                                    {
                                        "signal_key": key,
                                        "status": "success",
                                        "source": "founder_material",
                                        "source_ref": "founder_material",
                                        "evidence": evidence,
                                        "confidence": confidence,
                                        "agent_role": agent_profile.get("role"),
                                        "agent_prompt": agent_profile.get("prompt"),
                                        "agent_tools": agent_profile.get("tools"),
                                    },
                                    actor="agent",
                                )
                            except Exception:
                                pass
                            saved += 1
                            continue

            # Step 2: Try company website text if enabled
            if website_text and allow_site:
                est_cost = _estimate_cost(website_text)
                if total_est_cost + est_cost > ai_budget_usd:
                    return saved, f"Estimated AI cost ${total_est_cost + est_cost:.4f} exceeds budget ${ai_budget_usd:.2f}"
                total_est_cost += est_cost
                try:
                    signals = extraction.extract_single_signal_with_openai(
                        website_text,
                        d,
                        model=model,
                        company_name=company["name"],
                        agent_profile=agent_profile,
                    )
                except Exception:
                    signals = {}
                payload = signals.get(key)
                if payload is not None:
                    value_num, value_text, value_bool, value_json, confidence, evidence, evidence_page = extraction.normalize_extracted_signal(
                        payload, d["value_type"]
                    )
                    raw_value = payload.get("value") if isinstance(payload, dict) else payload
                    if raw_value is not None:
                        db.add_signal_value(
                            conn,
                            company_id,
                            key,
                            value_num,
                            value_text,
                            value_bool,
                            value_json,
                            source_type="company_website",
                            source_ref=website_url or company["website"],
                            notes=f"model={model};search_per_signal=yes;source=company_website",
                            confidence=confidence,
                            evidence_text=evidence,
                            evidence_page=evidence_page,
                        )
                        try:
                            db.add_activity(
                                conn,
                                "agent_signal",
                                "company",
                                company_id,
                                {
                                    "signal_key": key,
                                    "status": "success",
                                    "source": "company_website",
                                    "source_ref": website_url or company["website"],
                                    "evidence": evidence,
                                    "confidence": confidence,
                                    "agent_role": agent_profile.get("role"),
                                    "agent_prompt": agent_profile.get("prompt"),
                                    "agent_tools": agent_profile.get("tools"),
                                },
                                actor="agent",
                            )
                        except Exception:
                            pass
                        saved += 1
                        continue

            # Step 2: Web search agent for this signal (only if founder materials had no value)
            if not allow_web:
                try:
                    db.add_activity(
                        conn,
                        "agent_signal",
                        "company",
                        company_id,
                        {
                            "signal_key": key,
                            "status": "failed",
                            "reason": "web_tool_disabled" if allow_web_fallback else "founder_only_no_value",
                            "agent_role": agent_profile.get("role"),
                            "agent_prompt": agent_profile.get("prompt"),
                            "agent_tools": agent_profile.get("tools"),
                        },
                        actor="agent",
                    )
                except Exception:
                    pass
                continue
            urls = web_discovery.serpapi_search(query, num=search_max_pages)
            context_parts = []
            for url in urls:
                try:
                    text = extraction.fetch_website_text(url)
                except Exception:
                    continue
                if text:
                    context_parts.append(f"SOURCE: {url}\n{text}")
            if not context_parts:
                try:
                    db.add_activity(
                        conn,
                        "agent_signal",
                        "company",
                        company_id,
                        {
                            "signal_key": key,
                            "status": "failed",
                            "reason": "no_sources_found",
                            "agent_role": agent_profile.get("role"),
                            "agent_prompt": agent_profile.get("prompt"),
                            "agent_tools": agent_profile.get("tools"),
                        },
                        actor="agent",
                    )
                except Exception:
                    pass
                continue
            context = "\n\n".join(context_parts)
            est_cost = _estimate_cost(context)
            if total_est_cost + est_cost > ai_budget_usd:
                return saved, f"Estimated AI cost ${total_est_cost + est_cost:.4f} exceeds budget ${ai_budget_usd:.2f}"
            total_est_cost += est_cost
            try:
                signals = extraction.extract_single_signal_with_openai(
                    context,
                    d,
                    model=model,
                    company_name=company["name"],
                    agent_profile=agent_profile,
                )
            except Exception:
                continue
            payload = signals.get(key)
            if payload is None:
                try:
                    db.add_activity(
                        conn,
                        "agent_signal",
                        "company",
                        company_id,
                        {
                            "signal_key": key,
                            "status": "failed",
                            "reason": "no_value_extracted",
                            "agent_role": agent_profile.get("role"),
                            "agent_prompt": agent_profile.get("prompt"),
                            "agent_tools": agent_profile.get("tools"),
                        },
                        actor="agent",
                    )
                except Exception:
                    pass
                continue
            value_num, value_text, value_bool, value_json, confidence, evidence, evidence_page = extraction.normalize_extracted_signal(
                payload, d["value_type"]
            )
            raw_value = payload.get("value") if isinstance(payload, dict) else payload
            if raw_value is None:
                try:
                    db.add_activity(
                        conn,
                        "agent_signal",
                        "company",
                        company_id,
                        {
                            "signal_key": key,
                            "status": "failed",
                            "reason": "empty_value",
                            "agent_role": agent_profile.get("role"),
                            "agent_prompt": agent_profile.get("prompt"),
                            "agent_tools": agent_profile.get("tools"),
                        },
                        actor="agent",
                    )
                except Exception:
                    pass
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
                source_ref=urls[0] if urls else "web_search",
                notes=f"model={model};search_per_signal=yes;urls={len(urls)}",
                confidence=confidence,
                evidence_text=evidence,
                evidence_page=evidence_page,
            )
            try:
                db.add_activity(
                    conn,
                    "agent_signal",
                    "company",
                    company_id,
                    {
                        "signal_key": key,
                        "status": "success",
                        "source": "web_search",
                        "source_ref": urls[0] if urls else "web_search",
                        "evidence": evidence,
                        "confidence": confidence,
                        "agent_role": agent_profile.get("role"),
                        "agent_prompt": agent_profile.get("prompt"),
                        "agent_tools": agent_profile.get("tools"),
                    },
                    actor="agent",
                )
            except Exception:
                pass
            saved += 1
    else:
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

    if saved:
        try:
            from app import auto_update

            auto_update.rescore_company(conn, company_id, actor="signal_update")
        except Exception:
            pass
        try:
            db.add_activity(
                conn,
                "ai_scoring",
                "company",
                company_id,
                {"signals_saved": saved, "model": model, "search_per_signal": search_per_signal},
                actor="system",
            )
        except Exception:
            pass

    return saved, None
