import os
from typing import List

from app import db
from app import extraction
from app import web_discovery


INTENSITY_LIMITS = {
    "light": 5,
    "medium": 15,
    "heavy": 40,
}


def _dedupe(urls: List[str]) -> List[str]:
    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def discover_urls(company, mode: str, allowlist: List[str], intensity: str, news_domains: List[str]) -> List[str]:
    urls = []
    if company["website"]:
        urls.append(company["website"])

    if mode == "closed":
        urls.extend([u for u in allowlist if u])
        if news_domains:
            urls.extend(web_discovery.serpapi_search_domains(company["name"], news_domains, num_per_domain=INTENSITY_LIMITS.get(intensity, 5)))
        return _dedupe(urls)

    # open search
    query = f"{company['name']} competitors peers market sector news"
    results = web_discovery.serpapi_search(query, num=INTENSITY_LIMITS.get(intensity, 5))
    urls.extend(results)
    if news_domains:
        urls.extend(web_discovery.serpapi_search_domains(company["name"], news_domains, num_per_domain=INTENSITY_LIMITS.get(intensity, 5)))
    return _dedupe(urls)


def crawl_and_extract(conn, company_id: int, mode: str, intensity: str, allowlist: List[str], news_domains: List[str]):
    company = db.get_company(conn, company_id)
    if not company:
        return 0, "Company not found"

    max_pages = INTENSITY_LIMITS.get(intensity, 5)
    urls = discover_urls(company, mode, allowlist, intensity, news_domains)[:max_pages]
    if not urls:
        return 0, "No URLs discovered"

    defs = [dict(d) for d in db.list_signal_definitions(conn)]
    saved = 0
    all_pages = []
    for url in urls:
        try:
            # search-only: fetch a single page per result, no site crawl
            text = extraction.fetch_website_text(url)
            pages = [url]
        except Exception:
            continue
        if not text:
            continue
        signals = extraction.extract_signals_with_openai(text, defs, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        for key, payload in signals.items():
            value_num, value_text, value_bool, value_json, confidence, evidence, evidence_page = extraction.normalize_extracted_signal(
                payload, next((d["value_type"] for d in defs if d["key"] == key), "text")
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
                source_type="crawl",
                source_ref=url,
                notes=f"mode={mode};intensity={intensity}",
                confidence=confidence,
                evidence_text=evidence,
                evidence_page=evidence_page,
            )
            saved += 1
        all_pages.extend(pages)

    db.add_activity(
        conn,
        "crawl_and_extract",
        "company",
        company_id,
        {"mode": mode, "intensity": intensity, "url_count": len(urls), "signals_saved": saved},
        actor="system",
    )
    return saved, None
