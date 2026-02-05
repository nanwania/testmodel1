import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlencode

import requests

OC_BASE = "https://api.opencorporates.com/v0.4"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"


def _parse_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _age_years(inc_date: Optional[str]) -> Optional[float]:
    dt = _parse_date(inc_date)
    if not dt:
        return None
    delta = datetime.utcnow() - dt
    return round(delta.days / 365.25, 2)


def _count_field(obj, key) -> Optional[int]:
    if not obj:
        return None
    value = obj.get(key)
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        for k in ("total_count", "count"):
            if k in value and isinstance(value[k], int):
                return value[k]
    return None


def opencorporates_search(query: str, api_token: str, jurisdiction_code: Optional[str] = None, per_page: int = 5) -> List[dict]:
    params = {"q": query, "api_token": api_token, "per_page": per_page}
    if jurisdiction_code:
        params["jurisdiction_code"] = jurisdiction_code
    url = f"{OC_BASE}/companies/search"
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    companies = data.get("results", {}).get("companies", [])
    return [c.get("company", {}) for c in companies]


def opencorporates_fetch_company(jurisdiction_code: str, company_number: str, api_token: str) -> dict:
    url = f"{OC_BASE}/companies/{jurisdiction_code}/{company_number}"
    resp = requests.get(url, params={"api_token": api_token}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("results", {}).get("company", {})


def opencorporates_signals(company: dict) -> Dict[str, object]:
    status = company.get("current_status") or ""
    inc_date = company.get("incorporation_date")
    age_years = _age_years(inc_date)

    signals = {
        "company_age_years": age_years,
        "oc_is_active": True if "active" in status.lower() else False if status else None,
        "oc_status": status or None,
        "oc_jurisdiction": company.get("jurisdiction_code"),
        "oc_company_number": company.get("company_number"),
        "oc_officers_count": _count_field(company, "officers"),
        "oc_filings_count": _count_field(company, "filings"),
        "oc_incorporation_date": inc_date,
    }
    return signals


def sec_lookup_cik(ticker: str, user_agent: str) -> Optional[str]:
    headers = {"User-Agent": user_agent}
    resp = requests.get(SEC_TICKERS, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    ticker = ticker.upper()
    for _, record in data.items():
        if record.get("ticker", "").upper() == ticker:
            cik = str(record.get("cik_str", "")).zfill(10)
            return cik
    return None


def sec_fetch_submissions(cik: str, user_agent: str) -> dict:
    headers = {"User-Agent": user_agent}
    url = SEC_SUBMISSIONS.format(cik=str(cik).zfill(10))
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def sec_signals(submissions: dict) -> Dict[str, object]:
    filings = submissions.get("filings", {}).get("recent", {})
    dates = filings.get("filingDate", []) or []

    parsed = [_parse_date(d) for d in dates if d]
    parsed = [d for d in parsed if d]
    latest = max(parsed).strftime("%Y-%m-%d") if parsed else None

    cutoff = datetime.utcnow() - timedelta(days=365)
    count_12m = sum(1 for d in parsed if d >= cutoff) if parsed else 0

    signals = {
        "sec_is_public": True,
        "sec_latest_filing_date": latest,
        "sec_filing_count_12m": count_12m,
        "sec_cik": submissions.get("cik") or None,
    }
    return signals


def gdelt_timeline_vol_raw(query: str, days: int = 30) -> Tuple[Optional[int], List[dict], str]:
    params = {
        "query": query,
        "mode": "timelinevolraw",
        "timespan": f"{days}d",
        "format": "json",
    }
    url = f"{GDELT_DOC}?{urlencode(params)}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    timeline = data.get("timeline") or data.get("data") or []
    total = 0
    for row in timeline:
        for key in ("value", "count", "volume", "numarticles"):
            if key in row:
                try:
                    total += int(float(row[key]))
                    break
                except (TypeError, ValueError):
                    pass
    return total if timeline else None, timeline, url


def gdelt_signals(total: Optional[int], days: int) -> Dict[str, object]:
    if total is None:
        return {"news_volume": None, "news_daily_avg": None, "news_timespan_days": days}
    daily_avg = round(total / max(days, 1), 2)
    return {
        "news_volume": total,
        "news_daily_avg": daily_avg,
        "news_timespan_days": days,
    }

