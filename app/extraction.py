import json
import os
import re
from typing import Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import requests

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_TIMEOUT = 20
MAX_CHARS = 12000
MAX_EVIDENCE_WORDS = 20
MAX_URLS_STORED = 20


def fetch_website_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (AI Sourcing MVP)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    if trafilatura:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        if extracted:
            return extracted.strip()

    # fallback: remove tags
    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_links(html: str, base_url: str) -> List[str]:
    links = []
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag.get("href")
            if href:
                links.append(urljoin(base_url, href))
    else:
        # fallback: regex-based href extraction
        for match in re.findall(r'href=[\"\\\'](.*?)[\"\\\']', html, flags=re.IGNORECASE):
            links.append(urljoin(base_url, match))
    return links


def _is_valid_link(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    if lower.startswith("mailto:") or lower.startswith("tel:"):
        return False
    if lower.endswith((".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip")):
        return False
    return lower.startswith("http://") or lower.startswith("https://")


def _same_domain(url_a: str, url_b: str) -> bool:
    try:
        return urlparse(url_a).netloc == urlparse(url_b).netloc
    except Exception:
        return False


def _score_link(url: str) -> int:
    # Higher score = earlier crawl
    keywords = [
        "about",
        "team",
        "company",
        "mission",
        "product",
        "pricing",
        "careers",
        "jobs",
        "press",
        "news",
        "blog",
        "faq",
        "customers",
        "case-study",
        "case-studies",
    ]
    score = 0
    path = urlparse(url).path.lower()
    for kw in keywords:
        if kw in path:
            score += 1
    return score


def crawl_site_text(
    base_url: str,
    max_pages: int = 5,
    same_domain: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[str, List[str]]:
    visited = []
    queue = [base_url]
    texts = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        if not _is_valid_link(url):
            continue
        if same_domain and not _same_domain(base_url, url):
            continue

        try:
            headers = {"User-Agent": "Mozilla/5.0 (AI Sourcing MVP)"}
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            html = resp.text
        except Exception:
            continue

        visited.append(url)

        if trafilatura:
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
            if extracted:
                texts.append(extracted.strip())
        else:
            text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\\s+", " ", text)
            text = text.strip()
            if text:
                texts.append(text)

        links = _extract_links(html, url)
        links = [l for l in links if _is_valid_link(l)]
        if same_domain:
            links = [l for l in links if _same_domain(base_url, l)]
        links = sorted(set(links), key=_score_link, reverse=True)

        for link in links:
            if link not in visited and link not in queue and len(queue) + len(visited) < max_pages * 3:
                queue.append(link)

    combined = "\n\n".join(texts)
    return combined[:MAX_CHARS], visited[:MAX_URLS_STORED]


def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _clip_evidence(text: str) -> str:
    if not text:
        return ""
    words = text.strip().split()
    if len(words) <= MAX_EVIDENCE_WORDS:
        return text.strip()
    return " ".join(words[:MAX_EVIDENCE_WORDS])


def extract_signals_with_openai(
    text: str,
    signal_defs,
    model: str = DEFAULT_MODEL,
) -> Dict[str, object]:
    signal_defs = [dict(d) for d in signal_defs]
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()

    allowed = [d["key"] for d in signal_defs if not d["disabled"]]
    types = {d["key"]: d["value_type"] for d in signal_defs}
    catalog_lines = []
    for d in signal_defs:
        if d["disabled"]:
            continue
        name = d.get("name") or ""
        desc = d.get("description") or ""
        prompt = d.get("automation_prompt") or ""
        extra = f" | {prompt}" if prompt else ""
        catalog_lines.append(f"{d['key']}: {name} - {desc}{extra} ({d['value_type']})")
    catalog = "\n".join(catalog_lines)

    system_msg = (
        "You extract structured startup signals from raw website text. "
        "Return a JSON object with only the requested keys. "
        "Each key maps to an object with fields: value, confidence, evidence, page. "
        "confidence is between 0 and 1. evidence is a short quote from the text (<=20 words). "
        "page is an integer if evidence comes from a [Page N] tag; otherwise null. "
        "Use numbers for numeric values, booleans for true/false, and strings for text. "
        "If a value is unknown, set value to null and confidence to 0."
    )

    short_text = _truncate(text)
    user_msg = (
        "Extract the following signals from the text and return JSON only. "
        f"Keys: {allowed}. "
        f"Value types: {types}. "
        f"Signals:\n{catalog}\n\nText: {short_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    if not isinstance(data, dict):
        raise RuntimeError("Model did not return a JSON object")

    # Normalize to {key: {value, confidence, evidence}}
    return normalize_extracted_signals(data, allowed)


def extract_single_signal_with_openai(
    text: str,
    signal_def: Dict[str, object],
    model: str = DEFAULT_MODEL,
    company_name: str | None = None,
    agent_profile: Dict[str, str] | None = None,
) -> Dict[str, Dict[str, object]]:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()

    d = dict(signal_def)
    key = d["key"]
    name = d.get("name") or key
    desc = d.get("description") or ""
    value_type = d.get("value_type") or "text"
    prompt = d.get("automation_prompt") or ""

    profile = agent_profile or {}
    role = profile.get("role") or f"{name} Signal Agent"
    goal = profile.get("goal") or f"Extract the {name} signal for a startup from provided text."
    backstory = profile.get("backstory") or "You are precise, evidence-driven, and only return values supported by the text."
    extra_instructions = profile.get("prompt") or ""

    system_msg = (
        f"You are {role}. Goal: {goal}. Backstory: {backstory}. "
        "Return a JSON object with only the requested key. "
        "The value must be an object with fields: value, confidence, evidence, page. "
        "confidence is between 0 and 1. evidence is a short quote from the text (<=20 words). "
        "page is an integer if evidence comes from a [Page N] tag; otherwise null. "
        "Use numbers for numeric values, booleans for true/false, and strings for text. "
        "If a value is unknown, set value to null and confidence to 0."
    )

    short_text = _truncate(text)
    company_line = f"Company: {company_name}." if company_name else ""
    signal_line = f"Signal: {key} — {name}. Description: {desc}. Type: {value_type}."
    custom = f"Custom instructions: {extra_instructions or prompt}." if (extra_instructions or prompt) else ""
    user_msg = (
        f"{company_line} {signal_line} {custom} "
        "Extract only this signal and return JSON only. "
        f"Text: {short_text}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    if not isinstance(data, dict):
        raise RuntimeError("Model did not return a JSON object")

    return normalize_extracted_signals(data, [key])


def normalize_extracted_signals(data: Dict[str, object], allowed_keys) -> Dict[str, Dict[str, object]]:
    normalized = {}
    for key in allowed_keys:
        raw = data.get(key)
        if isinstance(raw, dict):
            value = raw.get("value")
            confidence = raw.get("confidence")
            evidence = raw.get("evidence")
            page = raw.get("page")
        else:
            value = raw
            confidence = None
            evidence = None
            page = None

        if confidence is None:
            confidence = 0.0 if value is None else 0.5
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        if evidence:
            evidence = _clip_evidence(str(evidence))

        normalized[key] = {
            "value": value,
            "confidence": confidence,
            "evidence": evidence,
            "page": page,
        }
    return normalized


def normalize_value(value, value_type):
    if value is None:
        return None, None, None, None
    if value_type == "number":
        try:
            return float(value), None, None, None
        except (TypeError, ValueError):
            return None, None, None, None
    if value_type == "bool":
        if isinstance(value, bool):
            return None, None, int(value), None
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ("true", "yes", "1"):
                return None, None, 1, None
            if val in ("false", "no", "0"):
                return None, None, 0, None
        if isinstance(value, (int, float)):
            return None, None, 1 if value else 0, None
        return None, None, None, None
    if value_type == "json":
        try:
            return None, None, None, json.dumps(value)
        except TypeError:
            return None, None, None, None
    # default text
    return None, str(value), None, None


def normalize_extracted_signal(item, value_type) -> Tuple[float, str, int, str, float, str, int]:
    if isinstance(item, dict):
        value = item.get("value")
        confidence = item.get("confidence")
        evidence = item.get("evidence")
        page = item.get("page")
    else:
        value = item
        confidence = None
        evidence = None
        page = None

    value_num, value_text, value_bool, value_json = normalize_value(value, value_type)
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None

    if evidence:
        evidence = _clip_evidence(str(evidence))

    try:
        page = int(page) if page is not None else None
    except (TypeError, ValueError):
        page = None

    return value_num, value_text, value_bool, value_json, confidence, evidence, page
