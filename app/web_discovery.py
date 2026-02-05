import os
import requests

SERP_API_URL = "https://serpapi.com/search.json"


def serpapi_search(query: str, num: int = 10, gl: str = "us", hl: str = "en"):
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set")

    params = {
        "q": query,
        "num": num,
        "gl": gl,
        "hl": hl,
        "api_key": api_key,
        "engine": "google",
    }
    resp = requests.get(SERP_API_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("organic_results", [])
    urls = []
    for r in results:
        link = r.get("link")
        if link:
            urls.append(link)
    return urls


def serpapi_search_domains(company_name: str, domains, num_per_domain: int = 5):
    urls = []
    for domain in domains:
        if not domain:
            continue
        query = f"site:{domain} {company_name}"
        try:
            urls.extend(serpapi_search(query, num=num_per_domain))
        except Exception:
            continue
    return urls
