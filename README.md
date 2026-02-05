# AI Sourcing MVP

This is a beginner-friendly MVP for an auditable startup sourcing tool.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
export OPENCORPORATES_API_TOKEN=your_token_here
export SEC_USER_AGENT="Your Name youremail@example.com"
streamlit run app.py
```

## What you get

- Companies list
- Company detail + ratings (1–5)
- Signals Manager (definitions + values + history)
- Criteria Manager (versioned criteria)
- Transparent scoring with breakdown
- Confidence-weighted, normalized scoring
- Optimization (AI-guided weight suggestions)
- AI signal extraction from websites
- AI extraction confidence + evidence highlights
- Multi-page crawl for richer evidence
- Founder materials RAG (PDF/TXT/EML/DOCX/PPTX)
- Public data ingestion: OpenCorporates, SEC EDGAR, GDELT
- Benchmarking (peer percentiles)
- Risk detection (rule-based red flags)
- CSV upload

The data lives in `data/app.db` (SQLite).

AI extraction requires `OPENAI_API_KEY` and an optional `OPENAI_MODEL` env var.
Public data ingestion uses OpenCorporates (API token) and SEC EDGAR (User-Agent required).

## Next steps

- Expand crawling coverage and extraction prompts
- Add Affinity sync
- Move to Postgres for scale
