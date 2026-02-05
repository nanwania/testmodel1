import html
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from app import db
from app import scoring
from app import optimization
from app import extraction
from app import rag
from app import public_data
from app import benchmark
from app import risk_detector
from app import criteria_import
from app import criteria_ai
from app import simple_benchmark
from app import auto_update
from app import discovery
from app import agent_orchestrator


def _utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def _highlight_evidence(text, evidences, max_chars=4000):
    if not text:
        return ""
    preview = html.escape(text[:max_chars])
    for ev in evidences:
        if not ev:
            continue
        ev_esc = html.escape(ev)
        preview = preview.replace(ev_esc, f"<mark>{ev_esc}</mark>")
    return preview


def _merge_preview(existing, incoming):
    fields = ["name", "website", "description", "industry", "location", "founder_names"]
    to_fill = []
    for f in fields:
        current = existing.get(f) if existing else None
        new_val = incoming.get(f)
        if (current is None or str(current).strip() == "") and new_val:
            to_fill.append(f)
    return to_fill


def _render_run_tooltips():
    st.sidebar.subheader("Run Steps")
    st.sidebar.caption("Hover each item to see a plain-English explanation.")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.95rem; line-height: 1.6;">
          <div><span title="We discover relevant pages using the selected crawl mode (open or closed) and fetch text from those pages.">Auto-crawl</span></div>
          <div><span title="We use AI to extract signals from the fetched text and save the source URL and evidence quote.">AI extraction</span></div>
          <div><span title="We compute the score using your criteria, weights, and confidence values.">Scoring</span></div>
          <div><span title="We evaluate rule-based, anomaly, and text-based risks and store all flagged evidence.">Risk scan</span></div>
          <div><span title="Whenever new signals are saved, we re-score and re-run risks for that company.">Auto-rescore on new data</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def seed_defaults(conn):
    def ensure_signal_definitions(defs):
        existing = {d["key"] for d in db.list_signal_definitions(conn)}
        missing = [d for d in defs if d["key"] not in existing]
        if missing:
            db.upsert_signal_definitions(conn, missing, actor="system")

    base_defs = [
        {
            "key": "founder_exits",
            "name": "Founder exits",
            "description": "Number of prior exits across founding team",
            "value_type": "number",
            "unit": "count",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "founder_experience_years",
            "name": "Founder experience (years)",
            "description": "Total years of relevant experience",
            "value_type": "number",
            "unit": "years",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "traction_metric",
            "name": "Traction metric",
            "description": "Numeric traction signal (users, revenue, etc.)",
            "value_type": "number",
            "unit": "count",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "product_stage",
            "name": "Product stage",
            "description": "Idea, MVP, Beta, Live",
            "value_type": "text",
            "unit": None,
            "allowed_range_json": None,
            "disabled": False,
        },
        {
            "key": "market_size_est",
            "name": "Market size estimate",
            "description": "Estimated TAM in USD",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
    ]
    ensure_signal_definitions(base_defs)

    public_defs = [
        {
            "key": "company_age_years",
            "name": "Company age (years)",
            "description": "Derived from incorporation date",
            "value_type": "number",
            "unit": "years",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "news_volume",
            "name": "News volume",
            "description": "News volume from GDELT",
            "value_type": "number",
            "unit": "count",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "news_daily_avg",
            "name": "News daily average",
            "description": "News volume divided by days",
            "value_type": "number",
            "unit": "count",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "news_timespan_days",
            "name": "News timespan (days)",
            "description": "Timespan used for news query",
            "value_type": "number",
            "unit": "days",
            "allowed_range_json": json.dumps({"min": 1}),
            "disabled": False,
        },
        {
            "key": "gross_margin_current",
            "name": "Current gross margin",
            "description": "Current gross margin percentage",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": 0, "max": 100}),
            "disabled": False,
        },
        {
            "key": "gross_margin_target",
            "name": "Target gross margin",
            "description": "Target gross margin percentage",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": 0, "max": 100}),
            "disabled": False,
        },
        {
            "key": "cash_needed_breakeven",
            "name": "Cash needed until breakeven",
            "description": "Estimated cash required to reach breakeven",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "cost_foak",
            "name": "Cost of FOAK",
            "description": "First-of-a-kind (FOAK) cost estimate",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "cost_noak",
            "name": "Cost of NOAK",
            "description": "Nth-of-a-kind (NOAK) cost estimate",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "scale_factor_to_noak",
            "name": "Scale factor to NOAK",
            "description": "Scale-up factor from current to NOAK",
            "value_type": "number",
            "unit": "x",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "co2_savings_vs_incumbent",
            "name": "CO2 savings vs incumbent",
            "description": "Estimated CO2 savings compared to incumbent solution",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": 0, "max": 100}),
            "disabled": False,
        },
        {
            "key": "first_revenue_date",
            "name": "First revenue date",
            "description": "Date of first recorded revenue",
            "value_type": "text",
            "unit": None,
            "allowed_range_json": None,
            "disabled": False,
        },
        {
            "key": "trl_level",
            "name": "TRL level",
            "description": "Technology readiness level (1-9)",
            "value_type": "number",
            "unit": "level",
            "allowed_range_json": json.dumps({"min": 1, "max": 9}),
            "disabled": False,
        },
        {
            "key": "first_revenue_year",
            "name": "Year of first revenue",
            "description": "Calendar year when first revenue was recorded",
            "value_type": "number",
            "unit": "year",
            "allowed_range_json": json.dumps({"min": 1990, "max": 2100}),
            "disabled": False,
        },
        {
            "key": "five_year_cagr_after_first_revenue",
            "name": "5-year CAGR after first revenue",
            "description": "Compound annual growth rate for 5 years after first revenue",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": -100, "max": 1000}),
            "disabled": False,
        },
        {
            "key": "existing_investor_quality",
            "name": "Existing investor quality",
            "description": "Assessment score of existing investor quality",
            "value_type": "number",
            "unit": "score",
            "allowed_range_json": json.dumps({"min": 0, "max": 10}),
            "disabled": False,
        },
    ]
    ensure_signal_definitions(public_defs)

    founder_material_defs = [
        {
            "key": "mrr",
            "name": "MRR",
            "description": "Monthly recurring revenue",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "arr",
            "name": "ARR",
            "description": "Annual recurring revenue",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "revenue_growth_rate",
            "name": "Revenue growth rate",
            "description": "Growth rate (percent)",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "churn_rate",
            "name": "Churn rate",
            "description": "Churn rate (percent)",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "gross_margin",
            "name": "Gross margin",
            "description": "Gross margin (percent)",
            "value_type": "number",
            "unit": "percent",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "burn_rate",
            "name": "Burn rate",
            "description": "Monthly burn rate",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "runway_months",
            "name": "Runway (months)",
            "description": "Cash runway in months",
            "value_type": "number",
            "unit": "months",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "cac",
            "name": "CAC",
            "description": "Customer acquisition cost",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "ltv",
            "name": "LTV",
            "description": "Customer lifetime value",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "payback_months",
            "name": "Payback (months)",
            "description": "CAC payback period",
            "value_type": "number",
            "unit": "months",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "funding_raised",
            "name": "Funding raised",
            "description": "Total funding raised",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
        {
            "key": "valuation",
            "name": "Valuation",
            "description": "Company valuation",
            "value_type": "number",
            "unit": "usd",
            "allowed_range_json": json.dumps({"min": 0}),
            "disabled": False,
        },
    ]
    ensure_signal_definitions(founder_material_defs)

    composite_defs = [
        {
            "key": "growth_momentum",
            "name": "Growth momentum",
            "description": "Composite of news change, hiring velocity, and launches",
            "value_type": "number",
            "unit": "score",
            "allowed_range_json": json.dumps({"min": 0, "max": 1}),
            "disabled": False,
        },
        {
            "key": "market_validation",
            "name": "Market validation",
            "description": "Composite of revenue growth, customer count, and press coverage",
            "value_type": "number",
            "unit": "score",
            "allowed_range_json": json.dumps({"min": 0, "max": 1}),
            "disabled": False,
        },
        {
            "key": "team_quality",
            "name": "Team quality",
            "description": "Composite of founder experience, advisor prestige, and hiring success",
            "value_type": "number",
            "unit": "score",
            "allowed_range_json": json.dumps({"min": 0, "max": 1}),
            "disabled": False,
        },
    ]
    ensure_signal_definitions(composite_defs)


def _auto_process_company(conn, company_id, website_url=None):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ai_budget = float(os.getenv("AI_BUDGET_USD", "0.2"))

    crawl_settings = db.get_crawl_settings(conn)
    mode = crawl_settings["mode"] if crawl_settings else "closed"
    intensity = crawl_settings["intensity"] if crawl_settings else "light"
    agent_mode = False
    if crawl_settings and "agent_mode" in crawl_settings.keys():
        agent_mode = bool(crawl_settings["agent_mode"])
    allowlist = []
    news_domains = []
    if crawl_settings and crawl_settings["allowlist_json"]:
        try:
            allowlist = json.loads(crawl_settings["allowlist_json"])
        except json.JSONDecodeError:
            allowlist = []
    if crawl_settings and crawl_settings["news_domains_json"]:
        try:
            news_domains = json.loads(crawl_settings["news_domains_json"])
        except json.JSONDecodeError:
            news_domains = []

    try:
        if agent_mode:
            orchestrator = agent_orchestrator.AgentOrchestrator(conn, mode, intensity, allowlist, news_domains)
            orchestrator.run(company_id)
            return None, None
        discovery.crawl_and_extract(conn, company_id, mode=mode, intensity=intensity, allowlist=allowlist, news_domains=news_domains)
    except Exception:
        pass

    saved, err = criteria_ai.run_ai_scoring(
        conn,
        company_id,
        model=model,
        ai_budget_usd=ai_budget,
        include_website=True,
        include_founder_materials=True,
        website_url=website_url,
        max_pages=5,
        same_domain=True,
        top_k=8,
    )

    version_id = db.get_active_criteria_version_id(conn)
    scoring.run_scoring_for_company(conn, version_id, company_id)

    engine = risk_detector.RiskDetectionEngine(conn, model=model, ai_budget_usd=ai_budget, enable_llm=True)
    engine.run_scan(company_ids=[company_id])

    return saved, err

    # Seed criteria if empty
    version_id = db.get_active_criteria_version_id(conn)
    criteria_rows = db.list_criteria(conn, version_id)
    if len(criteria_rows) == 0:
        now = _utc_now()
        cur = conn.cursor()
        seed_criteria = [
            {
                "name": "Founder exits",
                "description": "At least 1 prior exit",
                "signal_key": "founder_exits",
                "weight": 0.12,
                "enabled": 1,
                "scoring_method": "binary",
                "params_json": json.dumps({"op": "gte", "threshold": 1}),
                "missing_policy": "zero",
            },
            {
                "name": "Founder experience",
                "description": "10+ years experience",
                "signal_key": "founder_experience_years",
                "weight": 0.12,
                "enabled": 1,
                "scoring_method": "binary",
                "params_json": json.dumps({"op": "gte", "threshold": 10}),
                "missing_policy": "zero",
            },
            {
                "name": "Traction signal",
                "description": "Traction >= 1000",
                "signal_key": "traction_metric",
                "weight": 0.12,
                "enabled": 1,
                "scoring_method": "binary",
                "params_json": json.dumps({"op": "gte", "threshold": 1000}),
                "missing_policy": "zero",
            },
            {
                "name": "Product stage",
                "description": "At least MVP",
                "signal_key": "product_stage",
                "weight": 0.10,
                "enabled": 1,
                "scoring_method": "binary",
                "params_json": json.dumps({"op": "contains", "threshold": "mvp"}),
                "missing_policy": "zero",
            },
            {
                "name": "Market size",
                "description": "TAM >= $1B",
                "signal_key": "market_size_est",
                "weight": 0.10,
                "enabled": 1,
                "scoring_method": "binary",
                "params_json": json.dumps({"op": "gte", "threshold": 1000000000}),
                "missing_policy": "zero",
            },
        ]
        for c in seed_criteria:
            cur.execute(
                """
                INSERT INTO criteria (
                    criteria_set_version_id, name, description, signal_key, weight,
                    enabled, scoring_method, params_json, missing_policy, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    c["name"],
                    c["description"],
                    c["signal_key"],
                    c["weight"],
                    c["enabled"],
                    c["scoring_method"],
                    c["params_json"],
                    c["missing_policy"],
                    now,
                    "system",
                ),
            )
        conn.commit()


def page_dashboard(conn):
    st.title("AI Sourcing MVP")
    companies = db.list_companies(conn)
    st.metric("Companies", len(companies))
    latest_scores = db.list_latest_score_summary(conn)
    if latest_scores:
        top = [r for r in latest_scores if r["total_score"] is not None][:5]
        if top:
            st.subheader("Top Companies")
            st.dataframe(pd.DataFrame(top)[["name", "total_score", "website"]])


def page_simple(conn):
    st.markdown(
        """
        <style>
        .card {
          border: 1px solid #e6e6e6;
          border-radius: 12px;
          padding: 16px;
          margin-bottom: 16px;
          background: #fafafa;
        }
        .card h3 {
          margin: 0 0 8px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Sourcing Workflow")
    st.caption("A simple, 4-step flow: add company → run agents on founder deck → run agents on web → view results.")

    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="simple_model")
    ai_budget = st.number_input("AI budget per company (USD)", min_value=0.0, max_value=5.0, value=0.2, step=0.05, key="simple_budget")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1) Add Company + Optional Founder Deck")
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Company name", key="simple_name")
        website = st.text_input("Website", key="simple_website")
        industry = st.text_input("Industry", key="simple_industry")
    with col2:
        location = st.text_input("Location", key="simple_location")
        founders = st.text_input("Founder names", key="simple_founders")
        description = st.text_area("Short description", key="simple_description")

    files = st.file_uploader(
        "Upload founder deck/materials (PDF, PPTX, DOCX, TXT, EML)",
        type=["pdf", "pptx", "docx", "txt", "eml"],
        accept_multiple_files=True,
        key="simple_founder_files",
    )
    merge_duplicates = st.checkbox("Merge if company already exists by domain", value=True, key="simple_merge")

    if st.button("Create / Add company", key="simple_create"):
        if not name:
            st.error("Company name is required.")
        else:
            company_id, created = db.add_company(
                conn,
                name=name,
                website=website,
                description=description,
                industry=industry,
                location=location,
                founder_names=founders,
                source="simple",
            )
            existing = db.get_company(conn, company_id)
            if created:
                db.add_activity(conn, "company_added", "company", company_id, {"source": "simple"}, actor="user")
                st.success(f"Company created: {existing['name']}")
            else:
                preview_fields = _merge_preview(
                    dict(existing),
                    {
                        "name": name,
                        "website": website,
                        "description": description,
                        "industry": industry,
                        "location": location,
                        "founder_names": founders,
                    },
                )
                if merge_duplicates:
                    did_merge = db.merge_company_fields(
                        conn,
                        company_id,
                        name=name,
                        website=website,
                        description=description,
                        industry=industry,
                        location=location,
                        founder_names=founders,
                    )
                    if did_merge and preview_fields:
                        st.warning(f"Duplicate domain. Merged into {existing['name']}. Filled: {', '.join(preview_fields)}")
                    else:
                        st.warning(f"Duplicate domain. No empty fields to fill for {existing['name']}.")
                else:
                    st.warning(f"Company already exists (domain match): {existing['name']}")

            if files:
                processed = 0
                for file_obj in files:
                    doc_type = rag.detect_doc_type(file_obj.name)
                    if doc_type == "unknown":
                        st.warning(f"Unsupported file type: {file_obj.name}")
                        continue
                    try:
                        doc_id, pages, chunks = rag.store_document(
                            conn,
                            file_obj,
                            file_obj.name,
                            doc_type=doc_type,
                            source_type="founder_material",
                            is_global=False,
                            company_id=company_id,
                        )
                        processed += 1
                    except Exception as exc:
                        st.error(f"Failed to process {file_obj.name}: {exc}")
                if processed:
                    st.success(f"Processed {processed} founder materials.")

            st.session_state["simple_company_id"] = company_id
    st.markdown("</div>", unsafe_allow_html=True)

    companies = db.list_companies(conn)
    if not companies:
        st.info("Add a company to continue.")
        return
    default_company_id = st.session_state.get("simple_company_id") or companies[0]["id"]
    company_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
    default_label = next((k for k, v in company_map.items() if v == default_company_id), list(company_map.keys())[0])
    selected_label = st.selectbox("Active company", list(company_map.keys()), index=list(company_map.keys()).index(default_label), key="simple_company_select")
    company_id = company_map[selected_label]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2) Run Agents on Founder Deck")
    if st.button("Run founder-deck agents", key="simple_founder_agents"):
        saved, err = criteria_ai.run_ai_scoring(
            conn,
            company_id,
            model=model,
            ai_budget_usd=ai_budget,
            include_website=False,
            include_founder_materials=True,
            search_per_signal=True,
            search_max_pages=0,
            allow_web_fallback=False,
        )
        if err:
            st.error(err)
        else:
            st.success(f"Saved {saved} signals from founder materials.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("3) Run Agents on Web Sources")
    search_pages = st.slider("Search pages per signal", min_value=1, max_value=5, value=3, key="simple_search_pages")
    if st.button("Run web agents", key="simple_web_agents"):
        saved, err = criteria_ai.run_ai_scoring(
            conn,
            company_id,
            model=model,
            ai_budget_usd=ai_budget,
            include_website=False,
            include_founder_materials=False,
            search_per_signal=True,
            search_max_pages=search_pages,
            allow_web_fallback=True,
        )
        if err:
            st.error(err)
        else:
            st.success(f"Saved {saved} signals from web sources.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("4) View Numbers Agents Found")
    signals = db.list_latest_signal_values(conn, company_id)
    if signals:
        st.dataframe(
            pd.DataFrame(signals)[
                [
                    "signal_key",
                    "value_num",
                    "value_text",
                    "confidence",
                    "evidence_text",
                    "source_type",
                    "source_ref",
                    "observed_at",
                ]
            ],
            use_container_width=True,
        )
    else:
        st.info("No signals yet.")
    score_item = db.list_latest_score_item(conn, company_id)
    if score_item:
        st.metric("Latest Score", round(score_item["normalized_total"] or score_item["total_score"] or 0.0, 4))
    st.markdown("</div>", unsafe_allow_html=True)


def page_companies(conn):
    st.header("Companies")
    if st.button("Run scoring for all companies"):
        version_id = db.get_active_criteria_version_id(conn)
        scoring.run_scoring(conn, version_id)
        st.success("Scoring complete")

    tabs = st.tabs(["Active Companies", "Archived Companies"])

    with tabs[0]:
        st.subheader("Archive Company")
        companies = db.list_companies(conn)
        if companies:
            delete_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
            delete_selection = st.selectbox("Select company to archive", list(delete_map.keys()), key="delete_company_select")
            delete_confirm = st.checkbox("I understand this will archive the company and hide it from lists.", key="delete_company_confirm")
            if st.button("Archive company", key="delete_company_btn"):
                if not delete_confirm:
                    st.warning("Please confirm archiving.")
                else:
                    company_id = delete_map[delete_selection]
                    db.delete_company(conn, company_id, actor="user")
                    db.add_activity(conn, "company_archived", "company", company_id, {"source": "user_archive"}, actor="user")
                    st.success("Company archived.")
        else:
            st.info("No companies to archive.")

    with tabs[1]:
        st.subheader("Archived Companies")
        archived = [c for c in db.list_companies(conn, include_deleted=True) if c["deleted_at"] is not None]
        if archived:
            arch_map = {f"{c['name']} (#{c['id']})": c["id"] for c in archived}
            restore_selection = st.selectbox("Select company to restore", list(arch_map.keys()), key="restore_company_select")
            if st.button("Restore company", key="restore_company_btn"):
                company_id = arch_map[restore_selection]
                db.restore_company(conn, company_id, actor="user")
                db.add_activity(conn, "company_restored", "company", company_id, {"source": "user_restore"}, actor="user")
                st.success("Company restored.")
            st.dataframe(pd.DataFrame(archived)[["id", "name", "website", "industry", "location", "deleted_at", "deleted_by"]])
        else:
            st.info("No archived companies.")

    rows = db.list_latest_score_summary(conn)
    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        cols = ["id", "name", "website", "industry", "location", "total_score", "scored_at"]
        available = [c for c in cols if c in df.columns]
        if available:
            st.dataframe(df[available])
        else:
            st.dataframe(df)
    else:
        st.info("No companies yet. Upload a CSV in Upload/Crawl.")


def page_company_detail(conn):
    st.header("Company Detail")
    companies = db.list_companies(conn)
    if not companies:
        st.info("No companies yet.")
        return

    company_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
    selection = st.selectbox("Select company", list(company_map.keys()))
    company_id = company_map[selection]
    company = db.get_company(conn, company_id)

    st.subheader(company["name"])
    st.write(company["description"] or "No description")
    st.write(f"Website: {company['website'] or 'N/A'}")
    st.write(f"Industry: {company['industry'] or 'N/A'}")
    st.write(f"Location: {company['location'] or 'N/A'}")
    st.write(f"Founders: {company['founder_names'] or 'N/A'}")

    st.subheader("Archive This Company")
    delete_confirm = st.checkbox("I understand this will archive this company and hide it from lists.", key="delete_company_detail_confirm")
    if st.button("Archive this company", key="delete_company_detail_btn"):
        if not delete_confirm:
            st.warning("Please confirm deletion.")
        else:
            db.delete_company(conn, company_id, actor="user")
            db.add_activity(conn, "company_archived", "company", company_id, {"source": "detail_archive"}, actor="user")
            st.success("Company archived. Please refresh the page.")
            return

    rating = db.list_latest_rating(conn, company_id)
    current_rating = rating["rating_int"] if rating else 3
    new_rating = st.slider("Rating (1–5)", 1, 5, int(current_rating))
    if st.button("Save rating"):
        db.add_rating(conn, company_id, new_rating)
        st.success("Rating saved")

    st.subheader("Agent Run Status")
    agent_events = {
        "agent_discovery": "Discovery",
        "agent_extraction": "Extraction",
        "agent_criteria_ai": "AI Criteria",
        "agent_scoring": "Scoring",
        "agent_risk": "Risk Scan",
    }
    logs = db.list_activity_for_company(conn, company_id, limit=200)
    if logs:
        logs = [dict(r) for r in logs]
        agent_logs = [l for l in logs if l.get("event_type") in agent_events]
    else:
        agent_logs = []

    if agent_logs:
        rows = []
        for event_type, label in agent_events.items():
            latest = next((l for l in agent_logs if l.get("event_type") == event_type), None)
            if latest:
                details = latest.get("details_json")
                rows.append(
                    {
                        "Step": label,
                        "Status": "Completed",
                        "Last Run": latest.get("created_at"),
                        "Details": details,
                    }
                )
            else:
                rows.append({"Step": label, "Status": "Not run", "Last Run": None, "Details": None})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No agent runs yet for this company.")

    st.subheader("Per-Signal Agent Status")
    if logs:
        signal_logs = [l for l in logs if l.get("event_type") == "agent_signal"]
    else:
        signal_logs = []
    if signal_logs:
        rows = []
        for entry in signal_logs[:200]:
            details = entry.get("details_json")
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {"raw": details}
            rows.append(
                {
                    "Signal": (details or {}).get("signal_key"),
                    "Status": (details or {}).get("status"),
                    "Source": (details or {}).get("source") or (details or {}).get("source_ref"),
                    "Found in Founder Materials": (details or {}).get("source") == "founder_material",
                    "Evidence": (details or {}).get("evidence"),
                    "When": entry.get("created_at"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No per-signal agent logs yet for this company.")

    st.subheader("Latest Signals")
    signals = db.list_latest_signal_values(conn, company_id)
    if signals:
        st.dataframe(
            pd.DataFrame(signals)[
                [
                    "signal_key",
                    "value_num",
                    "value_text",
                    "value_bool",
                    "value_json",
                    "confidence",
                    "evidence_text",
                    "evidence_page",
                    "source_type",
                    "observed_at",
                ]
            ]
        )
    else:
        st.info("No signals yet for this company.")

    st.subheader("AI-Assisted Criteria Scoring")
    st.caption("Scores AI-automatable criteria (0–5) using website + founder materials and saves them as signals.")
    ai_include_website = st.checkbox("Include website", value=True, key="ai_score_website")
    ai_url = st.text_input("Website URL", value=company["website"] or "", key="ai_score_url")
    ai_max_pages = st.slider("Max pages to crawl", min_value=1, max_value=10, value=5, key="ai_score_pages")
    ai_same_domain = st.checkbox("Only crawl same domain", value=True, key="ai_score_same_domain")
    ai_include_founder = st.checkbox("Include founder materials", value=True, key="ai_score_founder")
    ai_top_k = st.slider("Founder materials top chunks", min_value=3, max_value=15, value=8, key="ai_score_topk")
    ai_search_per_signal = st.checkbox("Use web search per signal (agents)", value=False, key="ai_score_search_per_signal")
    ai_search_pages = st.slider("Web search pages per signal", min_value=1, max_value=5, value=3, key="ai_score_search_pages")
    ai_model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), key="ai_score_model")
    ai_budget = st.number_input("AI budget per company (USD)", min_value=0.0, max_value=5.0, value=0.2, step=0.05, key="ai_score_budget")
    ai_defs = [d for d in db.list_signal_definitions(conn) if (d.get("automation_type") or "").lower() == "ai" and not d.get("disabled")]
    ai_signal_options = [f"{d['key']} — {d.get('name') or d['key']}" for d in ai_defs]
    selected_ai_signals = st.multiselect("Select signals to run (optional)", ai_signal_options)
    selected_keys = [s.split(" — ")[0] for s in selected_ai_signals]
    if st.button("Run AI criteria scoring", key="ai_score_run"):
        saved, err = criteria_ai.run_ai_scoring(
            conn,
            company_id,
            model=ai_model,
            ai_budget_usd=ai_budget,
            include_website=ai_include_website,
            include_founder_materials=ai_include_founder,
            website_url=ai_url or None,
            max_pages=ai_max_pages,
            same_domain=ai_same_domain,
            top_k=ai_top_k,
            search_per_signal=ai_search_per_signal,
            search_max_pages=ai_search_pages,
            signal_keys=selected_keys or None,
        )
        if err:
            st.error(err)
        else:
            st.success(f"Saved {saved} AI-scored signals.")

    if st.button("Run agents for selected signals", key="ai_score_run_selected"):
        if not selected_keys:
            st.warning("Select at least one signal to run.")
        else:
            saved, err = criteria_ai.run_ai_scoring(
                conn,
                company_id,
                model=ai_model,
                ai_budget_usd=ai_budget,
                include_website=ai_include_website,
                include_founder_materials=ai_include_founder,
                website_url=ai_url or None,
                max_pages=ai_max_pages,
                same_domain=ai_same_domain,
                top_k=ai_top_k,
                search_per_signal=True,
                search_max_pages=ai_search_pages,
                signal_keys=selected_keys,
            )
            if err:
                st.error(err)
            else:
                st.success(f"Saved {saved} AI-scored signals.")

    st.subheader("Score Breakdown")
    score_item = db.list_latest_score_item(conn, company_id)
    if score_item:
        col1, col2, col3 = st.columns(3)
        col1.metric("Normalized Score", round(score_item["normalized_total"] or score_item["total_score"] or 0.0, 4))
        col2.metric("Raw Score", round(score_item["raw_total"] or 0.0, 4))
        col3.metric("Weight Used", round(score_item["weight_used"] or 0.0, 4))
        if score_item["our_angle_score"] is not None:
            st.metric("Our Angle Score", round(score_item["our_angle_score"], 4))

        with st.expander("Scoring transparency"):
            st.write(
                "Main score = sum(raw_score × effective_weight × confidence) / sum(effective_weight × confidence). "
                "Effective weights are normalized per theme (excluding 'Our Angle'). "
                "Our Angle score is computed separately from its own theme weights."
            )
        components = db.list_score_components(conn, company_id)
        if components:
            comp_df = pd.DataFrame(components).rename(
                columns={"weight": "effective_weight", "criterion_weight_raw": "raw_weight"}
            )
            st.dataframe(
                comp_df[
                    [
                        "criterion_name",
                        "criterion_theme",
                        "criterion_subtheme",
                        "raw_weight",
                        "raw_score",
                        "effective_weight",
                        "contribution",
                        "signal_confidence",
                        "signal_evidence_page",
                        "signal_evidence",
                        "explanation",
                    ]
                ]
            )
    else:
        st.info("No scores yet. Run scoring in Companies page.")

    st.subheader("Benchmark Results (Latest Run)")
    bench_rows = db.list_latest_benchmark_results(conn, company_id)
    if bench_rows:
        st.dataframe(
            pd.DataFrame(bench_rows)[
                [
                    "metric_key",
                    "metric_value",
                    "percentile_band",
                    "vs_median",
                    "sector",
                    "stage",
                    "geo",
                ]
            ]
        )
    else:
        st.info("No benchmark results yet. Run Benchmarking.")

    st.subheader("Simple Benchmarks (Quick Win)")
    industry_key = (company["industry"] or "").lower().replace(" ", "_")
    if industry_key in simple_benchmark.SECTOR_BENCHMARKS:
        signals_map = db.get_all_signals(conn, company_id)
        benchmarks = simple_benchmark.SECTOR_BENCHMARKS[industry_key]
        rows = []
        for metric_key, metric_bench in benchmarks.items():
            value = signals_map.get(metric_key)
            if value is None:
                continue
            rows.append(
                {
                    "metric": metric_key,
                    "value": value,
                    "percentile": simple_benchmark.percentile_rank(value, metric_bench),
                }
            )
        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.info("No matching metrics for quick benchmarks.")
    else:
        st.info("No quick benchmarks configured for this industry.")

    st.subheader("Intelligent Risk (Latest Run)")
    risk_item = db.get_latest_risk_run_item(conn, company_id)
    if risk_item:
        st.metric("Risk Score", f"{risk_item['risk_score']:.2f}", risk_item["risk_level"].title())
        risk_flags = db.list_latest_risk_flags(conn, company_id)
        if risk_flags:
            st.dataframe(
                pd.DataFrame(risk_flags)[
                    [
                        "category",
                        "risk_type",
                        "severity",
                        "confidence",
                        "description",
                        "evidence_text",
                        "source_ref",
                        "evidence_page",
                        "method",
                    ]
                ]
            )
        else:
            st.info("No risk flags in latest intelligent scan.")
    else:
        st.info("No intelligent risk scans yet. Run Risk Detection.")


def page_signals_manager(conn):
    st.header("Signals Manager")
    tabs = st.tabs(["Signal Definitions", "Signal Values"])

    with tabs[0]:
        st.subheader("Definitions")
        defs = db.list_signal_definitions(conn)
        if defs:
            df = pd.DataFrame([dict(d) for d in defs])
        else:
            df = pd.DataFrame(columns=["key", "name", "description", "value_type", "unit", "allowed_range_json", "disabled"])
        desired_cols = [
            "key",
            "name",
            "description",
            "value_type",
            "unit",
            "allowed_range_json",
            "disabled",
            "automation_type",
            "automation_detail",
            "automation_prompt",
        ]
        available_cols = [c for c in desired_cols if c in df.columns]
        editor = st.data_editor(
            df[available_cols],
            num_rows="dynamic",
            use_container_width=True,
        )
        if st.button("Save definitions"):
            records = editor.fillna("").to_dict(orient="records")
            cleaned = []
            for r in records:
                if not r.get("key"):
                    continue
                cleaned.append({
                    "key": r.get("key").strip(),
                    "name": r.get("name") or r.get("key"),
                    "description": r.get("description"),
                    "value_type": r.get("value_type") or "text",
                    "unit": r.get("unit"),
                    "allowed_range_json": r.get("allowed_range_json"),
                    "disabled": bool(r.get("disabled")),
                    "automation_type": r.get("automation_type"),
                    "automation_detail": r.get("automation_detail"),
                    "automation_prompt": r.get("automation_prompt"),
                })
            db.upsert_signal_definitions(conn, cleaned)
            st.success("Definitions saved")

    with tabs[1]:
        st.subheader("Values")
        companies = db.list_companies(conn)
        if not companies:
            st.info("No companies available.")
            return
        company_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
        company_selection = st.selectbox("Select company", list(company_map.keys()))
        company_id = company_map[company_selection]

        st.write("Latest signal values")
        latest = db.list_latest_signal_values(conn, company_id)
        if latest:
            st.dataframe(
                pd.DataFrame(latest)[
                    [
                        "signal_key",
                        "value_num",
                        "value_text",
                        "value_bool",
                        "value_json",
                        "confidence",
                        "evidence_text",
                        "evidence_page",
                        "source_type",
                        "observed_at",
                    ]
                ]
            )

        defs = db.list_signal_definitions(conn)
        if defs:
            signal_keys = [d["key"] for d in defs if not d["disabled"]]
        else:
            signal_keys = []
        if signal_keys:
            signal_key = st.selectbox("Signal key", signal_keys)
            selected_def = next((d for d in defs if d["key"] == signal_key), None)
            value_type = selected_def["value_type"] if selected_def else "text"

            value_num = None
            value_text = None
            value_bool = None
            value_json = None

            if value_type == "number":
                value_num = st.number_input("Value (number)", value=0.0, step=1.0)
            elif value_type == "bool":
                value_bool = st.checkbox("Value (bool)")
            elif value_type == "json":
                value_json = st.text_area("Value (json)", value="{}")
            else:
                value_text = st.text_input("Value (text)")

            source_ref = st.text_input("Source reference (optional)")
            confidence_input = st.text_input("Confidence (0-1, optional)")
            evidence_input = st.text_input("Evidence (optional)")
            evidence_page_input = st.text_input("Evidence page (optional)")
            if st.button("Add signal value"):
                if value_type == "json":
                    try:
                        json.loads(value_json or "{}")
                    except json.JSONDecodeError:
                        st.error("Invalid JSON")
                        return
                confidence = None
                if confidence_input:
                    try:
                        confidence = max(0.0, min(1.0, float(confidence_input)))
                    except ValueError:
                        st.error("Confidence must be a number between 0 and 1")
                        return
                evidence_page = None
                if evidence_page_input:
                    try:
                        evidence_page = int(evidence_page_input)
                    except ValueError:
                        st.error("Evidence page must be an integer")
                        return
                db.add_signal_value(
                    conn,
                    company_id,
                    signal_key,
                    value_num,
                    value_text,
                    1 if value_bool else 0 if value_type == "bool" else None,
                    value_json,
                    "manual",
                    source_ref or None,
                    confidence=confidence,
                    evidence_text=evidence_input or None,
                    evidence_page=evidence_page,
                )
                auto_update.rescore_company(conn, company_id, actor="signal_update")
                st.success("Signal value added")

        st.subheader("Signal history")
        history_key = st.selectbox("History for signal key", signal_keys if signal_keys else [])
        if history_key:
            history = db.list_signal_history(conn, company_id, history_key)
            if history:
                st.dataframe(
                    pd.DataFrame(history)[
                        [
                            "signal_key",
                            "value_num",
                            "value_text",
                            "value_bool",
                            "value_json",
                            "confidence",
                            "evidence_text",
                            "evidence_page",
                            "source_type",
                            "observed_at",
                        ]
                    ]
                )


def _validate_params_json(params_json):
    if not params_json:
        return True, None
    try:
        json.loads(params_json)
        return True, None
    except json.JSONDecodeError as exc:
        return False, str(exc)


def page_criteria_manager(conn):
    st.header("Criteria Manager")

    st.subheader("Import from Excel")
    default_excel = os.path.join("data", "Quantitative Investment Checklist - ANON.xlsx")
    excel_path = st.text_input("Excel file path", value=default_excel)
    if st.button("Preview Excel criteria"):
        try:
            parsed = criteria_import.parse_excel_criteria(excel_path)
            st.session_state["excel_criteria_preview"] = parsed
            st.success(f"Parsed {len(parsed)} criteria from Excel")
        except Exception as exc:
            st.error(f"Failed to parse Excel: {exc}")

    preview = st.session_state.get("excel_criteria_preview")
    if preview:
        st.dataframe(pd.DataFrame(preview)[["theme", "subtheme", "name", "weight", "signal_key", "automation_type"]])
        if st.button("Import as new criteria version"):
            defs = []
            for row in preview:
                defs.append(
                    {
                        "key": row["signal_key"],
                        "name": row["name"],
                        "description": f"{row['theme']} / {row['subtheme']}".strip(" /"),
                        "value_type": "number",
                        "unit": "score",
                        "allowed_range_json": json.dumps({"min": 0, "max": 5}),
                        "disabled": False,
                        "automation_type": row.get("automation_type"),
                        "automation_detail": "llm" if row.get("automation_type") == "ai" else "manual",
                        "automation_prompt": None,
                    }
                )
            db.upsert_signal_definitions(conn, defs, actor="system")

            criteria_rows = []
            for row in preview:
                criteria_rows.append(
                    {
                        "name": row["name"],
                        "description": row.get("description") or row.get("subtheme"),
                        "signal_key": row["signal_key"],
                        "weight": float(row["weight"] or 0),
                        "enabled": True,
                        "scoring_method": "linear",
                        "params_json": json.dumps({"min": row.get("score_min", 0), "max": row.get("score_max", 5), "clamp": True}),
                        "missing_policy": "exclude",
                        "theme": row.get("theme"),
                        "subtheme": row.get("subtheme"),
                        "score_min": row.get("score_min", 0),
                        "score_max": row.get("score_max", 5),
                        "display_order": row.get("display_order"),
                    }
                )

            version_id = db.get_active_criteria_version_id(conn)
            new_version_id = db.replace_criteria_version(conn, version_id, criteria_rows, actor="system")
            st.success(f"Imported criteria into new version {new_version_id}")

    version_id = db.get_active_criteria_version_id(conn)
    versions = db.list_criteria_versions(conn)
    current_version = next((v for v in versions if v["id"] == version_id), None)
    if current_version:
        st.caption(f"Active version: v{current_version['version']} ({current_version['created_at']})")

    criteria = db.list_criteria(conn, version_id)
    if criteria:
        df = pd.DataFrame([dict(c) for c in criteria])
    else:
        df = pd.DataFrame(
            columns=[
                "name",
                "description",
                "signal_key",
                "weight",
                "enabled",
                "scoring_method",
                "params_json",
                "missing_policy",
                "theme",
                "subtheme",
                "score_min",
                "score_max",
                "display_order",
            ]
        )

    desired_cols = [
        "name",
        "description",
        "signal_key",
        "weight",
        "enabled",
        "scoring_method",
        "params_json",
        "missing_policy",
        "theme",
        "subtheme",
        "score_min",
        "score_max",
        "display_order",
    ]
    available_cols = [c for c in desired_cols if c in df.columns]
    editor = st.data_editor(
        df[available_cols],
        num_rows="dynamic",
        use_container_width=True,
    )

    if st.button("Save as new version"):
        records = editor.fillna("").to_dict(orient="records")
        cleaned = []
        for r in records:
            if not r.get("name") or not r.get("signal_key"):
                continue
            valid, err = _validate_params_json(r.get("params_json"))
            if not valid:
                st.error(f"Invalid params_json for {r.get('name')}: {err}")
                return
            cleaned.append({
                "name": r.get("name"),
                "description": r.get("description"),
                "signal_key": r.get("signal_key"),
                "weight": float(r.get("weight") or 0),
                "enabled": bool(r.get("enabled")),
                "scoring_method": r.get("scoring_method") or "binary",
                "params_json": r.get("params_json") or "{}",
                "missing_policy": r.get("missing_policy") or "zero",
                "theme": r.get("theme"),
                "subtheme": r.get("subtheme"),
                "score_min": r.get("score_min"),
                "score_max": r.get("score_max"),
                "display_order": r.get("display_order"),
            })
        new_version_id = db.replace_criteria_version(conn, version_id, cleaned)
        st.success(f"Saved new version (id {new_version_id})")

    st.subheader("Previous versions")
    if versions:
        version_df = pd.DataFrame(versions)
        st.dataframe(version_df[["id", "version", "created_at", "notes", "active_version_id"]])


def page_upload_crawl(conn):
    st.header("Upload / Crawl")
    st.subheader("Upload CSV")
    st.caption("Expected columns: name, website, description, industry, location, founder_names")
    merge_duplicates = st.checkbox("Merge duplicates by domain (fill missing fields)", value=True, key="merge_duplicates_upload")
    show_merge_preview = st.checkbox("Show merge preview", value=True, key="merge_preview_upload")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        added = 0
        skipped = 0
        merged = 0
        duplicate_names = []
        merge_previews = []
        for _, row in df.iterrows():
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            company_id, created = db.add_company(
                conn,
                name=name,
                website=row.get("website"),
                description=row.get("description"),
                industry=row.get("industry"),
                location=row.get("location"),
                founder_names=row.get("founder_names"),
                source="upload",
            )
            if not created:
                skipped += 1
                existing = db.get_company(conn, company_id)
                if existing:
                    duplicate_names.append(existing["name"])
                    if show_merge_preview:
                        preview_fields = _merge_preview(
                            dict(existing),
                            {
                                "name": name,
                                "website": row.get("website"),
                                "description": row.get("description"),
                                "industry": row.get("industry"),
                                "location": row.get("location"),
                                "founder_names": row.get("founder_names"),
                            },
                        )
                        if preview_fields:
                            merge_previews.append(
                                {
                                    "Existing Company": existing["name"],
                                    "Incoming Name": name,
                                    "Fields to Fill": ", ".join(preview_fields),
                                }
                            )
                if merge_duplicates:
                    did_merge = db.merge_company_fields(
                        conn,
                        company_id,
                        name=name,
                        website=row.get("website"),
                        description=row.get("description"),
                        industry=row.get("industry"),
                        location=row.get("location"),
                        founder_names=row.get("founder_names"),
                    )
                    if did_merge:
                        merged += 1
                continue
            db.add_activity(conn, "company_added", "company", company_id, {"source": "upload"}, actor="user")
            try:
                _auto_process_company(conn, company_id, website_url=row.get("website"))
            except Exception as exc:
                st.warning(f"Auto-processing failed for {name}: {exc}")
            added += 1
        st.success(f"Added {added} companies (skipped {skipped} duplicates by domain, merged {merged})")
        if duplicate_names:
            preview = ", ".join(duplicate_names[:5])
            st.warning(f"Duplicates matched by domain: {preview}" + (" ..." if len(duplicate_names) > 5 else ""))
        if merge_previews:
            st.subheader("Merge Preview")
            st.dataframe(pd.DataFrame(merge_previews), use_container_width=True)

    st.subheader("AI Signal Extraction from Website")
    st.caption("Requires OPENAI_API_KEY and an OpenAI model name (default: gpt-4o-mini).")
    with st.expander("Quick add company from URL"):
        quick_url = st.text_input("New company website URL", key="quick_add_url")
        quick_merge = st.checkbox("Merge if duplicate (fill missing fields)", value=True, key="quick_add_merge")
        if st.button("Add company", key="quick_add_btn"):
            if quick_url:
                name = quick_url.replace("https://", "").replace("http://", "").split("/")[0]
                company_id, created = db.add_company(conn, name=name, website=quick_url, source="crawl")
                if created:
                    db.add_activity(conn, "company_added", "company", company_id, {"source": "quick_add"}, actor="user")
                    try:
                        _auto_process_company(conn, company_id, website_url=quick_url)
                    except Exception as exc:
                        st.warning(f"Auto-processing failed for {name}: {exc}")
                    st.success("Company created. Select it below.")
                else:
                    existing = db.get_company(conn, company_id)
                    if quick_merge:
                        preview_fields = _merge_preview(
                            dict(existing),
                            {"name": name, "website": quick_url, "description": None, "industry": None, "location": None, "founder_names": None},
                        )
                        db.merge_company_fields(conn, company_id, website=quick_url, name=name)
                        if preview_fields:
                            st.warning(
                                f"Company already exists (domain match). Merged into {existing['name']}. Filled: {', '.join(preview_fields)}"
                            )
                        else:
                            st.warning(f"Company already exists (domain match). Merged into {existing['name']}.")
                    else:
                        st.warning(f"Company already exists (domain match): {existing['name']}.")
            else:
                st.error("Please enter a URL")

    companies = db.list_companies(conn)
    if not companies:
        st.info("Add at least one company to attach extracted signals.")
        return

    company_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
    company_selection = st.selectbox("Select company", list(company_map.keys()))
    company_id = company_map[company_selection]
    company = db.get_company(conn, company_id)

    default_url = company["website"] or ""
    url = st.text_input("Company website URL", value=default_url)
    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    max_pages = st.slider("Max pages to crawl", min_value=1, max_value=10, value=5)
    same_domain = st.checkbox("Only crawl same domain", value=True)

    if st.button("Fetch & extract signals"):
        if not url:
            st.error("Please enter a URL")
        else:
            try:
                text, pages = extraction.crawl_site_text(url, max_pages=max_pages, same_domain=same_domain)
                if not text:
                    st.error("Could not extract readable text from the website")
                else:
                    defs = db.list_signal_definitions(conn)
                    signals = extraction.extract_signals_with_openai(text, defs, model=model)
                    st.session_state["extracted_signals"] = signals
                    st.session_state["extraction_url"] = url
                    st.session_state["extraction_company_id"] = company_id
                    st.session_state["extraction_text"] = text
                    st.session_state["extraction_pages"] = pages
                    st.success("Signals extracted. Review below and save.")
            except Exception as exc:
                st.error(f"Extraction failed: {exc}")

    extracted = st.session_state.get("extracted_signals")
    if extracted:
        st.subheader("Extracted Signals")
        rows = []
        evidences = []
        for key, payload in extracted.items():
            value = payload.get("value") if isinstance(payload, dict) else payload
            confidence = payload.get("confidence") if isinstance(payload, dict) else None
            evidence = payload.get("evidence") if isinstance(payload, dict) else None
            rows.append(
                {
                    "signal_key": key,
                    "value": value,
                    "confidence": confidence,
                    "evidence": evidence,
                }
            )
            if evidence:
                evidences.append(str(evidence))

        st.dataframe(pd.DataFrame(rows)[["signal_key", "value", "confidence", "evidence"]])

        if evidences:
            st.subheader("Evidence Highlights")
            for row in rows:
                if row["evidence"]:
                    conf = row.get("confidence")
                    conf_display = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
                    st.markdown(
                        f"**{row['signal_key']}** – confidence {conf_display}\n\n<mark>{html.escape(str(row['evidence']))}</mark>",
                        unsafe_allow_html=True,
                    )

        source_text = st.session_state.get("extraction_text")
        if source_text:
            with st.expander("Source text preview (highlighted)"):
                highlighted = _highlight_evidence(source_text, evidences)
                st.markdown(
                    f"<div style='white-space: pre-wrap'>{highlighted}</div>",
                    unsafe_allow_html=True,
                )
        pages = st.session_state.get("extraction_pages") or []
        if pages:
            with st.expander("Pages crawled"):
                for page in pages:
                    st.write(page)

        if st.button("Save extracted signals"):
            defs = db.list_signal_definitions(conn)
            def_map = {d["key"]: d for d in defs}
            saved = 0
            pages = st.session_state.get("extraction_pages") or []
            notes = f"model={model};pages={len(pages)}"
            for key, payload in extracted.items():
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
                    st.session_state["extraction_company_id"],
                    key,
                    value_num,
                    value_text,
                    value_bool,
                    value_json,
                    source_type="model",
                    source_ref=st.session_state.get("extraction_url"),
                    notes=notes,
                    confidence=confidence,
                    evidence_text=evidence,
                    evidence_page=evidence_page,
                )
                saved += 1
            if saved:
                auto_update.rescore_company(conn, st.session_state["extraction_company_id"], actor="signal_update")
            st.success(f"Saved {saved} extracted signals")
            st.session_state.pop("extracted_signals", None)
            st.session_state.pop("extraction_text", None)
            st.session_state.pop("extraction_pages", None)


    st.subheader("Founder Materials Upload & Extraction (RAG)")
    st.caption("Supported: PDF, TXT, EML, DOCX, PPTX. Files are stored in data/uploads/.")
    files = st.file_uploader("Upload founder materials", type=["pdf", "txt", "eml", "docx", "pptx"], accept_multiple_files=True, key="fm_upload")
    col1, col2 = st.columns(2)
    add_global = col1.checkbox("Add to global library", value=False, key="fm_global")
    link_company = col2.checkbox("Link to selected company", value=True, key="fm_link")

    fm_company_id = None
    if link_company:
        fm_company_selection = st.selectbox("Company to link", list(company_map.keys()), key="fm_company")
        fm_company_id = company_map[fm_company_selection]

    if st.button("Process founder materials", key="process_fm"):
        if not files:
            st.error("Please upload at least one file")
        elif not add_global and not link_company:
            st.error("Select at least one: global library or company link")
        else:
            processed = 0
            for file_obj in files:
                doc_type = rag.detect_doc_type(file_obj.name)
                if doc_type == "unknown":
                    st.warning(f"Unsupported file type: {file_obj.name}")
                    continue
                try:
                    doc_id, pages, chunks = rag.store_document(
                        conn,
                        file_obj,
                        file_obj.name,
                        doc_type=doc_type,
                        source_type="founder_material",
                        is_global=add_global,
                        company_id=fm_company_id,
                    )
                    st.success(f"Stored {file_obj.name} as document #{doc_id} ({pages} pages, {chunks} chunks)")
                    processed += 1
                except Exception as exc:
                    st.error(f"Failed to process {file_obj.name}: {exc}")
            if processed:
                st.success(f"Processed {processed} founder materials")

    st.subheader("Extract Signals from Founder Materials (RAG)")
    fm_extract_company = st.selectbox("Company for founder-material extraction", list(company_map.keys()), key="fm_extract_company")
    fm_company_id = company_map[fm_extract_company]
    top_k = st.slider("Top chunks to use", min_value=3, max_value=15, value=8, key="fm_topk")

    if st.button("Retrieve + extract from founder materials", key="extract_fm"):
        try:
            chunks = rag.get_company_chunks(conn, fm_company_id, source_types=["founder_material"])
            if not chunks:
                st.error("No founder-material chunks found for this company. Upload files first.")
            else:
                defs = db.list_signal_definitions(conn)
                query = rag.build_query_from_signals(defs)
                top_chunks = rag.retrieve_top_chunks(chunks, query, top_k=top_k)
                if not top_chunks:
                    st.error("No relevant chunks found. Try increasing top_k or add more documents.")
                else:
                    context = rag.build_context(top_chunks)
                    signals = extraction.extract_signals_with_openai(context, defs, model=model)
                    st.session_state["fm_extracted_signals"] = signals
                    st.session_state["fm_context"] = context
                    st.session_state["fm_chunks"] = top_chunks
                    st.session_state["fm_company_id"] = fm_company_id
                    st.success("Founder-material signals extracted. Review below and save.")
        except Exception as exc:
            st.error(f"Founder-material extraction failed: {exc}")

    fm_extracted = st.session_state.get("fm_extracted_signals")
    if fm_extracted:
        st.subheader("Extracted Signals (Founder Materials)")
        fm_rows = []
        fm_evidences = []
        for key, payload in fm_extracted.items():
            value = payload.get("value") if isinstance(payload, dict) else payload
            confidence = payload.get("confidence") if isinstance(payload, dict) else None
            evidence = payload.get("evidence") if isinstance(payload, dict) else None
            page = payload.get("page") if isinstance(payload, dict) else None
            fm_rows.append(
                {
                    "signal_key": key,
                    "value": value,
                    "confidence": confidence,
                    "evidence": evidence,
                    "page": page,
                }
            )
            if evidence:
                fm_evidences.append(str(evidence))

        st.dataframe(pd.DataFrame(fm_rows)[["signal_key", "value", "confidence", "evidence", "page"]])

        if fm_evidences:
            st.subheader("Founder Materials Evidence Highlights")
            for row in fm_rows:
                if row["evidence"]:
                    conf = row.get("confidence")
                    conf_display = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
                    st.markdown(
                        f"**{row['signal_key']}** – confidence {conf_display} (page {row.get('page')})\n\n<mark>{html.escape(str(row['evidence']))}</mark>",
                        unsafe_allow_html=True,
                    )

        fm_context = st.session_state.get("fm_context")
        if fm_context:
            with st.expander("Founder materials context preview (highlighted)"):
                highlighted = _highlight_evidence(fm_context, fm_evidences)
                st.markdown(
                    f"<div style='white-space: pre-wrap'>{highlighted}</div>",
                    unsafe_allow_html=True,
                )

        fm_chunks = st.session_state.get("fm_chunks") or []
        if fm_chunks:
            with st.expander("Top chunks used"):
                chunk_rows = [
                    {
                        "filename": ch.get("filename"),
                        "page": ch.get("page_num"),
                        "chunk_index": ch.get("chunk_index"),
                        "doc_type": ch.get("doc_type"),
                    }
                    for ch in fm_chunks
                ]
                st.dataframe(pd.DataFrame(chunk_rows))

        if st.button("Save extracted founder-material signals", key="save_fm_signals"):
            defs = db.list_signal_definitions(conn)
            def_map = {d["key"]: d for d in defs}
            saved = 0
            fm_chunks = st.session_state.get("fm_chunks") or []
            doc_ids = sorted({ch.get("document_id") for ch in fm_chunks if ch.get("document_id")})
            notes = f"model={model};docs={','.join(map(str, doc_ids))}"
            for key, payload in fm_extracted.items():
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
                    st.session_state["fm_company_id"],
                    key,
                    value_num,
                    value_text,
                    value_bool,
                    value_json,
                    source_type="founder_material",
                    source_ref="founder_material",
                    notes=notes,
                    confidence=confidence,
                    evidence_text=evidence,
                    evidence_page=evidence_page,
                )
                saved += 1
            if saved:
                auto_update.rescore_company(conn, st.session_state["fm_company_id"], actor="signal_update")
            st.success(f"Saved {saved} extracted signals from founder materials")
            st.session_state.pop("fm_extracted_signals", None)
            st.session_state.pop("fm_context", None)
            st.session_state.pop("fm_chunks", None)


def page_optimization(conn):
    st.header("Optimization")
    st.caption("Uses your 1–5 ratings to suggest updated criteria weights (manual approval required).")

    ratings = db.list_latest_ratings(conn)
    defs = db.list_signal_definitions(conn)
    numeric_defs = [d for d in defs if d["disabled"] == 0 and d["value_type"] in ("number", "bool")]

    col1, col2 = st.columns(2)
    col1.metric("Rated companies", len(ratings))
    col2.metric("Numeric/bool signals", len(numeric_defs))

    version_id = db.get_active_criteria_version_id(conn)

    if st.button("Run optimization"):
        result, err = optimization.run_optimization(conn, version_id)
        if err:
            st.error(err)
        else:
            st.session_state["opt_run_id"] = result.run_id
            st.success("Optimization complete")

    runs = db.list_optimization_runs(conn)
    if runs:
        st.subheader("Runs")
        runs_df = pd.DataFrame(runs)
        st.dataframe(runs_df[["id", "criteria_set_version_id", "model_type", "rows_used", "created_at"]])

    run_id = st.session_state.get("opt_run_id")
    if runs:
        run_options = [r["id"] for r in runs]
        if run_id not in run_options:
            run_id = run_options[0]
        run_id = st.selectbox("View run", run_options, index=run_options.index(run_id))

        run_row = next((r for r in runs if r["id"] == run_id), None)
        if run_row and run_row["metrics_json"]:
            try:
                metrics = json.loads(run_row["metrics_json"])
                st.write(f"MAE: {metrics.get('mae'):.3f} | R²: {metrics.get('r2'):.3f}")
            except Exception:
                pass

    if run_id:
        suggestions = db.list_optimization_suggestions(conn, run_id)
        if suggestions:
            st.subheader(f"Suggestions (Run #{run_id})")
            suggestions_df = pd.DataFrame(suggestions)
            suggestions_df["accept"] = suggestions_df["accepted"].fillna(0).astype(int).astype(bool)

            editor = st.data_editor(
                suggestions_df[
                    [
                        "accept",
                        "criterion_name",
                        "signal_key",
                        "current_weight",
                        "suggested_weight",
                        "delta",
                        "confidence",
                        "notes",
                    ]
                ],
                use_container_width=True,
            )

            if st.button("Apply selected suggestions"):
                accepted_ids = suggestions_df.loc[editor["accept"] == True, "id"].tolist()
                db.mark_suggestions(conn, run_id, accepted_ids)
                new_version_id = db.apply_optimization_suggestions(conn, run_id, accepted_ids)
                if new_version_id:
                    st.success(f"Applied {len(accepted_ids)} suggestions. New criteria version: {new_version_id}")
                else:
                    st.error("No suggestions applied.")
        else:
            st.info("No suggestions available for this run yet.")




def page_benchmarking(conn):
    st.header("Benchmarking")
    st.caption("Upload a peer dataset CSV with columns: sector, stage, geo (optional), and numeric metrics like mrr, arr, churn_rate, etc.")

    upload = st.file_uploader("Upload benchmark CSV", type=["csv"], key="benchmark_upload")
    bench_name = st.text_input("Benchmark set name", value="Default Benchmarks")
    bench_desc = st.text_input("Description", value="")

    if upload is not None:
        df = pd.read_csv(upload)
        st.write("Preview")
        st.dataframe(df.head(10))

        columns = df.columns.tolist()
        sector_col = st.selectbox("Sector column", options=columns, index=columns.index("sector") if "sector" in columns else 0)
        stage_col = st.selectbox("Stage column", options=columns, index=columns.index("stage") if "stage" in columns else 0)
        geo_col = st.selectbox("Geo column (optional)", options=[""] + columns, index=(columns.index("geo") + 1) if "geo" in columns else 0)

        non_metric = {sector_col, stage_col}
        if geo_col:
            non_metric.add(geo_col)
        metric_cols = [c for c in columns if c not in non_metric and pd.api.types.is_numeric_dtype(df[c])]
        metric_selection = st.multiselect("Metric columns", options=metric_cols, default=metric_cols)

        if st.button("Create benchmark set"):
            if not metric_selection:
                st.error("Select at least one metric column")
            else:
                group_cols = ["sector", "stage"]
                df = df.rename(columns={sector_col: "sector", stage_col: "stage"})
                if geo_col:
                    df = df.rename(columns={geo_col: "geo"})
                    group_cols.append("geo")
                else:
                    df["geo"] = None

                stats_rows = benchmark.compute_benchmark_stats(df, group_cols, metric_selection)
                if not stats_rows:
                    st.error("No benchmark stats computed. Check your data.")
                else:
                    set_id = db.create_benchmark_set(conn, bench_name, bench_desc)
                    db.add_benchmark_stats(conn, set_id, stats_rows)
                    st.success(f"Created benchmark set #{set_id} with {len(stats_rows)} stats rows")

    st.subheader("Run Benchmarking")
    sets = db.list_benchmark_sets(conn)
    if not sets:
        st.info("No benchmark sets yet. Upload a CSV above.")
        return

    set_map = {f"{s['name']} (#{s['id']})": s["id"] for s in sets}
    set_sel = st.selectbox("Benchmark set", list(set_map.keys()))
    set_id = set_map[set_sel]

    if st.button("Run benchmarking"):
        run_id, err = benchmark.run_benchmark(conn, set_id)
        if err:
            st.error(err)
        else:
            st.success(f"Benchmark run completed (id {run_id})")


def page_risk_detection(conn):
    st.header("Risk Detection")
    st.caption("Intelligent red flag detection with rules, anomalies, patterns, cross-source checks, and LLM.")

    model = st.text_input("OpenAI model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    ai_budget = st.number_input("AI budget per company (USD)", min_value=0.0, max_value=5.0, value=0.2, step=0.05)
    enable_llm = st.checkbox("Enable LLM text risk extraction", value=True)

    if st.button("Run intelligent risk scan"):
        engine = risk_detector.RiskDetectionEngine(conn, model=model, ai_budget_usd=ai_budget, enable_llm=enable_llm)
        run_id, err = engine.run_scan()
        if err:
            st.error(err)
        else:
            st.success(f"Intelligent risk scan completed (id {run_id})")

    companies = db.list_companies(conn)
    if not companies:
        st.info("No companies yet.")
        return

    company_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
    selection = st.selectbox("View findings for company", list(company_map.keys()), key="risk_company")
    company_id = company_map[selection]
    run_item = db.get_latest_risk_run_item(conn, company_id)
    if run_item:
        st.metric("Risk Score", f"{run_item['risk_score']:.2f}", run_item["risk_level"].title())

    flags = db.list_latest_risk_flags(conn, company_id)
    if flags:
        st.dataframe(
            pd.DataFrame(flags)[
                [
                    "category",
                    "risk_type",
                    "severity",
                    "confidence",
                    "description",
                    "evidence_text",
                    "source_ref",
                    "evidence_page",
                    "method",
                ]
            ]
        )
    else:
        st.info("No risk flags for this company in the latest intelligent run.")

    report = db.get_risk_report(conn, company_id)
    if report:
        st.download_button(
            "Export Risk Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"risk_report_company_{company_id}.json",
            mime="application/json",
        )


def page_public_data(conn):
    st.header("Crawl Manager")
    st.caption("Control open/closed web discovery and search intensity. All extracted signals are source-attributed.")

    settings = db.get_crawl_settings(conn)
    settings_dict = dict(settings) if settings else {}
    mode_default = settings_dict.get("mode") or "closed"
    intensity_default = settings_dict.get("intensity") or "light"
    agent_default = bool(settings_dict.get("agent_mode") or 0)
    allowlist_default = []
    if settings_dict.get("allowlist_json"):
        try:
            allowlist_default = json.loads(settings_dict["allowlist_json"])
        except json.JSONDecodeError:
            allowlist_default = []

    mode = st.selectbox("Search mode", ["closed", "open"], index=0 if mode_default == "closed" else 1)
    intensity = st.selectbox("Search intensity", ["light", "medium", "heavy"], index=["light", "medium", "heavy"].index(intensity_default))
    agent_mode = st.checkbox("Use AI agents for discovery → extraction → scoring → risk", value=agent_default)
    allowlist_text = st.text_area("Closed-mode allowlist (one URL per line)", value="\\n".join(allowlist_default))
    news_domains_default = []
    if settings_dict.get("news_domains_json"):
        try:
            news_domains_default = json.loads(settings_dict["news_domains_json"])
        except json.JSONDecodeError:
            news_domains_default = []
    news_domains_text = st.text_area("News domains to search (one domain per line)", value="\\n".join(news_domains_default))

    st.write("Paid sources (will be crawled only if included in allowlist)")
    st.checkbox("PitchBook", value=True, disabled=True)
    st.checkbox("Affinity", value=True, disabled=True)
    st.checkbox("FT.com", value=True, disabled=True)

    if st.button("Save crawl settings"):
        allowlist = [line.strip() for line in allowlist_text.splitlines() if line.strip()]
        news_domains = [line.strip() for line in news_domains_text.splitlines() if line.strip()]
        db.upsert_crawl_settings(conn, mode, intensity, allowlist, news_domains, agent_mode, actor="user")
        db.add_activity(conn, "crawl_settings_saved", None, None, {"mode": mode, "intensity": intensity, "agent_mode": agent_mode, "allowlist_count": len(allowlist), "news_domain_count": len(news_domains)}, actor="user")
        st.success("Crawl settings saved")

    st.subheader("Run web search for a company")
    companies = db.list_companies(conn)
    if not companies:
        st.info("Add a company first in Upload / Crawl.")
        return
    company_map = {f"{c['name']} (#{c['id']})": c["id"] for c in companies}
    company_selection = st.selectbox("Select company", list(company_map.keys()), key="crawl_company")
    company_id = company_map[company_selection]
    allowlist = [line.strip() for line in allowlist_text.splitlines() if line.strip()]
    news_domains = [line.strip() for line in news_domains_text.splitlines() if line.strip()]

    if st.button("Run discovery search now"):
        if agent_mode:
            orchestrator = agent_orchestrator.AgentOrchestrator(conn, mode, intensity, allowlist, news_domains)
            orchestrator.run(company_id)
            st.success("Agent pipeline complete (discovery → extraction → scoring → risk)")
        else:
            saved, err = discovery.crawl_and_extract(conn, company_id, mode=mode, intensity=intensity, allowlist=allowlist, news_domains=news_domains)
            if err:
                st.error(err)
            else:
                auto_update.rescore_company(conn, company_id, actor="signal_update")
                st.success(f"Saved {saved} signals from crawl")

    st.subheader("Activity Log")
    logs = db.list_activity(conn, limit=200)
    if logs:
        df = pd.DataFrame([dict(r) for r in logs])
        st.dataframe(df[["created_at", "event_type", "entity_type", "entity_id", "actor", "details_json"]])
    else:
        st.info("No activity logs yet.")


def main():
    st.set_page_config(page_title="AI Sourcing MVP", layout="wide")
    db.init_db()
    conn = db.get_conn()
    seed_defaults(conn)

    st.sidebar.title("Navigation")
    _render_run_tooltips()
    show_advanced = st.sidebar.checkbox("Show advanced pages", value=False)
    pages = ["Simple Workflow"]
    if show_advanced:
        pages.extend(
            [
                "Dashboard",
                "Companies",
                "Company Detail",
                "Signals Manager",
                "Criteria Manager",
                "Upload / Crawl",
                "Optimization",
                "Benchmarking",
                "Risk Detection",
                "Crawl Manager",
            ]
        )
    page = st.sidebar.radio("Go to", pages)

    if page == "Simple Workflow":
        page_simple(conn)
    elif page == "Dashboard":
        page_dashboard(conn)
    elif page == "Companies":
        page_companies(conn)
    elif page == "Company Detail":
        page_company_detail(conn)
    elif page == "Signals Manager":
        page_signals_manager(conn)
    elif page == "Criteria Manager":
        page_criteria_manager(conn)
    elif page == "Upload / Crawl":
        page_upload_crawl(conn)
    elif page == "Optimization":
        page_optimization(conn)
    elif page == "Benchmarking":
        page_benchmarking(conn)
    elif page == "Risk Detection":
        page_risk_detection(conn)
    elif page == "Crawl Manager":
        page_public_data(conn)

    conn.close()


if __name__ == "__main__":
    main()
