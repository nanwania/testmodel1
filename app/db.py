import json
import os
import sqlite3
from urllib.parse import urlparse
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "app.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            website TEXT,
            description TEXT,
            industry TEXT,
            location TEXT,
            founder_names TEXT,
            source TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_definitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            value_type TEXT NOT NULL,
            unit TEXT,
            allowed_range_json TEXT,
            automation_type TEXT,
            automation_detail TEXT,
            automation_prompt TEXT,
            disabled INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            created_by TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_values (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            signal_key TEXT NOT NULL,
            value_num REAL,
            value_text TEXT,
            value_bool INTEGER,
            value_json TEXT,
            confidence REAL,
            evidence_text TEXT,
            evidence_page INTEGER,
            source_type TEXT NOT NULL,
            source_ref TEXT,
            observed_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT,
            notes TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS criteria_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            active_version_id INTEGER,
            created_at TEXT NOT NULL,
            created_by TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS criteria_set_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            criteria_set_id INTEGER NOT NULL,
            version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT,
            notes TEXT,
            FOREIGN KEY(criteria_set_id) REFERENCES criteria_sets(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS criteria (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            criteria_set_version_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            signal_key TEXT NOT NULL,
            weight REAL NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            scoring_method TEXT NOT NULL,
            params_json TEXT,
            missing_policy TEXT NOT NULL,
            theme TEXT,
            subtheme TEXT,
            score_min REAL,
            score_max REAL,
            display_order INTEGER,
            created_at TEXT NOT NULL,
            created_by TEXT,
            FOREIGN KEY(criteria_set_version_id) REFERENCES criteria_set_versions(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS score_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            criteria_set_version_id INTEGER NOT NULL,
            run_started_at TEXT NOT NULL,
            run_completed_at TEXT,
            triggered_by TEXT,
            notes TEXT,
            FOREIGN KEY(criteria_set_version_id) REFERENCES criteria_set_versions(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS score_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score_run_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            total_score REAL NOT NULL,
            raw_total REAL,
            normalized_total REAL,
            weight_used REAL,
            our_angle_score REAL,
            scored_at TEXT NOT NULL,
            FOREIGN KEY(score_run_id) REFERENCES score_runs(id),
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS score_components (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            score_item_id INTEGER NOT NULL,
            criterion_id INTEGER NOT NULL,
            signal_value_id INTEGER,
            raw_value TEXT,
            raw_score REAL,
            weight REAL NOT NULL,
            contribution REAL NOT NULL,
            passed INTEGER,
            explanation TEXT,
            evaluated_at TEXT NOT NULL,
            FOREIGN KEY(score_item_id) REFERENCES score_items(id),
            FOREIGN KEY(criterion_id) REFERENCES criteria(id),
            FOREIGN KEY(signal_value_id) REFERENCES signal_values(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            rating_int INTEGER NOT NULL,
            labeled_at TEXT NOT NULL,
            labeled_by TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            criteria_set_version_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            model_type TEXT NOT NULL,
            rows_used INTEGER NOT NULL,
            metrics_json TEXT,
            created_by TEXT,
            FOREIGN KEY(criteria_set_version_id) REFERENCES criteria_set_versions(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS optimization_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            optimization_run_id INTEGER NOT NULL,
            criterion_id INTEGER NOT NULL,
            current_weight REAL NOT NULL,
            suggested_weight REAL NOT NULL,
            delta REAL NOT NULL,
            confidence REAL,
            accepted INTEGER,
            notes TEXT,
            FOREIGN KEY(optimization_run_id) REFERENCES optimization_runs(id),
            FOREIGN KEY(criterion_id) REFERENCES criteria(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            old_value TEXT,
            new_value TEXT,
            actor TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_set_id INTEGER NOT NULL,
            sector TEXT,
            stage TEXT,
            geo TEXT,
            metric_key TEXT NOT NULL,
            p10 REAL,
            p25 REAL,
            p50 REAL,
            p75 REAL,
            p90 REAL,
            sample_size INTEGER,
            FOREIGN KEY(benchmark_set_id) REFERENCES benchmark_sets(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_set_id INTEGER NOT NULL,
            run_started_at TEXT NOT NULL,
            run_completed_at TEXT,
            triggered_by TEXT,
            notes TEXT,
            FOREIGN KEY(benchmark_set_id) REFERENCES benchmark_sets(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            benchmark_run_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            metric_key TEXT NOT NULL,
            metric_value REAL,
            percentile_band TEXT,
            vs_median REAL,
            sector TEXT,
            stage TEXT,
            geo TEXT,
            FOREIGN KEY(benchmark_run_id) REFERENCES benchmark_runs(id),
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            rule_type TEXT NOT NULL,
            signal_key TEXT,
            params_json TEXT,
            severity TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            created_by TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_started_at TEXT NOT NULL,
            run_completed_at TEXT,
            triggered_by TEXT,
            notes TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_run_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            rule_id INTEGER NOT NULL,
            signal_value_id INTEGER,
            value REAL,
            severity TEXT NOT NULL,
            explanation TEXT,
            evidence_text TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(risk_run_id) REFERENCES risk_runs(id),
            FOREIGN KEY(company_id) REFERENCES companies(id),
            FOREIGN KEY(rule_id) REFERENCES risk_rules(id),
            FOREIGN KEY(signal_value_id) REFERENCES signal_values(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_run_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_run_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            FOREIGN KEY(risk_run_id) REFERENCES risk_runs(id),
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            risk_run_item_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            risk_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            confidence REAL,
            description TEXT,
            evidence_text TEXT,
            source_type TEXT,
            source_ref TEXT,
            evidence_page INTEGER,
            method TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            FOREIGN KEY(risk_run_item_id) REFERENCES risk_run_items(id)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crawl_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT NOT NULL,
            intensity TEXT NOT NULL,
            allowlist_json TEXT,
            news_domains_json TEXT,
            agent_mode INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            entity_type TEXT,
            entity_id INTEGER,
            details_json TEXT,
            created_at TEXT NOT NULL,
            actor TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS company_highlights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER NOT NULL,
            highlights_json TEXT NOT NULL,
            model TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT,
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )

    conn.commit()
    _ensure_column(conn, "companies", "canonical_domain", "TEXT")
    _ensure_column(conn, "companies", "deleted_at", "TEXT")
    _ensure_column(conn, "companies", "deleted_by", "TEXT")
    _ensure_column(conn, "score_items", "raw_total", "REAL")
    _ensure_column(conn, "score_items", "normalized_total", "REAL")
    _ensure_column(conn, "score_items", "weight_used", "REAL")
    _ensure_column(conn, "score_items", "our_angle_score", "REAL")
    _ensure_column(conn, "signal_values", "confidence", "REAL")
    _ensure_column(conn, "signal_values", "evidence_text", "TEXT")
    _ensure_column(conn, "signal_values", "evidence_page", "INTEGER")
    _ensure_column(conn, "criteria", "theme", "TEXT")
    _ensure_column(conn, "criteria", "subtheme", "TEXT")
    _ensure_column(conn, "criteria", "score_min", "REAL")
    _ensure_column(conn, "criteria", "score_max", "REAL")
    _ensure_column(conn, "criteria", "display_order", "INTEGER")
    _ensure_column(conn, "signal_definitions", "automation_type", "TEXT")
    _ensure_column(conn, "signal_definitions", "automation_detail", "TEXT")
    _ensure_column(conn, "signal_definitions", "automation_prompt", "TEXT")
    _ensure_column(conn, "crawl_settings", "allowlist_json", "TEXT")
    _ensure_column(conn, "crawl_settings", "news_domains_json", "TEXT")
    _ensure_column(conn, "crawl_settings", "agent_mode", "INTEGER")
    _ensure_document_tables(conn)
    _ensure_default_risk_rules(conn)
    _ensure_default_criteria_set(conn)
    _backfill_company_domains(conn)
    conn.close()


def _ensure_default_criteria_set(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, active_version_id FROM criteria_sets WHERE name = ?", ("Default",))
    row = cur.fetchone()
    if row:
        if row["active_version_id"] is None:
            version_id = _create_criteria_version(conn, row["id"], notes="Initial version")
            cur.execute("UPDATE criteria_sets SET active_version_id = ? WHERE id = ?", (version_id, row["id"]))
            conn.commit()
        return

    now = _utc_now()
    cur.execute(
        """
        INSERT INTO criteria_sets (name, description, created_at, created_by)
        VALUES (?, ?, ?, ?)
        """,
        ("Default", "Default criteria set", now, "system"),
    )
    criteria_set_id = cur.lastrowid
    version_id = _create_criteria_version(conn, criteria_set_id, notes="Initial version")
    cur.execute("UPDATE criteria_sets SET active_version_id = ? WHERE id = ?", (version_id, criteria_set_id))
    conn.commit()


def _create_criteria_version(conn, criteria_set_id, notes=None):
    cur = conn.cursor()
    cur.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM criteria_set_versions WHERE criteria_set_id = ?", (criteria_set_id,))
    next_version = cur.fetchone()[0]
    now = _utc_now()
    cur.execute(
        """
        INSERT INTO criteria_set_versions (criteria_set_id, version, created_at, created_by, notes)
        VALUES (?, ?, ?, ?, ?)
        """,
        (criteria_set_id, next_version, now, "system", notes),
    )
    conn.commit()
    return cur.lastrowid


def _ensure_column(conn, table, column, col_type):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def _ensure_document_tables(conn):
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            doc_type TEXT,
            source_type TEXT,
            extracted_text TEXT,
            is_global INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            created_by TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS document_company_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT,
            FOREIGN KEY(document_id) REFERENCES documents(id),
            FOREIGN KEY(company_id) REFERENCES companies(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            page_num INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
        """
    )
    conn.commit()
    _ensure_column(conn, "documents", "doc_type", "TEXT")
    _ensure_column(conn, "documents", "source_type", "TEXT")
    _ensure_column(conn, "documents", "extracted_text", "TEXT")


def get_active_criteria_version_id(conn):
    cur = conn.cursor()
    cur.execute("SELECT active_version_id FROM criteria_sets WHERE name = ?", ("Default",))
    row = cur.fetchone()
    return row["active_version_id"] if row else None


def list_criteria_versions(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT v.id, v.version, v.created_at, v.notes,
               s.active_version_id
        FROM criteria_set_versions v
        JOIN criteria_sets s ON s.id = v.criteria_set_id
        WHERE s.name = ?
        ORDER BY v.version DESC
        """,
        ("Default",),
    )
    return cur.fetchall()


def set_active_criteria_version(conn, version_id):
    cur = conn.cursor()
    cur.execute("SELECT criteria_set_id FROM criteria_set_versions WHERE id = ?", (version_id,))
    row = cur.fetchone()
    if not row:
        return
    cur.execute("UPDATE criteria_sets SET active_version_id = ? WHERE id = ?", (version_id, row["criteria_set_id"]))
    conn.commit()


def list_criteria(conn, version_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM criteria
        WHERE criteria_set_version_id = ?
        ORDER BY id ASC
        """,
        (version_id,),
    )
    return cur.fetchall()


def replace_criteria_version(conn, version_id, criteria_rows, actor="user"):
    cur = conn.cursor()
    cur.execute("SELECT criteria_set_id FROM criteria_set_versions WHERE id = ?", (version_id,))
    row = cur.fetchone()
    if not row:
        return None

    new_version_id = _create_criteria_version(conn, row["criteria_set_id"], notes="Updated via UI")
    now = _utc_now()

    for c in criteria_rows:
        cur.execute(
            """
            INSERT INTO criteria (
                criteria_set_version_id, name, description, signal_key, weight,
                enabled, scoring_method, params_json, missing_policy,
                theme, subtheme, score_min, score_max, display_order,
                created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_version_id,
                c.get("name"),
                c.get("description"),
                c.get("signal_key"),
                float(c.get("weight", 0)),
                1 if c.get("enabled") else 0,
                c.get("scoring_method"),
                c.get("params_json"),
                c.get("missing_policy"),
                c.get("theme"),
                c.get("subtheme"),
                c.get("score_min"),
                c.get("score_max"),
                c.get("display_order"),
                now,
                actor,
            ),
        )

    cur.execute("UPDATE criteria_sets SET active_version_id = ? WHERE id = ?", (new_version_id, row["criteria_set_id"]))
    conn.commit()
    return new_version_id


def add_audit_log(conn, entity_type, entity_id, action, old_value, new_value, actor="user"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_log (entity_type, entity_id, action, old_value, new_value, actor, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entity_type,
            entity_id,
            action,
            json.dumps(old_value) if old_value is not None else None,
            json.dumps(new_value) if new_value is not None else None,
            actor,
            _utc_now(),
        ),
    )
    conn.commit()


def upsert_signal_definitions(conn, defs, actor="user"):
    cur = conn.cursor()
    now = _utc_now()
    for d in defs:
        cur.execute(
            "SELECT id, name, description, value_type, unit, allowed_range_json, disabled, automation_type, automation_detail, automation_prompt FROM signal_definitions WHERE key = ?",
            (d["key"],),
        )
        existing = cur.fetchone()
        if existing:
            old = dict(existing)
            cur.execute(
                """
                UPDATE signal_definitions
                SET name = ?, description = ?, value_type = ?, unit = ?, allowed_range_json = ?, disabled = ?,
                    automation_type = ?, automation_detail = ?, automation_prompt = ?
                WHERE key = ?
                """,
                (
                    d.get("name"),
                    d.get("description"),
                    d.get("value_type"),
                    d.get("unit"),
                    d.get("allowed_range_json"),
                    1 if d.get("disabled") else 0,
                    d.get("automation_type"),
                    d.get("automation_detail"),
                    d.get("automation_prompt"),
                    d.get("key"),
                ),
            )
            add_audit_log(conn, "signal_definitions", existing["id"], "update", old, d, actor=actor)
        else:
            cur.execute(
                """
                INSERT INTO signal_definitions (key, name, description, value_type, unit, allowed_range_json, disabled,
                                                automation_type, automation_detail, automation_prompt, created_at, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d.get("key"),
                    d.get("name"),
                    d.get("description"),
                    d.get("value_type"),
                    d.get("unit"),
                    d.get("allowed_range_json"),
                    1 if d.get("disabled") else 0,
                    d.get("automation_type"),
                    d.get("automation_detail"),
                    d.get("automation_prompt"),
                    now,
                    actor,
                ),
            )
            add_audit_log(conn, "signal_definitions", cur.lastrowid, "create", None, d, actor=actor)
    conn.commit()


def list_signal_definitions(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM signal_definitions ORDER BY key ASC")
    return cur.fetchall()


def add_signal_value(
    conn,
    company_id,
    signal_key,
    value_num,
    value_text,
    value_bool,
    value_json,
    source_type,
    source_ref,
    actor="user",
    notes=None,
    confidence=None,
    evidence_text=None,
    evidence_page=None,
):
    now = _utc_now()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO signal_values (
            company_id, signal_key, value_num, value_text, value_bool, value_json,
            confidence, evidence_text, evidence_page, source_type, source_ref, observed_at, created_at, created_by, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            company_id,
            signal_key,
            value_num,
            value_text,
            value_bool,
            value_json,
            confidence,
            evidence_text,
            evidence_page,
            source_type,
            source_ref,
            now,
            now,
            actor,
            notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def list_companies(conn, include_deleted=False):
    cur = conn.cursor()
    if include_deleted:
        cur.execute("SELECT * FROM companies ORDER BY id DESC")
    else:
        cur.execute("SELECT * FROM companies WHERE deleted_at IS NULL ORDER BY id DESC")
    return cur.fetchall()


def get_company(conn, company_id):
    cur = conn.cursor()
    cur.execute("SELECT * FROM companies WHERE id = ?", (company_id,))
    return cur.fetchone()


def _normalize_domain(website):
    if not website:
        return None
    raw = website.strip()
    if not raw:
        return None
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    host = parsed.netloc or parsed.path
    if not host:
        return None
    host = host.split("/")[0].split(":")[0].lower().strip(".")
    if host.startswith("www."):
        host = host[4:]
    if host.endswith(".com"):
        host = host[:-4]
    return host or None


def _backfill_company_domains(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, website, canonical_domain FROM companies")
    rows = cur.fetchall()
    updates = []
    for r in rows:
        if r["canonical_domain"]:
            continue
        canonical = _normalize_domain(r["website"])
        if canonical:
            updates.append((canonical, r["id"]))
    if updates:
        cur.executemany("UPDATE companies SET canonical_domain = ? WHERE id = ?", updates)
        conn.commit()


def get_company_by_domain(conn, canonical_domain):
    cur = conn.cursor()
    cur.execute("SELECT * FROM companies WHERE canonical_domain = ?", (canonical_domain,))
    return cur.fetchone()


def add_company(conn, name, website=None, description=None, industry=None, location=None, founder_names=None, source="upload"):
    now = _utc_now()
    canonical_domain = _normalize_domain(website)
    if canonical_domain:
        existing = get_company_by_domain(conn, canonical_domain)
        if existing:
            return existing["id"], False
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO companies (name, website, canonical_domain, description, industry, location, founder_names, source, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, website, canonical_domain, description, industry, location, founder_names, source, now),
    )
    conn.commit()
    return cur.lastrowid, True


def delete_company(conn, company_id, actor="user"):
    cur = conn.cursor()
    cur.execute(
        "UPDATE companies SET deleted_at = ?, deleted_by = ? WHERE id = ?",
        (_utc_now(), actor, company_id),
    )
    conn.commit()


def restore_company(conn, company_id, actor="user"):
    cur = conn.cursor()
    cur.execute(
        "UPDATE companies SET deleted_at = NULL, deleted_by = NULL WHERE id = ?",
        (company_id,),
    )
    conn.commit()


def merge_company_fields(conn, company_id, name=None, website=None, description=None, industry=None, location=None, founder_names=None):
    cur = conn.cursor()
    cur.execute("SELECT * FROM companies WHERE id = ?", (company_id,))
    row = cur.fetchone()
    if not row:
        return False
    updates = {
        "name": name,
        "website": website,
        "description": description,
        "industry": industry,
        "location": location,
        "founder_names": founder_names,
    }
    set_parts = []
    values = []
    for field, incoming in updates.items():
        current = row[field]
        if (current is None or str(current).strip() == "") and incoming:
            set_parts.append(f"{field} = ?")
            values.append(incoming)
    if not set_parts:
        return False
    values.append(company_id)
    cur.execute(f"UPDATE companies SET {', '.join(set_parts)} WHERE id = ?", values)
    conn.commit()
    return True


def list_latest_signal_values(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sv.*
        FROM signal_values sv
        JOIN (
            SELECT signal_key, MAX(id) AS max_id
            FROM signal_values
            WHERE company_id = ?
            GROUP BY signal_key
        ) latest ON sv.id = latest.max_id
        WHERE sv.company_id = ?
        ORDER BY sv.signal_key ASC
        """,
        (company_id, company_id),
    )
    return cur.fetchall()


def _parse_signal_row(row):
    if row is None:
        return None
    if row.get("value_num") is not None:
        return float(row["value_num"])
    if row.get("value_bool") is not None:
        return bool(row["value_bool"])
    if row.get("value_text") is not None:
        return row["value_text"]
    if row.get("value_json"):
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            return row["value_json"]
    return None


def get_all_signals(conn, company_id):
    rows = list_latest_signal_values(conn, company_id)
    return {row["signal_key"]: _parse_signal_row(dict(row)) for row in rows}


def upsert_signal_value(conn, company_id, signal_key, value, source_type="derived", source_ref=None, actor="system", confidence=0.8, notes=None):
    cur = conn.cursor()
    cur.execute("SELECT value_type FROM signal_definitions WHERE key = ?", (signal_key,))
    row = cur.fetchone()
    value_type = row["value_type"] if row else "number"

    value_num = None
    value_text = None
    value_bool = None
    value_json = None

    if value_type == "number":
        try:
            value_num = float(value)
        except (TypeError, ValueError):
            value_num = None
    elif value_type == "bool":
        if isinstance(value, bool):
            value_bool = 1 if value else 0
        elif isinstance(value, (int, float)):
            value_bool = 1 if value else 0
        elif isinstance(value, str):
            value_bool = 1 if value.strip().lower() in ("true", "yes", "1") else 0
    elif value_type == "json":
        try:
            value_json = json.dumps(value)
        except TypeError:
            value_json = None
    else:
        value_text = str(value) if value is not None else None

    add_signal_value(
        conn,
        company_id,
        signal_key,
        value_num,
        value_text,
        value_bool,
        value_json,
        source_type,
        source_ref,
        actor=actor,
        notes=notes,
        confidence=confidence,
        evidence_text=None,
        evidence_page=None,
    )


def list_signal_history(conn, company_id, signal_key):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM signal_values
        WHERE company_id = ? AND signal_key = ?
        ORDER BY id DESC
        """,
        (company_id, signal_key),
    )
    return cur.fetchall()


def add_rating(conn, company_id, rating_int, actor="user"):
    now = _utc_now()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO ratings (company_id, rating_int, labeled_at, labeled_by)
        VALUES (?, ?, ?, ?)
        """,
        (company_id, int(rating_int), now, actor),
    )
    conn.commit()


def list_latest_rating(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM ratings
        WHERE company_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (company_id,),
    )
    return cur.fetchone()

def list_latest_ratings(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT r.*
        FROM ratings r
        JOIN (
            SELECT company_id, MAX(id) AS max_id
            FROM ratings
            GROUP BY company_id
        ) latest ON r.id = latest.max_id
        """
    )
    return cur.fetchall()


def list_score_components(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sc.*, c.name AS criterion_name,
               c.weight AS criterion_weight_raw,
               c.theme AS criterion_theme,
               c.subtheme AS criterion_subtheme,
               c.scoring_method AS criterion_method,
               c.missing_policy AS criterion_missing_policy,
               sv.confidence AS signal_confidence,
               sv.evidence_text AS signal_evidence,
               sv.evidence_page AS signal_evidence_page,
               sv.signal_key AS signal_key,
               sv.source_type AS signal_source_type,
               sv.source_ref AS signal_source_ref
        FROM score_components sc
        JOIN score_items si ON si.id = sc.score_item_id
        JOIN score_runs sr ON sr.id = si.score_run_id
        JOIN criteria c ON c.id = sc.criterion_id
        LEFT JOIN signal_values sv ON sv.id = sc.signal_value_id
        WHERE si.company_id = ?
        ORDER BY sc.id DESC
        """,
        (company_id,),
    )
    return cur.fetchall()


def list_latest_score_item(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT si.*, sr.run_completed_at
        FROM score_items si
        JOIN score_runs sr ON sr.id = si.score_run_id
        WHERE si.company_id = ?
        ORDER BY si.id DESC
        LIMIT 1
        """,
        (company_id,),
    )
    return cur.fetchone()


def list_latest_score_summary(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT c.*, s.total_score, s.raw_total, s.normalized_total, s.weight_used, s.scored_at
        FROM companies c
        LEFT JOIN (
            SELECT si.company_id, si.total_score, si.raw_total, si.normalized_total, si.weight_used, si.scored_at
            FROM score_items si
            JOIN score_runs sr ON sr.id = si.score_run_id
            WHERE sr.id = (SELECT MAX(id) FROM score_runs)
        ) s ON s.company_id = c.id
        WHERE c.deleted_at IS NULL
        ORDER BY (s.total_score IS NULL), s.total_score DESC, c.name ASC
        """
    )
    return cur.fetchall()


def create_benchmark_set(conn, name, description=None, actor="user"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO benchmark_sets (name, description, created_at, created_by)
        VALUES (?, ?, ?, ?)
        """,
        (name, description, _utc_now(), actor),
    )
    conn.commit()
    return cur.lastrowid


def add_benchmark_stats(conn, benchmark_set_id, rows):
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO benchmark_stats (
                benchmark_set_id, sector, stage, geo, metric_key,
                p10, p25, p50, p75, p90, sample_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                benchmark_set_id,
                r.get("sector"),
                r.get("stage"),
                r.get("geo"),
                r.get("metric_key"),
                r.get("p10"),
                r.get("p25"),
                r.get("p50"),
                r.get("p75"),
                r.get("p90"),
                r.get("sample_size"),
            ),
        )
    conn.commit()


def list_benchmark_sets(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM benchmark_sets ORDER BY id DESC")
    return cur.fetchall()


def list_benchmark_stats(conn, benchmark_set_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM benchmark_stats
        WHERE benchmark_set_id = ?
        """,
        (benchmark_set_id,),
    )
    return cur.fetchall()


def create_benchmark_run(conn, benchmark_set_id, actor="user"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO benchmark_runs (benchmark_set_id, run_started_at, triggered_by, notes)
        VALUES (?, ?, ?, ?)
        """,
        (benchmark_set_id, _utc_now(), actor, "manual run"),
    )
    conn.commit()
    return cur.lastrowid


def complete_benchmark_run(conn, run_id):
    cur = conn.cursor()
    cur.execute(
        "UPDATE benchmark_runs SET run_completed_at = ? WHERE id = ?",
        (_utc_now(), run_id),
    )
    conn.commit()


def add_benchmark_results(conn, run_id, rows):
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """
            INSERT INTO benchmark_results (
                benchmark_run_id, company_id, metric_key, metric_value,
                percentile_band, vs_median, sector, stage, geo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                r.get("company_id"),
                r.get("metric_key"),
                r.get("metric_value"),
                r.get("percentile_band"),
                r.get("vs_median"),
                r.get("sector"),
                r.get("stage"),
                r.get("geo"),
            ),
        )
    conn.commit()


def list_latest_benchmark_results(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT br.*
        FROM benchmark_results br
        JOIN benchmark_runs r ON r.id = br.benchmark_run_id
        WHERE br.company_id = ?
          AND r.id = (SELECT MAX(id) FROM benchmark_runs)
        ORDER BY br.metric_key ASC
        """,
        (company_id,),
    )
    return cur.fetchall()


def list_benchmark_runs(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM benchmark_runs ORDER BY id DESC")
    return cur.fetchall()


def create_risk_run(conn, actor="user"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO risk_runs (run_started_at, triggered_by, notes)
        VALUES (?, ?, ?)
        """,
        (_utc_now(), actor, "manual run"),
    )
    conn.commit()
    return cur.lastrowid


def complete_risk_run(conn, run_id):
    cur = conn.cursor()
    cur.execute(
        "UPDATE risk_runs SET run_completed_at = ? WHERE id = ?",
        (_utc_now(), run_id),
    )
    conn.commit()


def add_risk_findings(conn, run_id, findings):
    cur = conn.cursor()
    for f in findings:
        cur.execute(
            """
            INSERT INTO risk_findings (
                risk_run_id, company_id, rule_id, signal_value_id, value, severity, explanation, evidence_text, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                f.get("company_id"),
                f.get("rule_id"),
                f.get("signal_value_id"),
                f.get("value"),
                f.get("severity"),
                f.get("explanation"),
                f.get("evidence_text"),
                _utc_now(),
            ),
        )
    conn.commit()


def list_risk_rules(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM risk_rules WHERE enabled = 1 ORDER BY id ASC")
    return cur.fetchall()


def list_latest_risk_findings(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rf.*, rr.name AS rule_name
        FROM risk_findings rf
        JOIN risk_rules rr ON rr.id = rf.rule_id
        WHERE rf.company_id = ?
          AND rf.risk_run_id = (SELECT MAX(id) FROM risk_runs)
        ORDER BY rf.id DESC
        """,
        (company_id,),
    )
    return cur.fetchall()


def create_risk_run_item(conn, run_id, company_id, risk_score, risk_level):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO risk_run_items (risk_run_id, company_id, risk_score, risk_level, computed_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, company_id, float(risk_score), str(risk_level), _utc_now()),
    )
    conn.commit()
    return cur.lastrowid


def add_risk_flags(conn, run_item_id, flags):
    cur = conn.cursor()
    for f in flags:
        cur.execute(
            """
            INSERT INTO risk_flags (
                risk_run_item_id, category, risk_type, severity, confidence,
                description, evidence_text, source_type, source_ref, evidence_page,
                method, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_item_id,
                f.get("category"),
                f.get("risk_type"),
                f.get("severity"),
                f.get("confidence"),
                f.get("description"),
                f.get("evidence_text"),
                f.get("source_type"),
                f.get("source_ref"),
                f.get("evidence_page"),
                f.get("method"),
                f.get("detected_at") or _utc_now(),
            ),
        )
    conn.commit()


def get_latest_risk_run_item(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM risk_run_items
        WHERE company_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (company_id,),
    )
    return cur.fetchone()


def list_latest_risk_flags(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rf.*
        FROM risk_flags rf
        JOIN risk_run_items ri ON ri.id = rf.risk_run_item_id
        WHERE ri.company_id = ?
          AND ri.id = (SELECT MAX(id) FROM risk_run_items WHERE company_id = ?)
        ORDER BY rf.id DESC
        """,
        (company_id, company_id),
    )
    return cur.fetchall()


def get_risk_report(conn, company_id):
    run_item = get_latest_risk_run_item(conn, company_id)
    if not run_item:
        return None
    run_item = dict(run_item)
    flags = [dict(f) for f in list_latest_risk_flags(conn, company_id)]
    output_flags = []
    for f in flags:
        output_flags.append(
            {
                "category": f.get("category"),
                "type": f.get("risk_type"),
                "severity": f.get("severity"),
                "confidence": f.get("confidence"),
                "description": f.get("description"),
                "evidence": f.get("evidence_text"),
                "source": f.get("source_ref") or f.get("source_type"),
                "page": f.get("evidence_page"),
                "method": f.get("method"),
                "detected_at": f.get("detected_at"),
            }
        )
    return {
        "company_id": company_id,
        "risk_score": run_item.get("risk_score"),
        "risk_level": run_item.get("risk_level"),
        "flags": output_flags,
    }


def get_crawl_settings(conn):
    cur = conn.cursor()
    cur.execute("SELECT * FROM crawl_settings ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    return row


def upsert_crawl_settings(conn, mode, intensity, allowlist, news_domains, agent_mode, actor="user"):
    cur = conn.cursor()
    now = _utc_now()
    cur.execute(
        """
        INSERT INTO crawl_settings (mode, intensity, allowlist_json, news_domains_json, agent_mode, updated_at, updated_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (mode, intensity, json.dumps(allowlist or []), json.dumps(news_domains or []), 1 if agent_mode else 0, now, actor),
    )
    conn.commit()


def add_activity(conn, event_type, entity_type=None, entity_id=None, details=None, actor="system"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO activity_log (event_type, entity_type, entity_id, details_json, created_at, actor)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            event_type,
            entity_type,
            entity_id,
            json.dumps(details) if details is not None else None,
            _utc_now(),
            actor,
        ),
    )
    conn.commit()


def list_activity(conn, limit=200):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM activity_log
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )
    return cur.fetchall()


def list_activity_for_company(conn, company_id, limit=200):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM activity_log
        WHERE entity_type = 'company' AND entity_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (company_id, limit),
    )
    return cur.fetchall()


def add_company_highlights(conn, company_id, highlights, model=None, actor="system"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO company_highlights (company_id, highlights_json, model, created_at, created_by)
        VALUES (?, ?, ?, ?, ?)
        """,
        (company_id, json.dumps(highlights), model, _utc_now(), actor),
    )
    conn.commit()


def get_latest_company_highlights(conn, company_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM company_highlights
        WHERE company_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (company_id,),
    )
    return cur.fetchone()


def _ensure_default_risk_rules(conn):
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM risk_rules")
    count = cur.fetchone()[0]
    if count > 0:
        return

    rules = [
        {
            "name": "High churn",
            "description": "Churn rate above 10%",
            "rule_type": "threshold",
            "signal_key": "churn_rate",
            "params_json": json.dumps({"op": "gt", "threshold": 10}),
            "severity": "high",
        },
        {
            "name": "Short runway",
            "description": "Runway less than 6 months",
            "rule_type": "threshold",
            "signal_key": "runway_months",
            "params_json": json.dumps({"op": "lt", "threshold": 6}),
            "severity": "high",
        },
        {
            "name": "Low gross margin",
            "description": "Gross margin below 40%",
            "rule_type": "threshold",
            "signal_key": "gross_margin",
            "params_json": json.dumps({"op": "lt", "threshold": 40}),
            "severity": "medium",
        },
        {
            "name": "Long payback",
            "description": "Payback period above 18 months",
            "rule_type": "threshold",
            "signal_key": "payback_months",
            "params_json": json.dumps({"op": "gt", "threshold": 18}),
            "severity": "medium",
        },
        {
            "name": "Negative growth",
            "description": "Revenue growth rate below 0%",
            "rule_type": "threshold",
            "signal_key": "revenue_growth_rate",
            "params_json": json.dumps({"op": "lt", "threshold": 0}),
            "severity": "high",
        },
        {
            "name": "Low LTV/CAC",
            "description": "LTV/CAC ratio below 3",
            "rule_type": "ltv_cac_ratio",
            "signal_key": None,
            "params_json": json.dumps({"threshold": 3}),
            "severity": "high",
        },
        {
            "name": "ARR/MRR mismatch",
            "description": "ARR not within 10-14x MRR",
            "rule_type": "arr_mrr_mismatch",
            "signal_key": None,
            "params_json": json.dumps({"min": 10, "max": 14}),
            "severity": "medium",
        },
        {
            "name": "Burn vs MRR",
            "description": "Burn rate exceeds 2x MRR",
            "rule_type": "burn_vs_mrr",
            "signal_key": None,
            "params_json": json.dumps({"multiplier": 2}),
            "severity": "medium",
        },
    ]

    now = _utc_now()
    for r in rules:
        cur.execute(
            """
            INSERT INTO risk_rules (name, description, rule_type, signal_key, params_json, severity, enabled, created_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["name"],
                r["description"],
                r["rule_type"],
                r.get("signal_key"),
                r.get("params_json"),
                r["severity"],
                1,
                now,
                "system",
            ),
        )
    conn.commit()


def create_document(
    conn,
    filename,
    storage_path,
    is_global=0,
    actor="user",
    doc_type=None,
    source_type=None,
    extracted_text=None,
):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO documents (filename, storage_path, doc_type, source_type, extracted_text, is_global, created_at, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            filename,
            storage_path,
            doc_type,
            source_type,
            extracted_text,
            1 if is_global else 0,
            _utc_now(),
            actor,
        ),
    )
    conn.commit()
    return cur.lastrowid


def link_document_to_company(conn, document_id, company_id, actor="user"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO document_company_links (document_id, company_id, created_at, created_by)
        VALUES (?, ?, ?, ?)
        """,
        (document_id, company_id, _utc_now(), actor),
    )
    conn.commit()


def add_document_chunks(conn, document_id, chunks):
    cur = conn.cursor()
    now = _utc_now()
    for ch in chunks:
        cur.execute(
            """
            INSERT INTO document_chunks (document_id, page_num, chunk_index, chunk_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (document_id, ch["page_num"], ch["chunk_index"], ch["chunk_text"], now),
        )
    conn.commit()


def list_documents(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT d.*,
               COUNT(DISTINCT l.company_id) AS linked_companies,
               COUNT(DISTINCT c.id) AS chunks
        FROM documents d
        LEFT JOIN document_company_links l ON l.document_id = d.id
        LEFT JOIN document_chunks c ON c.document_id = d.id
        GROUP BY d.id
        ORDER BY d.id DESC
        """
    )
    return cur.fetchall()


def list_document_chunks_for_company(conn, company_id, source_types=None):
    cur = conn.cursor()
    if source_types:
        placeholders = ",".join("?" * len(source_types))
        query = f"""
            SELECT DISTINCT dc.*, d.filename, d.storage_path, d.is_global, d.source_type, d.doc_type, d.id AS document_id
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            LEFT JOIN document_company_links l ON l.document_id = d.id
            WHERE (d.is_global = 1 OR l.company_id = ?)
              AND d.source_type IN ({placeholders})
            ORDER BY dc.id ASC
        """
        params = (company_id, *source_types)
        cur.execute(query, params)
    else:
        cur.execute(
            """
            SELECT DISTINCT dc.*, d.filename, d.storage_path, d.is_global, d.source_type, d.doc_type, d.id AS document_id
            FROM document_chunks dc
            JOIN documents d ON d.id = dc.document_id
            LEFT JOIN document_company_links l ON l.document_id = d.id
            WHERE d.is_global = 1 OR l.company_id = ?
            ORDER BY dc.id ASC
            """,
            (company_id,),
        )
    return cur.fetchall()


def create_optimization_run(conn, criteria_version_id, model_type, rows_used, metrics_json, actor="user"):
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO optimization_runs (criteria_set_version_id, created_at, model_type, rows_used, metrics_json, created_by)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (criteria_version_id, _utc_now(), model_type, rows_used, metrics_json, actor),
    )
    conn.commit()
    return cur.lastrowid


def add_optimization_suggestions(conn, run_id, suggestions):
    cur = conn.cursor()
    for s in suggestions:
        cur.execute(
            """
            INSERT INTO optimization_suggestions (
                optimization_run_id, criterion_id, current_weight, suggested_weight, delta, confidence, accepted, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                s["criterion_id"],
                s["current_weight"],
                s["suggested_weight"],
                s["delta"],
                s.get("confidence"),
                s.get("accepted"),
                s.get("notes"),
            ),
        )
    conn.commit()


def list_optimization_runs(conn):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT * FROM optimization_runs
        ORDER BY id DESC
        """
    )
    return cur.fetchall()


def list_optimization_suggestions(conn, run_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT os.*, c.name AS criterion_name, c.signal_key
        FROM optimization_suggestions os
        JOIN criteria c ON c.id = os.criterion_id
        WHERE os.optimization_run_id = ?
        ORDER BY os.id ASC
        """,
        (run_id,),
    )
    return cur.fetchall()


def mark_suggestions(conn, run_id, accepted_ids):
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE optimization_suggestions
        SET accepted = CASE
            WHEN id IN ({placeholders}) THEN 1
            ELSE 0
        END
        WHERE optimization_run_id = ?
        """.format(placeholders=",".join("?" * len(accepted_ids))) if accepted_ids else
        "UPDATE optimization_suggestions SET accepted = 0 WHERE optimization_run_id = ?",
        (*accepted_ids, run_id) if accepted_ids else (run_id,),
    )
    conn.commit()


def apply_optimization_suggestions(conn, run_id, accepted_ids, actor="user"):
    cur = conn.cursor()
    cur.execute("SELECT criteria_set_version_id FROM optimization_runs WHERE id = ?", (run_id,))
    run_row = cur.fetchone()
    if not run_row:
        return None

    criteria_version_id = run_row["criteria_set_version_id"]
    criteria_rows = list_criteria(conn, criteria_version_id)
    if not criteria_rows:
        return None

    if accepted_ids:
        cur.execute(
            f"""
            SELECT id, criterion_id, suggested_weight
            FROM optimization_suggestions
            WHERE optimization_run_id = ? AND id IN ({",".join("?" * len(accepted_ids))})
            """,
            (run_id, *accepted_ids),
        )
        suggestion_rows = cur.fetchall()
        suggestion_map = {r["criterion_id"]: r["suggested_weight"] for r in suggestion_rows}
    else:
        suggestion_map = {}

    updated = []
    for c in criteria_rows:
        new_weight = suggestion_map.get(c["id"], c["weight"])
        updated.append(
            {
                "name": c["name"],
                "description": c["description"],
                "signal_key": c["signal_key"],
                "weight": new_weight,
                "enabled": bool(c["enabled"]),
                "scoring_method": c["scoring_method"],
                "params_json": c["params_json"],
                "missing_policy": c["missing_policy"],
            }
        )

    new_version_id = replace_criteria_version(conn, criteria_version_id, updated, actor=actor)
    return new_version_id
