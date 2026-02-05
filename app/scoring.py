import json
from datetime import datetime

from app import db


def _utc_now():
    return datetime.utcnow().isoformat(timespec="seconds")


def _parse_signal_value(signal_row):
    if signal_row is None:
        return None
    if signal_row["value_num"] is not None:
        return signal_row["value_num"]
    if signal_row["value_bool"] is not None:
        return bool(signal_row["value_bool"])
    if signal_row["value_text"] is not None:
        return signal_row["value_text"]
    if signal_row["value_json"]:
        try:
            return json.loads(signal_row["value_json"])
        except json.JSONDecodeError:
            return signal_row["value_json"]
    return None


def _apply_missing_policy(policy):
    if policy == "neutral":
        return 0.5, "Missing signal; applied neutral score"
    if policy == "exclude":
        return None, "Missing signal; excluded from total"
    return 0.0, "Missing signal; applied zero score"


def _binary_score(value, params):
    op = params.get("op", "gte")
    threshold = params.get("threshold")
    if threshold is None:
        return 0.0, "Missing threshold in params"

    passed = False
    try:
        if op == "gte":
            passed = value >= threshold
        elif op == "gt":
            passed = value > threshold
        elif op == "lte":
            passed = value <= threshold
        elif op == "lt":
            passed = value < threshold
        elif op == "eq":
            passed = value == threshold
        elif op == "contains":
            passed = str(threshold).lower() in str(value).lower()
    except TypeError:
        passed = False

    return (1.0 if passed else 0.0), f"Binary rule {op} {threshold} => {'pass' if passed else 'fail'}"


def _linear_score(value, params):
    min_v = params.get("min")
    max_v = params.get("max")
    clamp = params.get("clamp", True)
    if min_v is None or max_v is None or max_v == min_v:
        return 0.0, "Invalid linear params"
    score = (value - min_v) / (max_v - min_v)
    if clamp:
        score = max(0.0, min(1.0, score))
    return float(score), f"Linear score between {min_v} and {max_v}"


def _bucket_score(value, params):
    buckets = params.get("buckets", [])
    for b in buckets:
        if value >= b.get("min", float("-inf")) and value <= b.get("max", float("inf")):
            return float(b.get("score", 0.0)), "Bucket score"
    return 0.0, "No bucket match"


def _theme_key(theme):
    if not theme:
        return "Uncategorized"
    return str(theme).strip()


def _is_our_angle(theme):
    if not theme:
        return False
    return "our angle" in str(theme).lower()


def _compute_effective_weights(criteria):
    groups = {}
    for c in criteria:
        if not c.get("enabled"):
            continue
        theme = _theme_key(c.get("theme"))
        groups.setdefault(theme, []).append(c)

    main_themes = [t for t in groups.keys() if not _is_our_angle(t)]
    our_angle_themes = [t for t in groups.keys() if _is_our_angle(t)]

    theme_totals = {t: sum(float(c.get("weight", 0)) for c in groups[t]) for t in main_themes}
    active_themes = [t for t, total in theme_totals.items() if total > 0]
    theme_count = len(active_themes)

    weight_map = {}
    if theme_count == 0:
        total_weight = sum(float(c.get("weight", 0)) for t in main_themes for c in groups[t])
        for t in main_themes:
            for c in groups[t]:
                if total_weight > 0:
                    weight_map[c["id"]] = float(c.get("weight", 0)) / total_weight
                else:
                    weight_map[c["id"]] = 0.0
    else:
        for t in active_themes:
            total = theme_totals[t]
            for c in groups[t]:
                weight_map[c["id"]] = (float(c.get("weight", 0)) / total) * (1 / theme_count)

    our_angle_map = {}
    our_angle_total = sum(float(c.get("weight", 0)) for t in our_angle_themes for c in groups[t])
    if our_angle_total > 0:
        for t in our_angle_themes:
            for c in groups[t]:
                our_angle_map[c["id"]] = float(c.get("weight", 0)) / our_angle_total

    return weight_map, our_angle_map


def evaluate_criterion(criterion, signal_row):
    value = _parse_signal_value(signal_row)
    if value is None:
        return _apply_missing_policy(criterion["missing_policy"])

    params = {}
    if criterion["params_json"]:
        try:
            params = json.loads(criterion["params_json"])
        except json.JSONDecodeError:
            params = {}

    method = criterion["scoring_method"]
    if method == "binary":
        return _binary_score(value, params)
    if method == "linear":
        return _linear_score(value, params)
    if method == "bucket":
        return _bucket_score(value, params)

    # fallback
    if isinstance(value, bool):
        return (1.0 if value else 0.0), "Boolean signal"
    if isinstance(value, (int, float)):
        return (1.0 if value > 0 else 0.0), "Numeric fallback"
    return 0.0, "Unsupported scoring method"


def calculate_composite_signals(company_id, conn):
    signals = db.get_all_signals(conn, company_id)

    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    news_vol = _to_float(signals.get("news_volume") or 0)
    prev_news = _to_float(signals.get("news_volume_prev_month") or 0)
    news_trend = (news_vol - prev_news) / max(prev_news, 1)

    hiring_velocity = _to_float(signals.get("hiring_velocity") or 0)
    product_launches = _to_float(signals.get("product_launches") or 0)
    growth_momentum = (news_trend * 0.5) + (hiring_velocity * 0.3) + (min(product_launches / 3, 1) * 0.2)

    revenue = _to_float(signals.get("mrr") or 0)
    customers = _to_float(signals.get("customer_count") or 0)
    validation = (revenue / 10000) * 0.5 + (customers / 100) * 0.3 + news_trend * 0.2

    founder_exits = _to_float(signals.get("founder_exits") or 0)
    founder_exp = _to_float(signals.get("founder_experience_years") or 0)
    team_size = _to_float(signals.get("team_size") or 0)
    advisor_prestige = _to_float(signals.get("advisor_prestige") or 0)
    team_quality = (min(founder_exp / 10, 1) * 0.4) + (min(founder_exits / 1, 1) * 0.3) + (min(team_size / 10, 1) * 0.2) + (min(advisor_prestige / 10, 1) * 0.1)

    db.upsert_signal_value(conn, company_id, "growth_momentum", growth_momentum, source_type="derived", source_ref="composite")
    db.upsert_signal_value(conn, company_id, "market_validation", validation, source_type="derived", source_ref="composite")
    db.upsert_signal_value(conn, company_id, "team_quality", team_quality, source_type="derived", source_ref="composite")


def run_scoring(conn, criteria_version_id, actor="user"):
    cur = conn.cursor()
    now = _utc_now()
    cur.execute(
        """
        INSERT INTO score_runs (criteria_set_version_id, run_started_at, triggered_by, notes)
        VALUES (?, ?, ?, ?)
        """,
        (criteria_version_id, now, actor, "manual run"),
    )
    score_run_id = cur.lastrowid

    cur.execute("SELECT * FROM companies")
    companies = cur.fetchall()
    criteria = db.list_criteria(conn, criteria_version_id)
    effective_weights, our_angle_weights = _compute_effective_weights(criteria)

    for company in companies:
        calculate_composite_signals(company["id"], conn)
        total = 0.0
        used_weight = 0.0
        our_angle_total = 0.0
        our_angle_weight = 0.0
        score_item_id = None

        for criterion in criteria:
            if not criterion["enabled"]:
                continue
            cur.execute(
                """
                SELECT * FROM signal_values
                WHERE company_id = ? AND signal_key = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (company["id"], criterion["signal_key"]),
            )
            signal_row = cur.fetchone()
            raw_score, explanation = evaluate_criterion(criterion, signal_row)
            if raw_score is None:
                continue

            theme = criterion.get("theme")
            is_our_angle = _is_our_angle(theme)

            if is_our_angle:
                weight = our_angle_weights.get(criterion["id"], 0.0)
            else:
                weight = effective_weights.get(criterion["id"], 0.0)
            confidence = 1.0
            if signal_row is not None and signal_row["confidence"] is not None:
                try:
                    confidence = float(signal_row["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))

            effective_weight = weight * confidence
            contribution = float(raw_score) * effective_weight
            if is_our_angle:
                our_angle_total += contribution
                our_angle_weight += effective_weight
            else:
                total += contribution
                used_weight += effective_weight

            explanation = f"{explanation} | confidence {confidence:.2f}"
            if theme:
                explanation = f"{explanation} | theme {theme}"
            if is_our_angle:
                explanation = f"{explanation} | our_angle"

            if score_item_id is None:
                cur.execute(
                    """
                    INSERT INTO score_items (score_run_id, company_id, total_score, raw_total, normalized_total, weight_used, our_angle_score, scored_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (score_run_id, company["id"], 0.0, 0.0, 0.0, 0.0, None, now),
                )
                score_item_id = cur.lastrowid

            cur.execute(
                """
                INSERT INTO score_components (
                    score_item_id, criterion_id, signal_value_id, raw_value, raw_score, weight,
                    contribution, passed, explanation, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    score_item_id,
                    criterion["id"],
                    signal_row["id"] if signal_row else None,
                    json.dumps(_parse_signal_value(signal_row)) if signal_row else None,
                    float(raw_score),
                    float(weight),
                    contribution,
                    1 if raw_score >= 1.0 else 0,
                    explanation,
                    now,
                ),
            )

        raw_total = total
        normalized_total = raw_total / used_weight if used_weight > 0 else 0.0
        our_angle_score = our_angle_total / our_angle_weight if our_angle_weight > 0 else None

        if score_item_id is None:
            cur.execute(
                """
                INSERT INTO score_items (score_run_id, company_id, total_score, raw_total, normalized_total, weight_used, our_angle_score, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (score_run_id, company["id"], normalized_total, raw_total, normalized_total, used_weight, our_angle_score, now),
            )
            score_item_id = cur.lastrowid

        cur.execute(
            """
            UPDATE score_items
            SET total_score = ?, raw_total = ?, normalized_total = ?, weight_used = ?, our_angle_score = ?
            WHERE id = ?
            """,
            (normalized_total, raw_total, normalized_total, used_weight, our_angle_score, score_item_id),
        )

    cur.execute("UPDATE score_runs SET run_completed_at = ? WHERE id = ?", (now, score_run_id))
    conn.commit()
    return score_run_id


def run_scoring_for_company(conn, criteria_version_id, company_id, actor="user"):
    cur = conn.cursor()
    now = _utc_now()
    cur.execute(
        """
        INSERT INTO score_runs (criteria_set_version_id, run_started_at, triggered_by, notes)
        VALUES (?, ?, ?, ?)
        """,
        (criteria_version_id, now, actor, "auto run (single company)"),
    )
    score_run_id = cur.lastrowid

    company = db.get_company(conn, company_id)
    if company is None:
        return None

    criteria = db.list_criteria(conn, criteria_version_id)
    effective_weights, our_angle_weights = _compute_effective_weights(criteria)

    calculate_composite_signals(company["id"], conn)
    total = 0.0
    used_weight = 0.0
    our_angle_total = 0.0
    our_angle_weight = 0.0
    score_item_id = None

    for criterion in criteria:
        if not criterion["enabled"]:
            continue
        cur.execute(
            """
            SELECT * FROM signal_values
            WHERE company_id = ? AND signal_key = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (company["id"], criterion["signal_key"]),
        )
        signal_row = cur.fetchone()
        raw_score, explanation = evaluate_criterion(criterion, signal_row)
        if raw_score is None:
            continue

        theme = criterion.get("theme")
        is_our_angle = _is_our_angle(theme)

        if is_our_angle:
            weight = our_angle_weights.get(criterion["id"], 0.0)
        else:
            weight = effective_weights.get(criterion["id"], 0.0)
        confidence = 1.0
        if signal_row is not None and signal_row["confidence"] is not None:
            try:
                confidence = float(signal_row["confidence"])
            except (TypeError, ValueError):
                confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))

        effective_weight = weight * confidence
        contribution = float(raw_score) * effective_weight
        if is_our_angle:
            our_angle_total += contribution
            our_angle_weight += effective_weight
        else:
            total += contribution
            used_weight += effective_weight

        explanation = f"{explanation} | confidence {confidence:.2f}"
        if theme:
            explanation = f"{explanation} | theme {theme}"
        if is_our_angle:
            explanation = f"{explanation} | our_angle"

        if score_item_id is None:
            cur.execute(
                """
                INSERT INTO score_items (score_run_id, company_id, total_score, raw_total, normalized_total, weight_used, our_angle_score, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (score_run_id, company["id"], 0.0, 0.0, 0.0, 0.0, None, now),
            )
            score_item_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO score_components (
                score_item_id, criterion_id, signal_value_id, raw_value, raw_score, weight,
                contribution, passed, explanation, evaluated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                score_item_id,
                criterion["id"],
                signal_row["id"] if signal_row else None,
                json.dumps(_parse_signal_value(signal_row)) if signal_row else None,
                float(raw_score),
                float(weight),
                contribution,
                1 if raw_score >= 1.0 else 0,
                explanation,
                now,
            ),
        )

    raw_total = total
    normalized_total = raw_total / used_weight if used_weight > 0 else 0.0
    our_angle_score = our_angle_total / our_angle_weight if our_angle_weight > 0 else None

    if score_item_id is None:
        cur.execute(
            """
            INSERT INTO score_items (score_run_id, company_id, total_score, raw_total, normalized_total, weight_used, our_angle_score, scored_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (score_run_id, company["id"], normalized_total, raw_total, normalized_total, used_weight, our_angle_score, now),
        )
        score_item_id = cur.lastrowid

    cur.execute(
        """
        UPDATE score_items
        SET total_score = ?, raw_total = ?, normalized_total = ?, weight_used = ?, our_angle_score = ?
        WHERE id = ?
        """,
        (normalized_total, raw_total, normalized_total, used_weight, our_angle_score, score_item_id),
    )

    cur.execute("UPDATE score_runs SET run_completed_at = ? WHERE id = ?", (now, score_run_id))
    conn.commit()
    try:
        db.add_activity(conn, "score_company", "company", company_id, {"criteria_version_id": criteria_version_id}, actor=actor)
    except Exception:
        pass
    return score_run_id


def evaluate_criterion_with_confidence(criterion, signal_row):
    value = _parse_signal_value(signal_row)
    if value is None:
        missing_score, missing_expl = _apply_missing_policy(criterion["missing_policy"])
        if missing_score is None:
            return None, missing_expl, 0.0
        return missing_score, missing_expl, 0.0

    raw_score, explanation = evaluate_criterion(criterion, signal_row)
    if raw_score is None:
        return None, explanation, 0.0

    confidence = 1.0
    if signal_row and signal_row.get("confidence") is not None:
        try:
            confidence = float(signal_row["confidence"])
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 1.0

    effective_weight = float(criterion["weight"]) * confidence
    explanation = f"{explanation} | confidence={confidence:.2f}, effective_weight={effective_weight:.2f}"
    return raw_score, explanation, effective_weight


def run_scoring_v2(conn, criteria_version_id, actor="user"):
    cur = conn.cursor()
    now = _utc_now()
    cur.execute(
        """
        INSERT INTO score_runs (criteria_set_version_id, run_started_at, triggered_by, notes)
        VALUES (?, ?, ?, ?)
        """,
        (criteria_version_id, now, actor, "manual run v2"),
    )
    score_run_id = cur.lastrowid

    cur.execute("SELECT * FROM companies")
    companies = cur.fetchall()
    criteria = db.list_criteria(conn, criteria_version_id)

    for company in companies:
        total = 0.0
        used_weight = 0.0
        score_item_id = None

        for criterion in criteria:
            if not criterion["enabled"]:
                continue
            cur.execute(
                """
                SELECT * FROM signal_values
                WHERE company_id = ? AND signal_key = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (company["id"], criterion["signal_key"]),
            )
            signal_row = cur.fetchone()
            raw_score, explanation, effective_weight = evaluate_criterion_with_confidence(criterion, signal_row)
            if raw_score is None:
                continue

            contribution = float(raw_score) * effective_weight
            total += contribution
            used_weight += effective_weight

            if score_item_id is None:
                cur.execute(
                    """
                    INSERT INTO score_items (score_run_id, company_id, total_score, raw_total, normalized_total, weight_used, scored_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (score_run_id, company["id"], 0.0, 0.0, 0.0, 0.0, now),
                )
                score_item_id = cur.lastrowid

            cur.execute(
                """
                INSERT INTO score_components (
                    score_item_id, criterion_id, signal_value_id, raw_value, raw_score, weight,
                    contribution, passed, explanation, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    score_item_id,
                    criterion["id"],
                    signal_row["id"] if signal_row else None,
                    json.dumps(_parse_signal_value(signal_row)) if signal_row else None,
                    float(raw_score),
                    float(effective_weight),
                    contribution,
                    1 if raw_score >= 1.0 else 0,
                    explanation,
                    now,
                ),
            )

        raw_total = total
        normalized_total = raw_total / used_weight if used_weight > 0 else 0.0

        if score_item_id is None:
            cur.execute(
                """
                INSERT INTO score_items (score_run_id, company_id, total_score, raw_total, normalized_total, weight_used, scored_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (score_run_id, company["id"], normalized_total, raw_total, normalized_total, used_weight, now),
            )
            score_item_id = cur.lastrowid

        cur.execute(
            """
            UPDATE score_items
            SET total_score = ?, raw_total = ?, normalized_total = ?, weight_used = ?
            WHERE id = ?
            """,
            (normalized_total, raw_total, normalized_total, used_weight, score_item_id),
        )

    cur.execute("UPDATE score_runs SET run_completed_at = ? WHERE id = ?", (now, score_run_id))
    conn.commit()
    return score_run_id
