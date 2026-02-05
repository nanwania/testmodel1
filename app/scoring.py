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
            raw_score, explanation = evaluate_criterion(criterion, signal_row)
            if raw_score is None:
                continue

            weight = float(criterion["weight"])
            confidence = 1.0
            if signal_row is not None and signal_row["confidence"] is not None:
                try:
                    confidence = float(signal_row["confidence"])
                except (TypeError, ValueError):
                    confidence = 1.0
            confidence = max(0.0, min(1.0, confidence))

            effective_weight = weight * confidence
            contribution = float(raw_score) * effective_weight
            total += contribution
            used_weight += effective_weight

            explanation = f"{explanation} | confidence {confidence:.2f}"

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
                    float(criterion["weight"]),
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
