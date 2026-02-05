import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app import db


MIN_LABELS = 50
MAX_RELATIVE_CHANGE = 0.3
MAX_ABS_CHANGE_IF_ZERO = 0.05


@dataclass
class OptimizationResult:
    run_id: int
    metrics: dict
    suggestions_df: pd.DataFrame
    rows_used: int
    feature_keys: list


def _load_training_data(conn):
    ratings = db.list_latest_ratings(conn)
    if not ratings:
        return None

    ratings_df = pd.DataFrame(ratings)[["company_id", "rating_int"]]

    defs = db.list_signal_definitions(conn)
    defs_df = pd.DataFrame(defs)
    if defs_df.empty:
        return None

    eligible = defs_df[(defs_df["disabled"] == 0) & (defs_df["value_type"].isin(["number", "bool"]))]
    if eligible.empty:
        return None

    feature_keys = eligible["key"].tolist()

    placeholders = ",".join(["?"] * len(feature_keys))
    query = f"""
        SELECT sv.company_id, sv.signal_key, sv.value_num, sv.value_bool
        FROM signal_values sv
        JOIN (
            SELECT company_id, signal_key, MAX(id) AS max_id
            FROM signal_values
            GROUP BY company_id, signal_key
        ) latest ON sv.id = latest.max_id
        WHERE sv.signal_key IN ({placeholders})
    """
    signals_df = pd.read_sql_query(query, conn, params=feature_keys)

    if signals_df.empty:
        return None

    signals_df["value"] = signals_df["value_num"].fillna(signals_df["value_bool"].fillna(0))

    matrix = signals_df.pivot_table(
        index="company_id",
        columns="signal_key",
        values="value",
        aggfunc="first",
    )

    matrix = matrix.reindex(columns=feature_keys)
    matrix = matrix.fillna(0)

    data = ratings_df.merge(matrix, left_on="company_id", right_index=True, how="left")
    data = data.fillna(0)

    X = data[feature_keys].astype(float)
    y = data["rating_int"].astype(float)

    return X, y, feature_keys


def _normalize_weights(coefs, current_total_weight):
    magnitudes = np.abs(coefs)
    total_mag = magnitudes.sum()
    if total_mag == 0:
        return np.zeros_like(coefs)
    return (magnitudes / total_mag) * current_total_weight * np.sign(coefs)


def _cap_weight_change(current, suggested):
    if abs(current) < 1e-6:
        max_delta = MAX_ABS_CHANGE_IF_ZERO
        delta = suggested
        delta = max(-max_delta, min(max_delta, delta))
        return delta
    max_delta = abs(current) * MAX_RELATIVE_CHANGE
    delta = suggested - current
    delta = max(-max_delta, min(max_delta, delta))
    return current + delta


def run_optimization(conn, criteria_version_id, actor="user"):
    data = _load_training_data(conn)
    if data is None:
        return None, "No ratings or numeric/bool signals available yet."

    X, y, feature_keys = data
    rows_used = len(y)
    if rows_used < MIN_LABELS:
        return None, f"Need at least {MIN_LABELS} rated companies; currently {rows_used}."

    pipeline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }

    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridge"]

    scale = np.where(scaler.scale_ == 0, 1.0, scaler.scale_)
    coefs_unscaled = ridge.coef_ / scale

    criteria_rows = db.list_criteria(conn, criteria_version_id)
    if not criteria_rows:
        return None, "No criteria to optimize."

    total_weight = sum(abs(c["weight"]) for c in criteria_rows if c["enabled"])
    if total_weight == 0:
        total_weight = 1.0

    suggested_weights = _normalize_weights(coefs_unscaled, total_weight)

    suggestions = []
    max_coef = float(np.max(np.abs(coefs_unscaled))) if coefs_unscaled.size else 0.0

    coef_map = dict(zip(feature_keys, suggested_weights))
    raw_coef_map = dict(zip(feature_keys, coefs_unscaled))

    for c in criteria_rows:
        if not c["enabled"]:
            continue
        if c["signal_key"] not in coef_map:
            continue

        current_weight = float(c["weight"])
        suggested_weight = float(coef_map[c["signal_key"]])
        capped_weight = _cap_weight_change(current_weight, suggested_weight)
        delta = capped_weight - current_weight
        raw_coef = float(raw_coef_map.get(c["signal_key"], 0.0))
        confidence = abs(raw_coef) / max_coef if max_coef else 0.0

        suggestions.append(
            {
                "criterion_id": c["id"],
                "criterion_name": c["name"],
                "signal_key": c["signal_key"],
                "current_weight": current_weight,
                "suggested_weight": capped_weight,
                "delta": delta,
                "confidence": confidence,
                "notes": "negative coefficient" if raw_coef < 0 else None,
            }
        )

    run_id = db.create_optimization_run(
        conn,
        criteria_version_id,
        model_type="ridge_regression",
        rows_used=rows_used,
        metrics_json=json.dumps(metrics),
        actor=actor,
    )
    db.add_optimization_suggestions(conn, run_id, suggestions)

    suggestions_df = pd.DataFrame(suggestions)
    return OptimizationResult(run_id, metrics, suggestions_df, rows_used, feature_keys), None

