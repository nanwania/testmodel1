import math
from typing import Dict, List, Optional, Tuple

import pandas as pd

from app import db


def _norm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def compute_benchmark_stats(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str]):
    stats_rows = []
    df = df.copy()

    for col in group_cols:
        df[col] = df[col].apply(_norm)

    # global stats
    stats_rows.extend(_stats_for_group(df, metric_cols, sector=None, stage=None, geo=None))

    if group_cols:
        grouped = df.groupby(group_cols)
        for group_vals, group_df in grouped:
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)
            group_data = dict(zip(group_cols, group_vals))
            stats_rows.extend(
                _stats_for_group(
                    group_df,
                    metric_cols,
                    sector=group_data.get("sector"),
                    stage=group_data.get("stage"),
                    geo=group_data.get("geo"),
                )
            )

    return stats_rows


def _stats_for_group(df: pd.DataFrame, metric_cols: List[str], sector=None, stage=None, geo=None):
    rows = []
    for metric in metric_cols:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue
        quantiles = series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
        rows.append(
            {
                "sector": sector,
                "stage": stage,
                "geo": geo,
                "metric_key": metric,
                "p10": float(quantiles.get(0.1)),
                "p25": float(quantiles.get(0.25)),
                "p50": float(quantiles.get(0.5)),
                "p75": float(quantiles.get(0.75)),
                "p90": float(quantiles.get(0.9)),
                "sample_size": int(series.shape[0]),
            }
        )
    return rows


def run_benchmark(conn, benchmark_set_id, actor="user"):
    stats_rows = db.list_benchmark_stats(conn, benchmark_set_id)
    if not stats_rows:
        return None, "No benchmark stats found. Upload benchmarks first."

    stats_df = pd.DataFrame(stats_rows)

    companies = db.list_companies(conn)
    run_id = db.create_benchmark_run(conn, benchmark_set_id, actor=actor)
    results = []

    for company in companies:
        signal_rows = db.list_latest_signal_values(conn, company["id"])
        signal_map = {s["signal_key"]: s for s in signal_rows}

        sector = _norm(company["industry"])
        stage = _norm(_signal_value(signal_map.get("product_stage")))
        geo = _norm(company["location"])

        for metric_key in stats_df["metric_key"].unique():
            signal_row = signal_map.get(metric_key)
            if not signal_row:
                continue
            value = _signal_value(signal_row)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            stats = _match_stats(stats_df, metric_key, sector, stage, geo)
            if stats is None:
                continue

            band = _percentile_band(value, stats)
            vs_median = _vs_median(value, stats)
            results.append(
                {
                    "company_id": company["id"],
                    "metric_key": metric_key,
                    "metric_value": value,
                    "percentile_band": band,
                    "vs_median": vs_median,
                    "sector": sector,
                    "stage": stage,
                    "geo": geo,
                }
            )

    if results:
        db.add_benchmark_results(conn, run_id, results)
    db.complete_benchmark_run(conn, run_id)
    return run_id, None


def _signal_value(signal_row):
    if signal_row is None:
        return None
    if signal_row["value_num"] is not None:
        return signal_row["value_num"]
    if signal_row["value_bool"] is not None:
        return float(signal_row["value_bool"])
    if signal_row["value_text"] is not None:
        return signal_row["value_text"]
    return None


def _match_stats(stats_df, metric_key, sector, stage, geo):
    candidates = [
        {"sector": sector, "stage": stage, "geo": geo},
        {"sector": sector, "stage": stage, "geo": None},
        {"sector": sector, "stage": None, "geo": None},
        {"sector": None, "stage": None, "geo": None},
    ]
    for cand in candidates:
        subset = stats_df[
            (stats_df["metric_key"] == metric_key)
            & (stats_df["sector"].fillna("") == (cand["sector"] or ""))
            & (stats_df["stage"].fillna("") == (cand["stage"] or ""))
            & (stats_df["geo"].fillna("") == (cand["geo"] or ""))
        ]
        if not subset.empty:
            return subset.iloc[0]
    return None


def _percentile_band(value: float, stats_row) -> str:
    if value < stats_row["p10"]:
        return "<10th"
    if value < stats_row["p25"]:
        return "10-25th"
    if value < stats_row["p50"]:
        return "25-50th"
    if value < stats_row["p75"]:
        return "50-75th"
    if value < stats_row["p90"]:
        return "75-90th"
    return ">90th"


def _vs_median(value: float, stats_row) -> Optional[float]:
    median = stats_row.get("p50")
    if median in (None, 0):
        return None
    return round(float(value) / float(median), 2)
