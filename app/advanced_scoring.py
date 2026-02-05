import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app import db


@dataclass
class SignalValue:
    key: str
    value: Optional[float]
    value_text: Optional[str]
    value_bool: Optional[int]
    value_json: Optional[str]
    confidence: Optional[float]
    source_type: Optional[str]
    observed_at: Optional[str]
    evidence_text: Optional[str]
    evidence_page: Optional[int]
    raw_row: object


class SignalRepository:
    def __init__(self, conn):
        self.conn = conn

    def get_latest_signals(self, company_id: int) -> Dict[str, SignalValue]:
        rows = db.list_latest_signal_values(self.conn, company_id)
        return {r["signal_key"]: self._row_to_signal(r) for r in rows}

    def get_signal_history(self, company_id: int, signal_key: str, limit: int = 50) -> List[SignalValue]:
        rows = db.list_signal_history(self.conn, company_id, signal_key)
        rows = rows[:limit]
        return [self._row_to_signal(r) for r in rows][::-1]

    def get_signal_history_multi(self, company_id: int, keys: List[str], limit: int = 50) -> Dict[str, List[SignalValue]]:
        history = {}
        for key in keys:
            history[key] = self.get_signal_history(company_id, key, limit=limit)
        return history

    def _row_to_signal(self, row) -> SignalValue:
        def _safe_get(key, default=None):
            try:
                return row[key]
            except Exception:
                return default

        return SignalValue(
            key=_safe_get("signal_key"),
            value=_safe_get("value_num"),
            value_text=_safe_get("value_text"),
            value_bool=_safe_get("value_bool"),
            value_json=_safe_get("value_json"),
            confidence=_safe_get("confidence"),
            source_type=_safe_get("source_type"),
            observed_at=_safe_get("observed_at"),
            evidence_text=_safe_get("evidence_text"),
            evidence_page=_safe_get("evidence_page"),
            raw_row=row,
        )


class ConfidenceModel:
    DEFAULT_MAP = {
        "sec": 0.9,
        "opencorporates": 0.9,
        "gdelt": 0.7,
        "model": 0.5,
        "founder_material": 0.5,
        "crawl": 0.6,
        "upload": 0.8,
        "manual": 0.8,
        "derived": 0.8,
    }

    def __init__(self, default_confidence: float = 0.6):
        self.default_confidence = default_confidence

    def confidence_for(self, signal: SignalValue) -> float:
        if signal is None:
            return self.default_confidence
        if signal.confidence is not None:
            try:
                return max(0.0, min(1.0, float(signal.confidence)))
            except (TypeError, ValueError):
                pass
        return self.DEFAULT_MAP.get(signal.source_type or "", self.default_confidence)


class TrendAnalyzer:
    def __init__(self, drop_threshold: float = 0.3):
        self.drop_threshold = drop_threshold

    def analyze(self, series: List[SignalValue]) -> Dict[str, object]:
        values = [s.value for s in series if s.value is not None]
        if len(values) < 3:
            return {"trend": None, "acceleration": None, "drop_flag": False}

        x = np.arange(len(values))
        y = np.array(values, dtype=float)
        slope = np.polyfit(x, y, 1)[0]

        mid = len(values) // 2
        slope_first = np.polyfit(x[:mid], y[:mid], 1)[0] if mid >= 2 else slope
        slope_second = np.polyfit(x[mid:], y[mid:], 1)[0] if len(values[mid:]) >= 2 else slope
        acceleration = slope_second - slope_first

        recent = values[-1]
        prev_mean = float(np.mean(values[:-1])) if len(values) > 1 else recent
        drop_flag = recent < prev_mean * (1 - self.drop_threshold)

        return {
            "trend": slope,
            "acceleration": acceleration,
            "drop_flag": bool(drop_flag),
        }


class CompositeSignalGenerator:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def compute(self, signals: Dict[str, SignalValue], history: Dict[str, List[SignalValue]]) -> Dict[str, Dict[str, object]]:
        def val(key):
            s = signals.get(key)
            return s.value if s and s.value is not None else None

        composites = {}

        news_change = self._latest_change(history.get("news_volume"))
        growth_momentum = self._weighted_sum(
            {
                "news_volume_change": news_change,
                "hiring_velocity": val("hiring_velocity"),
                "product_launches": val("product_launches"),
            }
        )
        composites["growth_momentum"] = growth_momentum

        market_validation = self._weighted_sum(
            {
                "revenue_growth_rate": val("revenue_growth_rate"),
                "customer_count": val("customer_count"),
                "press_coverage": val("news_volume"),
            }
        )
        composites["market_validation"] = market_validation

        team_quality = self._weighted_sum(
            {
                "founder_experience_years": val("founder_experience_years"),
                "advisor_prestige": val("advisor_prestige"),
                "hiring_success": val("hiring_success"),
            }
        )
        composites["team_quality"] = team_quality

        return composites

    def _latest_change(self, series: Optional[List[SignalValue]]) -> Optional[float]:
        if not series or len(series) < 2:
            return None
        values = [s.value for s in series if s.value is not None]
        if len(values) < 2:
            return None
        prev, curr = values[-2], values[-1]
        if prev == 0:
            return None
        return (curr - prev) / abs(prev) * 100

    def _weighted_sum(self, mapping: Dict[str, Optional[float]]) -> Dict[str, object]:
        items = {k: v for k, v in mapping.items() if v is not None}
        if not items:
            return {"value": None, "explanation": "missing inputs"}
        scores = []
        for key, value in items.items():
            scores.append(self.normalizer.normalize(key, value))
        composite = float(np.mean(scores)) if scores else None
        return {
            "value": composite,
            "explanation": f"avg of {list(items.keys())}",
        }


class BenchmarkNormalizer:
    def __init__(self, stats_df: Optional[pd.DataFrame] = None):
        self.stats_df = stats_df

    def normalize(self, key: str, value: float) -> float:
        if self.stats_df is None or self.stats_df.empty:
            return self._default_scale(value)
        subset = self.stats_df[self.stats_df["metric_key"] == key]
        if subset.empty:
            return self._default_scale(value)

        row = subset.iloc[0]
        p10, p50, p90 = row["p10"], row["p50"], row["p90"]
        if p10 is None or p50 is None or p90 is None:
            return self._default_scale(value)

        if value <= p10:
            return 0.1
        if value >= p90:
            return 0.9
        if value == p50:
            return 0.5
        if value < p50:
            return 0.1 + (value - p10) / (p50 - p10) * 0.4
        return 0.5 + (value - p50) / (p90 - p50) * 0.4

    def _default_scale(self, value: float) -> float:
        return float(1 / (1 + np.exp(-value / 10)))


class ContextScorer:
    EARLY_STAGE = {"pre-seed", "seed"}
    GROWTH_STAGE = {"series a", "series b", "series c", "growth"}

    def resolve(self, company, signals: Dict[str, SignalValue]) -> Dict[str, object]:
        stage = None
        if signals.get("product_stage") and signals["product_stage"].value_text:
            stage = signals["product_stage"].value_text.lower()
        sector = company["industry"].lower() if company["industry"] else None
        return {"stage": stage, "sector": sector}

    def adjust_weights(self, criteria_rows, context: Dict[str, object]):
        stage = (context.get("stage") or "").lower()
        sector = (context.get("sector") or "").lower()

        weight_map = {c["id"]: float(c["weight"]) for c in criteria_rows}

        def boost(signal_keys, multiplier):
            for c in criteria_rows:
                if c["signal_key"] in signal_keys:
                    weight_map[c["id"]] *= multiplier

        if any(s in stage for s in self.EARLY_STAGE):
            boost({"team_quality", "market_size_est", "founder_experience_years"}, 1.3)
        if any(s in stage for s in self.GROWTH_STAGE):
            boost({"mrr", "arr", "revenue_growth_rate", "ltv", "cac"}, 1.3)

        if "deep" in sector or "biotech" in sector:
            boost({"technical_moat_score", "patent_count"}, 1.4)
        if "consumer" in sector:
            boost({"viral_coefficient", "user_growth_rate"}, 1.4)

        # normalize weights to sum to 1
        total = sum(weight_map.values()) or 1.0
        for k in weight_map:
            weight_map[k] = weight_map[k] / total

        return weight_map


class AnomalyDetector:
    def __init__(self, stats_df: Optional[pd.DataFrame] = None, inconsistency_threshold: float = 0.5):
        self.stats_df = stats_df
        self.inconsistency_threshold = inconsistency_threshold

    def detect_outliers(self, signals: Dict[str, SignalValue]) -> List[Dict[str, object]]:
        findings = []
        if self.stats_df is None or self.stats_df.empty:
            return findings

        for key, signal in signals.items():
            if signal.value is None:
                continue
            stats = self.stats_df[self.stats_df["metric_key"] == key]
            if stats.empty:
                continue
            row = stats.iloc[0]
            p10, p90 = row.get("p10"), row.get("p90")
            if p10 is None or p90 is None:
                continue
            if signal.value < p10 * 0.5 or signal.value > p90 * 2:
                findings.append(
                    {
                        "type": "outlier",
                        "signal_key": key,
                        "value": signal.value,
                        "explanation": "Value outside peer range",
                    }
                )
        return findings

    def detect_inconsistencies(self, history: Dict[str, List[SignalValue]]) -> List[Dict[str, object]]:
        findings = []
        for key, series in history.items():
            if len(series) < 2:
                continue
            # compare latest two values from different sources
            latest = series[-1]
            prev = series[-2]
            if latest.source_type == prev.source_type:
                continue
            if latest.value is None or prev.value is None:
                continue
            if prev.value == 0:
                continue
            diff = abs(latest.value - prev.value) / abs(prev.value)
            if diff >= self.inconsistency_threshold:
                findings.append(
                    {
                        "type": "inconsistency",
                        "signal_key": key,
                        "value": latest.value,
                        "explanation": f"{key} differs across sources by {diff:.0%}",
                    }
                )
        return findings


class CompositeScoringAgent:
    def __init__(self, conn, criteria_version_id: Optional[int] = None, ai_budget_usd: float = 0.2):
        self.conn = conn
        self.criteria_version_id = criteria_version_id or db.get_active_criteria_version_id(conn)
        self.ai_budget_usd = ai_budget_usd
        self.ai_cost_usd = 0.0

        self.repo = SignalRepository(conn)
        self.confidence_model = ConfidenceModel()

    def score_company(self, company_id: int, persist_derived: bool = True) -> Dict[str, object]:
        company = db.get_company(self.conn, company_id)
        signals = self.repo.get_latest_signals(company_id)

        # Benchmark stats for normalization and outliers
        bench_id = self._latest_benchmark_set_id()
        stats_df = pd.DataFrame(db.list_benchmark_stats(self.conn, bench_id)) if bench_id else pd.DataFrame()
        normalizer = BenchmarkNormalizer(stats_df if not stats_df.empty else None)

        # Temporal analysis
        temporal_keys = ["news_volume", "revenue_growth_rate", "churn_rate", "mrr", "arr"]
        history = self.repo.get_signal_history_multi(company_id, temporal_keys)
        trend_analyzer = TrendAnalyzer()
        temporal = {key: trend_analyzer.analyze(series) for key, series in history.items()}

        # Composite signals
        composite_gen = CompositeSignalGenerator(normalizer)
        composites = composite_gen.compute(signals, history)

        # Persist derived composite signals
        if persist_derived:
            self._persist_composites(company_id, composites)
            # refresh latest signals
            signals = self.repo.get_latest_signals(company_id)

        # Contextual weighting
        context = ContextScorer().resolve(company, signals)
        criteria_rows = db.list_criteria(self.conn, self.criteria_version_id)
        weight_map = ContextScorer().adjust_weights(criteria_rows, context)

        # Score criteria
        scored = []
        total = 0.0
        weight_used = 0.0

        for criterion in criteria_rows:
            if not criterion["enabled"]:
                continue
            signal_key = criterion["signal_key"]
            signal = signals.get(signal_key)

            raw_score, explanation = self._evaluate(criterion, signal)
            if raw_score is None:
                continue

            confidence = self.confidence_model.confidence_for(signal)
            weight = weight_map.get(criterion["id"], float(criterion["weight"]))
            contribution = raw_score * confidence * weight

            total += contribution
            weight_used += weight * confidence

            scored.append(
                {
                    "criterion": criterion["name"],
                    "signal_key": signal_key,
                    "base_score": raw_score,
                    "confidence": confidence,
                    "weight": weight,
                    "contribution": contribution,
                    "explanation": explanation,
                }
            )

        normalized_score = total / weight_used if weight_used > 0 else 0.0

        # Anomalies
        anomaly = AnomalyDetector(stats_df if not stats_df.empty else None)
        outliers = anomaly.detect_outliers(signals)
        inconsistencies = anomaly.detect_inconsistencies(history)

        temporal_anomalies = []
        for key, result in temporal.items():
            if result.get("drop_flag"):
                temporal_anomalies.append(
                    {
                        "type": "sudden_drop",
                        "signal_key": key,
                        "explanation": f"{key} dropped sharply in recent period",
                    }
                )

        return {
            "company_id": company_id,
            "company_name": company["name"],
            "context": context,
            "composites": composites,
            "temporal": temporal,
            "criteria_scores": scored,
            "total_score": normalized_score,
            "ai_cost_usd": self.ai_cost_usd,
            "anomalies": outliers + inconsistencies + temporal_anomalies,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }

    def _evaluate(self, criterion, signal: Optional[SignalValue]):
        if signal is None or signal.value is None:
            policy = criterion["missing_policy"]
            if policy == "neutral":
                return 0.5, "missing signal; neutral score"
            if policy == "exclude":
                return None, "missing signal; excluded"
            return 0.0, "missing signal; zero score"

        params = {}
        if criterion["params_json"]:
            try:
                params = json.loads(criterion["params_json"])
            except json.JSONDecodeError:
                params = {}

        method = criterion["scoring_method"]
        value = signal.value

        if method == "binary":
            op = params.get("op", "gte")
            threshold = params.get("threshold")
            if threshold is None:
                return 0.0, "missing threshold"
            passed = False
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
            return (1.0 if passed else 0.0), f"binary {op} {threshold}"

        if method == "linear":
            min_v = params.get("min")
            max_v = params.get("max")
            if min_v is None or max_v is None or max_v == min_v:
                return 0.0, "invalid linear params"
            score = (value - min_v) / (max_v - min_v)
            score = max(0.0, min(1.0, float(score)))
            return score, f"linear {min_v}-{max_v}"

        if method == "bucket":
            buckets = params.get("buckets", [])
            for b in buckets:
                if value >= b.get("min", float("-inf")) and value <= b.get("max", float("inf")):
                    return float(b.get("score", 0.0)), "bucket"
            return 0.0, "no bucket match"

        return 0.0, "unsupported scoring method"

    def _persist_composites(self, company_id: int, composites: Dict[str, Dict[str, object]]):
        for key, payload in composites.items():
            value = payload.get("value")
            if value is None:
                continue
            db.add_signal_value(
                self.conn,
                company_id,
                key,
                value_num=float(value),
                value_text=None,
                value_bool=None,
                value_json=None,
                source_type="derived",
                source_ref="composite",
                notes=payload.get("explanation"),
                confidence=0.8,
                evidence_text=None,
                evidence_page=None,
            )

    def _latest_benchmark_set_id(self) -> Optional[int]:
        sets = db.list_benchmark_sets(self.conn)
        return sets[0]["id"] if sets else None
