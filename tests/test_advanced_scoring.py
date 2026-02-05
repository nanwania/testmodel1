import unittest

import numpy as np

from app.advanced_scoring import (
    AnomalyDetector,
    CompositeSignalGenerator,
    ConfidenceModel,
    SignalValue,
    TrendAnalyzer,
    ContextScorer,
)


class DummyNormalizer:
    def normalize(self, key, value):
        return value / 100.0


class AdvancedScoringTests(unittest.TestCase):
    def test_composite_signal_generator(self):
        normalizer = DummyNormalizer()
        generator = CompositeSignalGenerator(normalizer)

        signals = {
            "revenue_growth_rate": SignalValue("revenue_growth_rate", 50, None, None, None, None, None, None, None, None, None),
            "customer_count": SignalValue("customer_count", 100, None, None, None, None, None, None, None, None, None),
            "news_volume": SignalValue("news_volume", 200, None, None, None, None, None, None, None, None, None),
        }
        history = {"news_volume": [signals["news_volume"]]}

        composites = generator.compute(signals, history)
        self.assertIn("market_validation", composites)
        value = composites["market_validation"]["value"]
        self.assertAlmostEqual(value, np.mean([0.5, 1.0, 2.0]))

    def test_trend_analyzer_drop_flag(self):
        analyzer = TrendAnalyzer(drop_threshold=0.3)
        series = [
            SignalValue("mrr", 100, None, None, None, None, None, None, None, None, None),
            SignalValue("mrr", 110, None, None, None, None, None, None, None, None, None),
            SignalValue("mrr", 60, None, None, None, None, None, None, None, None, None),
        ]
        result = analyzer.analyze(series)
        self.assertTrue(result["drop_flag"])

    def test_confidence_model(self):
        model = ConfidenceModel()
        signal = SignalValue("mrr", 100, None, None, None, None, "sec", None, None, None, None)
        self.assertGreaterEqual(model.confidence_for(signal), 0.8)

    def test_context_weight_adjustment(self):
        scorer = ContextScorer()
        criteria = [
            {"id": 1, "signal_key": "team_quality", "weight": 0.2},
            {"id": 2, "signal_key": "market_size_est", "weight": 0.2},
            {"id": 3, "signal_key": "mrr", "weight": 0.2},
        ]
        context = {"stage": "seed", "sector": "saas"}
        weights = scorer.adjust_weights(criteria, context)
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertGreater(weights[1], 0.2)

    def test_anomaly_detector_inconsistency(self):
        detector = AnomalyDetector(stats_df=None, inconsistency_threshold=0.5)
        history = {
            "mrr": [
                SignalValue("mrr", 100, None, None, None, None, "manual", None, None, None, None),
                SignalValue("mrr", 200, None, None, None, None, "model", None, None, None, None),
            ]
        }
        findings = detector.detect_inconsistencies(history)
        self.assertTrue(findings)


if __name__ == "__main__":
    unittest.main()
