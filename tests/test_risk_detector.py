import math
import unittest

from app.risk_detector import RiskDetectionEngine, RuleBasedDetector, PatternDetector


class RiskDetectorTests(unittest.TestCase):
    def test_risk_score_formula(self):
        flags = [
            {"severity": "medium", "confidence": 1.0, "weight": 1.0},
        ]
        score = RiskDetectionEngine.compute_risk_score(flags)
        expected = 1 - math.exp(-0.4)
        self.assertAlmostEqual(score, expected, places=5)

    def test_rule_based_ratio(self):
        rules = {
            "financial": [
                {
                    "type": "burn_multiple",
                    "signal_keys": ["burn_rate", "revenue"],
                    "calc": "ratio",
                    "rule": {"op": "gt", "threshold": 10},
                    "severity": "high",
                    "weight": 1.0,
                    "description": "Burn exceeds revenue by >10x",
                }
            ]
        }
        detector = RuleBasedDetector(rules)
        signal_map = {
            "burn_rate": {"value_num": 500, "source_type": "manual"},
            "revenue": {"value_num": 40, "source_type": "manual"},
        }
        flags = detector.detect(signal_map)
        self.assertTrue(flags)

    def test_round_number_pattern(self):
        detector = PatternDetector(conn=None)
        signal_map = {
            "user_count": {"value_num": 10000, "source_type": "manual"}
        }
        flags = detector._round_numbers(signal_map)
        self.assertEqual(len(flags), 1)


if __name__ == "__main__":
    unittest.main()
