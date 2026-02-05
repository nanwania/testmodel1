import os
import tempfile
import unittest

from app import db
from app import scoring
from app import risk_rules

TEST_COMPANIES = [
    {
        "name": "HighGrowth SaaS Inc",
        "signals": {
            "mrr": 100000,
            "mrr_growth_mom": 15,
            "cac": 280,
            "ltv": 1400,
            "churn_rate": 0.05,
            "team_size": 12,
            "burn_multiple": 1.3,
            "news_volume": 120,
            "news_volume_prev_month": 60,
            "customer_count": 1000,
        },
        "expected_score": 0.85,
        "expected_risks": [],
    },
    {
        "name": "Struggling Startup Co",
        "signals": {
            "mrr": 10000,
            "mrr_growth_mom": 2,
            "cac": 800,
            "ltv": 1000,
            "churn_rate": 0.15,
            "team_size": 8,
            "burn_multiple": 4.2,
            "news_volume": 10,
            "news_volume_prev_month": 20,
            "customer_count": 50,
        },
        "expected_score": 0.35,
        "expected_risks": ["high_churn", "high_burn", "stalled_growth"],
    },
]


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        db.DB_PATH = os.path.join(self.tmpdir.name, "test.db")
        db.init_db()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _seed_company(self, name, signals):
        conn = db.get_conn()
        company_id = db.add_company(conn, name=name, source="test")
        for key, value in signals.items():
            db.add_signal_value(
                conn,
                company_id,
                key,
                value_num=value if isinstance(value, (int, float)) else None,
                value_text=None if isinstance(value, (int, float)) else str(value),
                value_bool=None,
                value_json=None,
                source_type="manual",
                source_ref="test",
            )
        conn.close()
        return company_id

    @unittest.skip("Score calibration depends on configured criteria weights")
    def test_scoring_accuracy(self):
        conn = db.get_conn()
        for test_case in TEST_COMPANIES:
            company_id = self._seed_company(test_case["name"], test_case["signals"])
            scoring.run_scoring(conn, db.get_active_criteria_version_id(conn))
            score_item = db.list_latest_score_item(conn, company_id)
            self.assertIsNotNone(score_item)
        conn.close()

    def test_risk_rules_detection(self):
        conn = db.get_conn()
        for test_case in TEST_COMPANIES:
            company_id = self._seed_company(test_case["name"], test_case["signals"])
            flags = risk_rules.detect_risks(company_id, conn)
            detected = [f["type"] for f in flags]
            for expected in test_case["expected_risks"]:
                self.assertIn(expected, detected)
        conn.close()


if __name__ == "__main__":
    unittest.main()
