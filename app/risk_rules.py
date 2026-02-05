from app import db

RISK_RULES = {
    "high_burn": {
        "condition": lambda s: s.get("burn_multiple", 0) > 3,
        "category": "financial",
        "severity": "high",
        "message": "Burn rate >3x revenue is concerning",
    },
    "high_churn": {
        "condition": lambda s: s.get("churn_rate", 0) > 0.10,
        "category": "traction",
        "severity": "critical",
        "message": "Monthly churn >10% indicates product-market fit issues",
    },
    "stalled_growth": {
        "condition": lambda s: s.get("mrr_growth_mom", 100) < 5,
        "category": "traction",
        "severity": "medium",
        "message": "MoM growth <5% suggests plateau",
    },
}


def detect_risks(company_id, conn):
    signals = db.get_all_signals(conn, company_id)
    flags = []

    for risk_id, rule in RISK_RULES.items():
        if rule["condition"](signals):
            flags.append(
                {
                    "type": risk_id,
                    "category": rule.get("category"),
                    "severity": rule["severity"],
                    "message": rule["message"],
                }
            )

    return flags
