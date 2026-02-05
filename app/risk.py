import json

from app import db


def _signal_value(signal_row):
    if signal_row is None:
        return None
    if signal_row["value_num"] is not None:
        return float(signal_row["value_num"])
    if signal_row["value_bool"] is not None:
        return float(signal_row["value_bool"])
    return None


def _row_id(signal_row):
    if signal_row is None:
        return None
    try:
        return signal_row["id"]
    except Exception:
        return None


def _evidence(signal_row):
    if signal_row is None:
        return None
    try:
        return signal_row["evidence_text"]
    except Exception:
        return None


def run_risk_scan(conn, actor="user"):
    rules = db.list_risk_rules(conn)
    if not rules:
        return None, "No risk rules available."

    companies = db.list_companies(conn)
    run_id = db.create_risk_run(conn, actor=actor)
    findings = []

    for company in companies:
        signal_rows = db.list_latest_signal_values(conn, company["id"])
        signal_map = {s["signal_key"]: s for s in signal_rows}

        for rule in rules:
            rule_type = rule["rule_type"]
            params = {}
            if rule["params_json"]:
                try:
                    params = json.loads(rule["params_json"])
                except json.JSONDecodeError:
                    params = {}

            if rule_type == "threshold":
                signal_row = signal_map.get(rule["signal_key"])
                value = _signal_value(signal_row)
                if value is None:
                    continue

                op = params.get("op")
                threshold = params.get("threshold")
                if threshold is None:
                    continue

                hit = False
                if op == "gt":
                    hit = value > threshold
                elif op == "gte":
                    hit = value >= threshold
                elif op == "lt":
                    hit = value < threshold
                elif op == "lte":
                    hit = value <= threshold

                if hit:
                    findings.append(
                        {
                            "company_id": company["id"],
                            "rule_id": rule["id"],
                            "signal_value_id": _row_id(signal_row),
                            "value": value,
                            "severity": rule["severity"],
                            "explanation": f"{rule['name']}: value {value} {op} {threshold}",
                            "evidence_text": _evidence(signal_row),
                        }
                    )

            elif rule_type == "ltv_cac_ratio":
                ltv = _signal_value(signal_map.get("ltv"))
                cac = _signal_value(signal_map.get("cac"))
                if ltv is None or cac in (None, 0):
                    continue
                ratio = ltv / cac
                threshold = params.get("threshold", 3)
                if ratio < threshold:
                    findings.append(
                        {
                            "company_id": company["id"],
                            "rule_id": rule["id"],
                            "signal_value_id": _row_id(signal_map.get("ltv")),
                            "value": ratio,
                            "severity": rule["severity"],
                            "explanation": f"LTV/CAC ratio {ratio:.2f} below {threshold}",
                            "evidence_text": _evidence(signal_map.get("ltv")),
                        }
                    )

            elif rule_type == "arr_mrr_mismatch":
                arr = _signal_value(signal_map.get("arr"))
                mrr = _signal_value(signal_map.get("mrr"))
                if arr is None or mrr in (None, 0):
                    continue
                ratio = arr / mrr
                min_v = params.get("min", 10)
                max_v = params.get("max", 14)
                if ratio < min_v or ratio > max_v:
                    findings.append(
                        {
                            "company_id": company["id"],
                            "rule_id": rule["id"],
                            "signal_value_id": _row_id(signal_map.get("arr")),
                            "value": ratio,
                            "severity": rule["severity"],
                            "explanation": f"ARR/MRR ratio {ratio:.2f} outside {min_v}-{max_v}",
                            "evidence_text": _evidence(signal_map.get("arr")),
                        }
                    )

            elif rule_type == "burn_vs_mrr":
                burn = _signal_value(signal_map.get("burn_rate"))
                mrr = _signal_value(signal_map.get("mrr"))
                if burn is None or mrr in (None, 0):
                    continue
                multiplier = params.get("multiplier", 2)
                if burn > mrr * multiplier:
                    findings.append(
                        {
                            "company_id": company["id"],
                            "rule_id": rule["id"],
                            "signal_value_id": _row_id(signal_map.get("burn_rate")),
                            "value": burn,
                            "severity": rule["severity"],
                            "explanation": f"Burn rate {burn:.2f} exceeds {multiplier}x MRR",
                            "evidence_text": _evidence(signal_map.get("burn_rate")),
                        }
                    )

    if findings:
        db.add_risk_findings(conn, run_id, findings)
    db.complete_risk_run(conn, run_id)
    return run_id, None
