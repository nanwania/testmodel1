SECTOR_BENCHMARKS = {
    "b2b_saas": {
        "cac": {"p25": 200, "p50": 320, "p75": 500},
        "ltv_cac": {"p25": 3.0, "p50": 4.5, "p75": 6.0},
        "burn_multiple": {"p25": 1.2, "p50": 1.5, "p75": 2.0},
    }
}


def percentile_rank(value, benchmarks):
    if value < benchmarks["p25"]:
        return "<25th"
    if value < benchmarks["p50"]:
        return "25-50th"
    if value < benchmarks["p75"]:
        return "50-75th"
    return ">75th"
