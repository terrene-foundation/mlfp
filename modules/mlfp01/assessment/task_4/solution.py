# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 4: Profile, Clean and Justify with DataExplorer
(Reference Solution)

Withheld from students. Verified to pass grader.py.

What profiling the raw quarterly slice reveals (run_profile(raw_q).alerts):
  - duplicates: quarter 2019-2 appears twice as an identical row
  - high_nulls: inflation_rate (8 of 101 = 7.9%) and trade_balance_sgd_bn
    (6 of 101 = 5.9%)
  - constant: period_type (it is "quarterly" on every row of the slice)
  - high_cardinality: period, gdp_growth_pct, property_price_index and
    tourist_arrivals (info level). tourist_arrivals is text, with thousands
    separators on some rows, so it is profiled as a string
What describe()/value_counts() add: period is written three ways
("Q1 2000", "2001-Q1", "2001-2").

Imputation choice: these are quarterly time series with trends, so a gap is
filled by linear interpolation between the neighbouring quarters. One median
over 25 years would put a 2000-era value into a 2020 gap.
"""
from __future__ import annotations

import polars as pl

from kailash_ml import AlertConfig
from shared import MLFPDataLoader, run_profile

NUMERIC = [
    "gdp_growth_pct",
    "unemployment_rate",
    "inflation_rate",
    "trade_balance_sgd_bn",
    "property_price_index",
]
ACCEPTED_REASONS = {
    "high_cardinality": (
        "a continuous measurement: almost every quarter has a distinct value, "
        "which is expected and not an identifier problem"
    ),
    "high_correlation": (
        "a trend over time: the property index rises steadily with the year, "
        "so the two move together. Both are kept because the trend is the "
        "signal; a model would choose one of them later"
    ),
}


def alert_key(alert: dict) -> str:
    """Encode a DataExplorer alert as 'type', 'type:column' or 'type:a,b'."""
    if alert.get("columns"):
        return f"{alert['type']}:{','.join(sorted(alert['columns']))}"
    if alert.get("column"):
        return f"{alert['type']}:{alert['column']}"
    return alert["type"]


def _clean(raw_q: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    quarter = pl.coalesce(
        pl.col("period").str.extract(r"Q(\d)", 1),
        pl.col("period").str.extract(r"^\d{4}-(\d)$", 1),
    ).cast(pl.Int64)
    df = raw_q.with_columns(
        pl.col("period").str.extract(r"(\d{4})", 1).cast(pl.Int64).alias("period_year"),
        quarter.alias("period_quarter"),
        pl.col("tourist_arrivals").str.replace_all(",", "").str.strip_chars().cast(pl.Int64),
    )
    # One row per quarter: the same quarter can be recorded twice, possibly
    # under different period spellings, so deduplicate on the parsed key.
    df = (
        df.unique(subset=["period_year", "period_quarter"], keep="first", maintain_order=True)
        .sort("period_year", "period_quarter")
    )
    nulls_filled = int(df.select(pl.col("inflation_rate", "trade_balance_sgd_bn").null_count()).sum_horizontal().item())
    df = df.with_columns(
        pl.col("inflation_rate").interpolate(),
        pl.col("trade_balance_sgd_bn").interpolate(),
    )
    cleaned = df.select("period_year", "period_quarter", *NUMERIC, "tourist_arrivals")
    return cleaned, nulls_filled


def audit_indicators(raw: pl.DataFrame) -> dict:
    """Profile, clean and justify the quarterly economic indicators.

    Imputation choice: linear interpolation between neighbouring quarters.
    Inflation and the trade balance drift over 25 years, so the quarters on
    either side of a gap are far better estimates than one median over the
    whole period, which would put a 2000-era value into a 2020 gap.
    """
    raw_q = raw.filter(pl.col("period_type") == "quarterly")
    raw_alerts = [alert_key(a) for a in run_profile(raw_q).alerts]

    cleaned, nulls_filled = _clean(raw_q)

    remaining = [alert_key(a) for a in run_profile(cleaned).alerts]
    accepted = {key: ACCEPTED_REASONS[key.split(":")[0]] for key in remaining}
    return {
        "raw_alerts": raw_alerts,
        "cleaned": cleaned,
        "imputation": "interpolate",
        # null_pct > threshold fires, so only 0.0 catches a single missing value.
        "null_alert_config": AlertConfig(high_null_pct_threshold=0.0),
        "accepted_alerts": accepted,
        "quality_delta": {
            "rows_removed": raw_q.height - cleaned.height,
            "nulls_filled": nulls_filled,
        },
    }


if __name__ == "__main__":
    out = audit_indicators(MLFPDataLoader().load("mlfp01", "economic_indicators.csv"))
    print("Raw alerts:", out["raw_alerts"])
    print(out["cleaned"].head())
    print("Cleaned shape:", out["cleaned"].shape)
    print("Accepted alerts:", out["accepted_alerts"])
    print("Quality delta:", out["quality_delta"])
