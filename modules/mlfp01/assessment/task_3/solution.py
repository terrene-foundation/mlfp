# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 3: Town Price Trends with Correct Time Alignment
(Reference Solution)

Withheld from students. Verified to pass grader.py.

What inspecting the data reveals:
  - resale_price has 107 sales at S$10 and 144 at S$9,000,000. Every other
    price lies between about S$215,000 and S$1,800,000, so a plausibility
    band (S$100,000 to S$2,000,000) separates the errors by rule.
  - 4 of the 3,240 town-months have no sale (3 in BUKIT TIMAH, 1 in
    CENTRAL AREA). A row-based shift(12) or rolling_mean(3) silently
    compares the wrong months for those towns from the gap onwards.

The fix used here is to build the complete town x month calendar, compute
the windows on it (where 12 rows back IS the same month last year), then
keep only the months that have sales.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

PLAUSIBLE_PRICE = (100_000, 2_000_000)


def town_trends(hdb: pl.DataFrame) -> pl.DataFrame:
    """Return the monthly price-trend table for every town."""
    sales = hdb.filter(pl.col("resale_price").is_between(*PLAUSIBLE_PRICE)).with_columns(
        pl.col("month").str.to_date("%Y-%m").alias("month")
    )
    monthly = sales.group_by("town", "month").agg(
        pl.len().cast(pl.Int64).alias("n_sales"),
        pl.col("resale_price").median().cast(pl.Float64).alias("median_price"),
    )

    # Complete calendar: every town x every month in the covered period.
    months = pl.date_range(
        monthly["month"].min(), monthly["month"].max(), interval="1mo", eager=True
    ).alias("month")
    calendar = (
        monthly.select("town").unique().join(months.to_frame(), how="cross")
    )
    full = calendar.join(monthly, on=["town", "month"], how="left").sort(
        "town", "month"
    )

    full = full.with_columns(
        (
            100.0
            * (pl.col("median_price") / pl.col("median_price").shift(12).over("town") - 1.0)
        ).alias("yoy_pct"),
        # rolling_mean skips nulls, so empty months drop out of the window
        # while the window still spans exactly three calendar months.
        pl.col("median_price")
        .rolling_mean(window_size=3, min_samples=1)
        .over("town")
        .alias("rolling_3m_avg"),
    )

    out = full.filter(pl.col("n_sales").is_not_null()).with_columns(
        pl.col("median_price")
        .rank(method="min", descending=True)
        .over("month")
        .cast(pl.Int64)
        .alias("price_rank_in_month")
    )
    return out.select(
        "town",
        "month",
        "n_sales",
        "median_price",
        "yoy_pct",
        "rolling_3m_avg",
        "price_rank_in_month",
    ).sort("town", "month")


if __name__ == "__main__":
    out = town_trends(MLFPDataLoader().load("mlfp01", "hdb_resale.parquet"))
    print(out.head(15))
    print(f"Shape: {out.shape}")
    print(f"yoy_pct nulls: {out['yoy_pct'].null_count()}")
