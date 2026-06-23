# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 3: Window Functions & Price Trends (Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Build a per-town, per-year HDB price-trend table using window functions.

    For each (town, sale_year): the median resale price and number of sales,
    the year-over-year % change within the town, a 3-year trailing rolling
    average of the median price within the town, and the town's price rank
    within that year (1 = most expensive). Sorted by [town, sale_year].
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet").with_columns(
        pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year")
    )

    agg = (
        df.group_by(["town", "sale_year"])
        .agg(
            [
                pl.col("resale_price").median().alias("median_price"),
                pl.len().alias("n_sales"),
            ]
        )
        .sort(["town", "sale_year"])
    )

    agg = agg.with_columns(
        [
            # YoY % change within town (null for each town's first year).
            (
                100.0
                * (
                    pl.col("median_price")
                    - pl.col("median_price").shift(1).over("town")
                )
                / pl.col("median_price").shift(1).over("town")
            ).alias("yoy_pct"),
            # 3-year trailing rolling average within town.
            pl.col("median_price")
            .rolling_mean(window_size=3, min_samples=1)
            .over("town")
            .alias("rolling_3yr_avg"),
            # Rank within year, 1 = most expensive town.
            pl.col("median_price")
            .rank(method="min", descending=True)
            .over("sale_year")
            .cast(pl.Int64)
            .alias("price_rank_in_year"),
        ]
    )

    cols = [
        "town",
        "sale_year",
        "n_sales",
        "median_price",
        "yoy_pct",
        "rolling_3yr_avg",
        "price_rank_in_year",
    ]
    return agg.select(cols).sort(["town", "sale_year"])


if __name__ == "__main__":
    out = solve()
    print(out.head(12))
    print(f"\nShape: {out.shape}")
    print(f"yoy_pct nulls (one per town = first year): {out['yoy_pct'].null_count()}")
    print(
        f"rank range: {out['price_rank_in_year'].min()}–{out['price_rank_in_year'].max()}"
    )
