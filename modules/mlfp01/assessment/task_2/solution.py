# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB YoY Price Changes in Ang Mo Kio (Reference Solution)
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Compute YoY median resale price changes for Ang Mo Kio by flat type."""
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    ang_mo_kio = df.filter(pl.col("town") == "ANG MO KIO").with_columns(
        pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("year")
    )

    yearly = (
        ang_mo_kio.group_by(["flat_type", "year"])
        .agg(pl.col("resale_price").median().alias("median_price"))
        .sort(["flat_type", "year"])
    )

    result = (
        yearly.with_columns(
            pl.col("median_price").shift(1).over("flat_type").alias("prev_year_median")
        )
        .with_columns(
            (
                100.0
                * (pl.col("median_price") - pl.col("prev_year_median"))
                / pl.col("prev_year_median")
            ).alias("yoy_pct_change")
        )
        .drop_nulls(subset=["prev_year_median"])
        .select(
            [
                "flat_type",
                "year",
                "median_price",
                "prev_year_median",
                "yoy_pct_change",
            ]
        )
        .sort(["flat_type", "year"])
    )

    return result


if __name__ == "__main__":
    result = solve()
    print(result)
    print(f"\nShape: {result.shape}")
    print(f"Flat types: {result['flat_type'].unique().sort().to_list()}")
