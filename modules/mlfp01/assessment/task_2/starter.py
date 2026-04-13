# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 2: HDB YoY Price Changes in Ang Mo Kio

Complete the `solve()` function. Your solution will be auto-graded.

Run the grader with:
    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Compute YoY median resale price changes for Ang Mo Kio by flat type.

    Returns:
        A Polars DataFrame with 5 columns:
            flat_type, year, median_price, prev_year_median, yoy_pct_change

        Rows where prev_year_median is null (earliest year per flat_type) are dropped.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "hdb_resale.parquet")

    # TODO: filter the DataFrame to rows where town == "ANG MO KIO".
    # Hint: df.filter(pl.col("town") == "ANG MO KIO")

    # TODO: derive a `year` column (Int64) from the first 4 characters of `month`.
    # Hint: pl.col("month").str.slice(0, 4).cast(pl.Int64)

    # TODO: group by (flat_type, year) and compute the median resale_price.
    # Hint: .group_by(["flat_type", "year"]).agg(pl.col("resale_price").median().alias("median_price"))

    # TODO: sort by (flat_type, year), then add a `prev_year_median` column using
    # a lag-1 window within each flat_type.
    # Hint: pl.col("median_price").shift(1).over("flat_type") — but the DataFrame
    # must be sorted first so the shift follows chronological order.

    # TODO: compute yoy_pct_change = 100 * (median_price - prev_year_median) / prev_year_median

    # TODO: drop rows where prev_year_median is null, then select the 5 columns
    # in this exact order and sort by (flat_type, year):
    #   ["flat_type", "year", "median_price", "prev_year_median", "yoy_pct_change"]

    raise NotImplementedError("TODO: complete solve()")


if __name__ == "__main__":
    result = solve()
    print(result)
    print(f"\nShape: {result.shape}")
    print(f"Flat types: {result['flat_type'].unique().sort().to_list()}")
