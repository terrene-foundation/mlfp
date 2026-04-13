# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Monthly Weather Statistics

Complete the `solve()` function. Your solution will be auto-graded.

Run the grader with:
    python grader.py starter.py
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Compute monthly weather deviations from the annual mean.

    Returns:
        A Polars DataFrame with 12 rows and 6 columns:
            month, mean_temperature_c, total_rainfall_mm,
            temp_deviation_c, rainfall_vs_mean_pct, is_wet_month
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "sg_weather.csv")

    # TODO: compute the annual mean temperature and rainfall across all 12 rows.
    # Hint: use df["mean_temperature_c"].mean() and df["total_rainfall_mm"].mean()

    # TODO: add three new columns with `with_columns`:
    #   - temp_deviation_c     = mean_temperature_c - annual mean temperature
    #   - rainfall_vs_mean_pct = 100 * (total_rainfall_mm - annual mean rainfall) / annual mean rainfall
    #   - is_wet_month         = total_rainfall_mm > annual mean rainfall
    # Hint: pl.col("total_rainfall_mm") > <annual_mean> returns a Boolean expression

    # TODO: return a DataFrame with exactly these 6 columns in this order:
    #   ["month", "mean_temperature_c", "total_rainfall_mm",
    #    "temp_deviation_c", "rainfall_vs_mean_pct", "is_wet_month"]

    raise NotImplementedError("TODO: complete solve()")


if __name__ == "__main__":
    result = solve()
    print(result)
    print(f"\nShape: {result.shape}")
    print(f"Wet months: {result['is_wet_month'].sum()}")
