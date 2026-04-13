# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Monthly Weather Statistics (Reference Solution)
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def solve() -> pl.DataFrame:
    """Compute monthly weather deviations from the annual mean."""
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "sg_weather.csv")

    annual_mean_temp = df["mean_temperature_c"].mean()
    annual_mean_rain = df["total_rainfall_mm"].mean()

    result = df.with_columns(
        [
            (pl.col("mean_temperature_c") - annual_mean_temp).alias("temp_deviation_c"),
            (
                100.0
                * (pl.col("total_rainfall_mm") - annual_mean_rain)
                / annual_mean_rain
            ).alias("rainfall_vs_mean_pct"),
            (pl.col("total_rainfall_mm") > annual_mean_rain).alias("is_wet_month"),
        ]
    ).select(
        [
            "month",
            "mean_temperature_c",
            "total_rainfall_mm",
            "temp_deviation_c",
            "rainfall_vs_mean_pct",
            "is_wet_month",
        ]
    )

    return result


if __name__ == "__main__":
    result = solve()
    print(result)
    print(f"\nShape: {result.shape}")
    print(f"Wet months: {result['is_wet_month'].sum()}")
