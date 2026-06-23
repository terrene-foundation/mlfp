# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Taxi Trip Data Forensics (Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def _canonical_payment(col: pl.Expr) -> pl.Expr:
    """Collapse 15 raw payment spellings into {Card, Cash, NETS, Grab}."""
    low = col.str.to_lowercase().str.strip_chars()
    return (
        pl.when(low.str.contains("grab"))
        .then(pl.lit("Grab"))
        .when(low.str.contains("nets"))
        .then(pl.lit("NETS"))
        .when(low.str.contains("cash"))
        .then(pl.lit("Cash"))
        .when(
            low.str.contains("card")
            | low.str.contains("visa")
            | low.str.contains("mastercard")
            | low.str.contains("credit")
        )
        .then(pl.lit("Card"))
        .otherwise(pl.lit("Card"))
        .alias("payment_type")
    )


def solve() -> pl.DataFrame:
    """Clean the raw Singapore taxi-trip log into an analysis-ready table.

    Returns a 16-column Polars DataFrame sorted by ``pickup_datetime``:
    the 12 source columns (with parsed types + normalised payment + imputed
    nulls) plus four derived columns: ``trip_duration_min``,
    ``implied_speed_kmh``, ``fare_per_km``, ``is_airport``.
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp01", "sg_taxi_trips.parquet")

    fmt = "%Y-%m-%d %H:%M:%S"
    df = df.with_columns(
        [
            pl.col("pickup_datetime").str.strptime(pl.Datetime, fmt, strict=False),
            pl.col("dropoff_datetime").str.strptime(pl.Datetime, fmt, strict=False),
            _canonical_payment(pl.col("payment_type")),
            pl.col("tip_sgd").fill_null(0.0),
            pl.col("pickup_zone").fill_null("Unknown"),
            pl.col("dropoff_zone").fill_null("Unknown"),
        ]
    )

    # Derived quantities needed for the physical-plausibility filters.
    df = df.with_columns(
        (
            (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()
            / 60.0
        ).alias("trip_duration_min")
    ).with_columns(
        (pl.col("distance_km") / (pl.col("trip_duration_min") / 60.0)).alias(
            "implied_speed_kmh"
        )
    )

    # Physical-plausibility filter (all conjunctive — order-independent).
    df = df.filter(
        (pl.col("fare_sgd") > 0)
        & (pl.col("distance_km") > 0)
        & (pl.col("distance_km") <= 100)
        & (pl.col("passengers") >= 1)
        & (pl.col("trip_duration_min") > 0)
        & (pl.col("trip_duration_min") <= 180)
        & (pl.col("implied_speed_kmh") >= 2)
        & (pl.col("implied_speed_kmh") <= 120)
    )

    # Deterministic dedup: one row per trip_id, keep the highest fare,
    # tie-break on the latest dropoff.
    df = df.sort(["fare_sgd", "dropoff_datetime"], descending=[True, True]).unique(
        subset="trip_id", keep="first"
    )

    df = df.with_columns(
        [
            (pl.col("fare_sgd") / pl.col("distance_km")).alias("fare_per_km"),
            (
                (pl.col("pickup_zone") == "Changi Airport")
                | (pl.col("dropoff_zone") == "Changi Airport")
            ).alias("is_airport"),
        ]
    )

    cols = [
        "trip_id",
        "pickup_datetime",
        "dropoff_datetime",
        "pickup_zone",
        "dropoff_zone",
        "distance_km",
        "fare_sgd",
        "tip_sgd",
        "payment_type",
        "passengers",
        "pickup_latitude",
        "pickup_longitude",
        "trip_duration_min",
        "implied_speed_kmh",
        "fare_per_km",
        "is_airport",
    ]
    return df.select(cols).sort("pickup_datetime")


if __name__ == "__main__":
    out = solve()
    print(out.head())
    print(f"\nShape: {out.shape}")
    print(f"Payment categories: {sorted(out['payment_type'].unique().to_list())}")
    print(f"Airport trips: {out['is_airport'].sum()}")
    print(
        f"Speed range: {out['implied_speed_kmh'].min():.1f}–{out['implied_speed_kmh'].max():.1f} km/h"
    )
