# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Taxi Trip Data Forensics (Reference Solution)

Withheld from students. Verified to pass grader.py.

What inspecting the log reveals (describe(), null_count(), value_counts()):
  - the timestamps are strings, so they must be parsed before any time maths
  - payment_type uses 15 spellings of 4 methods (case and wording differ)
  - 250 rows have latitude and longitude swapped: a Singapore point with the
    two fields exchanged. This is unambiguous, so it is repaired
  - 1,000 non-positive fares and 500 rows with passengers < 1
  - 500 trips picked up in 2027, after the log was extracted
  - thousands of rows whose distance and duration imply an average speed below
    2 or above 120 km/h
  - 250 trip_ids are each issued to two different trips
  - tip_sgd is mostly null (no tip), and some zones are null
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from shared import MLFPDataLoader

EXTRACTED_AT = datetime(2025, 1, 1)
SG_LAT = (1.15, 1.47)
SG_LNG = (103.60, 104.05)
SPEED_KMH = (2.0, 120.0)
TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _canonical_payment() -> pl.Expr:
    """Map any spelling of the four accepted methods to its canonical label."""
    low = pl.col("payment_type").str.to_lowercase()
    return (
        pl.when(low.str.contains("grab"))
        .then(pl.lit("Grab"))
        .when(low.str.contains("nets"))
        .then(pl.lit("NETS"))
        .when(low.str.contains("cash"))
        .then(pl.lit("Cash"))
        .when(low.str.contains("card|visa|mastercard|credit"))
        .then(pl.lit("Card"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias("payment_type")
    )


def clean_trips(raw: pl.DataFrame) -> pl.DataFrame:
    """Return the usable trips in `raw`, cleaned to the data contract."""
    df = raw.with_columns(
        pl.col("pickup_datetime").str.to_datetime(TS_FORMAT),
        pl.col("dropoff_datetime").str.to_datetime(TS_FORMAT),
        _canonical_payment(),
        pl.col("tip_sgd").fill_null(0.0),
        pl.col("pickup_zone").fill_null("Unknown"),
        pl.col("dropoff_zone").fill_null("Unknown"),
    )

    # Repair: a "latitude" in Singapore's longitude band together with a
    # "longitude" in its latitude band is a Singapore point with the two
    # fields exchanged. The fix is unambiguous, so swap them back.
    swapped = pl.col("pickup_latitude").is_between(*SG_LNG) & pl.col(
        "pickup_longitude"
    ).is_between(*SG_LAT)
    df = df.with_columns(
        pl.when(swapped)
        .then(pl.col("pickup_longitude"))
        .otherwise(pl.col("pickup_latitude"))
        .alias("pickup_latitude"),
        pl.when(swapped)
        .then(pl.col("pickup_latitude"))
        .otherwise(pl.col("pickup_longitude"))
        .alias("pickup_longitude"),
    )

    df = df.with_columns(
        (
            (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()
            / 60.0
        ).alias("trip_duration_min")
    ).with_columns(
        (pl.col("distance_km") / (pl.col("trip_duration_min") / 60.0)).alias(
            "avg_speed_kmh"
        )
    )

    usable = (
        (pl.col("pickup_datetime") < EXTRACTED_AT)
        & (pl.col("fare_sgd") > 0)
        & (pl.col("passengers") >= 1)
        & pl.col("pickup_latitude").is_between(*SG_LAT)
        & pl.col("pickup_longitude").is_between(*SG_LNG)
        & pl.col("avg_speed_kmh").is_between(*SPEED_KMH)
        & pl.col("payment_type").is_not_null()
    )
    df = df.filter(usable)

    # Identity: an ID still shared by several usable records is ambiguous.
    return df.filter(~pl.col("trip_id").is_duplicated())


if __name__ == "__main__":
    log = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
    out = clean_trips(log)
    print(f"Raw {log.shape} -> cleaned {out.shape}")
    print(f"Payment methods: {sorted(out['payment_type'].unique().to_list())}")
    print(
        f"Speed range: {out['avg_speed_kmh'].min():.1f}-"
        f"{out['avg_speed_kmh'].max():.1f} km/h"
    )
