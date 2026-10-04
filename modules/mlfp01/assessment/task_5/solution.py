# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 5: From Clean Trips to Model Inputs and Charts
(Reference Solution)

Withheld from students. Verified to pass grader.py.

Part A decisions:
  - Only booking-time information can be a feature: tip, dropoff time,
    duration and average speed are known only after the trip, so they are
    not in the bookings frame. A pipeline fitted on them cannot transform a
    booking.
  - trip_id is present at booking but is an identifier, not information about
    the fare. PreprocessingPipeline would ordinal-encode it into a
    meaningless number, so it is left out.
  - The raw pickup timestamp is not a usable feature. The hour of day is, so
    it is derived (the same derivation is applied to bookings).
  - setup() learns imputation values, scaling statistics and categories from
    every row it is given, so it is given the training rows only. Bookings
    are only ever transform()ed.

Part B chart choices: distribution -> histogram; relationship between two
continuous variables -> scatter; comparing one statistic across categories
-> bar chart sorted by value, with a zero baseline; change over time -> line.
"""
from __future__ import annotations

import plotly.express as px
import polars as pl

from kailash_ml import ModelVisualizer, PreprocessingPipeline
from shared import MLFPDataLoader

FEATURES = [
    "pickup_zone",
    "dropoff_zone",
    "distance_km",
    "passengers",
    "payment_type",
    "pickup_latitude",
    "pickup_longitude",
    "pickup_hour",
]
TARGET = "fare_sgd"


def _booking_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pickup_datetime").dt.hour().cast(pl.Float64).alias("pickup_hour")
    ).select(FEATURES)


# ── Part A ───────────────────────────────────────────────────────────────
def fit_preprocessor(train: pl.DataFrame) -> PreprocessingPipeline:
    """Fit the preprocessing on the training trips only."""
    pipeline = PreprocessingPipeline()
    pipeline.setup(
        _booking_features(train).with_columns(train[TARGET]),
        target=TARGET,
        seed=42,
    )
    return pipeline


def prepare_bookings(fitted: PreprocessingPipeline, bookings: pl.DataFrame) -> pl.DataFrame:
    """Turn new bookings into model inputs with the already-fitted rules."""
    return fitted.transform(_booking_features(bookings))


# ── Part B ───────────────────────────────────────────────────────────────
def make_charts(trips: pl.DataFrame) -> dict:
    """Answer four questions about the cleaned trips with one chart each."""
    viz = ModelVisualizer()
    known_zone = trips.filter(pl.col("pickup_zone") != "Unknown")

    zone_fares = (
        known_zone.group_by("pickup_zone")
        .agg(pl.col("fare_sgd").median().alias("median_fare_sgd"))
        .sort("median_fare_sgd", descending=True)
    )
    monthly = (
        trips.group_by(pl.col("pickup_datetime").dt.truncate("1mo").alias("month"))
        .agg(pl.len().alias("trips"))
        .sort("month")
    )

    figures = {
        "fare_distribution": viz.histogram(
            trips, "fare_sgd", bins=60, title="How are fares distributed?"
        ),
        "fare_vs_distance": viz.scatter(
            trips, "distance_km", "fare_sgd", title="How does fare grow with distance?"
        ),
        "zone_median_fare": px.bar(
            zone_fares,
            x="pickup_zone",
            y="median_fare_sgd",
            title="Median fare by pickup zone",
            labels={"pickup_zone": "Pickup zone", "median_fare_sgd": "Median fare (S$)"},
        ),
        "monthly_trips": px.line(
            monthly,
            x="month",
            y="trips",
            title="Trips per month",
            labels={"month": "Month", "trips": "Trips"},
        ),
    }
    busiest = monthly.sort("trips", descending=True)["month"][0]
    findings = {
        "top_median_fare_zone": zone_fares["pickup_zone"][0],
        "busiest_month": busiest.strftime("%Y-%m"),
    }
    return {"figures": figures, "findings": findings}


if __name__ == "__main__":
    import importlib.util
    from pathlib import Path

    # Use the Task 1 reference cleaning to get clean trips for a local run.
    spec = importlib.util.spec_from_file_location(
        "task1", Path(__file__).parents[1] / "task_1" / "solution.py"
    )
    task1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task1)
    trips = task1.clean_trips(MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet"))

    test_mask = pl.col("trip_id").hash(3) % 5 == 0
    train, test = trips.filter(~test_mask), trips.filter(test_mask)
    fitted = fit_preprocessor(train)
    X = prepare_bookings(fitted, test.drop("fare_sgd", "tip_sgd", "dropoff_datetime"))
    print(f"Model inputs: {X.shape}")
    out = make_charts(trips)
    print(out["findings"])
