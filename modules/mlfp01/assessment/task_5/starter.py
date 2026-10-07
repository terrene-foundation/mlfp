# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 5: From Clean Trips to Model Inputs and Charts

Implement fit_preprocessor(), prepare_bookings() and make_charts(). problem.md
defines what each must return. The grader supplies its own clean trips and
keeps the bookings back, so derive everything from the frames you receive.

    python starter.py               # try your functions locally
"""
from __future__ import annotations

import plotly.express as px
import polars as pl

from kailash_ml import ModelVisualizer, PreprocessingPipeline
from shared import MLFPDataLoader

BOOKING_COLUMNS = [
    "trip_id",
    "pickup_datetime",
    "pickup_zone",
    "dropoff_zone",
    "distance_km",
    "passengers",
    "payment_type",
    "pickup_latitude",
    "pickup_longitude",
]


# ── Part A ───────────────────────────────────────────────────────────────
def fit_preprocessor(train: pl.DataFrame) -> object:
    """Learn the preprocessing rules from the completed training trips only."""
    raise NotImplementedError("Implement fit_preprocessor() — see problem.md")


def prepare_bookings(fitted: object, bookings: pl.DataFrame) -> pl.DataFrame:
    """Apply the learned rules to new bookings (BOOKING_COLUMNS only)."""
    raise NotImplementedError("Implement prepare_bookings() — see problem.md")


# ── Part B ───────────────────────────────────────────────────────────────
def make_charts(trips: pl.DataFrame) -> dict:
    """Return {"figures": {four keys -> Plotly figure}, "findings": {...}}."""
    raise NotImplementedError("Implement make_charts() — see problem.md")


if __name__ == "__main__":
    # Replace this with your own Task 1 clean_trips() to get clean trips:
    #   from task_1_starter import clean_trips   (adjust to where you keep it)
    raw = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
    trips = raw  # <- your clean_trips(raw)

    # A local train / held-back split, like the grader's.
    held_back = pl.col("trip_id").hash(1) % 5 == 0
    train, test = trips.filter(~held_back), trips.filter(held_back)
    fitted = fit_preprocessor(train)
    inputs = prepare_bookings(fitted, test.select(BOOKING_COLUMNS))
    print(f"Model inputs: {inputs.shape}")

    result = make_charts(trips)
    print(result["findings"])
    _ = (px, ModelVisualizer, PreprocessingPipeline)  # imported for your use
