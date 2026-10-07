# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP01 — Assessment Task 1: Taxi Trip Data Forensics

Implement `clean_trips()`. problem.md holds the data contract: what a usable
trip is and what the output must contain. It does not list the problems in the
log. Finding them is part of the task.

Your function is graded on the full log AND on unseen extracts from the same
dispatch systems, so it must apply rules. It must not remember particular rows.

    python grader.py starter.py     # (instructor) grade an attempt
    python starter.py               # (you) run your function on the full log
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def clean_trips(raw: pl.DataFrame) -> pl.DataFrame:
    """Return the usable trips in `raw`, cleaned to the data contract.

    Args:
        raw: a trip log with the same 12 columns and dtypes as
            sg_taxi_trips.parquet. Do not modify it in place.

    Returns:
        One row per usable trip: the 12 source columns (timestamps as
        Datetime) plus trip_duration_min and avg_speed_kmh.
    """
    raise NotImplementedError("Implement clean_trips() — see problem.md")


if __name__ == "__main__":
    log = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
    print(f"Raw log: {log.shape}")
    cleaned = clean_trips(log)
    print(f"Cleaned: {cleaned.shape}")
    print(cleaned.head())
