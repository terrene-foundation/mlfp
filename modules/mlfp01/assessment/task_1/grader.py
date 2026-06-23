#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 1 — Taxi Trip Data Forensics.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives the expected cleaned dataset independently and checks the
submission against strict invariants. All ten checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from shared import MLFPDataLoader

EXPECTED_COLUMNS = [
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
VALID_PAYMENTS = {"Card", "Cash", "NETS", "Grab"}


def _reference_count() -> int:
    """Independently re-derive the expected post-cleaning row count."""
    df = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
    fmt = "%Y-%m-%d %H:%M:%S"
    df = df.with_columns(
        [
            pl.col("pickup_datetime").str.strptime(pl.Datetime, fmt, strict=False),
            pl.col("dropoff_datetime").str.strptime(pl.Datetime, fmt, strict=False),
        ]
    ).with_columns(
        (
            (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()
            / 60.0
        ).alias("d")
    )
    df = df.with_columns((pl.col("distance_km") / (pl.col("d") / 60.0)).alias("s"))
    df = df.filter(
        (pl.col("fare_sgd") > 0)
        & (pl.col("distance_km") > 0)
        & (pl.col("distance_km") <= 100)
        & (pl.col("passengers") >= 1)
        & (pl.col("d") > 0)
        & (pl.col("d") <= 180)
        & (pl.col("s") >= 2)
        & (pl.col("s") <= 120)
    )
    return df.unique(subset="trip_id").height


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score
    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score
    try:
        r = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    c["returns_dataframe"] = isinstance(r, pl.DataFrame)
    if not c["returns_dataframe"]:
        return _finalize(score)

    c["columns_exact"] = r.columns == EXPECTED_COLUMNS
    c["datetime_dtypes"] = (
        r.schema.get("pickup_datetime") == pl.Datetime
        and r.schema.get("dropoff_datetime") == pl.Datetime
    )
    # Payment normalisation: only the 4 canonical labels, all 4 present.
    if "payment_type" in r.columns:
        pays = set(r["payment_type"].unique().to_list())
        c["payment_normalised"] = (
            pays.issubset(VALID_PAYMENTS) and pays == VALID_PAYMENTS
        )
    else:
        c["payment_normalised"] = False

    null_cols = ["tip_sgd", "pickup_zone", "dropoff_zone", "payment_type"]
    c["no_nulls_in_key_cols"] = all(
        col in r.columns and r[col].null_count() == 0 for col in null_cols
    )

    # Physical-plausibility invariants — any impossible row remaining fails.
    try:
        bad = r.filter(
            ~(
                (pl.col("fare_sgd") > 0)
                & (pl.col("distance_km") > 0)
                & (pl.col("distance_km") <= 100)
                & (pl.col("passengers") >= 1)
                & (pl.col("trip_duration_min") > 0)
                & (pl.col("trip_duration_min") <= 180)
                & (pl.col("implied_speed_kmh") >= 2)
                & (pl.col("implied_speed_kmh") <= 120)
            )
        ).height
        c["plausibility_invariants"] = bad == 0
    except Exception:
        c["plausibility_invariants"] = False

    c["no_duplicate_trip_id"] = r.height > 0 and r["trip_id"].n_unique() == r.height

    try:
        c["row_count_correct"] = r.height == _reference_count()
    except Exception:
        c["row_count_correct"] = False

    # Derived columns: fare_per_km exact; is_airport matches zone rule.
    try:
        chk = r.with_columns(
            [
                (pl.col("fare_sgd") / pl.col("distance_km")).alias("_fpk"),
                (
                    (pl.col("pickup_zone") == "Changi Airport")
                    | (pl.col("dropoff_zone") == "Changi Airport")
                ).alias("_air"),
            ]
        )
        fpk_ok = (
            chk.select((pl.col("fare_per_km") - pl.col("_fpk")).abs().max()).item()
            < 1e-9
        )
        air_ok = chk.select((pl.col("is_airport") == pl.col("_air")).all()).item()
        c["derived_columns_correct"] = bool(fpk_ok and air_ok)
    except Exception:
        c["derived_columns_correct"] = False

    try:
        pk = r["pickup_datetime"]
        c["sorted_by_pickup"] = pk.is_sorted()
    except Exception:
        c["sorted_by_pickup"] = False

    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["max"] > 0 and score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
