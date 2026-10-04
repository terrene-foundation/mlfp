#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 1 — Taxi Trip Data Forensics.

Usage:
    python grader.py starter.py     # grade a submission
    python grader.py solution.py    # verify the reference passes

The grader computes its own expected result from the raw log, then runs the
submission on (a) the full log and (b) an unseen extract built here from real
trips with planted defects and fresh trip IDs. Rows are compared by trip_id,
so a correct submission passes whatever row order it returns.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from shared import MLFPDataLoader

WEIGHT = 20
CHECKS = [
    "required_columns",
    "kept_trips_full_log",
    "payment_full_log",
    "repairs_full_log",
    "null_fill_full_log",
    "derived_full_log",
    "untouched_values_full_log",
    "kept_trips_unseen_extract",
    "values_unseen_extract",
    "input_not_mutated",
]
GATES = ("required_columns", "input_not_mutated")
REQUIRED = [
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
    "avg_speed_kmh",
]
TS = "%Y-%m-%d %H:%M:%S"


# ── Ground truth ─────────────────────────────────────────────────────────
def _payment(label: str | None) -> str | None:
    if label is None:
        return None
    s = label.lower()
    for key, canon in (("grab", "Grab"), ("nets", "NETS"), ("cash", "Cash")):
        if key in s:
            return canon
    if any(k in s for k in ("card", "visa", "mastercard", "credit")):
        return "Card"
    return None


def _expected(raw: pl.DataFrame) -> pl.DataFrame:
    """Independent reference for the data contract."""
    df = raw.with_columns(
        pl.col("pickup_datetime").str.strptime(pl.Datetime("us"), TS),
        pl.col("dropoff_datetime").str.strptime(pl.Datetime("us"), TS),
        pl.col("payment_type").map_elements(_payment, return_dtype=pl.String),
        pl.col("tip_sgd").fill_null(0.0),
        pl.col("pickup_zone").fill_null("Unknown"),
        pl.col("dropoff_zone").fill_null("Unknown"),
    )
    lat, lng = pl.col("pickup_latitude"), pl.col("pickup_longitude")
    swap = (lat >= 103.60) & (lat <= 104.05) & (lng >= 1.15) & (lng <= 1.47)
    df = df.with_columns(
        pl.when(swap).then(lng).otherwise(lat).alias("pickup_latitude"),
        pl.when(swap).then(lat).otherwise(lng).alias("pickup_longitude"),
    )
    dur = (pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds()
    df = df.with_columns((dur / 60.0).alias("trip_duration_min")).with_columns(
        (pl.col("distance_km") * 60.0 / pl.col("trip_duration_min")).alias(
            "avg_speed_kmh"
        )
    )
    lat, lng = pl.col("pickup_latitude"), pl.col("pickup_longitude")
    spd = pl.col("avg_speed_kmh")
    df = df.filter(
        (pl.col("pickup_datetime") < datetime(2025, 1, 1))
        & (pl.col("fare_sgd") > 0)
        & (pl.col("passengers") >= 1)
        & (lat >= 1.15)
        & (lat <= 1.47)
        & (lng >= 103.60)
        & (lng <= 104.05)
        & (spd >= 2.0)
        & (spd <= 120.0)
        & pl.col("payment_type").is_not_null()
    )
    counts = df.group_by("trip_id").len()
    unique_ids = counts.filter(pl.col("len") == 1).select("trip_id")
    return df.join(unique_ids, on="trip_id", how="semi")


def _unseen_extract(raw: pl.DataFrame) -> pl.DataFrame:
    """Real trips with fresh IDs and planted defects the student never sees."""
    ok = _expected(raw)
    pool = raw.filter(~pl.col("trip_id").is_duplicated()).join(
        ok.filter(pl.col("avg_speed_kmh").is_between(10, 80)).select("trip_id"),
        on="trip_id",
        how="semi",
    )
    # A deterministic pick that does not depend on a seeded sampler.
    base = (
        pool.with_columns(pl.col("trip_id").hash(7).alias("_h"))
        .sort("_h")
        .head(400)
        .drop("_h")
        .with_row_index("_i")
        .with_columns(
            pl.format("QA-{}", pl.col("_i").cast(pl.String).str.zfill(6)).alias(
                "trip_id"
            )
        )
    )
    rows = base.to_dicts()
    spellings = [
        "grabpay", "CREDIT CARD", "visa", "MASTERCARD", "nets",
        "CASH PAYMENT", "cash", "Grab", "card", "NETS",
    ]  # fmt: skip
    for r in rows:
        i = r.pop("_i")
        pick = datetime.strptime(r["pickup_datetime"], TS)
        drop = datetime.strptime(r["dropoff_datetime"], TS)
        if i < 40:  # swapped coordinates: repair and keep
            r["pickup_latitude"], r["pickup_longitude"] = (
                r["pickup_longitude"],
                r["pickup_latitude"],
            )
        elif i < 60:  # non-positive fare
            r["fare_sgd"] = -abs(r["fare_sgd"]) if i % 2 else 0.0
        elif i < 80:  # no passenger
            r["passengers"] = 0 if i % 2 else -1
        elif i < 100:  # picked up after the extraction date
            shift = datetime(2025, 3, 1) - pick.replace(hour=0, minute=0, second=0)
            pick, drop = pick + shift, drop + shift
        elif i < 110:  # impossibly fast
            r["distance_km"] = 45.0
            drop = pick + timedelta(minutes=4)
        elif i < 120:  # impossibly slow
            r["distance_km"] = 0.6
            drop = pick + timedelta(minutes=50)
        elif i < 130:  # pickup outside Singapore, not a swap
            r["pickup_latitude"] = 3.15
        elif i < 150:  # ten ids each issued to two usable trips
            r["trip_id"] = f"QA-DUP-{i % 10:03d}"
        elif i < 160:  # id shared with an unusable record: keep the usable one
            r["trip_id"] = f"QA-HALF-{i % 10:03d}"
        elif i < 170:
            r["trip_id"] = f"QA-HALF-{i % 10:03d}"
            r["fare_sgd"] = -5.0
        elif i < 200:  # unseen spellings and missing values
            r["payment_type"] = spellings[i % 10]
            r["tip_sgd"] = None
            r["pickup_zone"] = None if i % 2 else r["pickup_zone"]
            r["dropoff_zone"] = None if i % 3 == 0 else r["dropoff_zone"]
        elif i == 200:  # boundary: exactly at the extraction instant
            dur = drop - pick
            pick = datetime(2025, 1, 1)
            drop = pick + dur
        elif i == 201:  # boundary: just before the extraction instant
            dur = drop - pick
            pick = datetime(2024, 12, 31, 23, 0, 0)
            drop = pick + dur
        r["pickup_datetime"] = pick.strftime(TS)
        r["dropoff_datetime"] = drop.strftime(TS)
    return pl.DataFrame(rows, schema=raw.schema)


# ── Comparison helpers ───────────────────────────────────────────────────
def _paired(student: pl.DataFrame, expected: pl.DataFrame) -> pl.DataFrame:
    s = student.filter(~pl.col("trip_id").is_duplicated()).select(REQUIRED)
    s = s.rename({c: f"s_{c}" for c in REQUIRED[1:]})
    return expected.select(REQUIRED).join(s, on="trip_id", how="inner")


def _same(p: pl.DataFrame, cols: list[str], tol: float | None = None) -> bool:
    for c in cols:
        a, b = p[c], p[f"s_{c}"]
        if tol is None:
            if not (a == b).fill_null(False).all() or (a.is_null() != b.is_null()).any():
                return False
        else:
            if b.null_count() or ((a - b.cast(pl.Float64)).abs() > tol).any():
                return False
    return True


def _kept_matches(student: pl.DataFrame, expected: pl.DataFrame) -> bool:
    ids = student["trip_id"]
    return ids.n_unique() == ids.len() and set(ids.to_list()) == set(
        expected["trip_id"].to_list()
    )


# ── Grading ──────────────────────────────────────────────────────────────
def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _usable(out) -> bool:
    return (
        isinstance(out, pl.DataFrame)
        and all(c in out.columns for c in REQUIRED)
        and isinstance(out.schema["pickup_datetime"], pl.Datetime)
        and isinstance(out.schema["dropoff_datetime"], pl.Datetime)
    )


def _normalise_ts(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("pickup_datetime").cast(pl.Datetime("us")),
        pl.col("dropoff_datetime").cast(pl.Datetime("us")),
    )


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {c: False for c in CHECKS}}
    c = score["checks"]
    try:
        student = load_student_module(student_path)
        fn = student.clean_trips
    except Exception as e:
        score["error"] = f"Cannot load clean_trips(): {type(e).__name__}: {e}"
        return _finalize(score)

    raw = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
    snapshot = raw.clone()
    try:
        out = fn(raw)
    except Exception as e:
        score["error"] = f"clean_trips(full log) raised {type(e).__name__}: {e}"
        return _finalize(score)
    c["input_not_mutated"] = raw.equals(snapshot)

    c["required_columns"] = _usable(out)
    if not c["required_columns"]:
        c["input_not_mutated"] = False
        score["error"] = "Output is not a DataFrame with the 14 required columns"
        return _finalize(score)

    out = _normalise_ts(out)
    exp = _expected(snapshot)
    c["kept_trips_full_log"] = _kept_matches(out, exp)
    # Value checks run on the trips both sides kept, so one wrong keep/drop
    # decision does not also cost every value check. They need at least 95%
    # of the expected trips, and repaired trips must all be present.
    p = _paired(out, exp)
    covered = p.height >= 0.95 * exp.height
    raw_lat = snapshot.select("trip_id", pl.col("pickup_latitude").alias("_raw_lat"))
    repaired = exp.join(raw_lat, on="trip_id").filter(pl.col("_raw_lat") > 90)
    try:
        c["payment_full_log"] = covered and _same(p, ["payment_type"])
        c["repairs_full_log"] = (
            covered
            and repaired.height > 0
            and set(repaired["trip_id"].to_list()) <= set(p["trip_id"].to_list())
            and _same(p, ["pickup_latitude", "pickup_longitude"], tol=1e-9)
        )
        c["null_fill_full_log"] = (
            covered
            and _same(p, ["pickup_zone", "dropoff_zone"])
            and _same(p, ["tip_sgd"], tol=1e-9)
        )
        c["derived_full_log"] = covered and _same(
            p, ["trip_duration_min", "avg_speed_kmh"], tol=1e-6
        )
        c["untouched_values_full_log"] = (
            covered
            and _same(p, ["pickup_datetime", "dropoff_datetime", "passengers"])
            and _same(p, ["fare_sgd", "distance_km"], tol=1e-9)
        )
    except Exception as e:
        score["error"] = f"Value comparison failed: {type(e).__name__}: {e}"

    try:
        extract = _unseen_extract(snapshot)
        exp2 = _expected(extract)
        score["unseen_extract"] = {"rows": extract.height, "expected_kept": exp2.height}
        out2 = fn(extract.clone())
        if _usable(out2):
            out2 = _normalise_ts(out2)
            c["kept_trips_unseen_extract"] = _kept_matches(out2, exp2)
            p2 = _paired(out2, exp2)
            c["values_unseen_extract"] = (
                p2.height >= 0.95 * exp2.height
                and _same(p2, ["payment_type", "pickup_zone", "dropoff_zone"])
                and _same(p2, ["pickup_latitude", "pickup_longitude", "tip_sgd"], 1e-9)
            )
    except Exception as e:
        score["error"] = f"Unseen extract failed: {type(e).__name__}: {e}"

    score["full_log"] = {"expected_kept": exp.height, "submitted": out.height}
    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(CHECKS)
    # Gates earn no marks on their own: an output that breaks them scores 0.
    gates_ok = all(score["checks"][g] for g in GATES)
    earned = sum(1 for k, v in score["checks"].items() if v and k not in GATES)
    score["marks"] = (
        round(WEIGHT * earned / (len(CHECKS) - len(GATES)), 1) if gates_ok else 0.0
    )
    score["passed"] = score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
