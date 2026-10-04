#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 5 — Model Inputs and Charts.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader cleans the taxi log itself (an independent implementation of the
Task 1 contract), so Task 5 does not depend on the student's Task 1. Then:

Part A: it splits the clean trips by a hash of trip_id. The submission is
fitted on the training trips only. The held-out trips are reduced to
booking-time fields, without the fare, and passed to prepare_bookings(). The
student never sees these rows. The checks are: one numeric row per booking,
no identifier column, time of day represented, and scaling learned from the
training rows only.

Part B: the plotted data inside each Plotly figure is compared with
aggregates the grader computes, and the chart type must suit the question.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

WEIGHT = 25
CHECKS = [
    "one_numeric_row_per_booking",
    "no_identifier_feature",
    "time_of_day_feature",
    "fitted_on_training_rows_only",
    "chart_fare_distribution",
    "chart_fare_vs_distance",
    "chart_zone_median_fare",
    "chart_monthly_trips",
    "findings",
]
BOOKING_COLS = ["trip_id", "pickup_datetime", "pickup_zone", "dropoff_zone", "distance_km",
                "passengers", "payment_type", "pickup_latitude", "pickup_longitude"]  # fmt: skip
TS = "%Y-%m-%d %H:%M:%S"


# ── Clean trips (independent implementation of the Task 1 contract) ──────
def _clean_trips() -> pl.DataFrame:
    raw = MLFPDataLoader().load("mlfp01", "sg_taxi_trips.parquet")
    low = pl.col("payment_type").str.to_lowercase()
    lat, lng = pl.col("pickup_latitude"), pl.col("pickup_longitude")
    swap = lat.is_between(103.60, 104.05) & lng.is_between(1.15, 1.47)
    df = raw.with_columns(
        pl.col("pickup_datetime").str.strptime(pl.Datetime("us"), TS),
        pl.col("dropoff_datetime").str.strptime(pl.Datetime("us"), TS),
        pl.when(low.str.contains("grab")).then(pl.lit("Grab"))
        .when(low.str.contains("nets")).then(pl.lit("NETS"))
        .when(low.str.contains("cash")).then(pl.lit("Cash"))
        .otherwise(pl.lit("Card")).alias("payment_type"),
        pl.col("tip_sgd").fill_null(0.0),
        pl.col("pickup_zone").fill_null("Unknown"),
        pl.col("dropoff_zone").fill_null("Unknown"),
        pl.when(swap).then(lng).otherwise(lat).alias("pickup_latitude"),
        pl.when(swap).then(lat).otherwise(lng).alias("pickup_longitude"),
    )  # fmt: skip
    df = df.with_columns(
        ((pl.col("dropoff_datetime") - pl.col("pickup_datetime")).dt.total_seconds() / 60.0)
        .alias("trip_duration_min")
    ).with_columns((pl.col("distance_km") * 60.0 / pl.col("trip_duration_min")).alias("avg_speed_kmh"))
    df = df.filter(
        (pl.col("pickup_datetime") < datetime(2025, 1, 1)) & (pl.col("fare_sgd") > 0)
        & (pl.col("passengers") >= 1) & pl.col("pickup_latitude").is_between(1.15, 1.47)
        & pl.col("pickup_longitude").is_between(103.60, 104.05)
        & pl.col("avg_speed_kmh").is_between(2.0, 120.0)
    )  # fmt: skip
    return df.filter(pl.col("trip_id").is_unique())


# ── Helpers ──────────────────────────────────────────────────────────────
def _arr(v) -> list:
    if v is None:
        return []
    return list(np.asarray(v).tolist()) if not isinstance(v, (list, tuple)) else list(v)


def _titled(fig) -> bool:
    lay = fig.layout
    return bool((lay.xaxis.title.text or "").strip()) and bool((lay.yaxis.title.text or "").strip())


def _same_multiset(a: list, b: list, tol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False
    try:
        return bool(np.allclose(np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float)), atol=tol))
    except Exception:
        return False


def _month_key(v) -> str:
    if isinstance(v, (datetime, date)):
        return v.strftime("%Y-%m")
    if isinstance(v, np.datetime64):
        return str(v.astype("datetime64[M]"))
    return str(v)[:7]


# ── Part A ───────────────────────────────────────────────────────────────
def _grade_part_a(mod, trips: pl.DataFrame, c: dict, notes: dict) -> None:
    test_mask = pl.col("trip_id").hash(20261004) % 5 == 0
    train, test = trips.filter(~test_mask), trips.filter(test_mask)
    bookings = test.select(BOOKING_COLS)
    fitted = mod.fit_preprocessor(train.clone())
    X = mod.prepare_bookings(fitted, bookings.clone())
    if not isinstance(X, pl.DataFrame):
        notes["part_a"] = "prepare_bookings() did not return a DataFrame"
        return
    numeric = all(dt.is_numeric() or dt == pl.Boolean for dt in X.dtypes)
    complete = X.height == bookings.height and X.width > 0 and numeric
    complete = complete and X.null_count().sum_horizontal().item() == 0
    # A degenerate frame (constants, zeros) is not model input: the trip
    # distance, the strongest booking-time signal, must survive the transform.
    raw_dist = test["distance_km"].to_numpy()
    informative = complete and any(
        np.std(X[col].cast(pl.Float64).to_numpy()) > 0
        and abs(float(np.corrcoef(X[col].cast(pl.Float64).to_numpy(), raw_dist)[0, 1])) > 0.99
        for col in X.columns
    )
    c["one_numeric_row_per_booking"] = bool(informative)
    c["no_identifier_feature"] = c["one_numeric_row_per_booking"] and not any(
        "trip_id" in col.lower() for col in X.columns
    )
    if not numeric:
        return
    hour = test["pickup_datetime"].dt.hour().cast(pl.Float64).to_numpy()
    targets = [hour, np.sin(2 * math.pi * hour / 24), np.cos(2 * math.pi * hour / 24)]
    best = 0.0
    for col in X.columns:
        x = X[col].cast(pl.Float64).to_numpy()
        if np.std(x) == 0:
            continue
        for t in targets:
            best = max(best, abs(float(np.corrcoef(x, t)[0, 1])))
    c["time_of_day_feature"] = c["one_numeric_row_per_booking"] and best > 0.95
    notes["best_hour_correlation"] = round(best, 4)

    # Scaling learned from the training rows only: recover the affine map
    # applied to distance_km on the bookings, apply it to the TRAINING
    # distances, and require zero-mean/unit-sd (z-score) or [0, 1] (min-max).
    if "distance_km" in X.columns:
        raw_d = test["distance_km"].to_numpy()
        out_d = X["distance_km"].cast(pl.Float64).to_numpy()
        a, b = np.polyfit(raw_d, out_d, 1)
        exact = np.allclose(a * raw_d + b, out_d, atol=1e-6)
        t = a * train["distance_km"].to_numpy() + b
        zscore = abs(t.mean()) < 1e-6 and (abs(t.std() - 1) < 1e-6 or abs(t.std(ddof=1) - 1) < 1e-6)
        minmax = abs(t.min()) < 1e-6 and abs(t.max() - 1) < 1e-6
        c["fitted_on_training_rows_only"] = c["one_numeric_row_per_booking"] and bool(
            exact and (zscore or minmax)
        )
        notes["distance_scaling"] = {"exact_affine": bool(exact), "zscore_on_train": bool(zscore),
                                     "minmax_on_train": bool(minmax)}  # fmt: skip
    else:
        notes["distance_scaling"] = "no distance_km column in the model inputs"


# ── Part B ───────────────────────────────────────────────────────────────
def _grade_part_b(mod, trips: pl.DataFrame, c: dict, notes: dict) -> None:
    out = mod.make_charts(trips.clone())
    figs, found = out["figures"], out["findings"]

    fares = trips["fare_sgd"].to_list()
    f = figs["fare_distribution"]
    t = f.data[0] if len(f.data) == 1 else None
    if t is not None and t.type in {"histogram", "box", "violin"}:
        values = _arr(t.x) or _arr(t.y)
        lay = f.layout
        titled = bool((lay.xaxis.title.text or lay.yaxis.title.text or "").strip())
        c["chart_fare_distribution"] = titled and _same_multiset(values, fares)

    f = figs["fare_vs_distance"]
    t = f.data[0] if len(f.data) == 1 else None
    if t is not None and t.type in {"scatter", "scattergl"} and "lines" not in (t.mode or "markers"):
        x, y = _arr(t.x), _arr(t.y)
        pairs = sorted(zip(np.round(x, 9), np.round(y, 9))) if len(x) == len(y) else []
        truth = sorted(zip(np.round(trips["distance_km"].to_numpy(), 9),
                           np.round(trips["fare_sgd"].to_numpy(), 9)))  # fmt: skip
        c["chart_fare_vs_distance"] = _titled(f) and pairs == truth

    zone_truth = dict(
        trips.filter(pl.col("pickup_zone") != "Unknown")
        .group_by("pickup_zone").agg(pl.col("fare_sgd").median()).iter_rows()
    )  # fmt: skip
    f = figs["zone_median_fare"]
    t = f.data[0] if len(f.data) == 1 else None
    if t is not None and t.type == "bar":
        horizontal = t.orientation == "h"
        cats, vals = (_arr(t.y), _arr(t.x)) if horizontal else (_arr(t.x), _arr(t.y))
        axis = f.layout.xaxis if horizontal else f.layout.yaxis
        rng = axis.range
        zero_base = axis.type != "log" and (rng is None or rng[0] <= 0)
        ordered = vals == sorted(vals) or vals == sorted(vals, reverse=True)
        values_ok = len(cats) == len(zone_truth) and all(
            cat in zone_truth and abs(float(v) - zone_truth[cat]) < 1e-9 for cat, v in zip(cats, vals)
        )
        c["chart_zone_median_fare"] = _titled(f) and zero_base and ordered and values_ok
        notes["zone_chart"] = {"zero_baseline": zero_base, "sorted": ordered, "values": values_ok}

    month_truth = dict(
        trips.group_by(pl.col("pickup_datetime").dt.strftime("%Y-%m")).len().iter_rows()
    )
    f = figs["monthly_trips"]
    t = f.data[0] if len(f.data) == 1 else None
    if t is not None and t.type in {"scatter", "scattergl"} and "lines" in (t.mode or ""):
        months = [_month_key(v) for v in _arr(t.x)]
        counts = _arr(t.y)
        c["chart_monthly_trips"] = (
            _titled(f)
            and months == sorted(month_truth)
            and [int(v) for v in counts] == [month_truth[m] for m in months]
        )

    top_zone = max(zone_truth, key=zone_truth.get)
    busiest = max(month_truth, key=month_truth.get)
    c["findings"] = (
        found.get("top_median_fare_zone") == top_zone and found.get("busiest_month") == busiest
    )


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task5", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {c: False for c in CHECKS}, "notes": {}}
    c, notes = score["checks"], score["notes"]
    try:
        mod = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Cannot import submission: {type(e).__name__}: {e}"
        return _finalize(score)
    trips = _clean_trips()
    for part, fn in (("part_a", _grade_part_a), ("part_b", _grade_part_b)):
        try:
            fn(mod, trips, c, notes)
        except Exception as e:
            score.setdefault("errors", {})[part] = f"{type(e).__name__}: {e}"
    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(CHECKS)
    score["marks"] = round(WEIGHT * score["total"] / score["max"], 1)
    score["passed"] = score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["passed"] else 1)
