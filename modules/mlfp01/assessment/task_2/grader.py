#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 2 — HDB Feature Engineering.

Usage:
    python grader.py starter.py
    python grader.py solution.py

Re-derives every engineered feature independently and compares element-wise.
All ten checks must pass.
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
    "town",
    "flat_type",
    "flat_type_rooms",
    "sale_year",
    "storey_midpoint",
    "floor_area_sqm",
    "flat_age_years",
    "remaining_lease_years",
    "resale_price",
    "price_per_sqm",
]
_ROOMS = {
    "2 ROOM": 2,
    "3 ROOM": 3,
    "4 ROOM": 4,
    "5 ROOM": 5,
    "EXECUTIVE": 6,
    "MULTI-GENERATION": 7,
}


def _reference() -> pl.DataFrame:
    """Independent reference implementation of the engineered features."""
    df = MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")
    df = df.with_columns(
        pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year")
    )
    lo = (
        pl.col("storey_range")
        .str.extract(r"^\s*(\w+)\s+TO\s+", 1)
        .str.replace_all("O", "0")
        .cast(pl.Float64, strict=False)
    )
    hi = (
        pl.col("storey_range")
        .str.extract(r"\s+TO\s+(\w+)\s*$", 1)
        .str.replace_all("O", "0")
        .cast(pl.Float64, strict=False)
    )
    df = df.with_columns(((lo + hi) / 2.0).alias("storey_midpoint"))
    df = df.with_columns(
        [
            (pl.col("sale_year") - pl.col("lease_commence_date"))
            .cast(pl.Int64)
            .alias("flat_age_years"),
            (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("price_per_sqm"),
            pl.col("flat_type")
            .replace_strict(_ROOMS, default=None)
            .alias("flat_type_rooms"),
        ]
    )
    yrs = pl.col("remaining_lease").str.extract(r"(\d+)\s*year", 1).cast(pl.Float64)
    mons = (
        pl.col("remaining_lease")
        .str.extract(r"(\d+)\s*month", 1)
        .cast(pl.Float64)
        .fill_null(0.0)
    )
    plain = pl.col("remaining_lease").str.extract(r"^\s*(\d+)\s*$", 1).cast(pl.Float64)
    lease = pl.when(yrs.is_not_null()).then(yrs + mons / 12.0).otherwise(plain)
    df = df.with_columns(
        lease.fill_null(99.0 - pl.col("flat_age_years")).alias("remaining_lease_years")
    )
    return df.select(EXPECTED_COLUMNS).sort(["sale_year", "town"])


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a: pl.Series, b: pl.Series, tol: float = 1e-6) -> bool:
    try:
        return bool((a.cast(pl.Float64) - b.cast(pl.Float64)).abs().max() < tol)
    except Exception:
        return False


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

    ref = _reference()
    c["columns_exact"] = r.columns == EXPECTED_COLUMNS
    c["row_count_50150"] = r.height == ref.height
    if not (c["columns_exact"] and c["row_count_50150"]):
        return _finalize(score)

    derived = [
        "storey_midpoint",
        "remaining_lease_years",
        "flat_type_rooms",
        "sale_year",
        "flat_age_years",
        "price_per_sqm",
    ]
    c["no_nulls_in_derived"] = all(r[col].null_count() == 0 for col in derived)

    c["storey_midpoint_correct"] = _close(r["storey_midpoint"], ref["storey_midpoint"])
    c["remaining_lease_correct"] = _close(
        r["remaining_lease_years"], ref["remaining_lease_years"]
    )
    c["flat_type_rooms_correct"] = _close(r["flat_type_rooms"], ref["flat_type_rooms"])
    c["flat_age_correct"] = _close(r["flat_age_years"], ref["flat_age_years"])
    c["price_per_sqm_correct"] = _close(
        r["price_per_sqm"], ref["price_per_sqm"], tol=1e-4
    )

    try:
        c["sorted_by_year_town"] = r.equals(r.sort(["sale_year", "town"]))
    except Exception:
        c["sorted_by_year_town"] = False

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
