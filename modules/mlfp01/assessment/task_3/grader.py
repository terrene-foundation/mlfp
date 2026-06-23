#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 3 — Window Functions & Trends.

Usage:
    python grader.py starter.py
    python grader.py solution.py

Re-derives the per-town/per-year trend table independently and compares.
All checks must pass.
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
    "sale_year",
    "n_sales",
    "median_price",
    "yoy_pct",
    "rolling_3yr_avg",
    "price_rank_in_year",
]


def _reference() -> pl.DataFrame:
    df = (
        MLFPDataLoader()
        .load("mlfp01", "hdb_resale.parquet")
        .with_columns(pl.col("month").str.slice(0, 4).cast(pl.Int64).alias("sale_year"))
    )
    agg = (
        df.group_by(["town", "sale_year"])
        .agg(
            [
                pl.col("resale_price").median().alias("median_price"),
                pl.len().alias("n_sales"),
            ]
        )
        .sort(["town", "sale_year"])
    )
    return (
        agg.with_columns(
            [
                (
                    100.0
                    * (
                        pl.col("median_price")
                        - pl.col("median_price").shift(1).over("town")
                    )
                    / pl.col("median_price").shift(1).over("town")
                ).alias("yoy_pct"),
                pl.col("median_price")
                .rolling_mean(window_size=3, min_samples=1)
                .over("town")
                .alias("rolling_3yr_avg"),
                pl.col("median_price")
                .rank(method="min", descending=True)
                .over("sale_year")
                .cast(pl.Int64)
                .alias("price_rank_in_year"),
            ]
        )
        .select(EXPECTED_COLUMNS)
        .sort(["town", "sale_year"])
    )


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a: pl.Series, b: pl.Series, tol: float = 1e-4) -> bool:
    try:
        # Compare non-null positions; null masks must also match.
        if a.null_count() != b.null_count():
            return False
        af = a.fill_null(0.0).cast(pl.Float64)
        bf = b.fill_null(0.0).cast(pl.Float64)
        return bool((af - bf).abs().max() < tol)
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
    c["row_count_correct"] = r.height == ref.height
    if not (c["columns_exact"] and c["row_count_correct"]):
        return _finalize(score)

    # Align both on (town, sale_year) so row order can't mask value errors.
    r2 = r.sort(["town", "sale_year"])
    c["keys_match"] = r2.select(["town", "sale_year"]).equals(
        ref.select(["town", "sale_year"])
    )
    if not c["keys_match"]:
        return _finalize(score)

    c["median_price_correct"] = _close(
        r2["median_price"], ref["median_price"], tol=1e-3
    )
    c["n_sales_correct"] = _close(r2["n_sales"], ref["n_sales"])
    c["yoy_pct_correct"] = _close(r2["yoy_pct"], ref["yoy_pct"], tol=1e-4)
    c["yoy_first_year_null"] = r2["yoy_pct"].null_count() == 27
    c["rolling_3yr_correct"] = _close(
        r2["rolling_3yr_avg"], ref["rolling_3yr_avg"], tol=1e-3
    )
    c["rank_correct"] = _close(r2["price_rank_in_year"], ref["price_rank_in_year"])
    c["rank_range_valid"] = (
        r2["price_rank_in_year"].min() == 1 and r2["price_rank_in_year"].max() == 27
    )

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
