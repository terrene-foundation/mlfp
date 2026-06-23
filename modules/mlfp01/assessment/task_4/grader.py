#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP01 Assessment Task 4 — Profile, Clean & Integrate.

Usage:
    python grader.py starter.py
    python grader.py solution.py

Re-derives the cleaned frame and the DataExplorer alert counts independently.
All checks must pass.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from kailash_ml import DataExplorer
from shared import MLFPDataLoader

EXPECTED_COLUMNS = [
    "period_year",
    "period_quarter",
    "gdp_growth_pct",
    "unemployment_rate",
    "inflation_rate",
    "trade_balance_sgd_bn",
    "property_price_index",
    "tourist_arrivals",
]


def _reference_clean(raw: pl.DataFrame) -> pl.DataFrame:
    q = raw.filter(pl.col("period_type") == "quarterly")
    q = q.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("period").str.extract(r"Q(\d)", 1),
                    pl.col("period").str.extract(r"-(\d)\s*$", 1),
                ]
            )
            .cast(pl.Int64)
            .alias("period_quarter"),
            pl.col("period")
            .str.extract(r"(\d{4})", 1)
            .cast(pl.Int64)
            .alias("period_year"),
        ]
    ).filter(
        pl.col("period_quarter").is_not_null() & pl.col("period_year").is_not_null()
    )
    q = q.with_columns(
        pl.col("tourist_arrivals")
        .str.replace_all(",", "")
        .str.strip_chars()
        .cast(pl.Int64)
    )
    q = q.with_columns(
        [
            pl.col("inflation_rate").fill_null(pl.col("inflation_rate").median()),
            pl.col("trade_balance_sgd_bn").fill_null(
                pl.col("trade_balance_sgd_bn").median()
            ),
        ]
    )
    return q.select(EXPECTED_COLUMNS).sort(["period_year", "period_quarter"])


async def _alerts(df: pl.DataFrame) -> int:
    return len((await DataExplorer().profile(df)).alerts)


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a: pl.Series, b: pl.Series, tol: float = 1e-4) -> bool:
    try:
        if a.null_count() != b.null_count():
            return False
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
        out = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    c["returns_dict"] = isinstance(out, dict) and {
        "cleaned",
        "raw_alert_count",
        "clean_alert_count",
    }.issubset(out)
    if not c["returns_dict"]:
        return _finalize(score)

    r = out["cleaned"]
    c["cleaned_is_dataframe"] = isinstance(r, pl.DataFrame)
    if not c["cleaned_is_dataframe"]:
        return _finalize(score)

    ref = _reference_clean(MLFPDataLoader().load("mlfp01", "economic_indicators.csv"))
    c["columns_exact"] = r.columns == EXPECTED_COLUMNS
    c["row_count_101"] = r.height == ref.height
    if not (c["columns_exact"] and c["row_count_101"]):
        return _finalize(score)

    r2 = r.sort(["period_year", "period_quarter"])
    c["tourist_arrivals_int"] = r2.schema.get("tourist_arrivals") == pl.Int64
    c["quarter_range_valid"] = (
        r2["period_quarter"].min() == 1 and r2["period_quarter"].max() == 4
    )
    c["no_nulls_imputed"] = (
        r2["inflation_rate"].null_count() == 0
        and r2["trade_balance_sgd_bn"].null_count() == 0
        and r2["tourist_arrivals"].null_count() == 0
    )
    c["tourist_arrivals_correct"] = _close(
        r2["tourist_arrivals"], ref["tourist_arrivals"], tol=0.5
    )
    c["inflation_correct"] = _close(
        r2["inflation_rate"], ref["inflation_rate"], tol=1e-3
    )
    c["keys_match"] = r2.select(["period_year", "period_quarter"]).equals(
        ref.select(["period_year", "period_quarter"])
    )

    # DataExplorer profiling: cleaning must REDUCE alerts, and the raw count
    # must match the independently-measured ground truth.
    try:
        ref_raw = asyncio.run(
            _alerts(
                MLFPDataLoader()
                .load("mlfp01", "economic_indicators.csv")
                .filter(pl.col("period_type") == "quarterly")
            )
        )
        c["raw_alert_count_correct"] = out["raw_alert_count"] == ref_raw
        c["cleaning_reduced_alerts"] = out["clean_alert_count"] < out["raw_alert_count"]
    except Exception:
        c["raw_alert_count_correct"] = False
        c["cleaning_reduced_alerts"] = False

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
