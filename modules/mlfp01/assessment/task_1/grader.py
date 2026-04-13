#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grade student submission for MLFP01 Task 1."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl


EXPECTED_COLUMNS = [
    "month",
    "mean_temperature_c",
    "total_rainfall_mm",
    "temp_deviation_c",
    "rainfall_vs_mean_pct",
    "is_wet_month",
]


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
        result = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    # Check 1: return type
    score["checks"]["returns_dataframe"] = isinstance(result, pl.DataFrame)
    if not score["checks"]["returns_dataframe"]:
        _finalize(score)
        return score

    # Check 2: row count
    score["checks"]["row_count_12"] = result.height == 12

    # Check 3: columns match exactly
    score["checks"]["columns_match"] = result.columns == EXPECTED_COLUMNS

    # Check 4: temp deviation sums ~ 0
    if "temp_deviation_c" in result.columns:
        try:
            dev_sum = float(result["temp_deviation_c"].sum())
            score["checks"]["temp_deviation_sums_zero"] = abs(dev_sum) < 0.01
        except Exception:
            score["checks"]["temp_deviation_sums_zero"] = False
    else:
        score["checks"]["temp_deviation_sums_zero"] = False

    # Check 5: January row has expected deviation (hidden ground truth)
    from shared import MLFPDataLoader

    loader = MLFPDataLoader()
    truth = loader.load("mlfp01", "sg_weather.csv")
    annual_mean_temp = float(truth["mean_temperature_c"].mean())
    annual_mean_rain = float(truth["total_rainfall_mm"].mean())
    jan_true_dev = float(
        truth.filter(pl.col("month") == "January")["mean_temperature_c"][0]
        - annual_mean_temp
    )

    try:
        jan_row = result.filter(pl.col("month") == "January")
        jan_dev = float(jan_row["temp_deviation_c"][0])
        score["checks"]["january_deviation_correct"] = abs(jan_dev - jan_true_dev) < 1e-6
    except Exception:
        score["checks"]["january_deviation_correct"] = False

    # Check 6: is_wet_month count matches expected
    expected_wet = int(
        (truth["total_rainfall_mm"] > annual_mean_rain).sum()
    )
    try:
        student_wet = int(result["is_wet_month"].sum())
        score["checks"]["wet_month_count_correct"] = student_wet == expected_wet
    except Exception:
        score["checks"]["wet_month_count_correct"] = False

    _finalize(score)
    return score


def _finalize(score: dict) -> None:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["total"] == score["max"] and score["max"] > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
