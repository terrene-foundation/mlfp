#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP02 Assessment Task 4 — Feature Engineering &
Feature Store.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader independently re-derives the full admission-level feature table from
the five raw ICU tables and compares the submission column-by-column (exact for
ints / strings, tight tolerance for floats). All twelve checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from shared import MLFPDataLoader

DT_FMT = "%Y-%m-%d %H:%M:%S"

FEATURE_COLUMNS = [
    "admission_id",
    "feature_timestamp",
    "age",
    "gender",
    "bmi",
    "diagnosis",
    "icu_type",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "n_vitals",
    "n_labs",
    "n_abnormal_labs",
    "mean_creatinine",
    "n_distinct_drugs",
    "n_iv_meds",
    "total_dose_mg",
    "los_days",
]
_MEDIAN_IMPUTE = [
    "age",
    "bmi",
    "mean_heart_rate",
    "mean_systolic_bp",
    "min_spo2",
    "max_temperature",
    "mean_creatinine",
]
_ZERO_IMPUTE_INT = ["n_vitals", "n_labs", "n_abnormal_labs", "n_distinct_drugs", "n_iv_meds"]
_FLOAT_COLS = _MEDIAN_IMPUTE + ["total_dose_mg", "los_days"]
_INT_COLS = _ZERO_IMPUTE_INT
_STR_COLS = ["admission_id", "gender", "diagnosis", "icu_type"]


def _reference() -> pl.DataFrame:
    loader = MLFPDataLoader()
    adm = loader.load("mlfp02", "icu_admissions.parquet")
    pat = loader.load("mlfp02", "icu_patients.parquet")
    vit = loader.load("mlfp02", "icu_vitals.parquet")
    labs = loader.load("mlfp02", "icu_labs.parquet")
    meds = loader.load("mlfp02", "icu_medications.parquet")

    base = adm.select(
        "admission_id",
        "patient_id",
        "diagnosis",
        "icu_type",
        "los_days",
        pl.col("admit_time").str.strptime(pl.Datetime, DT_FMT).alias("feature_timestamp"),
    ).join(pat.select("patient_id", "age", "gender", "bmi"), on="patient_id", how="left")

    vag = vit.group_by("admission_id").agg(
        pl.col("heart_rate").mean().alias("mean_heart_rate"),
        pl.col("systolic_bp").mean().alias("mean_systolic_bp"),
        pl.col("spo2").min().alias("min_spo2"),
        pl.col("temperature").max().alias("max_temperature"),
        pl.len().alias("n_vitals"),
    )
    labs_parsed = labs.with_columns(
        pl.col("value").cast(pl.Float64, strict=False).alias("val_num"),
        pl.col("flag").str.to_lowercase().alias("flag_l"),
    )
    lag = labs_parsed.group_by("admission_id").agg(
        pl.len().alias("n_labs"),
        (pl.col("flag_l") == "abnormal").sum().alias("n_abnormal_labs"),
        pl.col("val_num")
        .filter(pl.col("test_name") == "Creatinine")
        .mean()
        .alias("mean_creatinine"),
    )
    meds_parsed = meds.with_columns(
        pl.col("dose").str.extract(r"([0-9]+\.?[0-9]*)", 1).cast(pl.Float64).alias("dose_mg")
    )
    mag = meds_parsed.group_by("admission_id").agg(
        pl.col("drug_name").n_unique().alias("n_distinct_drugs"),
        (pl.col("route") == "IV").sum().alias("n_iv_meds"),
        pl.col("dose_mg").sum().alias("total_dose_mg"),
    )

    ft = (
        base.join(vag, on="admission_id", how="left")
        .join(lag, on="admission_id", how="left")
        .join(mag, on="admission_id", how="left")
    )
    ft = ft.with_columns(
        pl.col("gender").fill_null("Unknown"),
        pl.col("total_dose_mg").fill_null(0.0),
        *[pl.col(col).fill_null(0).cast(pl.Int64) for col in _ZERO_IMPUTE_INT],
        *[
            pl.col(col).cast(pl.Float64).fill_null(pl.col(col).median())
            for col in _MEDIAN_IMPUTE
        ],
    )
    return ft.select(FEATURE_COLUMNS).sort("admission_id")


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _floats_match(a: pl.Series, b: pl.Series, tol: float = 1e-6) -> bool:
    try:
        return bool((a.cast(pl.Float64) - b.cast(pl.Float64)).abs().max() < tol)
    except Exception:
        return False


def _ints_match(a: pl.Series, b: pl.Series) -> bool:
    try:
        return bool((a.cast(pl.Int64) == b.cast(pl.Int64)).all())
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

    c["columns_exact"] = r.columns == FEATURE_COLUMNS
    if not c["columns_exact"]:
        return _finalize(score)

    ref = _reference()
    c["row_count_8000"] = r.height == ref.height
    c["sorted_by_admission_id"] = bool(r["admission_id"].is_sorted())
    c["no_nulls_anywhere"] = sum(r[col].null_count() for col in r.columns) == 0
    if not (c["row_count_8000"] and c["sorted_by_admission_id"]):
        return _finalize(score)

    c["feature_timestamp_dtype"] = r.schema.get("feature_timestamp") == pl.Datetime
    try:
        c["entity_keys_match"] = bool((r["admission_id"] == ref["admission_id"]).all())
    except Exception:
        c["entity_keys_match"] = False

    # Demographics (median-imputed).
    c["demographics_correct"] = _floats_match(r["age"], ref["age"], 1e-6) and _floats_match(
        r["bmi"], ref["bmi"], 1e-6
    )
    # Vitals aggregates.
    c["vitals_features_correct"] = (
        _floats_match(r["mean_heart_rate"], ref["mean_heart_rate"], 1e-4)
        and _floats_match(r["mean_systolic_bp"], ref["mean_systolic_bp"], 1e-4)
        and _floats_match(r["min_spo2"], ref["min_spo2"], 1e-4)
        and _floats_match(r["max_temperature"], ref["max_temperature"], 1e-4)
        and _ints_match(r["n_vitals"], ref["n_vitals"])
    )
    # Labs aggregates (value parsing + flag normalisation + Creatinine pivot).
    c["labs_features_correct"] = (
        _ints_match(r["n_labs"], ref["n_labs"])
        and _ints_match(r["n_abnormal_labs"], ref["n_abnormal_labs"])
        and _floats_match(r["mean_creatinine"], ref["mean_creatinine"], 1e-4)
    )
    # Medication aggregates (dose parsing).
    c["meds_features_correct"] = (
        _ints_match(r["n_distinct_drugs"], ref["n_distinct_drugs"])
        and _ints_match(r["n_iv_meds"], ref["n_iv_meds"])
        and _floats_match(r["total_dose_mg"], ref["total_dose_mg"], 1e-4)
    )
    # Categorical features + label preserved exactly.
    c["categoricals_and_label_correct"] = (
        bool((r["gender"] == ref["gender"]).all())
        and bool((r["diagnosis"] == ref["diagnosis"]).all())
        and bool((r["icu_type"] == ref["icu_type"]).all())
        and _floats_match(r["los_days"], ref["los_days"], 1e-9)
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
