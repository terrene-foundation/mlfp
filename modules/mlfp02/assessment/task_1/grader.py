#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP02 Assessment Task 1 — Probability, Bayes & Experiment
Validation.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The grader re-derives every quantity independently from the raw data and compares
the submission against strict tolerances. All twelve checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
from scipy import stats

from shared import MLFPDataLoader

COHORT = ["control", "treatment_a"]
CONVERT_THRESHOLD = 50.0
FRAUD_BASE_RATE = 0.02
FRAUD_SENSITIVITY = 0.95
FRAUD_FPR = 0.03
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 20.0

REQUIRED_KEYS = [
    "p_convert_overall",
    "p_convert_control",
    "p_convert_treatment",
    "p_treatment_given_convert",
    "srm_chi2",
    "srm_p_value",
    "srm_flag",
    "p_fraud_given_flagged",
    "beta_post_alpha",
    "beta_post_beta",
    "posterior_mean",
    "cred_int_low",
    "cred_int_high",
]


def _reference() -> dict:
    """Independent re-derivation of the expected answer dict."""
    df = MLFPDataLoader().load("mlfp02", "experiment_data.parquet")
    co = df.filter(pl.col("experiment_group").is_in(COHORT)).with_columns(
        (pl.col("metric_value") >= CONVERT_THRESHOLD).alias("converted")
    )
    n_total = co.height
    n_control = co.filter(pl.col("experiment_group") == "control").height
    n_treatment = co.filter(pl.col("experiment_group") == "treatment_a").height

    p_overall = co["converted"].mean()
    p_ctrl = co.filter(pl.col("experiment_group") == "control")["converted"].mean()
    p_trt = co.filter(pl.col("experiment_group") == "treatment_a")["converted"].mean()
    p_treatment = n_treatment / n_total
    p_trt_given_conv = (p_trt * p_treatment) / p_overall

    expected = n_total / 2.0
    chi2 = ((n_control - expected) ** 2 / expected) + (
        (n_treatment - expected) ** 2 / expected
    )
    p_srm = float(stats.chi2.sf(chi2, df=1))

    p_flagged = FRAUD_SENSITIVITY * FRAUD_BASE_RATE + FRAUD_FPR * (
        1 - FRAUD_BASE_RATE
    )
    p_fraud = (FRAUD_SENSITIVITY * FRAUD_BASE_RATE) / p_flagged

    trt = co.filter(pl.col("experiment_group") == "treatment_a")
    s = int(trt["converted"].sum())
    f = int(trt.height - s)
    a_post = BETA_PRIOR_ALPHA + s
    b_post = BETA_PRIOR_BETA + f

    return {
        "p_convert_overall": float(p_overall),
        "p_convert_control": float(p_ctrl),
        "p_convert_treatment": float(p_trt),
        "p_treatment_given_convert": float(p_trt_given_conv),
        "srm_chi2": float(chi2),
        "srm_p_value": p_srm,
        "srm_flag": bool(p_srm < 1e-3),
        "p_fraud_given_flagged": float(p_fraud),
        "beta_post_alpha": float(a_post),
        "beta_post_beta": float(b_post),
        "posterior_mean": float(a_post / (a_post + b_post)),
        "cred_int_low": float(stats.beta.ppf(0.025, a_post, b_post)),
        "cred_int_high": float(stats.beta.ppf(0.975, a_post, b_post)),
    }


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a, b, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    try:
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
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
    c["returns_dict"] = isinstance(r, dict)
    if not c["returns_dict"]:
        return _finalize(score)
    c["has_all_keys"] = all(k in r for k in REQUIRED_KEYS)
    if not c["has_all_keys"]:
        return _finalize(score)

    ref = _reference()

    c["p_convert_overall"] = _close(r["p_convert_overall"], ref["p_convert_overall"])
    c["p_convert_control"] = _close(r["p_convert_control"], ref["p_convert_control"])
    c["p_convert_treatment"] = _close(
        r["p_convert_treatment"], ref["p_convert_treatment"]
    )
    c["bayes_inversion_correct"] = _close(
        r["p_treatment_given_convert"], ref["p_treatment_given_convert"]
    )
    c["srm_chi2_correct"] = _close(r["srm_chi2"], ref["srm_chi2"], rtol=1e-4)
    # p-value is astronomically small; accept any value within 1e-6 of the reference.
    c["srm_p_value_correct"] = _close(
        r["srm_p_value"], ref["srm_p_value"], rtol=1e-3, atol=1e-6
    )
    c["srm_flag_correct"] = bool(r["srm_flag"]) is ref["srm_flag"]
    c["fraud_bayes_correct"] = _close(
        r["p_fraud_given_flagged"], ref["p_fraud_given_flagged"]
    )
    c["beta_posterior_params_correct"] = _close(
        r["beta_post_alpha"], ref["beta_post_alpha"]
    ) and _close(r["beta_post_beta"], ref["beta_post_beta"])
    c["posterior_mean_correct"] = _close(r["posterior_mean"], ref["posterior_mean"])
    c["credible_interval_correct"] = (
        _close(r["cred_int_low"], ref["cred_int_low"], rtol=1e-4)
        and _close(r["cred_int_high"], ref["cred_int_high"], rtol=1e-4)
        and float(r["cred_int_low"]) < float(r["cred_int_high"])
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
