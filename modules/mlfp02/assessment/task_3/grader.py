#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP02 Assessment Task 3 — Regression Modelling &
Interpretation.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader re-derives the OLS fit (closed form), the partial F-test, and the
logistic MLE (IRLS) independently and compares against tight tolerances.
The logistic MLE is unique (convex), so any correct solver converges to the
same coefficients. All twelve checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

OLS_FEATURES = [
    "income_imp",
    "age",
    "employment_years",
    "debt_to_income",
    "credit_age_years",
    "num_dependents",
    "edu_ord",
]
LOGIT_FEATURES = [
    "credit_utilization",
    "num_late_payments",
    "previous_defaults",
    "debt_to_income",
    "num_hard_inquiries",
]
EDU_MAP = {
    "primary": 1.0,
    "secondary": 2.0,
    "diploma": 3.0,
    "degree": 4.0,
    "postgraduate": 5.0,
}
TARGET = "loan_amount_sgd"

REQUIRED_KEYS = [
    "n_obs",
    "coefficients",
    "t_stats",
    "p_values",
    "r_squared",
    "adj_r_squared",
    "f_statistic",
    "f_p_value",
    "partial_f",
    "partial_f_p_value",
    "delta_r_squared",
    "odds_ratios",
    "strongest_logit_predictor",
]


def _zscore(mat):
    return (mat - mat.mean(axis=0)) / mat.std(axis=0, ddof=0)


def _reference() -> dict:
    df = MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")
    income_median = df["income_sgd"].median()
    d = df.with_columns(
        pl.col("income_sgd").fill_null(income_median).alias("income_imp"),
        pl.col("education").replace_strict(EDU_MAP).cast(pl.Float64).alias("edu_ord"),
    )
    n = d.height
    y = d[TARGET].to_numpy().astype(float)

    Z = np.column_stack([d[f].to_numpy().astype(float) for f in OLS_FEATURES])
    Zs = _zscore(Z)
    X = np.column_stack([np.ones(n), Zs])
    p = X.shape[1]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float((resid**2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p)
    sigma2 = rss / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t_vals = beta / se
    p_vals = 2.0 * stats.t.sf(np.abs(t_vals), df=n - p)
    F = (r2 / (p - 1)) / ((1.0 - r2) / (n - p))
    Fp = float(stats.f.sf(F, p - 1, n - p))
    names = ["intercept"] + OLS_FEATURES

    inc_s = Zs[:, OLS_FEATURES.index("income_imp")]
    age_s = Zs[:, OLS_FEATURES.index("age")]
    emp_s = Zs[:, OLS_FEATURES.index("employment_years")]
    Xf = np.column_stack([X, inc_s**2, age_s * emp_s])
    pf = Xf.shape[1]
    bf, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    rss_f = float(((y - Xf @ bf) ** 2).sum())
    r2_f = 1.0 - rss_f / tss
    q = pf - p
    pF = ((rss - rss_f) / q) / (rss_f / (n - pf))
    pFp = float(stats.f.sf(pF, q, n - pf))

    Zl = _zscore(np.column_stack([d[f].to_numpy().astype(float) for f in LOGIT_FEATURES]))
    Xl = np.column_stack([np.ones(n), Zl])
    yl = d["default"].to_numpy().astype(float)
    bl = np.zeros(Xl.shape[1])
    for _ in range(100):
        eta = Xl @ bl
        pr = 1.0 / (1.0 + np.exp(-eta))
        w = pr * (1.0 - pr)
        step = np.linalg.solve((Xl * w[:, None]).T @ Xl, Xl.T @ (yl - pr))
        bl = bl + step
        if np.linalg.norm(step) < 1e-10:
            break
    logit_names = ["intercept"] + LOGIT_FEATURES
    strongest = logit_names[1 + int(np.argmax(np.abs(bl[1:])))]

    return {
        "n_obs": int(n),
        "coefficients": {nm: float(beta[i]) for i, nm in enumerate(names)},
        "t_stats": {nm: float(t_vals[i]) for i, nm in enumerate(names)},
        "p_values": {nm: float(p_vals[i]) for i, nm in enumerate(names)},
        "r_squared": float(r2),
        "adj_r_squared": float(adj),
        "f_statistic": float(F),
        "f_p_value": Fp,
        "partial_f": float(pF),
        "partial_f_p_value": pFp,
        "delta_r_squared": float(r2_f - r2),
        "odds_ratios": {nm: float(np.exp(bl[i])) for i, nm in enumerate(logit_names)},
        "strongest_logit_predictor": strongest,
        "_names": names,
        "_logit_names": logit_names,
    }


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _close(a, b, rtol: float = 1e-4, atol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    except Exception:
        return False


def _dict_close(rd, ref, names, rtol=1e-4, atol=1e-4) -> bool:
    if not isinstance(rd, dict):
        return False
    return all(k in rd and _close(rd[k], ref[k], rtol, atol) for k in names)


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
    names = ref["_names"]
    logit_names = ref["_logit_names"]

    c["n_obs_correct"] = int(r["n_obs"]) == ref["n_obs"]
    c["coefficients_correct"] = _dict_close(
        r["coefficients"], ref["coefficients"], names, rtol=1e-4, atol=1e-2
    )
    c["t_stats_correct"] = _dict_close(
        r["t_stats"], ref["t_stats"], names, rtol=1e-3, atol=1e-2
    )
    c["p_values_correct"] = _dict_close(
        r["p_values"], ref["p_values"], names, rtol=1e-3, atol=1e-6
    )
    c["r_squared_correct"] = _close(r["r_squared"], ref["r_squared"], rtol=1e-5)
    c["adj_r_squared_correct"] = _close(
        r["adj_r_squared"], ref["adj_r_squared"], rtol=1e-5
    )
    c["f_statistic_correct"] = _close(r["f_statistic"], ref["f_statistic"], rtol=1e-4)
    c["partial_f_correct"] = _close(r["partial_f"], ref["partial_f"], rtol=1e-3)
    c["partial_f_pvalue_correct"] = _close(
        r["partial_f_p_value"], ref["partial_f_p_value"], rtol=1e-2, atol=1e-6
    )
    # Significant addition, but a negligible R^2 gain (the key teaching point).
    c["delta_r2_correct"] = _close(
        r["delta_r_squared"], ref["delta_r_squared"], rtol=1e-3, atol=1e-6
    ) and (float(r["delta_r_squared"]) < 1e-3)
    c["odds_ratios_correct"] = _dict_close(
        r["odds_ratios"], ref["odds_ratios"], logit_names, rtol=1e-3, atol=1e-3
    )
    c["strongest_predictor_correct"] = (
        r["strongest_logit_predictor"] == ref["strongest_logit_predictor"]
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
