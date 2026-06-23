#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP02 Assessment Task 2 — Hypothesis Testing, Bootstrap
& CUPED.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader re-derives every statistic independently (including the seeded
bootstrap, which is bit-reproducible) and compares with tight tolerances.
All twelve checks must pass.
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

COHORT = ["control", "treatment_a"]
BOOT_SEED = 2024
BOOT_B = 2000
MT_P_VALUES = [0.03, 0.012, 0.04, 0.65, 0.009]
MT_ALPHA = 0.05

REQUIRED_KEYS = [
    "welch_t",
    "welch_p",
    "mean_diff",
    "boot_ci_low",
    "boot_ci_high",
    "cuped_theta",
    "var_metric",
    "var_adj",
    "cuped_var_reduction",
    "welch_t_cuped",
    "welch_p_cuped",
    "bonferroni_n_sig",
    "bh_n_sig",
]


def _reference() -> dict:
    df = MLFPDataLoader().load("mlfp02", "experiment_data.parquet")
    co = df.filter(pl.col("experiment_group").is_in(COHORT))
    t = (
        co.filter(pl.col("experiment_group") == "treatment_a")["metric_value"]
        .to_numpy()
        .astype(float)
    )
    c = (
        co.filter(pl.col("experiment_group") == "control")["metric_value"]
        .to_numpy()
        .astype(float)
    )
    welch_t, welch_p = stats.ttest_ind(t, c, equal_var=False)
    mean_diff = t.mean() - c.mean()

    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(BOOT_B)
    for b in range(BOOT_B):
        bt = rng.choice(t, size=t.size, replace=True)
        bc = rng.choice(c, size=c.size, replace=True)
        diffs[b] = bt.mean() - bc.mean()
    boot_lo, boot_hi = np.percentile(diffs, [2.5, 97.5])

    metric = co["metric_value"].to_numpy().astype(float)
    pre = co["pre_metric_value"].to_numpy().astype(float)
    theta = np.cov(metric, pre, ddof=1)[0, 1] / np.var(pre, ddof=1)
    metric_adj = metric - theta * (pre - pre.mean())
    var_metric = float(np.var(metric, ddof=1))
    var_adj = float(np.var(metric_adj, ddof=1))

    co_adj = co.with_columns(pl.Series("metric_adj", metric_adj))
    ta = (
        co_adj.filter(pl.col("experiment_group") == "treatment_a")["metric_adj"]
        .to_numpy()
        .astype(float)
    )
    ca = (
        co_adj.filter(pl.col("experiment_group") == "control")["metric_adj"]
        .to_numpy()
        .astype(float)
    )
    welch_t_cuped, welch_p_cuped = stats.ttest_ind(ta, ca, equal_var=False)

    m = len(MT_P_VALUES)
    bonf = int(sum(pv < MT_ALPHA / m for pv in MT_P_VALUES))
    p = np.sort(np.asarray(MT_P_VALUES, dtype=float))
    thr = MT_ALPHA * np.arange(1, m + 1) / m
    below = np.where(p <= thr)[0]
    bh = int(below.max() + 1) if below.size > 0 else 0

    return {
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mean_diff": float(mean_diff),
        "boot_ci_low": float(boot_lo),
        "boot_ci_high": float(boot_hi),
        "cuped_theta": float(theta),
        "var_metric": var_metric,
        "var_adj": var_adj,
        "cuped_var_reduction": float(1.0 - var_adj / var_metric),
        "welch_t_cuped": float(welch_t_cuped),
        "welch_p_cuped": float(welch_p_cuped),
        "bonferroni_n_sig": bonf,
        "bh_n_sig": bh,
    }


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task2", path)
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

    c["welch_test_correct"] = _close(r["welch_t"], ref["welch_t"], rtol=1e-4) and _close(
        r["welch_p"], ref["welch_p"], rtol=1e-3, atol=1e-9
    )
    c["mean_diff_correct"] = _close(r["mean_diff"], ref["mean_diff"], rtol=1e-6)
    # Seeded bootstrap is bit-reproducible — demand a tight match.
    c["bootstrap_ci_correct"] = (
        _close(r["boot_ci_low"], ref["boot_ci_low"], rtol=1e-6, atol=1e-6)
        and _close(r["boot_ci_high"], ref["boot_ci_high"], rtol=1e-6, atol=1e-6)
        and float(r["boot_ci_low"]) < float(r["boot_ci_high"])
    )
    # CI must straddle the point estimate and exclude zero (significant lift).
    c["bootstrap_ci_excludes_zero"] = float(r["boot_ci_low"]) > 0.0
    c["cuped_theta_correct"] = _close(r["cuped_theta"], ref["cuped_theta"], rtol=1e-5)
    c["cuped_variances_correct"] = _close(
        r["var_metric"], ref["var_metric"], rtol=1e-5
    ) and _close(r["var_adj"], ref["var_adj"], rtol=1e-5)
    c["cuped_reduction_correct"] = _close(
        r["cuped_var_reduction"], ref["cuped_var_reduction"], rtol=1e-5
    ) and (0.0 < float(r["cuped_var_reduction"]) < 1.0)
    c["cuped_test_correct"] = _close(
        r["welch_t_cuped"], ref["welch_t_cuped"], rtol=1e-4
    ) and _close(r["welch_p_cuped"], ref["welch_p_cuped"], rtol=1e-3, atol=1e-9)
    # CUPED removes covariate variance -> strictly larger |t| than unadjusted.
    c["cuped_increases_power"] = abs(float(r["welch_t_cuped"])) > abs(
        float(r["welch_t"])
    )
    c["bonferroni_correct"] = int(r["bonferroni_n_sig"]) == ref["bonferroni_n_sig"]
    c["bh_correct"] = int(r["bh_n_sig"]) == ref["bh_n_sig"]
    # BH is at least as powerful as Bonferroni.
    c["bh_at_least_bonferroni"] = int(r["bh_n_sig"]) >= int(r["bonferroni_n_sig"])

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
