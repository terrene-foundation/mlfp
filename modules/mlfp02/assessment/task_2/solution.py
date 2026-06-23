# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 2: Hypothesis Testing, Bootstrap & CUPED
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

COHORT = ["control", "treatment_a"]
BOOT_SEED = 2024            # np.random.default_rng(BOOT_SEED)
BOOT_B = 2000              # number of bootstrap resamples
MT_P_VALUES = [0.03, 0.012, 0.04, 0.65, 0.009]   # five simultaneous tests
MT_ALPHA = 0.05


def _bootstrap_diff_ci(t: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Percentile 95% CI of mean(treatment) − mean(control).

    Deterministic protocol (MUST be followed exactly):
        rng = np.random.default_rng(BOOT_SEED)
        for each of BOOT_B iterations:
            bt = rng.choice(t, size=t.size, replace=True)   # treatment first
            bc = rng.choice(c, size=c.size, replace=True)   # then control
            diff = bt.mean() - bc.mean()
        CI = np.percentile(diffs, [2.5, 97.5])
    """
    rng = np.random.default_rng(BOOT_SEED)
    diffs = np.empty(BOOT_B)
    for b in range(BOOT_B):
        bt = rng.choice(t, size=t.size, replace=True)
        bc = rng.choice(c, size=c.size, replace=True)
        diffs[b] = bt.mean() - bc.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def _bh_num_significant(pvalues: list[float], alpha: float) -> int:
    """Benjamini-Hochberg step-up: count rejected hypotheses at FDR = alpha."""
    p = np.sort(np.asarray(pvalues, dtype=float))
    m = p.size
    thresh = alpha * np.arange(1, m + 1) / m
    below = np.where(p <= thresh)[0]
    return int(below.max() + 1) if below.size > 0 else 0


def solve() -> dict:
    """Hypothesis testing, bootstrap CIs, CUPED variance reduction, and
    multiple-testing correction over the control + treatment_a cohort.

    Returns a dict with these exact keys (full-precision floats unless noted):
        welch_t, welch_p           Welch two-sample t-test on metric_value
        mean_diff                  mean(treatment) − mean(control)
        boot_ci_low, boot_ci_high  percentile 95% CI of mean_diff (seeded)
        cuped_theta                Cov(metric,pre)/Var(pre) on the cohort (ddof=1)
        var_metric, var_adj        Var (ddof=1) before / after CUPED adjustment
        cuped_var_reduction        1 − var_adj/var_metric
        welch_t_cuped, welch_p_cuped  Welch test on CUPED-adjusted metric
        bonferroni_n_sig (int)     # significant under Bonferroni at alpha
        bh_n_sig (int)             # significant under Benjamini-Hochberg FDR
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")
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

    # --- 2a: Welch two-sample t-test (unequal variances) ---
    welch_t, welch_p = stats.ttest_ind(t, c, equal_var=False)
    mean_diff = t.mean() - c.mean()

    # --- 2b: seeded percentile bootstrap CI for the mean difference ---
    boot_ci_low, boot_ci_high = _bootstrap_diff_ci(t, c)

    # --- 2c: CUPED variance reduction using pre_metric_value as covariate ---
    metric = co["metric_value"].to_numpy().astype(float)
    pre = co["pre_metric_value"].to_numpy().astype(float)
    cuped_theta = np.cov(metric, pre, ddof=1)[0, 1] / np.var(pre, ddof=1)
    metric_adj = metric - cuped_theta * (pre - pre.mean())
    var_metric = float(np.var(metric, ddof=1))
    var_adj = float(np.var(metric_adj, ddof=1))
    cuped_var_reduction = 1.0 - var_adj / var_metric

    # --- 2d: re-run the hypothesis test on the CUPED-adjusted metric ---
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

    # --- 2e: multiple-testing correction across 5 simultaneous tests ---
    m = len(MT_P_VALUES)
    bonferroni_n_sig = int(sum(pv < MT_ALPHA / m for pv in MT_P_VALUES))
    bh_n_sig = _bh_num_significant(MT_P_VALUES, MT_ALPHA)

    return {
        "welch_t": float(welch_t),
        "welch_p": float(welch_p),
        "mean_diff": float(mean_diff),
        "boot_ci_low": boot_ci_low,
        "boot_ci_high": boot_ci_high,
        "cuped_theta": float(cuped_theta),
        "var_metric": var_metric,
        "var_adj": var_adj,
        "cuped_var_reduction": float(cuped_var_reduction),
        "welch_t_cuped": float(welch_t_cuped),
        "welch_p_cuped": float(welch_p_cuped),
        "bonferroni_n_sig": bonferroni_n_sig,
        "bh_n_sig": bh_n_sig,
    }


if __name__ == "__main__":
    r = solve()
    print(f"Welch t = {r['welch_t']:.3f}, p = {r['welch_p']:.3e}")
    print(f"mean_diff = {r['mean_diff']:.4f}")
    print(f"bootstrap 95% CI = [{r['boot_ci_low']:.4f}, {r['boot_ci_high']:.4f}]")
    print(f"CUPED theta = {r['cuped_theta']:.4f}")
    print(
        f"Var: {r['var_metric']:.2f} -> {r['var_adj']:.2f} "
        f"(reduction {r['cuped_var_reduction']:.2%})"
    )
    print(f"CUPED Welch t = {r['welch_t_cuped']:.3f}, p = {r['welch_p_cuped']:.3e}")
    print(f"Bonferroni significant: {r['bonferroni_n_sig']}, BH significant: {r['bh_n_sig']}")
