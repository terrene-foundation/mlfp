# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 3 — A/B testing & multiple comparisons.

Contains: experiment data loading, group extraction, SRM check, common constants,
and small statistical helpers reused across the four technique files:

    01_bootstrap_power.py    — bootstrap CIs + MDE + power curves
    02_hypothesis_testing.py — two-proportion z-test + effect sizes
    03_multiple_testing.py   — Bonferroni + BH-FDR + FDR simulation
    04_permutation_test.py   — distribution-free alternative

Technique-specific code (the actual corrections, permutation loops, power
formulas) does NOT belong here — each technique file owns its own logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from shared.data_loader import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════

ALPHA: float = 0.05
POWER_TARGET: float = 0.80
N_BOOTSTRAP: int = 10_000
N_PERMUTATIONS: int = 10_000
RANDOM_SEED: int = 42

OUTPUT_DIR = Path("outputs") / "mlfp02_ex3_ab_testing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The experiment was DESIGNED with unequal allocation across four arms.
# SRM must be tested against this design, never against a default 50/50.
DESIGNED_ALLOCATION: dict[str, float] = {
    "control": 0.40,
    "treatment_a": 0.35,
    "treatment_b": 0.15,
    "variant_c": 0.10,
}

# Exercise 3 compares ONE treatment arm with control. Pooling several
# different treatments into one "treatment" group would estimate a
# meaningless mixture of effects.
TREATMENT_ARM: str = "treatment_a"

# Binary success event: a "qualifying order" — a basket of at least $50
# (metric_value is basket value in SGD). Nearly every user has
# metric_value > 0, so "> 0" would give a ~99% "conversion" rate.
CONVERSION_THRESHOLD: float = 50.0


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore e-commerce A/B test
# ════════════════════════════════════════════════════════════════════════


def load_experiment_all() -> pl.DataFrame:
    """Load every arm of the experiment (control + three variants).

    Columns: user_id, experiment_group, metric_value, pre_metric_value,
             revenue, timestamp, segment, platform, country, converted.

    `converted` = 1 when the user placed a qualifying order
    (metric_value >= CONVERSION_THRESHOLD).
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")
    return df.with_columns(
        (pl.col("metric_value") >= CONVERSION_THRESHOLD)
        .cast(pl.Int8)
        .alias("converted")
    )


def designed_control_share(arm: str = TREATMENT_ARM) -> float:
    """Designed share of control within the (control, arm) pair."""
    c = DESIGNED_ALLOCATION["control"]
    return c / (c + DESIGNED_ALLOCATION[arm])


def load_experiment(arm: str = TREATMENT_ARM) -> pl.DataFrame:
    """Control vs ONE treatment arm, with a `group` column in {control, treatment}.

    Only these two arms are kept, so every downstream comparison is a
    clean two-arm contrast. Run `srm_check_multi` on `load_experiment_all()`
    first to confirm the arms you analyse were allocated as designed.
    """
    df = load_experiment_all().filter(
        pl.col("experiment_group").is_in(["control", arm])
    )
    return df.with_columns(
        pl.when(pl.col("experiment_group") == "control")
        .then(pl.lit("control"))
        .otherwise(pl.lit("treatment"))
        .alias("group")
    )


def split_groups(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (control_df, treatment_df)."""
    control = df.filter(pl.col("group") == "control")
    treatment = df.filter(pl.col("group") == "treatment")
    return control, treatment


def conversion_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (control_converted, treatment_converted) as float64 arrays."""
    control, treatment = split_groups(df)
    c = control["converted"].to_numpy().astype(np.float64)
    t = treatment["converted"].to_numpy().astype(np.float64)
    return c, t


def revenue_arrays(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (control_revenue, treatment_revenue) as float64 arrays."""
    control, treatment = split_groups(df)
    c = control["revenue"].to_numpy().astype(np.float64)
    t = treatment["revenue"].to_numpy().astype(np.float64)
    return c, t


# ════════════════════════════════════════════════════════════════════════
# SANITY CHECKS — SRM
# ════════════════════════════════════════════════════════════════════════


def srm_check(
    n_control: int, n_treatment: int, expected_ratio: float
) -> dict[str, Any]:
    """χ² goodness-of-fit test for Sample Ratio Mismatch on a two-arm pair.

    `expected_ratio` is the DESIGNED share of control within the pair
    (see `designed_control_share`). There is deliberately no 50/50
    default: testing an unequal design against 50/50 always "detects" SRM.

    Returns dict with chi2, p_value, a boolean `srm`, and a verdict.
    SRM indicates randomisation bugs, bot traffic, or pipeline issues —
    if p < 0.01 do NOT trust downstream test results.
    """
    n_total = n_control + n_treatment
    expected = np.array([n_total * expected_ratio, n_total * (1 - expected_ratio)])
    observed = np.array([n_control, n_treatment])
    chi2, p = stats.chisquare(observed, f_exp=expected)
    verdict = (
        "SRM DETECTED — investigate randomisation"
        if p < 0.01
        else "OK — sample split consistent"
    )
    return {"chi2": float(chi2), "p_value": float(p), "srm": bool(p < 0.01), "verdict": verdict}


def srm_check_multi(
    counts: dict[str, int], allocation: dict[str, float] = DESIGNED_ALLOCATION
) -> dict[str, Any]:
    """χ² SRM test across ALL arms against the designed allocation.

    Returns chi2, p_value, srm flag, and a per-arm table (observed share,
    designed share, standardised residual) so you can see WHICH arm is
    mis-allocated, not just that something is wrong.
    """
    arms = list(allocation)
    observed = np.array([counts[a] for a in arms], dtype=np.float64)
    n_total = observed.sum()
    expected = np.array([allocation[a] for a in arms]) * n_total
    chi2, p = stats.chisquare(observed, f_exp=expected)
    per_arm = {
        a: {
            "observed": int(observed[i]),
            "observed_share": float(observed[i] / n_total),
            "designed_share": float(allocation[a]),
            "std_residual": float((observed[i] - expected[i]) / np.sqrt(expected[i])),
        }
        for i, a in enumerate(arms)
    }
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "srm": bool(p < 0.01),
        "per_arm": per_arm,
    }


# ════════════════════════════════════════════════════════════════════════
# SMALL REUSABLE STATS
# ════════════════════════════════════════════════════════════════════════


def two_proportion_ztest(
    p_control: float, p_treatment: float, n_control: int, n_treatment: int
) -> tuple[float, float]:
    """Pooled two-proportion z-test. Returns (z_stat, two_sided_p_value)."""
    p_pool = (p_control * n_control + p_treatment * n_treatment) / (
        n_control + n_treatment
    )
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = (p_treatment - p_control) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p_value)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return float(2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1))))


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Pooled Cohen's d effect size for two samples."""
    s_pool = np.sqrt((x1.var(ddof=1) + x2.var(ddof=1)) / 2)
    return float((x2.mean() - x1.mean()) / s_pool) if s_pool > 0 else 0.0


def interpret_magnitude(abs_effect: float) -> str:
    """Cohen convention: <0.2 negligible, <0.5 small, <0.8 medium, else large."""
    if abs_effect < 0.2:
        return "negligible"
    if abs_effect < 0.5:
        return "small"
    if abs_effect < 0.8:
        return "medium"
    return "large"


def print_header(title: str) -> None:
    """Consistent banner for each technique file."""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
