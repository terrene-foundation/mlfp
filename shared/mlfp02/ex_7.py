# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP02 Exercise 7 — CUPED and Causal Inference.

Contains: experiment data loading, designed-allocation SRM checks, naive
A/B baseline, CUPED reference helpers, Bayesian decision utilities, mSPRT
helpers, DiD panel simulator and parallel-trends test. Technique-specific narration and
checkpoints live in the per-technique files.

Importable from any cwd after `uv sync`:

    from shared.mlfp02.ex_7 import (
        load_experiment, compute_srm, naive_ab, single_cov_cuped, ...
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

# Output directory for visualisation artifacts
OUTPUT_DIR = Path("outputs") / "ex7_causal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Experiment data with pre-experiment covariates
# ════════════════════════════════════════════════════════════════════════


# The experiment was DESIGNED as a four-arm test with this traffic split.
# (variant_c was meant to receive 10% of users; a bucketing bug sent it 15%.)
# SRM checks must compare observed counts against THIS design, not 50/50.
DESIGNED_ALLOCATION: dict[str, float] = {
    "control": 0.40,
    "treatment_a": 0.35,
    "treatment_b": 0.15,
    "variant_c": 0.10,
}

# Exercise 7 analyses ONE treatment arm against control. Pooling arms would
# estimate a mixture of three different treatments, which answers no
# product question.
ANALYSIS_ARM = "treatment_a"


def load_experiment() -> pl.DataFrame:
    """Load the MLFP02 experiment dataset.

    Columns: user_id, experiment_group, metric_value, pre_metric_value,
    revenue, timestamp, segment, platform, country.

    ``pre_metric_value`` is measured BEFORE assignment (a valid CUPED
    covariate). ``metric_value`` is measured DURING the experiment and is
    affected by treatment — it must never be used as a CUPED covariate.
    """
    loader = MLFPDataLoader()
    return loader.load("mlfp02", "experiment_data.parquet")


def split_groups(
    experiment: pl.DataFrame, treatment_arm: str = ANALYSIS_ARM
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (control, treatment) sub-frames for control vs ONE named arm."""
    arms = set(experiment["experiment_group"].unique().to_list())
    if treatment_arm not in arms:
        raise ValueError(f"Unknown arm {treatment_arm!r}; available: {sorted(arms)}")
    control = experiment.filter(pl.col("experiment_group") == "control")
    treatment = experiment.filter(pl.col("experiment_group") == treatment_arm)
    return control, treatment


def get_revenue_arrays(
    control: pl.DataFrame, treatment: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Extract revenue arrays as float64 numpy arrays."""
    y_c = control["revenue"].to_numpy().astype(np.float64)
    y_t = treatment["revenue"].to_numpy().astype(np.float64)
    return y_c, y_t


def get_covariate_arrays(
    control: pl.DataFrame, treatment: pl.DataFrame, column: str = "pre_metric_value"
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a pre-experiment covariate as float64 numpy arrays."""
    x_c = control[column].to_numpy().astype(np.float64)
    x_t = treatment[column].to_numpy().astype(np.float64)
    return x_c, x_t


# ════════════════════════════════════════════════════════════════════════
# SAMPLE RATIO MISMATCH (SRM)
# ════════════════════════════════════════════════════════════════════════


def srm_allocation_check(
    experiment: pl.DataFrame,
    allocation: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Chi-square SRM test of ALL arms against the designed allocation.

    Returns the chi-square statistic, p-value, and a per-arm table of
    observed vs expected counts with standardised residuals
    (observed - expected) / sqrt(expected), which localise the faulty arm.
    """
    allocation = allocation or DESIGNED_ALLOCATION
    counts = experiment.group_by("experiment_group").len()
    observed = {r["experiment_group"]: int(r["len"]) for r in counts.iter_rows(named=True)}
    arms = list(allocation)
    total = sum(observed.get(a, 0) for a in arms)
    obs = np.array([observed.get(a, 0) for a in arms], dtype=np.float64)
    exp = np.array([allocation[a] for a in arms]) * total
    chi2, p = stats.chisquare(obs, f_exp=exp)
    residuals = (obs - exp) / np.sqrt(exp)
    table = [
        {
            "arm": a,
            "observed": int(o),
            "expected": float(e),
            "observed_share": float(o / total),
            "designed_share": float(allocation[a]),
            "std_residual": float(r),
        }
        for a, o, e, r in zip(arms, obs, exp, residuals)
    ]
    return {"chi2": float(chi2), "p_value": float(p), "arms": table}


def compute_srm(
    n_c: int,
    n_t: int,
    design_c: float = DESIGNED_ALLOCATION["control"],
    design_t: float = DESIGNED_ALLOCATION[ANALYSIS_ARM],
) -> float:
    """Pairwise chi-square SRM test against the DESIGNED ratio design_c:design_t.

    Returns the p-value; p < 0.01 indicates sample ratio mismatch and the
    comparison must not be analysed until the cause is found.
    """
    observed = np.array([n_c, n_t], dtype=np.float64)
    share_c = design_c / (design_c + design_t)
    expected = (n_c + n_t) * np.array([share_c, 1 - share_c])
    _, srm_p = stats.chisquare(observed, f_exp=expected)
    return float(srm_p)


# ════════════════════════════════════════════════════════════════════════
# STANDARD A/B (BASELINE)
# ════════════════════════════════════════════════════════════════════════


def naive_ab(y_c: np.ndarray, y_t: np.ndarray) -> dict[str, float]:
    """Standard Welch-style lift, SE, 95% CI, z, two-sided p-value."""
    n_c, n_t = len(y_c), len(y_t)
    mean_c, mean_t = y_c.mean(), y_t.mean()
    lift = mean_t - mean_c
    se = float(np.sqrt(y_c.var(ddof=1) / n_c + y_t.var(ddof=1) / n_t))
    ci_lo = lift - 1.96 * se
    ci_hi = lift + 1.96 * se
    z = lift / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        "mean_c": float(mean_c),
        "mean_t": float(mean_t),
        "lift": float(lift),
        "se": se,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "z": float(z),
        "p_value": float(p),
    }


# ════════════════════════════════════════════════════════════════════════
# CUPED — SINGLE COVARIATE
# ════════════════════════════════════════════════════════════════════════


def single_cov_cuped(
    y_c: np.ndarray, y_t: np.ndarray, x_c: np.ndarray, x_t: np.ndarray
) -> dict[str, Any]:
    """Single-covariate CUPED: Y_adj = Y - theta*(X - E[X]).

    theta = Cov(Y, X) / Var(X). Returns adjusted arrays, point estimate,
    SE, CI, p-value, theta, rho, and variance reduction.
    """
    x_all = np.concatenate([x_c, x_t])
    y_all = np.concatenate([y_c, y_t])
    var_x = np.var(x_all, ddof=1)
    theta = np.cov(y_all, x_all)[0, 1] / var_x if var_x > 0 else 0.0
    rho = np.corrcoef(y_all, x_all)[0, 1]
    x_mean = x_all.mean()

    y_c_adj = y_c - theta * (x_c - x_mean)
    y_t_adj = y_t - theta * (x_t - x_mean)

    n_c, n_t = len(y_c), len(y_t)
    lift_adj = y_t_adj.mean() - y_c_adj.mean()
    se = float(np.sqrt(y_c_adj.var(ddof=1) / n_c + y_t_adj.var(ddof=1) / n_t))
    ci_lo, ci_hi = lift_adj - 1.96 * se, lift_adj + 1.96 * se
    z = lift_adj / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "theta": float(theta),
        "rho": float(rho),
        "y_c_adj": y_c_adj,
        "y_t_adj": y_t_adj,
        "lift": float(lift_adj),
        "se": se,
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "z": float(z),
        "p_value": float(p),
        "theoretical_reduction": float(rho**2),
    }


# ════════════════════════════════════════════════════════════════════════
# CUPED — MULTI-COVARIATE
# ════════════════════════════════════════════════════════════════════════


def multi_cov_cuped(
    y_c: np.ndarray,
    y_t: np.ndarray,
    X_c: np.ndarray,
    X_t: np.ndarray,
) -> dict[str, Any]:
    """Multi-covariate CUPED via OLS on centered covariates.

    theta = (X'X)^-1 X'Y — multivariate regression coefficients.
    """
    y_all = np.concatenate([y_c, y_t])
    X_all = np.vstack([X_c, X_t])
    X_mean = X_all.mean(axis=0)
    X_centered = X_all - X_mean
    theta = np.linalg.lstsq(X_centered, y_all - y_all.mean(), rcond=None)[0]

    y_c_adj = y_c - (X_c - X_mean) @ theta
    y_t_adj = y_t - (X_t - X_mean) @ theta

    n_c, n_t = len(y_c), len(y_t)
    lift = y_t_adj.mean() - y_c_adj.mean()
    se = float(np.sqrt(y_c_adj.var(ddof=1) / n_c + y_t_adj.var(ddof=1) / n_t))
    return {
        "theta": theta,
        "y_c_adj": y_c_adj,
        "y_t_adj": y_t_adj,
        "lift": float(lift),
        "se": se,
        "ci_lo": float(lift - 1.96 * se),
        "ci_hi": float(lift + 1.96 * se),
    }


# ════════════════════════════════════════════════════════════════════════
# CUPED — STRATIFIED
# ════════════════════════════════════════════════════════════════════════


def stratify_by_covariate(
    x_c: np.ndarray, x_t: np.ndarray, percentiles: tuple[int, int] = (33, 67)
) -> dict[str, np.ndarray]:
    """Build Low/Medium/High strata masks over concatenated (x_c, x_t)."""
    x_all = np.concatenate([x_c, x_t])
    q_lo, q_hi = np.percentile(x_all, percentiles)
    return {
        "Low spenders": (x_all <= q_lo),
        "Medium spenders": (x_all > q_lo) & (x_all <= q_hi),
        "High spenders": (x_all > q_hi),
    }


def stratified_cuped(
    y_c: np.ndarray,
    y_t: np.ndarray,
    x_c: np.ndarray,
    x_t: np.ndarray,
    strata: dict[str, np.ndarray],
    min_per_cell: int = 30,
) -> dict[str, dict[str, float]]:
    """Apply CUPED within each stratum. Returns {name: {n_ctrl,n_treat,lift,se,p}}."""
    n_c = len(y_c)
    results: dict[str, dict[str, float]] = {}
    for name, mask in strata.items():
        ctrl_mask, treat_mask = mask[:n_c], mask[n_c:]
        y_c_s, y_t_s = y_c[ctrl_mask], y_t[treat_mask]
        x_c_s, x_t_s = x_c[ctrl_mask], x_t[treat_mask]
        if len(y_c_s) < min_per_cell or len(y_t_s) < min_per_cell:
            continue
        x_all_s = np.concatenate([x_c_s, x_t_s])
        y_all_s = np.concatenate([y_c_s, y_t_s])
        var_x = np.var(x_all_s, ddof=1)
        theta_s = np.cov(y_all_s, x_all_s)[0, 1] / var_x if var_x > 0 else 0.0
        x_mean = x_all_s.mean()
        y_c_adj = y_c_s - theta_s * (x_c_s - x_mean)
        y_t_adj = y_t_s - theta_s * (x_t_s - x_mean)
        lift = y_t_adj.mean() - y_c_adj.mean()
        se = float(
            np.sqrt(y_c_adj.var(ddof=1) / len(y_c_s) + y_t_adj.var(ddof=1) / len(y_t_s))
        )
        z = lift / se if se > 0 else 0.0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        results[name] = {
            "n_ctrl": len(y_c_s),
            "n_treat": len(y_t_s),
            "lift": float(lift),
            "se": se,
            "p_value": float(p),
        }
    return results


# ════════════════════════════════════════════════════════════════════════
# BAYESIAN A/B
# ════════════════════════════════════════════════════════════════════════


def bayesian_decision(
    y_c_adj: np.ndarray,
    y_t_adj: np.ndarray,
    lift: float,
    practical_threshold: float = 1.0,
) -> dict[str, float]:
    """Bayesian posterior using normal approximation on CUPED-adjusted arrays.

    Returns P(treatment > control), P(treatment > control + threshold),
    expected loss (both directions), and credible interval.
    """
    n_c, n_t = len(y_c_adj), len(y_t_adj)
    se_c = y_c_adj.std(ddof=1) / np.sqrt(n_c)
    se_t = y_t_adj.std(ddof=1) / np.sqrt(n_t)
    se_lift = float(np.sqrt(se_c**2 + se_t**2))

    prob_better = float(1 - stats.norm.cdf(0, loc=lift, scale=se_lift))
    prob_practical = float(
        1 - stats.norm.cdf(practical_threshold, loc=lift, scale=se_lift)
    )

    # Posterior lift L ~ Normal(m = lift, s = se_lift). With z = m / s:
    #   loss if we ship treatment  = E[max(0, -L)] = s*phi(z) - m*Phi(-z)
    #   loss if we keep control    = E[max(0,  L)] = s*phi(z) + m*Phi(z)
    z = lift / se_lift
    exp_loss_treat = float(se_lift * stats.norm.pdf(z) - lift * stats.norm.cdf(-z))
    exp_loss_ctrl = float(se_lift * stats.norm.pdf(z) + lift * stats.norm.cdf(z))

    return {
        "prob_treatment_better": prob_better,
        "prob_practical": prob_practical,
        "expected_loss_treatment": exp_loss_treat,
        "expected_loss_control": exp_loss_ctrl,
        "se_lift": se_lift,
        "ci_lo": float(lift - 1.96 * se_lift),
        "ci_hi": float(lift + 1.96 * se_lift),
    }


def bayesian_decision_rule(prob_better: float, exp_loss_treat: float) -> str:
    """Simple ship/continue/hold decision rule."""
    if prob_better > 0.95 and exp_loss_treat < 0.50:
        return "SHIP — high confidence + low expected loss"
    if prob_better > 0.80:
        return "CONTINUE — promising but need more data"
    return "HOLD — insufficient evidence"


# ════════════════════════════════════════════════════════════════════════
# SEQUENTIAL TESTING — mSPRT
# ════════════════════════════════════════════════════════════════════════


def msprt_lambda(diff: float, v_n: float, tau_sq: float) -> float:
    """mSPRT mixture likelihood ratio for a Normal mean difference.

    Lambda_n = sqrt(V / (V + tau^2)) * exp(tau^2 * diff^2 / (2 V (V + tau^2)))
    where V is the variance of the difference estimate at the current look.
    """
    return float(
        np.sqrt(v_n / (v_n + tau_sq))
        * np.exp(tau_sq * diff**2 / (2 * v_n * (v_n + tau_sq)))
    )


def msprt_sequential_pvalues(
    experiment: pl.DataFrame,
    tau_sq: float,
    treatment_arm: str = ANALYSIS_ARM,
    min_per_group: int = 100,
    skip_first_days: int = 3,
) -> list[dict[str, float]]:
    """Walk the experiment day by day, computing fixed and mSPRT p-values.

    tau_sq is the mSPRT hyperparameter — typically set to se_naive**2.
    The always-valid p-value is the running minimum of 1/Lambda_n, so it can
    only go down as evidence accumulates.
    """
    if experiment["timestamp"].dtype in [pl.Utf8, pl.String]:
        exp_daily = experiment.with_columns(
            pl.col("timestamp")
            .str.to_datetime("%Y-%m-%d %H:%M:%S")
            .dt.date()
            .alias("day")
        )
    else:
        exp_daily = experiment.with_columns(
            pl.col("timestamp").cast(pl.Date).alias("day")
        )

    days = sorted(exp_daily["day"].unique().to_list())
    results: list[dict[str, float]] = []
    p_seq = 1.0
    for i, day in enumerate(days):
        if i < skip_first_days:
            continue
        cumulative = exp_daily.filter(pl.col("day") <= day)
        c = (
            cumulative.filter(pl.col("experiment_group") == "control")["revenue"]
            .to_numpy()
            .astype(np.float64)
        )
        t = (
            cumulative.filter(pl.col("experiment_group") == treatment_arm)["revenue"]
            .to_numpy()
            .astype(np.float64)
        )
        if len(c) < min_per_group or len(t) < min_per_group:
            continue
        diff = t.mean() - c.mean()
        v_n = c.var(ddof=1) / len(c) + t.var(ddof=1) / len(t)
        se = float(np.sqrt(v_n))
        z = diff / se if se > 0 else 0.0
        p_fixed = float(2 * (1 - stats.norm.cdf(abs(z))))
        # mSPRT always-valid p-value: running minimum of 1 / Lambda_n
        lambda_n = msprt_lambda(diff, v_n, tau_sq)
        p_seq = float(min(p_seq, 1.0 / lambda_n))
        results.append(
            {
                "day": i + 1,
                "n": int(len(c) + len(t)),
                "lift": float(diff),
                "p_fixed": p_fixed,
                "p_sequential": p_seq,
            }
        )
    return results


# ════════════════════════════════════════════════════════════════════════
# PEEKING PROBLEM SIMULATION
# ════════════════════════════════════════════════════════════════════════


def simulate_peeking(
    n_sims: int = 1000,
    n_per_sim: int = 2000,
    n_checks: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Simulate A/A experiments (zero effect) with and without peeking.

    Returns dict with false-positive rates for: no-peek and fixed-p peeking.
    The looks are on ACCUMULATING data, so they are strongly correlated —
    the inflation must be measured by simulation; the independent-tests
    formula 1 - 0.95**k does not apply.
    """
    rng = np.random.default_rng(seed=seed)
    false_pos_fixed = 0
    false_pos_no_peek = 0

    for _ in range(n_sims):
        sim_ctrl = rng.normal(50, 10, size=n_per_sim)
        sim_treat = rng.normal(50, 10, size=n_per_sim)  # no real effect

        # no peeking
        z_end = (sim_treat.mean() - sim_ctrl.mean()) / np.sqrt(
            sim_ctrl.var(ddof=1) / n_per_sim + sim_treat.var(ddof=1) / n_per_sim
        )
        if 2 * (1 - stats.norm.cdf(abs(z_end))) < 0.05:
            false_pos_no_peek += 1

        # peeking at n_checks points, fixed p-values
        peeked_sig = False
        for check_n in np.linspace(100, n_per_sim, n_checks, dtype=int):
            sc = sim_ctrl[:check_n]
            st = sim_treat[:check_n]
            se_p = np.sqrt(sc.var(ddof=1) / check_n + st.var(ddof=1) / check_n)
            z_p = (st.mean() - sc.mean()) / se_p if se_p > 0 else 0
            if 2 * (1 - stats.norm.cdf(abs(z_p))) < 0.05:
                peeked_sig = True
                break
        if peeked_sig:
            false_pos_fixed += 1

    return {
        "n_sims": n_sims,
        "n_checks": n_checks,
        "rate_no_peek": false_pos_no_peek / n_sims,
        "rate_fixed_peek": false_pos_fixed / n_sims,
    }


# ════════════════════════════════════════════════════════════════════════
# DIFFERENCE-IN-DIFFERENCES — hypothetical HDB cooling measure (simulated)
# ════════════════════════════════════════════════════════════════════════


def simulate_hdb_cooling_panel(
    n_per_period: int = 200,
    n_pre: int = 6,
    n_post: int = 6,
    central_base: float = 550_000,
    noncentral_base: float = 450_000,
    growth_per_period: float = 2_000,
    central_extra_growth: float = 0.0,
    policy_effect: float = -20_000,
    seed: int = 99,
) -> pl.DataFrame:
    """Simulate quarterly HDB-style transactions around a HYPOTHETICAL measure.

    The scenario is illustrative: a cooling measure that applies only to
    Central-region flats (treated) and not to Non-Central flats (control).
    Both regions share a common price trend of ``growth_per_period``;
    ``central_extra_growth`` adds a Central-only pre-existing trend, which
    VIOLATES parallel trends (used to show the test can detect it).
    ``policy_effect`` is the true causal effect on Central prices after
    the measure (periods >= n_pre).

    Returns one row per transaction: period, central (0/1), post (0/1), price.
    """
    rng = np.random.default_rng(seed=seed)
    frames = []
    for t in range(n_pre + n_post):
        post = int(t >= n_pre)
        for central, base, sd in ((1, central_base, 80_000), (0, noncentral_base, 70_000)):
            mean = base + t * growth_per_period
            if central:
                mean += t * central_extra_growth + post * policy_effect
            prices = rng.normal(mean, sd, size=n_per_period)
            frames.append(
                pl.DataFrame(
                    {
                        "period": np.full(n_per_period, t, dtype=np.int64),
                        "central": np.full(n_per_period, central, dtype=np.int64),
                        "post": np.full(n_per_period, post, dtype=np.int64),
                        "price": prices,
                    }
                )
            )
    return pl.concat(frames)


def did_cells(panel: pl.DataFrame) -> dict[str, np.ndarray]:
    """Split the panel into the four DiD cells (group x pre/post) as arrays."""

    def cell(central: int, post: int) -> np.ndarray:
        return (
            panel.filter((pl.col("central") == central) & (pl.col("post") == post))[
                "price"
            ]
            .to_numpy()
            .astype(np.float64)
        )

    return {
        "pre_central": cell(1, 0),
        "post_central": cell(1, 1),
        "pre_noncentral": cell(0, 0),
        "post_noncentral": cell(0, 1),
    }


def diff_in_diff(cells: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute DiD estimate, SE, CI, z, p from four cell arrays."""
    y_tp = cells["pre_central"].mean()
    y_tq = cells["post_central"].mean()
    y_cp = cells["pre_noncentral"].mean()
    y_cq = cells["post_noncentral"].mean()

    did = (y_tq - y_tp) - (y_cq - y_cp)
    se = float(np.sqrt(sum(arr.var(ddof=1) / len(arr) for arr in cells.values())))
    z = did / se if se > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "y_treat_pre": float(y_tp),
        "y_treat_post": float(y_tq),
        "y_ctrl_pre": float(y_cp),
        "y_ctrl_post": float(y_cq),
        "did_estimate": float(did),
        "se": se,
        "ci_lo": float(did - 1.96 * se),
        "ci_hi": float(did + 1.96 * se),
        "z": float(z),
        "p_value": p,
    }


# ════════════════════════════════════════════════════════════════════════
# PARALLEL TRENDS TEST
# ════════════════════════════════════════════════════════════════════════


def parallel_trends_test(panel: pl.DataFrame, alpha: float = 0.05) -> dict[str, Any]:
    """Test parallel PRE-period trends with a group x time interaction.

    Fits OLS on the pre-period transactions of the SAME panel used for DiD:
        price = b0 + b1*period + b2*central + b3*(central*period) + e
    b3 is the difference in pre-period slopes (Central minus Non-Central).
    H0: b3 = 0 (parallel trends). The null distribution is centred at zero,
    so a real slope difference produces a small p-value.
    """
    pre = panel.filter(pl.col("post") == 0)
    t = pre["period"].to_numpy().astype(np.float64)
    g = pre["central"].to_numpy().astype(np.float64)
    y = pre["price"].to_numpy().astype(np.float64)
    X = np.column_stack([np.ones_like(t), t, g, g * t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se_b3 = float(np.sqrt(cov[3, 3]))
    t_stat = float(beta[3] / se_b3)
    p = float(2 * stats.t.sf(abs(t_stat), df=n - k))

    means = (
        pre.group_by(["period", "central"])
        .agg(pl.col("price").mean())
        .sort(["central", "period"])
    )
    pre_central = means.filter(pl.col("central") == 1)["price"].to_list()
    pre_noncentral = means.filter(pl.col("central") == 0)["price"].to_list()

    return {
        "pre_central": pre_central,
        "pre_noncentral": pre_noncentral,
        "time_points": np.arange(len(pre_central)),
        "slope_central": float(beta[1] + beta[3]),
        "slope_noncentral": float(beta[1]),
        "slope_diff": float(beta[3]),
        "slope_diff_se": se_b3,
        "t_stat": t_stat,
        "p_value": p,
        "passes": bool(p > alpha),
    }


# ════════════════════════════════════════════════════════════════════════
# VARIANCE REDUCTION REPORTING
# ════════════════════════════════════════════════════════════════════════


def variance_reduction(se_baseline: float, se_adjusted: float) -> dict[str, float]:
    """Report variance and CI-width reduction from baseline -> adjusted SE."""
    var_red = 1 - (se_adjusted**2) / (se_baseline**2)
    ci_red = 1 - (se_adjusted / se_baseline)
    return {
        "variance_reduction": float(var_red),
        "ci_width_reduction": float(ci_red),
        "effective_sample_multiplier": float(1 / max(1 - var_red, 1e-9)),
    }


def print_banner(title: str) -> None:
    """Consistent section header across technique files."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
