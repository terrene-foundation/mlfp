# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 1: Probability and Bayesian Thinking
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Apply Bayes' theorem using the Normal-Normal conjugate prior
#   - Compute MLE for a Normal distribution and quantify estimation uncertainty
#   - Distinguish Bayesian credible intervals from frequentist confidence intervals
#   - Explain why data overwhelms the prior as sample size grows
#   - Visualise posterior distributions and compare interval methods
#
# PREREQUISITES: Complete M1 — you should be comfortable loading data,
#   computing summary statistics, and reading Polars DataFrames.
#
# ESTIMATED TIME: 60-75 minutes
#
# TASKS:
#   1. Compute MLE for HDB resale price parameters (Normal distribution)
#   2. Specify conjugate priors (Normal-Normal for mean)
#   3. Derive and compute posterior distributions analytically
#   4. Compare Bayesian credible intervals with bootstrap confidence intervals
#   5. Visualise posteriors and bootstrap distributions using ModelVisualizer
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records
#   Filtered to: 4-ROOM flats, 2020 onwards
#   Key column: resale_price (SGD)
#
# THEORY:
#   Normal-Normal conjugate: prior μ ~ N(μ₀, σ₀²), likelihood x ~ N(μ, σ²)
#   Posterior: μ|x ~ N(μₙ, σₙ²) where:
#     μₙ = (μ₀/σ₀² + n*x̄/σ²) / (1/σ₀² + n/σ²)
#     σₙ² = 1 / (1/σ₀² + n/σ²)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 60)
print("  MLFP02 Exercise 1: Probability and Bayesian Thinking")
print("=" * 60)

# Focus on a specific flat type and recent period for clearer analysis
hdb_recent = hdb.filter(
    (pl.col("flat_type") == "4 ROOM")
    & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
)

prices = hdb_recent["resale_price"].to_numpy().astype(np.float64)
print(f"\n  Data loaded: {len(prices):,} 4-room HDB transactions (2020+)")
print(f"  Price range: ${prices.min():,.0f} – ${prices.max():,.0f}")
print(f"  Sample mean: ${prices.mean():,.0f}")
print(f"  Sample std:  ${prices.std():,.0f}\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Maximum Likelihood Estimation (MLE)
# ══════════════════════════════════════════════════════════════════════
# For X ~ N(μ, σ²): MLE gives μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ - x̄)²
# MLE is asymptotically efficient (achieves Cramér-Rao lower bound)

n = len(prices)
mle_mean = prices.mean()
mle_var = prices.var(ddof=0)  # MLE uses ddof=0 (biased estimator)
mle_std = np.sqrt(mle_var)

# Fisher information for Normal: I(μ) = n/σ² → Var(μ̂) ≥ σ²/n
fisher_info = n / mle_var
cramer_rao_bound = 1 / fisher_info  # Minimum variance for any unbiased estimator
mle_se = np.sqrt(cramer_rao_bound)

print(f"\n=== MLE Estimates ===")
print(f"μ̂ = ${mle_mean:,.0f}")
print(f"σ̂ = ${mle_std:,.0f}")
print(f"Fisher information I(μ) = {fisher_info:.4f}")
print(f"Cramér-Rao lower bound: Var(μ̂) ≥ {cramer_rao_bound:.2f}")
print(f"MLE standard error: ${mle_se:,.2f}")
# INTERPRETATION: The standard error (${mle_se:,.0f}) is the precision of
# our mean estimate. With n={n:,} transactions the MLE converges well —
# the Cramér-Rao bound guarantees this is the most efficient estimator.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n > 0, "No data loaded — check the filter conditions"
assert mle_mean > 0, "MLE mean should be positive (price cannot be zero)"
assert mle_std > 0, "MLE std should be positive"
assert mle_se > 0, "Standard error should be positive"
assert mle_se < mle_std, "SE of mean should be much smaller than std of prices"
print("\n✓ Checkpoint 1 passed — MLE estimates computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Specify conjugate priors
# ══════════════════════════════════════════════════════════════════════
# Prior belief: Singapore 4-room HDB prices centre around $500K
# with moderate uncertainty (σ₀ = $100K)
#
# Normal-Normal conjugate: prior for μ
#   μ ~ N(μ₀, σ₀²)
#
# We treat σ² as known (plug in MLE estimate) for analytical tractability.
# A full Bayesian treatment would use Normal-Inverse-Gamma, but the
# Normal-Normal conjugate is the key pedagogical concept here.

# Prior hyperparameters
mu_0 = 500_000.0    # Prior mean: $500K
sigma_0 = 100_000.0  # Prior std: moderate uncertainty

# Known variance (plug-in from MLE)
sigma_known = mle_std

print(f"\n=== Prior Specification ===")
print(f"Prior: μ ~ N(μ₀={mu_0:,.0f}, σ₀={sigma_0:,.0f})")
print(f"Likelihood: X|μ ~ N(μ, σ²) with σ={sigma_known:,.0f} (plug-in)")
# INTERPRETATION: A $500K prior mean reflects market knowledge before
# seeing this dataset. The $100K prior std encodes our uncertainty —
# we'd be surprised if the true mean were below $300K or above $700K.


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compute posterior analytically
# ══════════════════════════════════════════════════════════════════════
# Posterior: μ|x ~ N(μₙ, σₙ²)
#   precision_0 = 1/σ₀²     (prior precision)
#   precision_data = n/σ²    (data precision)
#   σₙ² = 1 / (precision_0 + precision_data)
#   μₙ = σₙ² * (μ₀/σ₀² + n*x̄/σ²)

precision_prior = 1.0 / sigma_0**2
precision_data = n / sigma_known**2

# Posterior precision = prior precision + data precision
precision_posterior = precision_prior + precision_data
sigma_n_sq = 1.0 / precision_posterior
sigma_n = np.sqrt(sigma_n_sq)

# Posterior mean = precision-weighted combination
mu_n = sigma_n_sq * (mu_0 * precision_prior + n * mle_mean / sigma_known**2)

print(f"\n=== Posterior Distribution ===")
print(f"Posterior: μ|data ~ N(μₙ={mu_n:,.0f}, σₙ={sigma_n:,.2f})")
print(f"Prior precision:     {precision_prior:.2e}")
print(f"Data precision:      {precision_data:.2e}")
print(f"Posterior precision:  {precision_posterior:.2e}")
print(f"Data-to-prior precision ratio: {precision_data / precision_prior:.0f}x")
print(f"  → Posterior is dominated by {'data' if precision_data > precision_prior else 'prior'}")
# INTERPRETATION: When data precision >> prior precision, the posterior mean
# is almost identical to the MLE. The prior is "washed out" by the data.
# This is why with large n, Bayesian and frequentist results converge.

# 95% credible interval (highest posterior density for symmetric Normal = quantile-based)
ci_95_lower = mu_n - 1.96 * sigma_n
ci_95_upper = mu_n + 1.96 * sigma_n
print(f"\n95% Bayesian credible interval: [${ci_95_lower:,.2f}, ${ci_95_upper:,.2f}]")
# INTERPRETATION: A credible interval has a direct probability interpretation:
# "Given the data, there is a 95% probability that the true mean price lies in
# this range." This is stronger than the frequentist CI interpretation.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert precision_data > 0, "Data precision must be positive"
assert sigma_n < sigma_0, "Posterior std should be narrower than prior (data adds info)"
assert ci_95_lower < mu_n < ci_95_upper, "Posterior mean must be within its own CI"
assert abs(mu_n - mle_mean) < sigma_0, "Posterior should be close to MLE with large n"
print("\n✓ Checkpoint 2 passed — posterior distribution computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Bootstrap confidence intervals for comparison
# ══════════════════════════════════════════════════════════════════════
# Non-parametric bootstrap: resample with replacement, compute statistic
# BCa (bias-corrected accelerated) is the gold standard but we'll also
# compute percentile CIs for comparison.

rng = np.random.default_rng(seed=42)
n_bootstrap = 10_000

# Bootstrap distribution of the sample mean
bootstrap_means = np.array([
    rng.choice(prices, size=n, replace=True).mean()
    for _ in range(n_bootstrap)
])

# Percentile confidence interval
boot_ci_lower = np.percentile(bootstrap_means, 2.5)
boot_ci_upper = np.percentile(bootstrap_means, 97.5)

# BCa confidence interval using scipy
bca_result = stats.bootstrap(
    (prices,),
    statistic=np.mean,
    n_resamples=n_bootstrap,
    confidence_level=0.95,
    method="BCa",
    random_state=42,
)
bca_ci_lower = bca_result.confidence_interval.low
bca_ci_upper = bca_result.confidence_interval.high

# Normal theory CI for reference
normal_ci_lower = mle_mean - 1.96 * mle_se
normal_ci_upper = mle_mean + 1.96 * mle_se

print(f"\n=== Confidence / Credible Intervals ===")
print(f"Normal theory 95% CI:    [${normal_ci_lower:,.2f}, ${normal_ci_upper:,.2f}]")
print(f"Bootstrap percentile CI: [${boot_ci_lower:,.2f}, ${boot_ci_upper:,.2f}]")
print(f"Bootstrap BCa CI:        [${bca_ci_lower:,.2f}, ${bca_ci_upper:,.2f}]")
print(f"Bayesian 95% credible:   [${ci_95_lower:,.2f}, ${ci_95_upper:,.2f}]")

# Interpretation
print(f"\n--- Interpretation ---")
print(f"With n={n:,} observations, the data overwhelms the prior.")
print(f"All intervals are very tight (±${(normal_ci_upper - normal_ci_lower)/2:,.0f})")
print(f"because the standard error of the mean is ${mle_se:,.2f}.")
print(f"The Bayesian posterior is almost identical to the MLE — data dominates.")
# INTERPRETATION: All four methods agree because n is large. Bootstrap percentile
# and BCa will diverge for small n or skewed distributions. BCa is preferred
# as it corrects for both bias and skewness in the bootstrap distribution.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert boot_ci_lower < boot_ci_upper, "Bootstrap CI lower must be below upper"
assert bca_ci_lower < bca_ci_upper, "BCa CI lower must be below upper"
assert normal_ci_lower < mle_mean < normal_ci_upper, "MLE mean should be within normal CI"
assert len(bootstrap_means) == n_bootstrap, "Should have exactly n_bootstrap samples"
print("\n✓ Checkpoint 3 passed — bootstrap CIs computed correctly\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualise with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: Prior vs Posterior distributions --
# We'll use training_history as a line plot utility
x_range = np.linspace(mu_0 - 3 * sigma_0, mu_0 + 3 * sigma_0, 500)
prior_pdf = stats.norm.pdf(x_range, mu_0, sigma_0)

# Posterior has much smaller variance — zoom in
x_posterior = np.linspace(mu_n - 5 * sigma_n, mu_n + 5 * sigma_n, 500)
posterior_pdf = stats.norm.pdf(x_posterior, mu_n, sigma_n)

# Use metric_comparison to show interval widths
interval_results = {
    "Normal Theory": {
        "lower_bound": normal_ci_lower,
        "upper_bound": normal_ci_upper,
        "width": normal_ci_upper - normal_ci_lower,
    },
    "Bootstrap Percentile": {
        "lower_bound": boot_ci_lower,
        "upper_bound": boot_ci_upper,
        "width": boot_ci_upper - boot_ci_lower,
    },
    "Bootstrap BCa": {
        "lower_bound": bca_ci_lower,
        "upper_bound": bca_ci_upper,
        "width": bca_ci_upper - bca_ci_lower,
    },
    "Bayesian Credible": {
        "lower_bound": ci_95_lower,
        "upper_bound": ci_95_upper,
        "width": ci_95_upper - ci_95_lower,
    },
}

fig_intervals = viz.metric_comparison(interval_results)
fig_intervals.update_layout(title="95% Interval Comparison (4-Room HDB Prices)")
fig_intervals.write_html("ex1_interval_comparison.html")
print("\nSaved: ex1_interval_comparison.html")

# -- Plot 2: Bootstrap distribution as histogram --
# Use training_history to show convergence of bootstrap mean estimate
# Compute running mean of bootstrap means to show convergence
running_means = np.cumsum(bootstrap_means) / np.arange(1, n_bootstrap + 1)
convergence_metrics = {
    "Bootstrap Mean (running avg)": running_means[::100].tolist(),
}
fig_convergence = viz.training_history(convergence_metrics, x_label="Bootstrap Iteration (×100)")
fig_convergence.update_layout(title="Bootstrap Mean Convergence")
fig_convergence.write_html("ex1_bootstrap_convergence.html")
print("Saved: ex1_bootstrap_convergence.html")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert abs(running_means[-1] - mle_mean) < mle_se * 3, "Bootstrap mean should converge to MLE mean"
print("\n✓ Checkpoint 4 passed — bootstrap convergence verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5b: Bayesian estimation across flat types
# ══════════════════════════════════════════════════════════════════════
# Apply the same Normal-Normal conjugate to each flat type
# to see how prior vs data balance differs with sample size

flat_types = ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"]

results_by_type = {}
for ft in flat_types:
    subset = hdb.filter(
        (pl.col("flat_type") == ft)
        & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
    )
    if subset.height == 0:
        continue

    p = subset["resale_price"].to_numpy().astype(np.float64)
    n_ft = len(p)
    xbar = p.mean()
    s = p.std()

    # Same prior for all types
    prec_prior = 1.0 / sigma_0**2
    prec_data = n_ft / s**2
    prec_post = prec_prior + prec_data
    mu_post = (mu_0 * prec_prior + n_ft * xbar / s**2) / prec_post
    sigma_post = np.sqrt(1.0 / prec_post)

    results_by_type[ft] = {
        "n": n_ft,
        "mle_mean": xbar,
        "posterior_mean": mu_post,
        "posterior_std": sigma_post,
        "prior_weight": (prec_prior / prec_post) * 100,
    }

print(f"\n=== Bayesian Estimates by Flat Type ===")
print(f"{'Type':<12} {'n':>8} {'MLE Mean':>12} {'Post Mean':>12} {'Post σ':>10} {'Prior %':>8}")
print("─" * 70)
for ft, r in results_by_type.items():
    print(
        f"{ft:<12} {r['n']:>8,} ${r['mle_mean']:>10,.0f} "
        f"${r['posterior_mean']:>10,.0f} ${r['posterior_std']:>8,.2f} {r['prior_weight']:>7.3f}%"
    )
# INTERPRETATION: Flat types with fewer transactions (e.g. 2 ROOM, EXECUTIVE)
# show a larger prior weight — the posterior is pulled toward $500K more
# than for 4 ROOM or 5 ROOM which have abundant data. This is Bayesian
# regularisation in action: less data → prior matters more.

# Visualise comparison across flat types
flat_type_metrics = {
    ft: {
        "MLE_Mean": r["mle_mean"],
        "Posterior_Mean": r["posterior_mean"],
    }
    for ft, r in results_by_type.items()
}
fig_flat = viz.metric_comparison(flat_type_metrics)
fig_flat.update_layout(title="MLE vs Bayesian Posterior Mean by Flat Type")
fig_flat.write_html("ex1_flat_type_comparison.html")
print("\nSaved: ex1_flat_type_comparison.html")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
for ft, r in results_by_type.items():
    assert r["posterior_std"] < sigma_0, f"{ft}: posterior std should shrink from prior"
    assert 0 < r["prior_weight"] < 100, f"{ft}: prior weight must be between 0 and 100%"
print("\n✓ Checkpoint 5 passed — flat type posteriors all valid\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print("""
  ✓ MLE: μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ - x̄)² — most efficient estimator
  ✓ Cramér-Rao bound: MLE achieves minimum possible variance
  ✓ Normal-Normal conjugate: prior + likelihood → posterior analytically
  ✓ Posterior precision = prior precision + data precision (additive)
  ✓ Credible interval: direct P(μ ∈ CI | data) — Bayesian statement
  ✓ Bootstrap CI: non-parametric alternative, BCa for small/skewed samples
  ✓ With large n: posterior ≈ MLE, prior is overwhelmed by data

  NEXT: In Exercise 2, you'll implement MLE from scratch using
  scipy.optimize.minimize, add a prior to get MAP estimation,
  and diagnose three cases where MLE fails (small n, multimodal
  data, misspecified likelihood). Singapore GDP growth data.
""")

print("\n✓ Exercise 1 complete — Bayesian inference with conjugate priors")
