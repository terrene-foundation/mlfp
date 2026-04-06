# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT02 — Exercise 1: Probability and Bayesian Thinking
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Apply Bayesian inference with conjugate priors, compute
#   posteriors analytically, and compare credible vs confidence intervals.
#   Visualise posteriors using ModelVisualizer.
#
# TASKS:
#   1. Compute MLE for HDB resale price parameters (Normal distribution)
#   2. Specify conjugate priors (Normal-Normal for mean, Gamma for precision)
#   3. Derive and compute posterior distributions analytically
#   4. Compare Bayesian credible intervals with bootstrap confidence intervals
#   5. Visualise posteriors and bootstrap distributions using ModelVisualizer
#
# THEORY:
#   Normal-Normal conjugate: prior μ ~ N(μ₀, σ₀²), likelihood x ~ N(μ, σ²)
#   Posterior: μ|x ~ N(μₙ, σₙ²) where:
#     μₙ = (μ₀/σ₀² + n*x̄/σ²) / (1/σ₀² + n/σ²)
#     σₙ² = 1 / (1/σ₀² + n/σ²)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from scipy import stats

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
hdb = loader.load("ascent01", "hdb_resale.parquet")

# Focus on a specific flat type and recent period for clearer analysis
hdb_recent = hdb.filter(
    (pl.col("flat_type") == "4 ROOM")
    & (pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1))
)

prices = hdb_recent["resale_price"].to_numpy().astype(np.float64)
print(f"Sample: {len(prices)} 4-room HDB transactions (2020+)")
print(f"Price range: ${prices.min():,.0f} – ${prices.max():,.0f}")
print(f"Sample mean: ${prices.mean():,.0f}")
print(f"Sample std:  ${prices.std():,.0f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1: Maximum Likelihood Estimation (MLE)
# ════════════════════════════════════════════════════════════════════════
# For X ~ N(μ, σ²): MLE gives μ̂ = x̄, σ̂² = (1/n)Σ(xᵢ - x̄)²
# MLE is asymptotically efficient (achieves Cramér-Rao lower bound)

n = len(prices)
mle_mean = prices.mean()
# TODO: Compute MLE variance using ddof=0 (MLE uses the biased estimator)
mle_var = ____  # Hint: prices.var(ddof=?)
mle_std = np.sqrt(mle_var)

# Fisher information for Normal: I(μ) = n/σ² → Var(μ̂) ≥ σ²/n
# TODO: Compute Fisher information = n / mle_var
fisher_info = ____  # Hint: n / mle_var
cramer_rao_bound = 1 / fisher_info
mle_se = np.sqrt(cramer_rao_bound)

print(f"\n=== MLE Estimates ===")
print(f"μ̂ = ${mle_mean:,.0f}")
print(f"σ̂ = ${mle_std:,.0f}")
print(f"Fisher information I(μ) = {fisher_info:.4f}")
print(f"Cramér-Rao lower bound: Var(μ̂) ≥ {cramer_rao_bound:.2f}")
print(f"MLE standard error: ${mle_se:,.2f}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2: Specify conjugate priors
# ════════════════════════════════════════════════════════════════════════
# Prior belief: Singapore 4-room HDB prices centre around $500K
# with moderate uncertainty (σ₀ = $100K)
#
# Normal-Normal conjugate: prior for μ
#   μ ~ N(μ₀, σ₀²)
#
# We treat σ² as known (plug in MLE estimate) for analytical tractability.

# TODO: Set the prior mean and prior std hyperparameters
mu_0 = ____  # Hint: prior mean in dollars (e.g. 500_000.0)
sigma_0 = ____  # Hint: prior std in dollars (e.g. 100_000.0)

# Known variance (plug-in from MLE)
sigma_known = mle_std

print(f"\n=== Prior Specification ===")
print(f"Prior: μ ~ N(μ₀={mu_0:,.0f}, σ₀={sigma_0:,.0f})")
print(f"Likelihood: X|μ ~ N(μ, σ²) with σ={sigma_known:,.0f} (plug-in)")


# ════════════════════════════════════════════════════════════════════════
# TASK 3: Compute posterior analytically
# ════════════════════════════════════════════════════════════════════════
# Posterior: μ|x ~ N(μₙ, σₙ²)
#   precision_0 = 1/σ₀²     (prior precision)
#   precision_data = n/σ²    (data precision)
#   σₙ² = 1 / (precision_0 + precision_data)
#   μₙ = σₙ² * (μ₀/σ₀² + n*x̄/σ²)

# TODO: Compute prior precision = 1 / sigma_0^2
precision_prior = ____  # Hint: 1.0 / sigma_0**2
# TODO: Compute data precision = n / sigma_known^2
precision_data = ____  # Hint: n / sigma_known**2

# Posterior precision = prior precision + data precision
precision_posterior = precision_prior + precision_data
sigma_n_sq = 1.0 / precision_posterior
sigma_n = np.sqrt(sigma_n_sq)

# TODO: Compute the posterior mean (precision-weighted combination)
# μₙ = σₙ² * (μ₀ * precision_prior + n * x̄ / sigma_known²)
mu_n = (
    ____  # Hint: sigma_n_sq * (mu_0 * precision_prior + n * mle_mean / sigma_known**2)
)

print(f"\n=== Posterior Distribution ===")
print(f"Posterior: μ|data ~ N(μₙ={mu_n:,.0f}, σₙ={sigma_n:,.2f})")
print(f"Prior precision:     {precision_prior:.2e}")
print(f"Data precision:      {precision_data:.2e}")
print(f"Posterior precision:  {precision_posterior:.2e}")
print(f"Data-to-prior precision ratio: {precision_data / precision_prior:.0f}x")
print(
    f"  → Posterior is dominated by {'data' if precision_data > precision_prior else 'prior'}"
)

# 95% credible interval
# TODO: Compute 95% credible interval bounds using mu_n ± 1.96 * sigma_n
ci_95_lower = ____  # Hint: mu_n - 1.96 * sigma_n
ci_95_upper = ____  # Hint: mu_n + 1.96 * sigma_n
print(f"\n95% Bayesian credible interval: [${ci_95_lower:,.2f}, ${ci_95_upper:,.2f}]")


# ════════════════════════════════════════════════════════════════════════
# TASK 4: Bootstrap confidence intervals for comparison
# ════════════════════════════════════════════════════════════════════════
# Non-parametric bootstrap: resample with replacement, compute statistic
# BCa (bias-corrected accelerated) is the gold standard.

rng = np.random.default_rng(seed=42)
n_bootstrap = 10_000

# TODO: Compute the bootstrap distribution of the sample mean
# Each iteration: draw n samples with replacement, compute mean
bootstrap_means = np.array(
    [
        ____  # Hint: rng.choice(prices, size=n, replace=True).mean()
        for _ in range(n_bootstrap)
    ]
)

# Percentile confidence interval
# TODO: Compute 2.5th and 97.5th percentiles of bootstrap_means
boot_ci_lower = ____  # Hint: np.percentile(bootstrap_means, 2.5)
boot_ci_upper = ____  # Hint: np.percentile(bootstrap_means, 97.5)

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

print(f"\n--- Interpretation ---")
print(f"With n={n:,} observations, the data overwhelms the prior.")
print(f"All intervals are very tight (±${(normal_ci_upper - normal_ci_lower)/2:,.0f})")
print(f"because the standard error of the mean is ${mle_se:,.2f}.")
print(f"The Bayesian posterior is almost identical to the MLE — data dominates.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5: Visualise with ModelVisualizer
# ════════════════════════════════════════════════════════════════════════

# TODO: Instantiate the ModelVisualizer
viz = ____  # Hint: ModelVisualizer()

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

# TODO: Call viz.metric_comparison to compare interval widths
fig_intervals = ____  # Hint: viz.metric_comparison(interval_results)
fig_intervals.update_layout(title="95% Interval Comparison (4-Room HDB Prices)")
fig_intervals.write_html("ex1_interval_comparison.html")
print("\nSaved: ex1_interval_comparison.html")

# Bootstrap convergence plot
running_means = np.cumsum(bootstrap_means) / np.arange(1, n_bootstrap + 1)
convergence_metrics = {
    "Bootstrap Mean (running avg)": running_means[::100].tolist(),
}
# TODO: Call viz.training_history with the convergence_metrics dict
fig_convergence = ____  # Hint: viz.training_history(convergence_metrics, x_label="Bootstrap Iteration (×100)")
fig_convergence.update_layout(title="Bootstrap Mean Convergence")
fig_convergence.write_html("ex1_bootstrap_convergence.html")
print("Saved: ex1_bootstrap_convergence.html")


# ════════════════════════════════════════════════════════════════════════
# TASK 5b: Bayesian estimation across flat types
# ════════════════════════════════════════════════════════════════════════
# Apply the same Normal-Normal conjugate to each flat type

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
print(
    f"{'Type':<12} {'n':>8} {'MLE Mean':>12} {'Post Mean':>12} {'Post σ':>10} {'Prior %':>8}"
)
print("─" * 70)
for ft, r in results_by_type.items():
    print(
        f"{ft:<12} {r['n']:>8,} ${r['mle_mean']:>10,.0f} "
        f"${r['posterior_mean']:>10,.0f} ${r['posterior_std']:>8,.2f} {r['prior_weight']:>7.3f}%"
    )

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

print("\n✓ Exercise 1 complete — Bayesian inference with conjugate priors")
