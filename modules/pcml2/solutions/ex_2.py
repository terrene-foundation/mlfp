# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT2 — Exercise 2: Estimation and Inference
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Derive maximum likelihood estimates for economic parameters,
#   compare MLE vs MAP estimation, and diagnose when MLE fails.
#   Optimize log-likelihoods with scipy and visualize with ModelVisualizer.
#
# TASKS:
#   1. Load Singapore economic indicators and profile the data
#   2. Derive and optimize the log-likelihood for GDP growth rate
#   3. Add a prior (MAP estimation) and compare to MLE
#   4. Diagnose MLE failure cases: small n, multimodal, misspecification
#   5. Visualise likelihood surfaces and posterior contours
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from scipy import stats
from scipy.optimize import minimize

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
econ = loader.load("ascent02", "economic_indicators.csv")

print("=== Singapore Economic Indicators ===")
print(f"Shape: {econ.shape}")
print(f"Columns: {econ.columns}")
print(econ.head(8))

# Extract GDP growth as our primary variable of interest
gdp_growth = econ["gdp_growth_pct"].drop_nulls().to_numpy().astype(np.float64)
inflation = econ["inflation_cpi_pct"].drop_nulls().to_numpy().astype(np.float64)
unemployment = econ["unemployment_rate_pct"].drop_nulls().to_numpy().astype(np.float64)

n_gdp = len(gdp_growth)
print(f"\nGDP growth observations: {n_gdp}")
print(f"GDP growth range: {gdp_growth.min():.2f}% to {gdp_growth.max():.2f}%")
print(f"Sample mean: {gdp_growth.mean():.3f}%")
print(f"Sample std:  {gdp_growth.std():.3f}%")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Understanding the data — model selection
# ══════════════════════════════════════════════════════════════════════
# Before fitting, we must choose a likelihood model.
# Normal distribution is the natural choice for GDP growth (symmetric,
# unbounded), but we should verify this assumption.

# Normality test (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(gdp_growth)
print(f"\n=== Normality Check (Shapiro-Wilk) ===")
print(f"Test statistic: {shapiro_stat:.4f}")
print(f"p-value: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("Cannot reject normality — Normal likelihood is plausible")
else:
    print("Normality rejected — consider heavier-tailed distributions (t, skew-Normal)")

# Skewness and kurtosis
skew = stats.skew(gdp_growth)
kurt = stats.kurtosis(gdp_growth)
print(f"Skewness: {skew:.3f} (|skew|>1 suggests non-Normal)")
print(f"Excess kurtosis: {kurt:.3f} (>0 → heavier tails than Normal)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: MLE via log-likelihood optimization
# ══════════════════════════════════════════════════════════════════════
# For X ~ N(μ, σ²), the log-likelihood is:
#   ℓ(μ, σ | x) = -n/2 * log(2π) - n * log(σ) - Σ(xᵢ - μ)² / (2σ²)
#
# Closed-form MLE: μ̂ = x̄, σ̂ = √(1/n Σ(xᵢ - μ̂)²)
# We also optimize numerically to demonstrate the scipy.optimize workflow.


def neg_log_likelihood_normal(params: np.ndarray, x: np.ndarray) -> float:
    """Negative log-likelihood for N(mu, sigma²) — minimized by scipy."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)  # Reparameterize to enforce sigma > 0
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))


# Analytical MLE (reference)
mle_mu_analytic = gdp_growth.mean()
mle_sigma_analytic = gdp_growth.std(ddof=0)  # MLE uses biased estimator

# Numerical MLE via L-BFGS-B
x0 = np.array([gdp_growth.mean(), np.log(gdp_growth.std())])
result_mle = minimize(
    neg_log_likelihood_normal,
    x0,
    args=(gdp_growth,),
    method="L-BFGS-B",
    options={"maxiter": 1000, "ftol": 1e-12},
)

mle_mu_numeric = result_mle.x[0]
mle_sigma_numeric = np.exp(result_mle.x[1])

print(f"\n=== MLE Results: GDP Growth Rate ===")
print(f"Analytical MLE:  μ = {mle_mu_analytic:.4f}%, σ = {mle_sigma_analytic:.4f}%")
print(f"Numerical MLE:   μ = {mle_mu_numeric:.4f}%, σ = {mle_sigma_numeric:.4f}%")
print(f"Optimizer converged: {result_mle.success} (message: {result_mle.message})")
print(f"Log-likelihood at MLE: {-result_mle.fun:.4f}")

# Standard errors via observed Fisher information (Hessian)
# Var(θ̂) ≈ [I(θ̂)]⁻¹ where I(θ) = -E[∂²ℓ/∂θ²]
# scipy.optimize does not return Hessian by default — use the formula for Normal:
# Var(μ̂) = σ²/n, Var(σ̂²) = 2σ⁴/n
mle_mu_se = mle_sigma_numeric / np.sqrt(n_gdp)
mle_sigma_se = mle_sigma_numeric / np.sqrt(2 * n_gdp)

print(f"\nStandard errors (asymptotic):")
print(f"  SE(μ̂) = {mle_mu_se:.4f}%")
print(f"  SE(σ̂) = {mle_sigma_se:.4f}%")

# 95% Wald confidence intervals
mle_mu_ci = (mle_mu_numeric - 1.96 * mle_mu_se, mle_mu_numeric + 1.96 * mle_mu_se)
print(f"\n95% CI for μ: [{mle_mu_ci[0]:.3f}%, {mle_mu_ci[1]:.3f}%]")

# Likelihood ratio interval (profile likelihood — more accurate for small n)
# Find all μ where 2*(ℓ(μ̂) - ℓ(μ)) < χ²(0.95, df=1) = 3.841
lr_threshold = stats.chi2.ppf(0.95, df=1) / 2
profile_loglik = lambda mu: -neg_log_likelihood_normal(
    [mu, np.log(mle_sigma_numeric)], gdp_growth
)
loglik_at_mle = -result_mle.fun

mu_grid = np.linspace(
    mle_mu_numeric - 4 * mle_mu_se, mle_mu_numeric + 4 * mle_mu_se, 500
)
lr_values = np.array([loglik_at_mle - profile_loglik(mu) for mu in mu_grid])
lr_ci_mask = lr_values <= lr_threshold
lr_ci = (mu_grid[lr_ci_mask][0], mu_grid[lr_ci_mask][-1])
print(f"Profile LR 95% CI for μ: [{lr_ci[0]:.3f}%, {lr_ci[1]:.3f}%]")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: MAP estimation — MLE with a prior
# ══════════════════════════════════════════════════════════════════════
# MAP = argmax p(θ | x) = argmax [ℓ(θ | x) + log p(θ)]
#
# Prior belief: Singapore GDP growth is positive but moderate.
# From economic literature: typical growth 2-5%, informative Normal prior.
#
# Prior: μ ~ N(μ₀=3.5%, σ₀=1.5%)  — realistic growth for small open economy
# MAP objective: -ℓ(μ, σ | x) - log p(μ)

mu_prior_mean = 3.5  # % — prior belief on Singapore growth
mu_prior_std = 1.5  # % — prior uncertainty


def neg_map_objective(params: np.ndarray, x: np.ndarray) -> float:
    """Negative MAP objective = NLL + negative log-prior."""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return np.inf
    # Negative log-likelihood
    nll = -np.sum(stats.norm.logpdf(x, loc=mu, scale=sigma))
    # Negative log-prior for mu (Normal prior)
    neg_log_prior = -stats.norm.logpdf(mu, loc=mu_prior_mean, scale=mu_prior_std)
    return nll + neg_log_prior


result_map = minimize(
    neg_map_objective,
    x0,
    args=(gdp_growth,),
    method="L-BFGS-B",
    options={"maxiter": 1000, "ftol": 1e-12},
)

map_mu = result_map.x[0]
map_sigma = np.exp(result_map.x[1])

print(f"\n=== MAP vs MLE Comparison ===")
print(f"Prior: μ ~ N({mu_prior_mean}, {mu_prior_std}²)")
print(f"n = {n_gdp} observations")
print(f"")
print(f"MLE: μ̂ = {mle_mu_numeric:.4f}%, σ̂ = {mle_sigma_numeric:.4f}%")
print(f"MAP: μ̂ = {map_mu:.4f}%, σ̂ = {map_sigma:.4f}%")
print(f"")
print(f"MAP shrinkage toward prior: {map_mu - mle_mu_numeric:+.4f}%")
print(
    f"Prior-to-posterior pull: {abs(map_mu - mle_mu_numeric) / abs(mu_prior_mean - mle_mu_numeric):.2%}"
)
print(
    f"  → With n={n_gdp}, data {'dominates' if n_gdp > 20 else 'is balanced with'} the prior"
)

# Demonstrate stronger effect with small n
rng = np.random.default_rng(seed=42)
for n_small in [5, 10, 20]:
    small_sample = rng.choice(gdp_growth, size=n_small, replace=False)

    res_mle_s = minimize(
        neg_log_likelihood_normal,
        [small_sample.mean(), np.log(small_sample.std() + 1e-6)],
        args=(small_sample,),
        method="L-BFGS-B",
    )
    res_map_s = minimize(
        neg_map_objective,
        [small_sample.mean(), np.log(small_sample.std() + 1e-6)],
        args=(small_sample,),
        method="L-BFGS-B",
    )
    mle_s = res_mle_s.x[0]
    map_s = res_map_s.x[0]
    print(
        f"n={n_small:>3}: MLE={mle_s:>6.3f}%, MAP={map_s:>6.3f}%, prior={mu_prior_mean:.1f}% | shrinkage={map_s-mle_s:+.3f}%"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: MLE failure cases — diagnostics and remedies
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== MLE Failure Case 1: Small n ===")
# With n=3, MLE variance estimate is unreliable (high variance of the variance)
small_n = 3
samples_3 = gdp_growth[:small_n]
mle_mu_3 = samples_3.mean()
mle_sigma_3 = samples_3.std(ddof=0)
# Relative efficiency: MLE sigma has high MSE for small n
# The unbiased estimator uses ddof=1; both converge as n→∞
unbiased_sigma_3 = samples_3.std(ddof=1)
print(f"n=3 sample: {samples_3}")
print(f"MLE σ (ddof=0): {mle_sigma_3:.4f}% (biased)")
print(f"Unbiased σ (ddof=1): {unbiased_sigma_3:.4f}%")
print(f"Bias = {mle_sigma_3 - gdp_growth.std():.4f}%")
print(f"Remedy: Use MAP (Bayesian shrinkage) or unbiased estimator for small n")

print(f"\n=== MLE Failure Case 2: Multimodal Data ===")
# Create a synthetic bimodal dataset (pre/post-COVID economic regimes)
rng = np.random.default_rng(seed=99)
pre_covid = rng.normal(loc=4.0, scale=1.2, size=30)  # Normal growth period
covid_shock = rng.normal(loc=-5.0, scale=3.0, size=10)  # COVID recession
bimodal_data = np.concatenate([pre_covid, covid_shock])

# MLE of a single Normal to bimodal data
bimodal_mle_mu = bimodal_data.mean()
bimodal_mle_sigma = bimodal_data.std(ddof=0)

# Bimodality coefficient (>0.555 suggests bimodality)
bimodality_coeff = (stats.skew(bimodal_data) ** 2 + 1) / (
    stats.kurtosis(bimodal_data, fisher=True)
    + 3
    * (
        (len(bimodal_data) - 1) ** 2
        / ((len(bimodal_data) - 2) * (len(bimodal_data) - 3))
    )
)

print(f"Bimodal data: {len(bimodal_data)} observations (two economic regimes)")
print(f"MLE Normal fit: μ={bimodal_mle_mu:.2f}%, σ={bimodal_mle_sigma:.2f}%")
print(f"Bimodality coefficient: {bimodality_coeff:.3f} (>0.555 → bimodal)")
print(f"Problem: MLE estimate {bimodal_mle_mu:.2f}% lies BETWEEN the two modes!")
print(f"Remedy: Gaussian mixture model or regime-switching model")

print(f"\n=== MLE Failure Case 3: Misspecified likelihood ===")
# GDP growth data for an economy with occasional large shocks follows a
# Student-t distribution (heavier tails), not Normal.
# Fitting Normal to t-distributed data underestimates tail risk.
rng_t = np.random.default_rng(seed=77)
shock_data = rng_t.standard_t(df=3, size=100) * 2.0 + 2.5

# Fit Normal MLE
normal_mle_mu = shock_data.mean()
normal_mle_sigma = shock_data.std(ddof=0)


# Fit t-distribution MLE
def neg_ll_t(params: np.ndarray, x: np.ndarray) -> float:
    df, mu, scale = params
    if df <= 0 or scale <= 0:
        return np.inf
    return -np.sum(stats.t.logpdf(x, df=df, loc=mu, scale=scale))


result_t = minimize(
    neg_ll_t,
    [5.0, shock_data.mean(), shock_data.std()],
    args=(shock_data,),
    method="Nelder-Mead",
)
t_df, t_mu, t_scale = result_t.x

# Compare 99th percentile (tail risk)
normal_99 = stats.norm.ppf(0.99, loc=normal_mle_mu, scale=normal_mle_sigma)
t_99 = stats.t.ppf(0.99, df=t_df, loc=t_mu, scale=t_scale)
actual_99 = np.percentile(shock_data, 99)

print(
    f"Simulated shock data: mean={shock_data.mean():.2f}%, std={shock_data.std():.2f}%"
)
print(f"Normal MLE 99th pct: {normal_99:.2f}% (underestimates tail risk)")
print(f"t-dist MLE 99th pct: {t_99:.2f}% (better tail estimate)")
print(f"Empirical 99th pct:  {actual_99:.2f}%")
print(f"t-distribution df = {t_df:.1f} (lower → heavier tails)")
print(f"Remedy: Likelihood ratio test to select between Normal and t")

# AIC comparison: Normal vs t (t has one extra parameter)
ll_normal = np.sum(stats.norm.logpdf(shock_data, normal_mle_mu, normal_mle_sigma))
ll_t = -result_t.fun
aic_normal = 2 * 2 - 2 * ll_normal  # k=2 params (mu, sigma)
aic_t = 2 * 3 - 2 * ll_t  # k=3 params (df, mu, scale)
print(f"AIC (Normal): {aic_normal:.2f}")
print(f"AIC (t-dist): {aic_t:.2f}")
print(
    f"Better fit: {'t-distribution' if aic_t < aic_normal else 'Normal'} (lower AIC wins)"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualise with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: Profile log-likelihood for GDP growth mean --
# Shows the log-likelihood as a function of mu, with MLE and MAP marked
ll_values = np.array(
    [
        -neg_log_likelihood_normal([mu, np.log(mle_sigma_numeric)], gdp_growth)
        for mu in mu_grid
    ]
)
map_ll_values = np.array(
    [-neg_map_objective([mu, np.log(map_sigma)], gdp_growth) for mu in mu_grid]
)

# Normalize to show relative curvature
ll_profile = {"Profile log-likelihood": ll_values.tolist()}
fig_ll = viz.training_history(ll_profile, x_label="mu (GDP growth %)")
fig_ll.update_layout(title="Profile Log-Likelihood: GDP Growth Rate")
fig_ll.write_html("ex2_profile_loglikelihood.html")
print("\nSaved: ex2_profile_loglikelihood.html")

# -- Plot 2: MLE vs MAP — effect of sample size --
# Show how MAP estimate changes with n (converges to MLE as n → ∞)
sample_sizes = [3, 5, 10, 15, 20, 30, 50, n_gdp]
mle_estimates = []
map_estimates = []

for n_s in sample_sizes:
    sample = gdp_growth[:n_s]
    if len(sample) < 2:
        mle_estimates.append(sample.mean())
        map_estimates.append(
            (
                mu_prior_mean / mu_prior_std**2
                + n_s * sample.mean() / sample.std(ddof=0) ** 2
            )
            / (1 / mu_prior_std**2 + n_s / sample.std(ddof=0) ** 2)
        )
        continue

    r_mle = minimize(
        neg_log_likelihood_normal,
        [sample.mean(), np.log(sample.std() + 1e-6)],
        args=(sample,),
        method="L-BFGS-B",
    )
    r_map = minimize(
        neg_map_objective,
        [sample.mean(), np.log(sample.std() + 1e-6)],
        args=(sample,),
        method="L-BFGS-B",
    )
    mle_estimates.append(r_mle.x[0])
    map_estimates.append(r_map.x[0])

comparison_metrics = {
    "MLE": {"mu_estimate": mle_estimates[-1], "converges_to_truth": mle_mu_analytic},
    "MAP": {"mu_estimate": map_estimates[-1], "converges_to_truth": map_mu},
    "Prior": {"mu_estimate": mu_prior_mean, "converges_to_truth": mu_prior_mean},
}
fig_compare = viz.metric_comparison(comparison_metrics)
fig_compare.update_layout(title="MLE vs MAP Estimation: GDP Growth Rate")
fig_compare.write_html("ex2_mle_vs_map.html")
print("Saved: ex2_mle_vs_map.html")

# -- Plot 3: AIC comparison across distribution families --
distribution_fits = {}
distributions = {
    "Normal": (stats.norm, mle_mu_numeric, mle_sigma_numeric, 2),
}
# Student-t on full GDP data
r_t_full = minimize(
    neg_ll_t,
    [5.0, gdp_growth.mean(), gdp_growth.std()],
    args=(gdp_growth,),
    method="Nelder-Mead",
)
t_df_full, t_mu_full, t_scale_full = r_t_full.x
ll_t_full = -r_t_full.fun
ll_norm_full = -result_mle.fun

distribution_fits["Normal"] = {
    "AIC": 2 * 2 - 2 * ll_norm_full,
    "log_likelihood": ll_norm_full,
}
distribution_fits["Student-t"] = {
    "AIC": 2 * 3 - 2 * ll_t_full,
    "log_likelihood": ll_t_full,
}

fig_dist = viz.metric_comparison(distribution_fits)
fig_dist.update_layout(title="Distribution Fit Comparison (AIC and Log-Likelihood)")
fig_dist.write_html("ex2_distribution_comparison.html")
print("Saved: ex2_distribution_comparison.html")

print(f"\n✓ Exercise 2 complete — MLE and MAP estimation for economic parameters")
print(
    f"  Key concepts: log-likelihood, scipy.optimize, MAP as MLE-with-prior, failure cases"
)
