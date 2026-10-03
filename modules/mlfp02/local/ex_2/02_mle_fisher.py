# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 2.2: MLE and Fisher Information
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Write and optimise a log-likelihood function using scipy.optimize
#   - Reparameterise (log sigma) to enforce positivity without bounds
#   - Derive standard errors from the Fisher information matrix
#   - Construct Wald CIs and profile likelihood CIs
#   - Interpret profile likelihood — invariant to reparameterisation
#
# PREREQUISITES: 01_clt_sampling.py (data profiling, CLT intuition)
# ESTIMATED TIME: ~30 minutes
#
# TASKS (5-phase R10):
#   1. Theory — log-likelihood, Fisher information, reparameterisation
#   2. Build — neg-log-likelihood function with log(sigma) trick
#   3. Train — fit Normal MLE via L-BFGS-B, compare to analytical
#   4. Visualise — profile log-likelihood surface with CI bands
#   5. Apply — planning baseline for a bank's credit-risk scenarios
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_2 import (
    OUTPUT_DIR,
    extract_series,
    fit_normal_mle,
    load_singapore_econ,
    neg_log_likelihood_normal,
    normal_fisher_standard_errors,
    profile_lr_ci_normal_mu,
    save_figure,
    wald_ci,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Log-Likelihood and Fisher Information
# ════════════════════════════════════════════════════════════════════════
#
# For X ~ N(mu, sigma^2), the log-likelihood is:
#   l(mu, sigma | x) = -n/2 * log(2pi) - n*log(sigma) - Sum(xi-mu)^2 / (2 sigma^2)
#
# REPARAMETERISATION: optimise [mu, log(sigma)] so exp(log(sigma)) > 0.
#
# FISHER INFORMATION:
#   SE(mu)    = sigma / sqrt(n)
#   SE(sigma) = sigma / sqrt(2n)
#
# WALD CI:    mu_hat +/- z * SE     (assumes quadratic log-likelihood)
# PROFILE CI: { mu : 2*(l_max - l_p(mu)) < chi2(0.95, 1) }, where
#   l_p(mu) = max over sigma of l(mu, sigma)  (sigma re-fitted at each mu)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Log-Likelihood and MLE
# ════════════════════════════════════════════════════════════════════════

econ = load_singapore_econ()
gdp_growth = extract_series(econ, "gdp_growth_pct")
n_gdp = len(gdp_growth)

# TODO: Compute the analytical MLE for mu and sigma.
# Hint: mu = gdp_growth.mean(), sigma = gdp_growth.std(ddof=0)
mle_mu_analytic = ____
mle_sigma_analytic = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Numerical MLE via L-BFGS-B
# ════════════════════════════════════════════════════════════════════════

# TODO: Use fit_normal_mle() from the shared module to fit the Normal
# MLE numerically. It returns a dict with keys: mu, sigma, loglik, converged.
# Hint: fit_normal_mle(gdp_growth)
mle_result = ____
mle_mu = mle_result["mu"]
mle_sigma = mle_result["sigma"]
loglik_at_mle = mle_result["loglik"]

print(f"\n=== MLE Results: GDP Growth Rate ===")
print(
    f"Analytical MLE:  mu = {mle_mu_analytic:.4f}%, sigma = {mle_sigma_analytic:.4f}%"
)
print(f"Numerical MLE:   mu = {mle_mu:.4f}%, sigma = {mle_sigma:.4f}%")
print(f"Optimizer converged: {mle_result['converged']}")
print(f"Log-likelihood at MLE: {loglik_at_mle:.4f}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert mle_result["converged"], "MLE optimizer should converge"
assert abs(mle_mu - mle_mu_analytic) < 0.01, "Numerical and analytical MLE should agree"
print("\n--- Checkpoint 1 passed --- MLE computed analytically and numerically\n")


# ════════════════════════════════════════════════════════════════════════
# Standard Errors and Confidence Intervals
# ════════════════════════════════════════════════════════════════════════

# TODO: Compute Fisher standard errors using normal_fisher_standard_errors.
# Hint: normal_fisher_standard_errors(mle_sigma, n_gdp) returns (se_mu, se_sigma)
se_mu, se_sigma = ____

print(f"\n=== Standard Errors and Confidence Intervals ===")
print(f"SE(mu_hat) = {se_mu:.4f}%")
print(f"SE(sigma_hat) = {se_sigma:.4f}%")

# TODO: Compute the 95% Wald CI using wald_ci().
# Hint: wald_ci(estimate, se, alpha=0.05)
mu_wald_ci = ____
print(f"\n95% Wald CI for mu: [{mu_wald_ci[0]:.3f}%, {mu_wald_ci[1]:.3f}%]")

# TODO: Compute the profile likelihood CI using profile_lr_ci_normal_mu().
# Hint: profile_lr_ci_normal_mu(gdp_growth, mle_mu, mle_sigma, loglik_at_mle)
# Returns (ci, mu_grid, ll_values)
profile_ci, mu_grid, ll_values = ____
print(f"Profile LR 95% CI for mu: [{profile_ci[0]:.3f}%, {profile_ci[1]:.3f}%]")
print(f"Wald CI width:    {mu_wald_ci[1] - mu_wald_ci[0]:.4f}%")
print(f"Profile CI width: {profile_ci[1] - profile_ci[0]:.4f}%")

# Small-n check: profile out sigma on just the first 8 quarters
x_small = gdp_growth[:8]
small_fit = fit_normal_mle(x_small)
small_se, _ = normal_fisher_standard_errors(small_fit["sigma"], len(x_small))
small_wald = wald_ci(small_fit["mu"], small_se)
# TODO: Profile-likelihood CI for the 8-quarter sample (same helper as above)
# Hint: pass x_small and the matching entries of small_fit
small_profile, _, _ = ____
full_ratio = (profile_ci[1] - profile_ci[0]) / (mu_wald_ci[1] - mu_wald_ci[0])
small_ratio = (small_profile[1] - small_profile[0]) / (small_wald[1] - small_wald[0])
print(f"\nProfile / Wald width ratio, n={n_gdp}: {full_ratio:.3f}")
print(f"Profile / Wald width ratio, n={len(x_small)}:   {small_ratio:.3f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert mu_wald_ci[0] < mle_mu < mu_wald_ci[1], "MLE mean must be inside Wald CI"
assert se_mu > 0, "Standard error must be positive"
assert profile_ci[0] < mle_mu < profile_ci[1], "MLE mean must be inside profile CI"
assert small_ratio > 1.0, "With n=8 the profile CI should be wider than Wald"
print("\n--- Checkpoint 2 passed --- SEs and CIs computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Profile Log-Likelihood
# ════════════════════════════════════════════════════════════════════════

# Compare with the log-likelihood when sigma is (wrongly) held fixed at
# sigma_hat — this curve is exactly the Wald quadratic.
ll_fixed_sigma = np.array(
    [-neg_log_likelihood_normal([mu, np.log(mle_sigma)], gdp_growth) for mu in mu_grid]
)
fig = go.Figure()
fig.add_trace(go.Scatter(x=mu_grid, y=ll_values, name="Profile log-likelihood"))
fig.add_trace(
    go.Scatter(
        x=mu_grid, y=ll_fixed_sigma, name="sigma fixed at MLE", line={"dash": "dash"}
    )
)
fig.add_vline(x=mle_mu, line_dash="dash", annotation_text="MLE")
fig.add_vline(x=profile_ci[0], line_dash="dot", line_color="red")
fig.add_vline(x=profile_ci[1], line_dash="dot", line_color="red")
fig.update_layout(
    title="Profile Log-Likelihood: GDP Growth Rate",
    xaxis_title="mu (GDP growth %)",
    yaxis_title="Log-likelihood",
)
save_figure(fig, "ex2_02_profile_loglikelihood.html")
print("Saved: ex2_02_profile_loglikelihood.html")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 3 passed --- profile log-likelihood visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Planning Baseline for a Bank's Credit-Risk Scenarios
# ════════════════════════════════════════════════════════════════════════
# A Singapore bank's credit-risk team (illustrative) anchors its
# loan-loss planning on a "baseline" quarterly GDP growth rate. They use
# the long-run mean — but how precisely is that baseline known, and how
# does that differ from how much any ONE quarter can deviate from it?
#
#   - SE(mu_hat) answers: how well do we know the long-run MEAN?
#   - sigma_hat answers:  how far can a single quarter land from it?

print(f"\n=== APPLY: Planning Baseline for GDP Growth ===")

print(f"Baseline (mean quarterly GDP growth): {mle_mu:.3f}%")
print(f"95% CI for the baseline: [{mu_wald_ci[0]:.3f}%, {mu_wald_ci[1]:.3f}%]")
print(
    f"A conservative planner who uses the lower CI bound shifts the baseline"
    f"\ndown by {mle_mu - mu_wald_ci[0]:.3f} percentage points."
)
single_q_lo = mle_mu - 1.96 * mle_sigma
single_q_hi = mle_mu + 1.96 * mle_sigma
print(
    f"\nBut ONE quarter's growth varies far more: roughly "
    f"[{single_q_lo:.2f}%, {single_q_hi:.2f}%] under the Normal fit."
)
print(
    "Stress scenarios must be built from that spread (and its heavy tails —"
    "\nsee 04_mle_failures.py), not from the CI for the mean."
)
print(
    f"\nWith {n_gdp} quarters of data, SE = {se_mu:.3f}%. With twice the"
    f"\ndata, SE would shrink to {se_mu / np.sqrt(2):.3f}% — but the spread of a"
    f"\nsingle quarter (sigma ≈ {mle_sigma:.2f}%) would not shrink at all."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert single_q_hi - single_q_lo > mu_wald_ci[1] - mu_wald_ci[0], (
    "Single-quarter spread must exceed the CI for the mean"
)
print("\n--- Checkpoint 4 passed --- planning-baseline application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  - Log-likelihood optimisation with scipy.optimize.minimize
  - L-BFGS-B reparameterisation (log sigma enforces sigma > 0)
  - Wald SE from Fisher information: SE(mu_hat) = sigma / sqrt(n)
  - Profile likelihood CI — invariant to reparameterisation
  - Profiling out sigma: Wald and profile agree for large n, and the
    profile CI is wider (more honest) for small n
  - Precision of a mean (SE) vs spread of one outcome (sigma)

  NEXT: In 03_map_estimation.py, you'll add a prior to the MLE
  objective, creating MAP estimation, and observe how the prior
  "shrinks" the estimate — the Bayesian regularisation effect.
"""
)

print("--- Exercise 2.2 complete --- MLE and Fisher Information")
