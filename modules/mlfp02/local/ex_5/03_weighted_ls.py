# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.3: Weighted Least Squares (WLS)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement WLS when heteroscedasticity is present
#   - Estimate variance weights from fitted values
#   - Compare OLS and WLS coefficients and standard errors
#   - Understand when WLS improves inference vs point estimates
#
# PREREQUISITES: Exercise 5.1-5.2 (OLS, diagnostics, Breusch-Pagan)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load data and fit baseline OLS
#   2. Estimate variance function from residuals
#   3. Implement WLS: beta = (X'WX)^{-1}X'Wy
#   4. Compare OLS and WLS results
#   5. Visualise the effect of weighting
#
# THEORY:
#   When Var(e_i) = sigma_i^2 (not constant), OLS gives unbiased but
#   inefficient estimates. WLS weights each observation by 1/sigma_i^2,
#   giving less weight to high-variance observations.
#   beta_wls = (X'WX)^{-1}X'Wy where W = diag(1/sigma_i^2)
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    load_hdb_clean,
    build_design_matrix,
    fit_ols,
    print_coef_table,
    save_actual_vs_predicted,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Fit Baseline OLS
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.3: Weighted Least Squares")
print("=" * 70)

hdb_clean = load_hdb_clean()
X, y, feature_names = build_design_matrix(hdb_clean)
n_obs, k = X.shape

fit_baseline = fit_ols(X, y)
beta_ols = fit_baseline["beta"]
residuals = fit_baseline["residuals"]
y_hat = fit_baseline["y_hat"]
r_squared = fit_baseline["R2"]
SST = fit_baseline["SST"]
SSR = fit_baseline["SSR"]

print(
    f"\n  Baseline OLS: R-squared={r_squared:.6f}, RMSE=${fit_baseline['sigma_hat']:,.0f}"
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Weight Observations?
# ════════════════════════════════════════════════════════════════════════
# OLS treats every observation equally. But in housing data, a $1.2M
# executive flat has more price variation than a $300K 3-room flat —
# the "noise" is louder for expensive properties.
#
# Analogy: Imagine averaging exam scores from two classes. Class A has
# 30 students with tightly clustered scores (low variance). Class B
# has 30 students with wildly spread scores (high variance). A simple
# average weights them equally, but you would trust Class A's average
# more. WLS does exactly this — it trusts low-variance observations
# more than high-variance ones.
#
# WHY THIS MATTERS: A real estate analytics firm building an automated
# valuation model (AVM) for mortgage approvals needs honest uncertainty
# for EACH flat. OLS assumes one noise level for every flat, so its
# prediction interval is the same width for a cheap 3-room and an
# expensive executive flat — too narrow for the noisy, expensive end
# and too wide for the quiet, cheap end. WLS models the noise level, so
# the interval can widen where prices are genuinely more variable.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Estimate Variance Function
# ════════════════════════════════════════════════════════════════════════
# We model variance as a function of X: Var(e_i) ~ (X @ gamma)^2
# Use |residuals| as a proxy for standard deviation.

print(f"\n=== Estimating Variance Function ===")

abs_resid = np.abs(residuals)

# TODO: Fit |residuals| ~ X to get a variance model
# Hint: use np.linalg.lstsq(X, abs_resid, rcond=None)[0]
w_beta = ____

# TODO: Compute estimated variance per observation
# Hint: variance_hat = np.maximum((X @ w_beta) ** 2, 1e-6)
variance_hat = ____

# TODO: Compute weights = 1 / variance_hat
weights = ____

print(f"  Weight range: [{weights.min():.3e}, {weights.max():.3e}]")
print(f"  Weight ratio (max/min): {weights.max() / weights.min():.1f}x")
print(
    f"  Observations with above-median weight: {np.sum(weights > np.median(weights)):,}"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Implement WLS: beta = (X'WX)^{-1}X'Wy
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Weighted Least Squares ===")

# Never build W = diag(weights): with n = 24,904 that dense n x n matrix
# is ~5 GB. Scaling each row of X by its weight gives the same X'W.
# TODO: Row-scale X by the weights (X'W without the n x n matrix)
# Hint: broadcast weights as a column — weights[:, None]
Xw = ____

# TODO: Compute X'WX and X'Wy from the row-scaled matrix
# Hint: Xw.T @ X and Xw.T @ y
XtWX = ____
XtWy = ____

# TODO: Solve for WLS coefficients
# Hint: np.linalg.solve(XtWX, XtWy)
beta_wls = ____

y_hat_wls = X @ beta_wls
residuals_wls = y - y_hat_wls
ssr_wls = float(np.sum(residuals_wls**2))
r2_wls = 1 - ssr_wls / SST

# WLS standard errors
sigma_sq_wls = float(np.sum(weights * residuals_wls**2)) / (n_obs - k)
XtWX_inv = np.linalg.inv(XtWX)
se_wls = np.sqrt(sigma_sq_wls * np.diag(XtWX_inv))

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(beta_wls) == k, "WLS must have same number of coefficients"
assert ssr_wls > 0, "WLS SSR must be positive"
print("--- Checkpoint 3 passed --- WLS fitted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Compare OLS and WLS
# ════════════════════════════════════════════════════════════════════════

print(f"{'Feature':<25} {'OLS beta':>12} {'WLS beta':>12} {'Delta':>10}")
print("-" * 62)
for i, name in enumerate(feature_names):
    delta = beta_wls[i] - beta_ols[i]
    print(f"{name:<25} {beta_ols[i]:>12,.2f} {beta_wls[i]:>12,.2f} {delta:>+10,.2f}")

print(f"\nOLS R-squared  = {r_squared:.6f}")
print(f"WLS R-squared  = {r2_wls:.6f}")
print(f"OLS RMSE = ${np.sqrt(SSR / n_obs):,.0f}")
print(f"WLS RMSE = ${np.sqrt(ssr_wls / n_obs):,.0f}")

# Standard error comparison
print(f"\n{'Feature':<25} {'OLS SE':>12} {'WLS SE':>12}")
print("-" * 52)
for i, name in enumerate(feature_names):
    print(f"{name:<25} {fit_baseline['se_beta'][i]:>12,.2f} {se_wls[i]:>12,.2f}")

# INTERPRETATION: WLS coefficients may differ from OLS when
# heteroscedasticity is present. WLS gives more reliable standard
# errors and confidence intervals. If OLS and WLS coefficients are
# similar, the heteroscedasticity doesn't much affect the point
# estimates — but the SEs are still more trustworthy from WLS.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert all(se > 0 for se in se_wls), "All WLS standard errors must be positive"
print("\n--- Checkpoint 4 passed --- OLS vs WLS compared\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — WLS Actual vs Predicted
# ════════════════════════════════════════════════════════════════════════

path = save_actual_vs_predicted(
    y,
    y_hat_wls,
    title="WLS: Actual vs Predicted Price",
    filename="03_wls_actual_vs_predicted.html",
)
print(f"Saved: {path}")

# --- Residual variance: OLS vs WLS (before/after weighting) ---
sample = min(3000, n_obs)
fig_var = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        "OLS Residuals vs Fitted",
        "WLS Residuals vs Fitted",
    ],
)
fig_var.add_trace(
    go.Scatter(
        x=y_hat[:sample].tolist(),
        y=residuals[:sample].tolist(),
        mode="markers",
        marker={"size": 2, "opacity": 0.3, "color": "steelblue"},
        name="OLS",
    ),
    row=1,
    col=1,
)
fig_var.add_trace(
    go.Scatter(
        x=y_hat_wls[:sample].tolist(),
        y=residuals_wls[:sample].tolist(),
        mode="markers",
        marker={"size": 2, "opacity": 0.3, "color": "#D97706"},
        name="WLS",
    ),
    row=1,
    col=2,
)
# TODO: Draw a zero-residual reference line on BOTH panels.
# Hint: fig_var.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=...)
____
____
fig_var.update_layout(
    title="Residual Spread: OLS vs WLS — Does Weighting Reduce the Fan Shape?",
    height=400,
    width=900,
    showlegend=False,
)
fig_var.update_xaxes(title_text="Predicted ($)", row=1, col=1)
fig_var.update_xaxes(title_text="Predicted ($)", row=1, col=2)
fig_var.update_yaxes(title_text="Residual ($)", row=1, col=1)
fig_var.update_yaxes(title_text="Residual ($)", row=1, col=2)
path_var = OUTPUT_DIR / "03_wls_residual_comparison.html"
fig_var.write_html(str(path_var))
print(f"Saved: {path_var}")

# --- Predicted-vs-actual comparison: OLS vs WLS side by side ---
fig_cmp = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=[
        f"OLS (R-sq={r_squared:.4f})",
        f"WLS (R-sq={r2_wls:.4f})",
    ],
)
lo, hi = float(y.min()), float(y.max())
for col_idx, (pred, label) in enumerate([(y_hat, "OLS"), (y_hat_wls, "WLS")], start=1):
    fig_cmp.add_trace(
        go.Scatter(
            x=y[:sample].tolist(),
            y=pred[:sample].tolist(),
            mode="markers",
            marker={"size": 2, "opacity": 0.3},
            name=label,
        ),
        row=1,
        col=col_idx,
    )
    fig_cmp.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            showlegend=False,
        ),
        row=1,
        col=col_idx,
    )
fig_cmp.update_layout(
    title="Actual vs Predicted: OLS vs WLS — How Does Weighting Change Predictions?",
    height=400,
    width=900,
)
fig_cmp.update_xaxes(title_text="Actual ($)", row=1, col=1)
fig_cmp.update_xaxes(title_text="Actual ($)", row=1, col=2)
fig_cmp.update_yaxes(title_text="Predicted ($)", row=1, col=1)
fig_cmp.update_yaxes(title_text="Predicted ($)", row=1, col=2)
path_cmp = OUTPUT_DIR / "03_wls_ols_comparison.html"
fig_cmp.write_html(str(path_cmp))
print(f"Saved: {path_cmp}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Automated Valuation Model (AVM) for Mortgage Approval
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore proptech firm builds an automated valuation
# model (AVM) for mortgage lenders. The AVM must state not just a point
# estimate but how far THIS flat's price could plausibly fall from it.
#
# Two different intervals answer two different questions:
#   - CONFIDENCE interval for the MEAN price of flats like this one:
#       se_mean = sqrt(sigma^2 * x'(X'X)^{-1}x)   — shrinks with n
#   - PREDICTION interval for ONE flat's price:
#       se_pred = sqrt(sigma^2 * (1 + x'(X'X)^{-1}x)) — never shrinks
#     below the noise sigma, however much data you have.
# A valuation is about one flat, so it needs the PREDICTION interval.
# Under WLS the "1" becomes that flat's own estimated noise variance,
# so the interval is wider for noisier (larger, pricier) flats.
#
# BUSINESS IMPACT: With a 75% loan-to-value cap, the loan size moves
# 1:1 with the valuation. Quoting the narrow mean-CI as if it were
# the uncertainty of one flat understates the lender's valuation risk
# by more than an order of magnitude (compare the columns below).

print(f"\n--- Business Application: AVM Intervals for One Flat ---")
example_flats = {
    "~3-room (67 sqm, floor 8, 74y lease)": np.array([1.0, 67.0, 8.0, 74.0]),
    "~4-room (92 sqm, floor 8, 75y lease)": np.array([1.0, 92.0, 8.0, 75.0]),
    "~Executive (145 sqm, floor 8, 73y lease)": np.array([1.0, 145.0, 8.0, 73.0]),
}
sigma2_ols = fit_baseline["sigma_hat"] ** 2
print(
    f"  {'Flat':<42} {'Model':>5} {'Point':>12} "
    f"{'+/- mean CI':>12} {'+/- pred. PI':>13}"
)
pi_halfwidths = {}
for label, x_new in example_flats.items():
    leverage_ols = float(x_new @ fit_baseline["XtX_inv"] @ x_new)
    ci_ols = 1.96 * np.sqrt(sigma2_ols * leverage_ols)
    # TODO: OLS 95% PREDICTION half-width for one flat.
    # Hint: like ci_ols, but add the flat's own noise: sigma^2 * (1 + leverage)
    pi_ols = ____

    # WLS: this flat's own noise variance comes from the variance model
    var_new = max(float(x_new @ w_beta) ** 2, 1e-6)
    leverage_wls = float(x_new @ XtWX_inv @ x_new)
    ci_wls = 1.96 * np.sqrt(sigma_sq_wls * leverage_wls)
    # TODO: WLS 95% PREDICTION half-width — the flat's own variance replaces the "1".
    # Hint: same shape as ci_wls, with var_new added inside the square root
    pi_wls = ____
    pi_halfwidths[label] = (pi_ols, pi_wls)

    print(f"  {label:<42} {'OLS':>5} ${x_new @ beta_ols:>11,.0f} ${ci_ols:>11,.0f} ${pi_ols:>12,.0f}")
    print(f"  {'':<42} {'WLS':>5} ${x_new @ beta_wls:>11,.0f} ${ci_wls:>11,.0f} ${pi_wls:>12,.0f}")

ols_pis = [v[0] for v in pi_halfwidths.values()]
wls_pis = [v[1] for v in pi_halfwidths.values()]
print(
    f"\n  OLS prediction intervals span ${min(ols_pis):,.0f}-${max(ols_pis):,.0f}: "
    f"essentially one width for every flat."
)
print(
    f"  WLS prediction intervals span ${min(wls_pis):,.0f}-${max(wls_pis):,.0f}: "
    f"they track each flat's estimated noise level."
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
for pi_ols, pi_wls in pi_halfwidths.values():
    assert pi_ols > 0 and pi_wls > 0, "Prediction intervals must be positive"
assert all(
    pi > 10 * 1.96 * np.sqrt(sigma2_ols * float(x @ fit_baseline["XtX_inv"] @ x))
    for pi, x in zip(ols_pis, example_flats.values())
), "A prediction interval for one flat must be far wider than the CI of the mean"
print("\n--- Checkpoint 5 passed --- confidence vs prediction intervals\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (5.3)")
print("=" * 70)
print(
    """
  - Variance function estimation from OLS residuals
  - WLS implementation: beta = (X'WX)^{-1}X'Wy
  - Comparing OLS and WLS: coefficients, SEs, and R-squared
  - When WLS matters: heteroscedastic data with varying noise levels
  - Confidence interval (mean) vs prediction interval (one flat) for valuations

  NEXT: In 04_model_enrichment.py you'll extend the model with
  polynomial terms, interaction effects, dummy variables, and
  train/test evaluation.
"""
)

print("--- Exercise 5.3 complete --- Weighted Least Squares")
