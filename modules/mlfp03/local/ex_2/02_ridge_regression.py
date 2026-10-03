# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.2: Ridge Regression (L2) and Its Bayesian Twin
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit Ridge regression at many α values and read the shrinkage effect
#   - Explain L2 geometry: sphere constraint, shrinkage that is strongest
#     along low-variance directions, coefficients that rarely hit zero
#   - Connect Ridge to a Gaussian prior on the coefficient vector (MAP)
#   - Pick α by cross-validation and touch the test set only once
#   - Measure how much Ridge stabilises a credit model's coefficients
#
# PREREQUISITES:
#   - 01_bias_variance.py (understand why we WANT to constrain complexity)
#   - MLFP02 Bayesian thinking (priors, likelihoods, posteriors)
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — why L2 shrinkage helps (geometry + Bayes)
#   2. Build — OLS baseline and Ridge models across the α sweep
#   3. Train — fit each α, choose α by 5-fold CV on the training set
#   4. Visualise — coefficient paths and error curves vs α
#   5. Apply — refit stability for a Singapore bank's credit model
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from shared.mlfp03.ex_2 import (
    ALPHAS,
    SEED,
    load_credit_data,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — L2 Regularisation
# ════════════════════════════════════════════════════════════════════════
# Ridge modifies the ordinary-least-squares objective by adding an L2
# penalty on the coefficient vector:
#
#     min_β  ||y - Xβ||²  +  α · ||β||²
#              ───────         ───────
#              fit the data    penalise large weights
#
# GEOMETRY: The α-term defines a BALL around the origin. As α grows the
# ball shrinks. Because the ball is smooth (no corners), coefficients get
# smaller but rarely land exactly at zero.
#
# Shrinkage is NOT uniform. In the eigenbasis of X'X, Ridge multiplies
# the OLS component along direction j by d_j / (d_j + α), where d_j is
# the variance of the data along that direction. Directions the data
# barely spans (small d_j) — e.g. the DIFFERENCE between two almost
# identical features — are shrunk hardest; well-supported directions are
# barely touched.
#
# CLOSED FORM: β = (X'X + αI)⁻¹X'y. The αI makes the matrix always
# invertible — Ridge is the standard fix for multicollinearity.
#
# BAYESIAN VIEW: If you assume Gaussian noise with variance σ² AND place a
# Gaussian prior N(0, τ²I) on β, the MAP (maximum a posteriori) estimate
# is EXACTLY Ridge regression with α = σ²/τ². So:
#   - Large α  ⇔  narrow prior  ⇔  strong belief "coefficients are small"
#   - Small α  ⇔  wide prior    ⇔  weak belief, OLS in the limit
#
# You're not "regularising" — you're encoding a belief about the world.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the OLS baseline and the Ridge sweep
# ════════════════════════════════════════════════════════════════════════
# We use the Singapore credit scoring dataset from MLFP02. The target is
# savings_balance (standardised, so MSE 1.0 = "no better than the mean").
# The training sample is deliberately SMALL (300 applicants) relative to
# the ~30 correlated features — the regime where OLS overfits.

print_header("Ridge Regression on Singapore Credit Data")

# TODO: Load the Singapore credit split. load_credit_data() returns
# X_train, y_train, X_test, y_test, feature_names.
X_train, y_train, X_test, y_test, feature_names = ____
print(
    f"Train: {X_train.shape[0]} rows  "
    f"Test: {X_test.shape[0]} rows  "
    f"Features: {len(feature_names)}"
)

# Baseline OLS for comparison (α → 0)
# TODO: Fit an unregularised OLS baseline (LinearRegression) on the training set.
ols = ____
ols_norm = float(np.linalg.norm(ols.coef_))
ols_train_mse = mean_squared_error(y_train, ols.predict(X_train))
ols_test_mse = mean_squared_error(y_test, ols.predict(X_test))
print(
    f"\nOLS baseline: ||β||₂ = {ols_norm:.4f}, "
    f"train MSE = {ols_train_mse:.4f}, test MSE = {ols_test_mse:.4f}"
)
print(
    f"  OLS fits the training rows {ols_test_mse - ols_train_mse:.4f} MSE better "
    "than unseen rows — that gap is what regularisation attacks."
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN Ridge across the α sweep; choose α by CV
# ════════════════════════════════════════════════════════════════════════
# For each α we record: train MSE, 5-fold CV MSE on the TRAINING set (used
# to choose α), test MSE (reported, never used for choosing), coefficient
# norm, and count of exact zeros (Ridge should have very few).

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
ridge_results: dict[float, dict[str, float]] = {}
coef_path: dict[float, np.ndarray] = {}
for alpha in ALPHAS:
    # TODO: Fit Ridge(alpha=alpha) on the training set, then get its 5-fold
    # CV MSE on the TRAINING set only.
    # Hint: cross_val_score(..., cv=cv, scoring="neg_mean_squared_error") returns
    # NEGATIVE MSEs — negate the mean.
    ridge = ____
    ridge.fit(____, ____)
    cv_mse = ____
    coef_path[alpha] = ridge.coef_.copy()
    ridge_results[alpha] = {
        "train_mse": float(mean_squared_error(y_train, ridge.predict(X_train))),
        "cv_mse": float(cv_mse),
        "test_mse": float(mean_squared_error(y_test, ridge.predict(X_test))),
        "coef_norm": float(np.linalg.norm(ridge.coef_)),
        "n_zero": int(np.sum(np.abs(ridge.coef_) < 1e-6)),
    }

print(
    f"\n{'alpha':>10} {'train MSE':>11} {'CV MSE':>9} {'test MSE':>10} "
    f"{'||β||₂':>9} {'zeros':>6}"
)
print("-" * 60)
for alpha, r in ridge_results.items():
    print(
        f"{alpha:>10.3f} {r['train_mse']:>11.4f} {r['cv_mse']:>9.4f} "
        f"{r['test_mse']:>10.4f} {r['coef_norm']:>9.4f} {r['n_zero']:>6}"
    )

# TODO: Choose α by the lowest CV MSE (never by test MSE).
best_alpha = ____
best_row = ridge_results[best_alpha]
print(
    f"\nCV-chosen Ridge α = {best_alpha}  "
    f"(CV MSE = {best_row['cv_mse']:.4f}, test MSE = {best_row['test_mse']:.4f}, "
    f"OLS test MSE = {ols_test_mse:.4f})"
)


# ── Checkpoint 1 ───────────────────────────────────────────────────────
assert (
    ridge_results[1000.0]["coef_norm"] < ridge_results[0.001]["coef_norm"]
), "Higher α must produce a smaller ||β||₂"
assert (
    ridge_results[1.0]["n_zero"] <= 2
), "Ridge should leave at most a handful of exact zeros (L2 ≠ Lasso)"
assert best_alpha in ALPHAS, "Best α must come from the sweep"
print("\n[ok] Checkpoint 1 passed — Ridge shrinkage behaviour confirmed")

# INTERPRETATION — computed, not assumed:
norm_ratio = ridge_results[1000.0]["coef_norm"] / ridge_results[0.001]["coef_norm"]
print(
    f"""
  - ||β||₂ falls from {ridge_results[0.001]['coef_norm']:.3f} (α=0.001, ≈ OLS) to
    {ridge_results[1000.0]['coef_norm']:.3f} (α=1000) — {norm_ratio:.0%} of its OLS size.
  - Train MSE rises monotonically with α (Ridge is pulled away from the
    training-set optimum) while CV MSE is lowest at α = {best_alpha}.
  - Test MSE at the CV-chosen α is {best_row['test_mse']:.4f} vs {ols_test_mse:.4f} for
    OLS: {'Ridge generalises better' if best_row['test_mse'] < ols_test_mse else 'no gain on this split'}.
  - Any 'zeros' come from a feature that is constant in this sample
    (its coefficient is 0 for every α) — not from the penalty.
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE coefficient paths + the Bayesian interpretation
# ════════════════════════════════════════════════════════════════════════

# Figure 1 — coefficient path: every feature's β against α (log x-axis)
fig_path = go.Figure()
coef_matrix = np.array([coef_path[a] for a in ALPHAS])  # (n_alphas, n_features)
for j, name in enumerate(feature_names):
    fig_path.add_trace(
        go.Scatter(x=ALPHAS, y=coef_matrix[:, j], mode="lines", name=name)
    )
fig_path.update_layout(
    title="Ridge coefficient paths — all shrink, none snap to zero",
    xaxis_title="Regularisation strength α (log scale)",
    yaxis_title="Coefficient (standardised units)",
    xaxis_type="log",
)
print(f"Saved: {save_html_plot(fig_path, 'ex2_02_ridge_coef_path.html')}")

# Figure 2 — train / CV / test MSE against α
fig_err = go.Figure()
for key, label in [("train_mse", "Train"), ("cv_mse", "5-fold CV"), ("test_mse", "Test")]:
    fig_err.add_trace(
        go.Scatter(
            x=ALPHAS,
            y=[ridge_results[a][key] for a in ALPHAS],
            mode="lines+markers",
            name=label,
        )
    )
fig_err.add_hline(y=ols_test_mse, line_dash="dot", annotation_text="OLS test MSE")
fig_err.update_layout(
    title="Ridge error vs α",
    xaxis_title="Regularisation strength α (log scale)",
    yaxis_title="MSE (standardised target)",
    xaxis_type="log",
)
print(f"Saved: {save_html_plot(fig_err, 'ex2_02_ridge_error_vs_alpha.html')}")

print_header("Bayesian Interpretation: Ridge as MAP with Gaussian Prior")

# σ² is estimated from the OLS residuals with the unbiased n − p − 1 divisor
n, p = X_train.shape
residuals = y_train - ols.predict(X_train)
# TODO: σ² = residual sum of squares / (n - p - 1); τ² = 1.0; α = σ² / τ².
sigma_sq = ____
tau_sq = 1.0  # Unit prior variance on standardised coefficients
alpha_bayes = ____

ridge_bayes = Ridge(alpha=alpha_bayes).fit(X_train, y_train)
bayes_mse = mean_squared_error(y_test, ridge_bayes.predict(X_test))

print(
    f"""
Prior beliefs:
  σ² (noise variance)     = {sigma_sq:.4f}   (estimated from OLS residuals)
  τ² (prior variance)     = {tau_sq:.4f}    (unit prior — "effects of order 1 SD")

Implied regularisation:
  α = σ² / τ²            = {alpha_bayes:.4f}

Ridge with Bayesian α:
  Test MSE              = {bayes_mse:.4f}

Compare to CV-chosen:
  α*                    = {best_alpha}
  Test MSE at α*        = {best_row["test_mse"]:.4f}
"""
)

# Show shrinkage on the two most collinear features
ridge_best = Ridge(alpha=best_alpha).fit(X_train, y_train)
ridge_over = Ridge(alpha=1000.0).fit(X_train, y_train)
corr = np.corrcoef(X_train, rowvar=False)
np.fill_diagonal(corr, 0.0)
corr = np.nan_to_num(corr)
i_max, j_max = np.unravel_index(np.argmax(np.abs(corr)), corr.shape)
print(
    f"Most correlated pair in the training sample: {feature_names[i_max]} / "
    f"{feature_names[j_max]} (r = {corr[i_max, j_max]:+.3f})"
)
print(f"\n{'Feature':<26} {'OLS':>10} {'Ridge(α*)':>12} {'Ridge(1000)':>13}")
print("-" * 64)
for i in (i_max, j_max):
    print(
        f"{feature_names[i]:<26} {ols.coef_[i]:>10.4f} {ridge_best.coef_[i]:>12.4f} "
        f"{ridge_over.coef_[i]:>13.4f}"
    )


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert alpha_bayes > 0, "Implied α should be positive"
ridge_best_norm = float(np.linalg.norm(ridge_best.coef_))
# Ridge shrinks the coefficient L2 norm vs OLS for any alpha > 0. When the
# CV-optimal alpha is tiny, Ridge ≈ OLS and the two norms match up to
# solver float-noise, so compare with a small relative tolerance.
assert ridge_best_norm <= ols_norm * (
    1 + 1e-3
), "OLS coefficients should have at least as large a norm as Ridge"
print("\n[ok] Checkpoint 2 passed — Bayesian Ridge interpretation verified")
# INTERPRETATION: The Bayesian view turns α from a "tuning knob" into a
# statement of belief. For a near-duplicate pair OLS can put a big
# positive weight on one and a big negative weight on the other — their
# DIFFERENCE is a direction the data barely spans, so Ridge shrinks it
# hardest and the two weights move towards a shared, smaller value.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: refit stability for a Singapore bank's credit model
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank's SME-lending team refits a
# linear risk model every quarter. Its 30+ bureau features are strongly
# correlated (several "debt service" variants, loan amount vs balance).
# Model validators reject a refit whose coefficients swing between
# quarters: unstable weights mean the model's explanation of a decision
# changes even when the customer did not.
#
# Instead of quoting numbers, MEASURE stability on our own data: refit on
# 50 bootstrap resamples of the training set and compare how much each
# coefficient moves for OLS vs Ridge at the CV-chosen α.

print_header("Refit Stability — OLS vs Ridge on Bootstrap Refits")

rng = np.random.default_rng(SEED)
boot_ols, boot_ridge = [], []
for _ in range(50):
    idx = rng.choice(n, n, replace=True)
    # TODO: Refit OLS and Ridge(alpha=best_alpha) on the resampled rows and
    # append each model's .coef_ to its list.
    boot_ols.append(____)
    boot_ridge.append(____)
ols_spread = float(np.mean(np.std(boot_ols, axis=0)))
ridge_spread = float(np.mean(np.std(boot_ridge, axis=0)))
print(
    f"""
Average coefficient standard deviation across 50 refits:
  OLS               : {ols_spread:.4f}
  Ridge (α = {best_alpha:<6}): {ridge_spread:.4f}
  → Ridge coefficients move {ols_spread / ridge_spread:.1f}× less between refits.
"""
)

# ── Checkpoint 3 ───────────────────────────────────────────────────────
assert ridge_spread <= ols_spread, "Ridge refits should be at least as stable as OLS"
print("[ok] Checkpoint 3 passed — stability measured, not assumed")
# BUSINESS READING: fewer coefficient swings means fewer refits bounced
# back by model validation and a model explanation that stays consistent
# for customers. When features are HIGHLY correlated, Ridge keeps ALL of
# them with moderate weights instead of letting one dominate — useful when
# every bureau feature must appear in the coefficient report.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED
======================================================================

  [x] Ridge objective: ||y-Xβ||² + α·||β||²
  [x] Closed-form stability: (X'X + αI)⁻¹ always invertible
  [x] Shrinkage strongest along low-variance directions; rarely exact zeros
  [x] MAP equivalence: Ridge ⇔ Gaussian prior with α = σ²/τ²
  [x] Choosing α by CV and measuring refit stability with the bootstrap

  KEY INSIGHT: Ridge is the default when you believe "many features
  contribute small amounts." If instead you believe "only a handful
  matter and the rest should be zero", Lasso is the right tool — and
  that's the next file.

  NEXT: 03_lasso_elasticnet.py — L1 sparsity, corner-of-the-diamond
  geometry, and ElasticNet as the pragmatic compromise.
"""
)
