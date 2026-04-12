# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6: Logistic Regression and Classification Foundations
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Implement the sigmoid function correctly (numerically stable)
#   - Build logistic regression from scratch using MLE (Bernoulli likelihood)
#   - Interpret coefficients as odds ratios: exp(β) = multiplicative change in odds
#   - Evaluate classification models with accuracy, confusion matrix, and AUC-ROC
#   - Perform one-way ANOVA and post-hoc Tukey HSD for multi-group comparison
#
# PREREQUISITES: Complete Exercise 5 — you should understand MLE, t-statistics,
#   the sigmoid function connection to linear models, and scipy.optimize.minimize.
#
# ESTIMATED TIME: 75 minutes
#
# TASKS:
#   1. Load HDB data and create a binary classification target
#   2. Implement the sigmoid function and verify its properties
#   3. Build logistic regression via MLE (scipy.optimize.minimize)
#   4. Compare with sklearn LogisticRegression
#   5. Interpret coefficients as odds ratios
#   6. Compute accuracy and confusion matrix
#   7. One-way ANOVA: compare resale prices across flat types
#   8. Post-hoc Tukey HSD test
#   9. Visualise: ROC curve, odds ratio forest plot
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records, 2020+
#   Binary target: high_price = 1 if resale_price > median, else 0
#
# THEORY:
#   Logistic regression models P(y=1|X) via the log-odds (logit) link:
#     log(P / (1-P)) = beta_0 + beta_1*x_1 + ... + beta_p*x_p
#   Inverting gives the sigmoid:
#     P(y=1|X) = 1 / (1 + exp(-(beta_0 + beta_1*x_1 + ...)))
#   MLE maximises the Bernoulli log-likelihood:
#     log L(beta) = sum[ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]
#   Odds ratio: exp(beta_j) = multiplicative change in odds per unit of x_j
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
hdb = loader.load("mlfp01", "hdb_resale.parquet")

print("=" * 60)
print("  MLFP02 Exercise 6: Logistic Regression and Classification")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet ({hdb.shape[0]:,} rows)")
print(f"  Columns: {hdb.columns}")
print(hdb.head(5))

# Focus on recent data for a cleaner signal
hdb_recent = hdb.filter(
    pl.col("month").str.to_date("%Y-%m") >= pl.date(2020, 1, 1)
)
print(f"\nFiltered to 2020+: {hdb_recent.height:,} transactions")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Create binary classification target
# ══════════════════════════════════════════════════════════════════════
# Target: high_price = 1 if resale_price > median, else 0
# This converts a continuous outcome into a classification problem.
# Why median? It gives a balanced 50/50 split, making evaluation
# interpretable (baseline accuracy = 50%).

median_price = hdb_recent["resale_price"].median()
print(f"\nMedian resale price: ${median_price:,.0f}")

# Parse storey range to a numeric midpoint
hdb_model = hdb_recent.with_columns(
    (pl.col("resale_price") > median_price).cast(pl.Int32).alias("high_price"),
    (
        (
            pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
            + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
        )
        / 2.0
    ).alias("storey_mid"),
).drop_nulls(subset=["floor_area_sqm", "storey_mid", "high_price"])

print(f"Dataset for modelling: {hdb_model.height:,} rows")
print(f"Target distribution:")
print(
    f"  high_price=1: {hdb_model.filter(pl.col('high_price') == 1).height:,} "
    f"({hdb_model.filter(pl.col('high_price') == 1).height / hdb_model.height:.1%})"
)
print(
    f"  high_price=0: {hdb_model.filter(pl.col('high_price') == 0).height:,} "
    f"({hdb_model.filter(pl.col('high_price') == 0).height / hdb_model.height:.1%})"
)
# INTERPRETATION: Using the median as threshold guarantees a 50/50 split.
# A model predicting "high" for everything would achieve 50% accuracy —
# this is our baseline. Any useful model must beat 50%.

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert hdb_model.height > 0, "Model dataset should not be empty"
assert "high_price" in hdb_model.columns, "high_price target column missing"
n_high = hdb_model.filter(pl.col("high_price") == 1).height
n_low = hdb_model.filter(pl.col("high_price") == 0).height
assert abs(n_high - n_low) / hdb_model.height < 0.01, \
    "high_price should be approximately balanced (within 1% of 50/50)"
print("\n✓ Checkpoint 1 passed — binary target created and balanced\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement the sigmoid function
# ══════════════════════════════════════════════════════════════════════
# THEORY: The sigmoid (logistic) function maps any real number to (0, 1):
#   sigma(z) = 1 / (1 + exp(-z))
#
# Properties:
#   - sigma(0) = 0.5 (decision boundary)
#   - sigma(z) -> 1 as z -> +inf
#   - sigma(z) -> 0 as z -> -inf
#   - sigma'(z) = sigma(z) * (1 - sigma(z))  (useful for gradient descent)
#   - sigma(-z) = 1 - sigma(z)  (symmetric about 0.5)
#
# Numerical stability: for very negative z, exp(-z) overflows.
# Use: sigma(z) = exp(z) / (1 + exp(z)) when z < 0.


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    For z >= 0:  1 / (1 + exp(-z))       -- standard form
    For z < 0:   exp(z) / (1 + exp(z))   -- avoids exp overflow
    """
    result = np.zeros_like(z, dtype=np.float64)
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))
    exp_z = np.exp(z[neg_mask])
    result[neg_mask] = exp_z / (1.0 + exp_z)
    return result


# Verify properties
z_test = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
sig_test = sigmoid(z_test)

print(f"\n=== Sigmoid Function Verification ===")
print(f"{'z':>8} {'sigma(z)':>12} {'1-sigma(-z)':>14}")
print("─" * 38)
for z_val, s_val in zip(z_test, sig_test):
    check = 1.0 - sigmoid(np.array([-z_val]))[0]
    print(f"{z_val:>8.1f} {s_val:>12.6f} {check:>14.6f}")

print(f"\nsigma(0) = {sigmoid(np.array([0.0]))[0]:.6f} (should be 0.5)")
print(f"Symmetry check: sigma(z) + sigma(-z) = {sig_test[0] + sig_test[-1]:.6f} (should be 1.0)")

# Derivative check: sigma'(z) = sigma(z) * (1 - sigma(z))
z_deriv = np.array([0.0])
sig_at_0 = sigmoid(z_deriv)[0]
deriv_at_0 = sig_at_0 * (1 - sig_at_0)
print(f"sigma'(0) = {deriv_at_0:.6f} (maximum derivative = 0.25, at z=0)")
# INTERPRETATION: The derivative of sigmoid is maximised at z=0 (the decision
# boundary). This means the model is most "sensitive" to inputs near the boundary.
# Far from the boundary (|z|>>0), the sigmoid saturates and gradients vanish —
# this is why deep networks with many sigmoid layers suffer from vanishing gradients.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert abs(sigmoid(np.array([0.0]))[0] - 0.5) < 1e-10, "sigmoid(0) must equal 0.5"
assert abs(sig_test[0] + sig_test[-1] - 1.0) < 1e-10, "sigmoid symmetry: sigma(z)+sigma(-z)=1"
assert sig_test[-1] > sig_test[0], "sigmoid must be increasing"
assert abs(deriv_at_0 - 0.25) < 1e-10, "sigmoid'(0) must equal 0.25"
print("\n✓ Checkpoint 2 passed — sigmoid properties verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Logistic regression via MLE (from scratch)
# ══════════════════════════════════════════════════════════════════════
# THEORY: The log-likelihood for logistic regression is:
#   log L(beta) = sum[ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]
# where p_i = sigma(X_i @ beta)
#
# We minimise the NEGATIVE log-likelihood (= binary cross-entropy loss).
# Unlike linear regression (closed-form OLS), logistic regression
# has no closed-form solution — we must use iterative optimisation.

# Prepare feature matrix — standardise for numerical stability
X_raw = hdb_model.select("floor_area_sqm", "storey_mid").to_numpy().astype(np.float64)
y = hdb_model["high_price"].to_numpy().astype(np.float64)

# Standardise features (zero mean, unit variance)
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X_scaled = (X_raw - X_mean) / X_std

# Add intercept column
n_samples = X_scaled.shape[0]
X = np.column_stack([np.ones(n_samples), X_scaled])  # [1, x1_scaled, x2_scaled]

feature_names = ["intercept", "floor_area_sqm", "storey_mid"]
print(f"\n=== Feature Matrix ===")
print(f"Shape: {X.shape} (n_samples, n_features including intercept)")
print(f"Feature means (raw): area={X_mean[0]:.1f} sqm, storey={X_mean[1]:.1f}")
print(f"Feature stds  (raw): area={X_std[0]:.1f} sqm, storey={X_std[1]:.1f}")


def neg_log_likelihood_logistic(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood for logistic regression.

    Args:
        beta: coefficient vector [intercept, beta_1, ..., beta_p]
        X: design matrix with intercept column (n x (p+1))
        y: binary target vector (n,)

    Returns:
        Negative log-likelihood (scalar to minimise).
    """
    z = X @ beta
    p = sigmoid(z)
    # Clip to avoid log(0)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    # Binary cross-entropy = negative log-likelihood
    nll = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return nll


def neg_log_likelihood_gradient(beta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of the negative log-likelihood.

    d(-logL)/d(beta) = -X^T (y - p) = X^T (p - y)
    This is a clean result because of the logistic-Bernoulli conjugacy.
    """
    z = X @ beta
    p = sigmoid(z)
    return X.T @ (p - y)


# Optimise with L-BFGS-B (quasi-Newton, uses gradient)
beta_init = np.zeros(X.shape[1])
result_mle = minimize(
    neg_log_likelihood_logistic,
    beta_init,
    args=(X, y),
    method="L-BFGS-B",
    jac=neg_log_likelihood_gradient,
    options={"maxiter": 5000, "ftol": 1e-12},
)

beta_mle = result_mle.x
print(f"\n=== MLE Logistic Regression (from scratch) ===")
print(f"Optimiser converged: {result_mle.success}")
print(f"Iterations: {result_mle.nit}")
print(f"Final NLL: {result_mle.fun:.4f}")
print(f"\nCoefficients (standardised scale):")
for name, coef in zip(feature_names, beta_mle):
    print(f"  {name:<20s}: {coef:>8.4f}")

# Standard errors via the observed Fisher information matrix
# The Hessian of NLL is: H = X^T diag(p*(1-p)) X
# SE(beta_j) = sqrt( [H^{-1}]_{jj} )
p_hat = sigmoid(X @ beta_mle)
W = p_hat * (1 - p_hat)  # Bernoulli variance at each observation
H = X.T @ (X * W[:, None])  # Hessian: X^T W X
cov_beta = np.linalg.inv(H)
se_beta = np.sqrt(np.diag(cov_beta))

print(f"\nStandard errors and Wald z-tests:")
print(f"{'Feature':<20s} {'beta':>8s} {'SE':>8s} {'z':>8s} {'p-value':>10s}")
print("─" * 60)
for name, coef, se in zip(feature_names, beta_mle, se_beta):
    z_stat = coef / se
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {name:<20s} {coef:>8.4f} {se:>8.4f} {z_stat:>8.2f} {p_val:>10.2e} {sig}")
# INTERPRETATION: The Wald z-test for logistic regression coefficients is
# mathematically identical to the t-test from Exercise 5 — coefficient divided
# by standard error. For logistic regression, we use the z-distribution (not t)
# because MLE is asymptotically Normal, not t-distributed.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert result_mle.success, "MLE logistic regression optimizer must converge"
assert len(beta_mle) == X.shape[1], "Should have one coefficient per feature"
assert all(se > 0 for se in se_beta), "All standard errors must be positive"
print("\n✓ Checkpoint 3 passed — logistic regression MLE converged\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare with sklearn LogisticRegression
# ══════════════════════════════════════════════════════════════════════
# sklearn uses the same MLE approach (LBFGS by default), but adds
# L2 regularisation by default (C=1.0). We set C very large to
# approximate unregularised MLE for a fair comparison.

lr_sklearn = LogisticRegression(
    solver="lbfgs",
    C=1e10,         # Very large C ~ no regularisation (C = 1/lambda)
    max_iter=5000,
    fit_intercept=True,
)
lr_sklearn.fit(X_scaled, y)

beta_sklearn = np.concatenate([[lr_sklearn.intercept_[0]], lr_sklearn.coef_[0]])

print(f"\n=== sklearn vs From-Scratch Comparison ===")
print(f"{'Feature':<20s} {'MLE (scratch)':>14s} {'sklearn':>14s} {'diff':>10s}")
print("─" * 64)
for name, b_mle, b_sk in zip(feature_names, beta_mle, beta_sklearn):
    print(f"  {name:<20s} {b_mle:>14.6f} {b_sk:>14.6f} {abs(b_mle - b_sk):>10.2e}")

print(f"\nCoefficients match within numerical tolerance: "
      f"{np.allclose(beta_mle, beta_sklearn, atol=1e-3)}")
# INTERPRETATION: Our from-scratch implementation should match sklearn's result
# within numerical precision. This validation is important — it confirms our
# implementation of the Bernoulli log-likelihood and L-BFGS-B optimizer is correct.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert np.allclose(beta_mle, beta_sklearn, atol=1e-2), \
    "From-scratch and sklearn coefficients should agree within 0.01"
print("\n✓ Checkpoint 4 passed — from-scratch matches sklearn\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Interpret coefficients as odds ratios
# ══════════════════════════════════════════════════════════════════════
# THEORY: The odds of the event y=1 are:
#   odds = P(y=1) / P(y=0) = P(y=1) / (1 - P(y=1))
#
# From the logistic model:
#   log(odds) = beta_0 + beta_1*x_1 + beta_2*x_2
#
# Therefore:
#   odds = exp(beta_0) * exp(beta_1*x_1) * exp(beta_2*x_2)
#
# The odds ratio for feature j is exp(beta_j):
#   - When x_j increases by 1 unit, odds multiply by exp(beta_j)
#   - exp(beta_j) > 1 => odds increase (feature pushes toward y=1)
#   - exp(beta_j) < 1 => odds decrease (feature pushes toward y=0)
#
# IMPORTANT: Our features are standardised, so "1 unit" = 1 standard
# deviation. To get odds ratios in original units, we need to rescale.

print(f"\n=== Odds Ratios ===")
print(f"{'Feature':<20s} {'OR (per 1 SD)':>14s} {'OR (per unit)':>14s} {'95% CI (per unit)':>22s}")
print("─" * 76)

odds_ratios_per_sd = []
odds_ratios_per_unit = []
or_ci_lower_list = []
or_ci_upper_list = []

for i, name in enumerate(feature_names):
    if name == "intercept":
        continue

    or_std = np.exp(beta_mle[i])
    odds_ratios_per_sd.append(or_std)

    # Original-scale odds ratio
    beta_orig = beta_mle[i] / X_std[i - 1]
    se_orig = se_beta[i] / X_std[i - 1]
    or_unit = np.exp(beta_orig)
    odds_ratios_per_unit.append(or_unit)

    # 95% CI for odds ratio (Wald): exp(beta +/- 1.96*SE)
    or_ci_lower = np.exp(beta_orig - 1.96 * se_orig)
    or_ci_upper = np.exp(beta_orig + 1.96 * se_orig)
    or_ci_lower_list.append(or_ci_lower)
    or_ci_upper_list.append(or_ci_upper)

    print(f"  {name:<20s} {or_std:>14.4f} {or_unit:>14.4f} [{or_ci_lower:.4f}, {or_ci_upper:.4f}]")

# Interpretation
print(f"\n--- Interpretation ---")
area_or = odds_ratios_per_unit[0]
storey_or = odds_ratios_per_unit[1]
print(f"floor_area_sqm: OR = {area_or:.4f}")
if area_or > 1:
    print(f"  Each additional sqm of floor area multiplies the odds of")
    print(f"  high price by {area_or:.4f}x. For a 10 sqm increase: {area_or**10:.2f}x odds.")
else:
    print(f"  Each additional sqm of floor area multiplies the odds of")
    print(f"  high price by {area_or:.4f}x (reduces odds).")

print(f"\nstorey_mid: OR = {storey_or:.4f}")
if storey_or > 1:
    print(f"  Each additional storey multiplies the odds of high price by {storey_or:.4f}x.")
    print(f"  Moving from floor 5 to floor 15 (10 storeys): {storey_or**10:.2f}x odds.")
else:
    print(f"  Each additional storey multiplies the odds of high price by {storey_or:.4f}x.")
# INTERPRETATION: Odds ratios > 1 mean the feature increases the probability of
# being in the "high price" category. An OR of 1.08 per sqm means each square
# metre multiplies the odds of high price by 1.08 — equivalent to an 8% increase
# in odds. This is the business language for logistic regression.

# ── Checkpoint 5 ─────────────────────────────────────────────────────
for or_val in odds_ratios_per_unit:
    assert or_val > 0, "Odds ratios must be positive"
for lower, upper in zip(or_ci_lower_list, or_ci_upper_list):
    assert lower < upper, "CI lower must be below upper"
print("\n✓ Checkpoint 5 passed — odds ratios computed and interpreted\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Accuracy and confusion matrix
# ══════════════════════════════════════════════════════════════════════
# Predictions: classify as 1 if P(y=1|X) > 0.5 (the sigmoid crosses 0.5
# at z=0, which is the natural decision boundary).

y_prob = sigmoid(X @ beta_mle)
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print(f"\n=== Classification Performance ===")
print(f"Accuracy: {acc:.4f} ({acc:.1%})")
print(f"Baseline (majority class): {max(y.mean(), 1 - y.mean()):.1%}")
print(f"Lift over baseline: {acc - max(y.mean(), 1 - y.mean()):+.1%}")

# Confusion matrix with labels
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix:")
print(f"{'':>20} {'Predicted 0':>14} {'Predicted 1':>14}")
print(f"{'Actual 0':<20} {tn:>14,} {fp:>14,}")
print(f"{'Actual 1':<20} {fn:>14,} {tp:>14,}")

# Derived metrics (preview — detailed in M3.5)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
print(f"\nPrecision (of predicted highs, how many are truly high): {precision:.4f}")
print(f"Recall (of actual highs, how many did we catch):         {recall:.4f}")
print(f"F1 score:                                                  {f1:.4f}")

# ROC curve data (for plotting in Task 9)
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
print(f"AUC-ROC: {roc_auc:.4f}")
print(f"  (1.0 = perfect, 0.5 = random, <0.5 = worse than random)")
# INTERPRETATION: AUC-ROC measures the model's ability to rank positives above
# negatives across all possible thresholds. Unlike accuracy (threshold-dependent),
# AUC is threshold-independent. An AUC of 0.85 means the model ranks a random
# high-price flat above a random low-price flat 85% of the time.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert 0.5 < acc <= 1.0, "Accuracy should exceed 50% baseline for a useful model"
assert 0.5 < roc_auc <= 1.0, "AUC-ROC should exceed 0.5 (random)"
assert cm.sum() == n_samples, "Confusion matrix entries should sum to total samples"
print("\n✓ Checkpoint 6 passed — classification metrics computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: One-way ANOVA — prices across flat types
# ══════════════════════════════════════════════════════════════════════
# THEORY: ANOVA generalises the two-sample t-test to 3+ groups.
#   H0: mu_1 = mu_2 = ... = mu_k (all group means are equal)
#   H1: at least one group mean differs
#
#   F = MS_between / MS_within
#   When to use: 2 groups → t-test, 3+ groups → ANOVA,
#   continuous predictor → regression

flat_types_of_interest = ["3 ROOM", "4 ROOM", "5 ROOM"]
groups = {}
for ft in flat_types_of_interest:
    prices = (
        hdb_recent.filter(pl.col("flat_type") == ft)["resale_price"]
        .to_numpy()
        .astype(np.float64)
    )
    groups[ft] = prices
    print(f"\n{ft}: n={len(prices):,}, mean=${prices.mean():,.0f}, std=${prices.std():,.0f}")

# One-way ANOVA using scipy
f_stat, anova_p = stats.f_oneway(*groups.values())

# Compute components manually for pedagogical clarity
group_values = list(groups.values())
group_names = list(groups.keys())
k = len(group_values)
N_total = sum(len(g) for g in group_values)
grand_mean = np.concatenate(group_values).mean()

# SS_between: sum of n_j * (mean_j - grand_mean)^2
ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_values)
# SS_within: sum of (x_ij - mean_j)^2 within each group
ss_within = sum(np.sum((g - g.mean()) ** 2) for g in group_values)
# Mean squares
ms_between = ss_between / (k - 1)
ms_within = ss_within / (N_total - k)
f_manual = ms_between / ms_within

# Effect size: eta-squared = SS_between / SS_total
ss_total = ss_between + ss_within
eta_squared = ss_between / ss_total

print(f"\n=== One-Way ANOVA: Resale Price by Flat Type ===")
print(f"H0: mean price is the same across {', '.join(flat_types_of_interest)}")
print(f"H1: at least one group mean differs")
print(f"\nGrand mean: ${grand_mean:,.0f}")
print(f"SS_between: {ss_between:,.0f}   (df = {k-1})")
print(f"SS_within:  {ss_within:,.0f}   (df = {N_total-k})")
print(f"MS_between: {ms_between:,.0f}")
print(f"MS_within:  {ms_within:,.0f}")
print(f"\nF-statistic (manual):  {f_manual:.2f}")
print(f"F-statistic (scipy):   {f_stat:.2f}")
print(f"p-value: {anova_p:.2e}")
print(f"eta-squared (effect size): {eta_squared:.4f} ({eta_squared:.1%} of variance explained)")
print(f"\nConclusion: {'Reject H0' if anova_p < 0.05 else 'Fail to reject H0'} at alpha=0.05")
if anova_p < 0.05:
    print("At least one flat type has a significantly different mean price.")
    print("But ANOVA does not tell us WHICH pairs differ — we need post-hoc tests.")
# INTERPRETATION: ANOVA's F-statistic compares between-group variance to
# within-group variance. A large F means group means are far apart relative
# to the noise within each group. eta-squared = 0.20 means flat type explains
# 20% of all price variation — that's a large effect size.

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert abs(f_manual - f_stat) < 0.1, "Manual and scipy F-statistics should agree"
assert 0 <= anova_p <= 1, "ANOVA p-value must be a valid probability"
assert 0 < eta_squared < 1, "eta-squared must be between 0 and 1"
assert anova_p < 0.001, "With large n, ANOVA should be highly significant across flat types"
print("\n✓ Checkpoint 7 passed — ANOVA completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Post-hoc Tukey HSD test
# ══════════════════════════════════════════════════════════════════════
# THEORY: After a significant ANOVA, Tukey's Honestly Significant
# Difference (HSD) tests all pairwise group differences while
# controlling the family-wise error rate.
#
# If |mean_i - mean_j| > HSD, the pair is significantly different.
# We implement using Bonferroni-corrected t-tests as a practical alternative.

# Harmonic mean of group sizes (handles unequal n)
group_sizes = np.array([len(g) for g in group_values])
n_harmonic = k / np.sum(1.0 / group_sizes)

# Standard error for Tukey comparison
se_tukey = np.sqrt(ms_within / n_harmonic)

n_comparisons = k * (k - 1) // 2
alpha_bonf = 0.05 / n_comparisons
t_crit = stats.t.ppf(1 - alpha_bonf / 2, df=N_total - k)

print(f"\n=== Post-Hoc Pairwise Comparisons (Tukey HSD) ===")
print(f"MS_within: {ms_within:,.0f}")
print(f"Harmonic mean n: {n_harmonic:,.0f}")
print(f"SE for comparisons: ${se_tukey:,.0f}")
print(f"Number of comparisons: {n_comparisons}")
print(f"Bonferroni-adjusted alpha: {alpha_bonf:.4f}")
print(f"Critical t-value: {t_crit:.4f}")
print(f"\n{'Pair':<20s} {'Mean Diff':>12s} {'t-stat':>10s} {'p-value':>12s} {'Significant':>12s}")
print("─" * 72)

tukey_results = []
for i in range(k):
    for j in range(i + 1, k):
        mean_diff = group_values[i].mean() - group_values[j].mean()
        n_i, n_j = len(group_values[i]), len(group_values[j])
        se_pair = np.sqrt(ms_within * (1.0 / n_i + 1.0 / n_j))
        t_stat = mean_diff / se_pair
        # Two-tailed p-value with Bonferroni correction
        p_raw = 2 * (1 - stats.t.cdf(abs(t_stat), df=N_total - k))
        p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni adjustment
        sig = "Yes" if p_adj < 0.05 else "No"

        pair_name = f"{group_names[i]} vs {group_names[j]}"
        print(f"  {pair_name:<20s} ${mean_diff:>10,.0f} {t_stat:>10.2f} {p_adj:>12.2e} {sig:>12s}")

        tukey_results.append({
            "pair": pair_name,
            "group_a": group_names[i],
            "group_b": group_names[j],
            "mean_diff": mean_diff,
            "t_stat": t_stat,
            "p_adj": p_adj,
            "significant": p_adj < 0.05,
        })

print(f"\n--- Interpretation ---")
for r in tukey_results:
    if r["significant"]:
        higher = r["group_a"] if r["mean_diff"] > 0 else r["group_b"]
        lower = r["group_b"] if r["mean_diff"] > 0 else r["group_a"]
        print(f"  {higher} prices are significantly higher than {lower} "
              f"(diff = ${abs(r['mean_diff']):,.0f}, p = {r['p_adj']:.2e})")
    else:
        print(f"  {r['group_a']} and {r['group_b']} prices do not significantly differ "
              f"(p = {r['p_adj']:.2e})")
# INTERPRETATION: Tukey HSD (here implemented via Bonferroni) answers "which
# specific pairs drive the significant ANOVA?" For HDB prices, we expect each
# flat type (3R, 4R, 5R) to differ significantly because each adds more space.
# The mean differences are practically meaningful ($50-100K+) as well as significant.

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(tukey_results) == n_comparisons, f"Should have {n_comparisons} pairwise comparisons"
for r in tukey_results:
    assert 0 <= r["p_adj"] <= 1, f"Adjusted p-value must be valid probability for {r['pair']}"
print("\n✓ Checkpoint 8 passed — Tukey HSD completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Visualisation — ROC curve and odds ratio forest plot
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: ROC Curve --
roc_metrics = {
    f"ROC Curve (AUC={roc_auc:.3f})": tpr.tolist(),
    "Random (AUC=0.5)": fpr.tolist(),  # Diagonal reference
}
fig_roc = viz.training_history(roc_metrics, x_label="False Positive Rate")
fig_roc.update_layout(
    title="ROC Curve: Logistic Regression for High-Price Classification",
    yaxis_title="True Positive Rate",
)
fig_roc.write_html("ex6_roc_curve.html")
print(f"\nSaved: ex6_roc_curve.html")

# -- Plot 2: Odds Ratio Forest Plot --
or_feature_names = [n for n in feature_names if n != "intercept"]
or_data = {}
for i, name in enumerate(or_feature_names):
    or_data[name] = {
        "odds_ratio": odds_ratios_per_unit[i],
        "ci_lower": or_ci_lower_list[i],
        "ci_upper": or_ci_upper_list[i],
    }

fig_or = viz.metric_comparison(or_data)
fig_or.update_layout(title="Odds Ratios per Original Unit (with 95% CI)")
fig_or.write_html("ex6_odds_ratios.html")
print("Saved: ex6_odds_ratios.html")

# -- Plot 3: ANOVA group comparison --
anova_data = {
    ft: {
        "mean_price": float(g.mean()),
        "std_price": float(g.std()),
        "n": len(g),
    }
    for ft, g in groups.items()
}
fig_anova = viz.metric_comparison(anova_data)
fig_anova.update_layout(title=f"ANOVA: Resale Price by Flat Type (F={f_stat:.1f}, p={anova_p:.2e})")
fig_anova.write_html("ex6_anova_comparison.html")
print("Saved: ex6_anova_comparison.html")


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Exercise 6 Summary ===")
print(f"Logistic Regression:")
print(f"  - Built from scratch via MLE (scipy.optimize.minimize)")
print(f"  - Matched sklearn coefficients within numerical tolerance")
print(f"  - Accuracy: {acc:.1%} (baseline: {max(y.mean(), 1-y.mean()):.1%})")
print(f"  - AUC-ROC: {roc_auc:.3f}")
print(f"  - Key driver: floor_area (OR={odds_ratios_per_unit[0]:.3f}/sqm), "
      f"storey (OR={odds_ratios_per_unit[1]:.3f}/floor)")
print(f"\nANOVA:")
print(f"  - Significant difference in prices across flat types (F={f_stat:.1f}, p={anova_p:.2e})")
print(f"  - Effect size (eta-squared): {eta_squared:.3f}")
sig_pairs = [r["pair"] for r in tukey_results if r["significant"]]
if sig_pairs:
    print(f"  - Significant pairwise differences: {', '.join(sig_pairs)}")

print(f"\nKey takeaways:")
print(f"  1. Sigmoid maps linear predictions to valid probabilities (0,1)")
print(f"  2. Logistic regression uses MLE, not OLS (Bernoulli likelihood)")
print(f"  3. exp(beta) gives odds ratio: multiplicative effect on odds")
print(f"  4. ANOVA extends the t-test to 3+ groups; post-hoc tests identify which pairs differ")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(f"""
  ✓ Sigmoid: numerically stable, maps ℝ → (0,1), symmetric, max grad at z=0
  ✓ Logistic MLE: minimise binary cross-entropy = negative log-likelihood
  ✓ Hessian-based SE: H = X^T diag(p*(1-p)) X, SE = sqrt(diag(H^-1))
  ✓ Odds ratio: exp(β) — >1 increases odds, <1 decreases odds of y=1
  ✓ OR per original unit vs per 1 SD: standardise beta by X_std
  ✓ Confusion matrix: TP, TN, FP, FN → precision, recall, F1
  ✓ AUC-ROC: threshold-independent ranking quality (0.5=random, 1.0=perfect)
  ✓ ANOVA: F = MS_between / MS_within generalises t-test to 3+ groups
  ✓ eta-squared: proportion of variance explained by group membership
  ✓ Tukey HSD: controls FWER for pairwise post-hoc comparisons

  NEXT: In Exercise 7 you'll implement CUPED — the single most impactful
  technique for reducing A/B test variance. Using pre-experiment revenue
  as a covariate, you'll shrink confidence intervals by up to 50%, log
  everything with ExperimentTracker, and implement sequential testing
  with always-valid p-values to safely monitor experiments in real time.
""")

print(f"\n✓ Exercise 6 complete — logistic regression + ANOVA foundations")
