# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2: Bias-Variance, Regularisation, and
#                        Cross-Validation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Demonstrate underfitting vs overfitting through polynomial experiments
#   - Decompose expected test error into Bias², Variance, and irreducible
#     noise via bootstrap
#   - Apply L1 (Lasso) and L2 (Ridge) regularisation and explain the
#     geometry (diamond vs circle constraint)
#   - Connect L2 regularisation to Bayesian Gaussian priors on coefficients
#   - Combine L1 and L2 with ElasticNet and select the mixing ratio
#   - Implement nested cross-validation for unbiased model selection
#   - Apply time-series cross-validation (walk-forward) for temporal data
#   - Use GroupKFold when observations are grouped (e.g., same patient)
#   - Visualise the regularisation path (coefficient trajectories)
#
# PREREQUISITES:
#   - MLFP03 Exercise 1 (feature engineering, dataset familiarity)
#   - MLFP02 Module (linear regression, Bayesian thinking from M2.1)
#
# ESTIMATED TIME: ~150-180 minutes
#
# TASKS:
#   1.  Demonstrate underfitting vs overfitting with polynomial models
#   2.  Visualise bias-variance decomposition across model complexity
#   3.  L2 regularisation (Ridge): geometry and coefficient shrinkage
#   4.  L1 regularisation (Lasso): sparsity and feature selection
#   5.  ElasticNet: combining L1 and L2
#   6.  Bayesian interpretation: L2 = Gaussian prior on weights
#   7.  Regularisation path visualisation (coefficient trajectories)
#   8.  Nested cross-validation for unbiased model selection
#   9.  Time-series cross-validation (walk-forward)
#   10. GroupKFold for grouped observations
#   11. Learning curve analysis
#   12. Compare all CV strategies and summarise
#
# DATASET: Singapore credit scoring data (from MLFP02)
#   Target: credit_utilisation (continuous — suitable for regression demo)
#   Rows: ~5,000 credit applicants | Features: financial + behavioural
#   Key insight: with 45 features and a small dataset, overfitting is real.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import (
    Ridge,
    Lasso,
    ElasticNet,
    LinearRegression,
    RidgeCV,
    LassoCV,
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    StratifiedKFold,
    GroupKFold,
    TimeSeriesSplit,
    learning_curve,
)
from sklearn.metrics import mean_squared_error, r2_score

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

print(f"=== Singapore Credit Data ===")
print(f"Shape: {credit.shape}")
print(f"Default rate: {credit['default'].mean():.2%}")

# For regression demonstration, predict credit utilisation ratio
target_col = (
    "credit_utilization" if "credit_utilization" in credit.columns else "income_sgd"
)

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    data=credit,
    target=target_col,
    train_size=0.8,
    seed=42,
    normalize=True,  # Ridge/Lasso require normalised features
    categorical_encoding="ordinal",
    imputation_strategy="median",
)

print(f"\nTask type: {result.task_type}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")

X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=[c for c in result.train_data.columns if c != target_col],
    target_column=target_col,
)
X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=[c for c in result.test_data.columns if c != target_col],
    target_column=target_col,
)
feature_names = col_info["feature_columns"]
print(f"Features: {len(feature_names)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Underfitting vs Overfitting — polynomial degree experiment
# ══════════════════════════════════════════════════════════════════════
# Use a 1D synthetic problem to illustrate cleanly.
# True function: y = sin(2πx) + noise

rng = np.random.default_rng(42)
n_pts = 100
x_1d = rng.uniform(0, 1, n_pts)
y_1d = np.sin(2 * np.pi * x_1d) + rng.normal(0, 0.2, n_pts)

x_1d_train, x_1d_test = x_1d[:80], x_1d[80:]
y_1d_train, y_1d_test = y_1d[:80], y_1d[80:]

x_train_2d = x_1d_train.reshape(-1, 1)
x_test_2d = x_1d_test.reshape(-1, 1)

print(f"\n=== Polynomial Degree Experiment ===")
print(f"{'Degree':>8} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10} {'Diagnosis':>15}")
print("─" * 62)

degree_results = {}
for degree in [1, 2, 4, 6, 9, 12, 15, 20]:
    poly_pipe = Pipeline(
        [
            ("poly", PolynomialFeatures(degree)),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    poly_pipe.fit(x_train_2d, y_1d_train)

    train_mse = mean_squared_error(y_1d_train, poly_pipe.predict(x_train_2d))
    test_mse = mean_squared_error(y_1d_test, poly_pipe.predict(x_test_2d))
    gap = test_mse - train_mse

    if degree <= 2:
        diagnosis = "Underfitting"
    elif degree <= 6:
        diagnosis = "Good fit"
    else:
        diagnosis = "Overfitting"

    degree_results[degree] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "gap": gap,
    }
    print(
        f"{degree:>8} {train_mse:>12.4f} {test_mse:>12.4f} "
        f"{gap:>10.4f} {diagnosis:>15}"
    )

print("\nKey insight:")
print("  Underfitting: high train AND test error (high bias)")
print("  Good fit: low train error, test error close to train")
print("  Overfitting: very low train error, high test error (high variance)")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert (
    degree_results[1]["test_mse"] > degree_results[4]["test_mse"]
), "Degree=1 should have higher test error than degree=4"
assert (
    degree_results[20]["train_mse"] < degree_results[2]["train_mse"]
), "Degree=20 should have lower train error (memorisation)"
# INTERPRETATION: The gap between train_mse and test_mse is the overfit
# penalty.  At degree=20, the model has memorised training data — it has
# learned the noise, not the signal.  This is high variance: the model
# would look completely different if we collected new training data.
print("\n✓ Checkpoint 1 passed — underfitting/overfitting pattern confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Bias-variance decomposition
# ══════════════════════════════════════════════════════════════════════
# Decompose expected test error = Bias² + Variance + Noise
# Measured empirically across bootstrap samples.
#
# FORMULA: E[(y - y_hat)^2] = Bias²(y_hat) + Var(y_hat) + σ²


def bias_variance_decomposition(
    degree: int,
    n_bootstrap: int = 50,
) -> dict[str, float]:
    """Estimate bias² and variance for a polynomial of given degree."""
    all_preds = []
    for _ in range(n_bootstrap):
        # Bootstrap resample training data
        idx = rng.choice(len(y_1d_train), len(y_1d_train), replace=True)
        x_b, y_b = x_train_2d[idx], y_1d_train[idx]

        model = Pipeline(
            [
                ("poly", PolynomialFeatures(degree)),
                ("scaler", StandardScaler()),
                ("lr", LinearRegression()),
            ]
        )
        model.fit(x_b, y_b)
        all_preds.append(model.predict(x_test_2d))

    preds = np.array(all_preds)  # (n_bootstrap, n_test)
    mean_pred = preds.mean(axis=0)

    # True function values at test points (noiseless)
    y_true_noiseless = np.sin(2 * np.pi * x_1d_test)

    bias_sq = np.mean((mean_pred - y_true_noiseless) ** 2)
    variance = np.mean(preds.var(axis=0))
    noise = 0.04  # Known noise variance (sigma=0.2)²
    expected_error = bias_sq + variance + noise

    return {
        "bias_sq": bias_sq,
        "variance": variance,
        "noise": noise,
        "expected_error": expected_error,
    }


print(f"\n=== Bias-Variance Decomposition ===")
print(
    f"{'Degree':>8} {'Bias²':>10} {'Variance':>10} {'Noise':>8} "
    f"{'Expected':>12} {'Dominant':>12}"
)
print("─" * 66)

bv_results = {}
for degree in [1, 2, 3, 6, 10, 15]:
    bv = bias_variance_decomposition(degree, n_bootstrap=30)
    bv_results[degree] = bv
    dominant = "Bias" if bv["bias_sq"] > bv["variance"] else "Variance"
    print(
        f"{degree:>8} {bv['bias_sq']:>10.4f} {bv['variance']:>10.4f} "
        f"{bv['noise']:>8.4f} {bv['expected_error']:>12.4f} {dominant:>12}"
    )

print("\nBias-Variance Trade-off:")
print("  As degree ↑: Bias² ↓, Variance ↑")
print("  Sweet spot = where Bias² ≈ Variance (minimum expected error)")
print("  Irreducible noise σ² = 0.04 sets the floor — no model can beat it.")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    bv_results[1]["bias_sq"] > bv_results[10]["bias_sq"]
), "Degree=1 should have higher bias than degree=10"
assert (
    bv_results[1]["variance"] < bv_results[15]["variance"]
), "Degree=1 should have lower variance than degree=15"
# INTERPRETATION: At degree=1, Bias² dominates — the model is too simple
# to capture the sine wave.  At degree=15, Variance dominates — each
# bootstrap sample gives a wildly different polynomial.  The sweet spot
# (degree=3-6) balances both.  In practice, cross-validation finds this
# automatically without knowing the true function.
print("\n✓ Checkpoint 2 passed — bias-variance decomposition computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: L2 Regularisation (Ridge) — geometry and shrinkage
# ══════════════════════════════════════════════════════════════════════
# Ridge objective: min ||y - Xβ||² + α||β||²
#
# Geometry: the L2 penalty defines a sphere in coefficient space.
# The solution is where the MSE ellipsoid first touches the sphere.
# Result: all coefficients shrink uniformly toward zero, none exactly
# zero.
#
# Bayesian interpretation: L2 penalty ≡ Gaussian prior N(0, 1/α) on β.

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ridge_results = {}
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, ridge.predict(X_train))
    test_mse = mean_squared_error(y_test, ridge.predict(X_test))
    coef_norm = np.linalg.norm(ridge.coef_)
    n_zero = (np.abs(ridge.coef_) < 1e-6).sum()

    ridge_results[alpha] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "coef_norm": coef_norm,
        "n_zero": n_zero,
        "coef": ridge.coef_.copy(),
    }

print(f"\n=== Ridge Regularisation (L2) ===")
print(
    f"{'Alpha':>10} {'Train MSE':>12} {'Test MSE':>12} "
    f"{'||β||₂':>10} {'Zero coefs':>12}"
)
print("─" * 60)
for alpha, r in ridge_results.items():
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['coef_norm']:>10.4f} {r['n_zero']:>12}"
    )

# Best Ridge alpha
best_alpha_ridge = min(ridge_results.items(), key=lambda x: x[1]["test_mse"])
print(
    f"\nBest Ridge α = {best_alpha_ridge[0]} "
    f"(test MSE={best_alpha_ridge[1]['test_mse']:.4f})"
)

print("\nL2 Key Properties:")
print("  1. Shrinks all coefficients toward zero (never exactly zero)")
print("  2. Equivalent to MAP estimate with Gaussian prior: β ~ N(0, σ²/α)")
print("  3. Ridge closed-form: β = (X'X + αI)⁻¹X'y")
print("  4. Always invertible — regularisation fixes multicollinearity")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert (
    ridge_results[1000.0]["coef_norm"] < ridge_results[0.001]["coef_norm"]
), "Higher alpha should produce smaller coefficient norm"
assert (
    ridge_results[1.0]["n_zero"] <= 2
), "Ridge should produce very few zero coefficients (unlike Lasso)"
# INTERPRETATION: Watch how the coefficient norm drops as alpha increases.
# At alpha=0.001, Ridge barely differs from OLS.  At alpha=1000, all
# coefficients are forced nearly to zero.  The best alpha balances
# fitting the data against keeping coefficients small (prior belief).
print("\n✓ Checkpoint 3 passed — Ridge regularisation behaviour confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: L1 Regularisation (Lasso) — sparsity and feature selection
# ══════════════════════════════════════════════════════════════════════
# Lasso objective: min ||y - Xβ||² + α||β||₁
#
# Geometry: the L1 penalty defines a diamond (hypercube corners).
# The MSE ellipsoid is likely to first touch the diamond at a corner,
# where some coordinates are exactly zero → SPARSITY.
#
# Bayesian interpretation: L1 penalty ≡ Laplace prior on β.

lasso_results = {}
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10_000)
    lasso.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, lasso.predict(X_train))
    test_mse = mean_squared_error(y_test, lasso.predict(X_test))
    coef_norm = np.linalg.norm(lasso.coef_, ord=1)
    n_zero = (np.abs(lasso.coef_) < 1e-6).sum()

    lasso_results[alpha] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "coef_norm": coef_norm,
        "n_zero": n_zero,
        "coef": lasso.coef_.copy(),
    }

print(f"\n=== Lasso Regularisation (L1) ===")
print(
    f"{'Alpha':>10} {'Train MSE':>12} {'Test MSE':>12} "
    f"{'||β||₁':>10} {'Zero coefs':>12}"
)
print("─" * 60)
for alpha, r in lasso_results.items():
    pct_sparse = r["n_zero"] / len(feature_names) * 100
    print(
        f"{alpha:>10.3f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['coef_norm']:>10.4f} {r['n_zero']:>10} ({pct_sparse:.0f}%)"
    )

best_alpha_lasso = min(lasso_results.items(), key=lambda x: x[1]["test_mse"])
print(
    f"\nBest Lasso α = {best_alpha_lasso[0]} "
    f"(test MSE={best_alpha_lasso[1]['test_mse']:.4f}, "
    f"zero coefs={best_alpha_lasso[1]['n_zero']})"
)

# Show which features Lasso keeps at best alpha
best_lasso = Lasso(alpha=best_alpha_lasso[0], max_iter=10_000)
best_lasso.fit(X_train, y_train)
lasso_kept = [
    (name, coef)
    for name, coef in zip(feature_names, best_lasso.coef_)
    if abs(coef) > 1e-6
]
lasso_kept.sort(key=lambda x: abs(x[1]), reverse=True)

print(f"\nLasso-selected features ({len(lasso_kept)}):")
for name, coef in lasso_kept[:15]:
    direction = "+" if coef > 0 else "-"
    print(f"  {direction} {name:<30} β={coef:+.4f}")

print("\nL1 Key Properties:")
print("  1. Produces SPARSE solutions — some β_i exactly = 0")
print("  2. Acts as built-in feature selection")
print("  3. Equivalent to MAP estimate with Laplace prior")
print("  4. No closed-form solution — requires coordinate descent")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert (
    lasso_results[100.0]["n_zero"] > lasso_results[0.001]["n_zero"]
), "Higher alpha Lasso should zero out more coefficients"
# INTERPRETATION: Lasso's superpower vs Ridge.  At alpha=10, Lasso might
# eliminate 70% of features entirely — automatic feature selection baked
# into the loss function.  Compare which features Lasso keeps to what
# domain knowledge says should matter.
print("\n✓ Checkpoint 4 passed — Lasso sparsity demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: ElasticNet — combining L1 + L2
# ══════════════════════════════════════════════════════════════════════
# ElasticNet: min ||y - Xβ||² + α₁||β||₁ + α₂||β||₂
# l1_ratio controls the mix: 0 = pure Ridge, 1 = pure Lasso

en_results = {}
for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    en = ElasticNet(alpha=0.1, l1_ratio=l1_ratio, max_iter=10_000)
    en.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, en.predict(X_train))
    test_mse = mean_squared_error(y_test, en.predict(X_test))
    n_zero = (np.abs(en.coef_) < 1e-6).sum()

    en_results[l1_ratio] = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "n_zero": n_zero,
    }

print(f"\n=== ElasticNet (α=0.1) ===")
print(f"{'L1 ratio':>10} {'Train MSE':>12} {'Test MSE':>12} {'Zero coefs':>12}")
print("─" * 50)
for l1_ratio, r in en_results.items():
    print(
        f"{l1_ratio:>10.1f} {r['train_mse']:>12.4f} "
        f"{r['test_mse']:>12.4f} {r['n_zero']:>12}"
    )

print("\nElasticNet Benefits:")
print("  - Groups correlated features together (unlike pure Lasso)")
print("  - More stable than Lasso when features are correlated")
print("  - l1_ratio is a hyperparameter to tune")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    en_results[0.9]["n_zero"] >= en_results[0.1]["n_zero"]
), "l1_ratio=0.9 should zero out at least as many as l1_ratio=0.1"
# INTERPRETATION: ElasticNet solves Lasso's instability with correlated
# features.  When two features are correlated, Lasso arbitrarily picks
# one and zeros the other.  ElasticNet keeps both with smaller coefs.
print("\n✓ Checkpoint 5 passed — ElasticNet l1_ratio effect confirmed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Bayesian Interpretation — L2 as Gaussian Prior
# ══════════════════════════════════════════════════════════════════════
# MAP (Maximum A Posteriori) estimate:
# β_MAP = argmax P(β|y) = argmax P(y|β) P(β)
#
# If P(y|β) = N(Xβ, σ²I) → log-likelihood = -||y-Xβ||²/(2σ²)
# If P(β) = N(0, τ²I)    → log-prior = -||β||²/(2τ²)
#
# Then:
# β_MAP = argmin ||y-Xβ||² + (σ²/τ²) ||β||²
#       = argmin ||y-Xβ||² + α||β||²   where α = σ²/τ²
#
# This is exactly Ridge regression!

print(f"\n=== Bayesian Interpretation of L2 Regularisation ===")
print(
    """
MAP estimate under Gaussian prior:

  Prior:      β ~ N(0, τ²I)   (belief that β is close to zero)
  Likelihood: y|β ~ N(Xβ, σ²I)

  MAP objective:
  β_MAP = argmin_{β} ||y - Xβ||² + (σ²/τ²)||β||²
                               ↑
                        This is exactly α in Ridge!

Interpretation table:
  ┌──────────────────────────────────────────────────────────┐
  │ α (Ridge) │  τ (prior std) │  Belief                    │
  │───────────┼────────────────┼────────────────────────────│
  │  large    │  small τ       │  Strong: β ≈ 0             │
  │  small    │  large τ       │  Weak: β can be anything   │
  │  0        │  τ → ∞         │  Flat prior (OLS)          │
  └──────────────────────────────────────────────────────────┘
"""
)

# Verify: Ridge with α = σ²/τ²
n, p = X_train.shape
sigma_sq = mean_squared_error(
    y_train, Ridge(alpha=0).fit(X_train, y_train).predict(X_train)
)
tau_sq = 1.0  # Unit prior variance
alpha_implied = sigma_sq / tau_sq

ridge_bayes = Ridge(alpha=alpha_implied)
ridge_bayes.fit(X_train, y_train)
test_mse_bayes = mean_squared_error(y_test, ridge_bayes.predict(X_test))
print(f"Prior σ²={sigma_sq:.4f}, τ²={tau_sq}, implied α={alpha_implied:.4f}")
print(f"Ridge with Bayesian α: Test MSE = {test_mse_bayes:.4f}")

# Coefficient comparison at optimal vs over-regularised
alpha_best = best_alpha_ridge[0]
alpha_over = alphas[-1]

ridge_best = Ridge(alpha=alpha_best).fit(X_train, y_train)
ridge_over = Ridge(alpha=alpha_over).fit(X_train, y_train)

print(f"\n=== Coefficient Shrinkage Comparison ===")
header = f"{'Feature':<30} {'OLS':>10} {'Ridge α={:.3f}'.format(alpha_best):>16} {'Ridge α={:.0f}'.format(alpha_over):>16}"
print(header)
print("─" * 75)
ols = LinearRegression().fit(X_train, y_train)
for i, name in enumerate(feature_names[:10]):
    print(
        f"{name:<30} {ols.coef_[i]:>10.4f} "
        f"{ridge_best.coef_[i]:>16.4f} {ridge_over.coef_[i]:>16.4f}"
    )

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert alpha_implied > 0, "Implied alpha should be positive"
assert test_mse_bayes > 0, "Bayesian Ridge should produce valid predictions"
ols_norm = np.linalg.norm(ols.coef_)
ridge_best_norm = np.linalg.norm(ridge_best.coef_)
assert (
    ols_norm >= ridge_best_norm
), "OLS coefficients should have at least as large a norm as Ridge"
# INTERPRETATION: The Bayesian view transforms regularisation from a
# computational trick into a statement about beliefs.  If you believe
# credit default is driven by a few large effects, prefer Lasso (Laplace
# prior — sparse).  If many small effects contribute equally, Ridge
# (Gaussian prior — smooth) fits better.
print("\n✓ Checkpoint 6 passed — Bayesian interpretation verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Regularisation path visualisation
# ══════════════════════════════════════════════════════════════════════
# The regularisation path shows how each coefficient changes as α varies.
# This is the definitive visual for understanding L1 vs L2 behaviour.

viz = ModelVisualizer()

# Ridge regularisation path — coefficient norms
print(f"\n=== Regularisation Path ===")
print(
    f"{'Alpha':>10} {'||β||₂ (Ridge)':>16} {'||β||₁ (Lasso)':>16} {'Non-zero (L)':>14}"
)
print("─" * 60)
for alpha in alphas:
    ridge_norm = ridge_results[alpha]["coef_norm"]
    lasso_norm = lasso_results[alpha]["coef_norm"]
    lasso_nz = len(feature_names) - lasso_results[alpha]["n_zero"]
    print(f"{alpha:>10.3f} {ridge_norm:>16.4f} {lasso_norm:>16.4f} {lasso_nz:>14}")

# Visualise coefficient trajectories
coef_matrix_ridge = np.array([ridge_results[a]["coef"] for a in alphas])
fig = viz.training_history(
    {f"||β||₂": [ridge_results[a]["coef_norm"] for a in alphas]},
    x_label="Regularisation Strength (α)",
)
fig.update_layout(title="Ridge: Coefficient Norm vs Regularisation Strength")
fig.write_html("ex2_ridge_path.html")

# Lasso sparsity
fig_sparse = viz.training_history(
    {
        "Non-zero coefs": [
            len(feature_names) - lasso_results[a]["n_zero"] for a in alphas
        ]
    },
    x_label="Regularisation Strength (α)",
)
fig_sparse.update_layout(title="Lasso: Sparsity vs Regularisation Strength")
fig_sparse.write_html("ex2_lasso_sparsity.html")

print("\nSaved: ex2_ridge_path.html, ex2_lasso_sparsity.html")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert coef_matrix_ridge.shape == (
    len(alphas),
    len(feature_names),
), "Coefficient matrix should have one row per alpha"
# INTERPRETATION: The regularisation path is the single most informative
# visualisation for understanding L1 vs L2.  Ridge: all coefficients
# shrink smoothly toward zero.  Lasso: coefficients snap to zero one by
# one as α increases — the "kink" in the path is the L1 diamond geometry.
print("\n✓ Checkpoint 7 passed — regularisation path visualised\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Nested cross-validation for unbiased model selection
# ══════════════════════════════════════════════════════════════════════
# PROBLEM: Using the same CV to select hyperparameters AND estimate
# performance gives an optimistic estimate (information leakage).
#
# SOLUTION: Nested CV
#   Outer loop: estimate generalisation performance (5 folds)
#   Inner loop: select best hyperparameters (3 folds within each outer fold)
#
# The outer score is an unbiased estimate of the selected model's
# performance on truly unseen data.

print(f"\n=== Nested Cross-Validation ===")

# Standard (non-nested) CV for comparison — BIASED
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)
standard_cv_score = ridge_cv.score(X_test, y_test)
print(f"Standard CV (biased): R²={standard_cv_score:.4f}, best α={ridge_cv.alpha_:.4f}")

# Nested CV — UNBIASED
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

nested_scores = []
selected_alphas = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train)):
    X_outer_train, X_outer_test = X_train[train_idx], X_train[test_idx]
    y_outer_train, y_outer_test = y_train[train_idx], y_train[test_idx]

    # Inner loop: select best alpha
    best_inner_score = -np.inf
    best_inner_alpha = alphas[0]
    for alpha in alphas:
        inner_scores = cross_val_score(
            Ridge(alpha=alpha),
            X_outer_train,
            y_outer_train,
            cv=inner_cv,
            scoring="r2",
        )
        if inner_scores.mean() > best_inner_score:
            best_inner_score = inner_scores.mean()
            best_inner_alpha = alpha

    # Outer loop: evaluate selected model on held-out fold
    ridge_selected = Ridge(alpha=best_inner_alpha)
    ridge_selected.fit(X_outer_train, y_outer_train)
    outer_score = r2_score(y_outer_test, ridge_selected.predict(X_outer_test))

    nested_scores.append(outer_score)
    selected_alphas.append(best_inner_alpha)
    print(
        f"  Fold {fold_idx + 1}: selected α={best_inner_alpha:.4f}, "
        f"outer R²={outer_score:.4f}"
    )

nested_mean = np.mean(nested_scores)
nested_std = np.std(nested_scores)
print(f"\nNested CV: R² = {nested_mean:.4f} ± {nested_std:.4f}")
print(f"Standard CV: R² = {standard_cv_score:.4f}")
print(f"Optimism bias: {standard_cv_score - nested_mean:.4f}")
print(f"Selected alphas across folds: {selected_alphas}")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
assert len(nested_scores) == 5, "Should have 5 outer fold scores"
assert all(isinstance(s, float) for s in nested_scores), "Scores should be floats"
# INTERPRETATION: The gap between standard CV and nested CV is the
# optimism bias from using the same data for selection and evaluation.
# In practice the gap may be small (well-behaved data) or large
# (high-dimensional, many hyperparameters).  Nested CV is always
# more honest — use it when you report final model performance.
print("\n✓ Checkpoint 8 passed — nested CV demonstrates optimism bias\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 9: Time-series cross-validation (walk-forward)
# ══════════════════════════════════════════════════════════════════════
# Standard k-fold shuffles data randomly.  For time-series, this leaks
# future information into the training set.  TimeSeriesSplit uses only
# past data to predict the future — walk-forward validation.

print(f"\n=== Time-Series Cross-Validation ===")

# Simulate temporal ordering (use the first column as a proxy for time)
# In real applications, you'd use an actual timestamp column.
tscv = TimeSeriesSplit(n_splits=5)

print("Walk-forward splits:")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
    print(
        f"  Fold {fold + 1}: train=[0:{train_idx[-1]+1}] "
        f"({len(train_idx)} samples), "
        f"test=[{test_idx[0]}:{test_idx[-1]+1}] ({len(test_idx)} samples)"
    )

# Compare standard k-fold vs time-series split
kfold_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=5, scoring="r2")
ts_scores = cross_val_score(Ridge(alpha=1.0), X_train, y_train, cv=tscv, scoring="r2")

print(f"\nStandard 5-fold: R² = {kfold_scores.mean():.4f} ± {kfold_scores.std():.4f}")
print(f"Time-series 5-fold: R² = {ts_scores.mean():.4f} ± {ts_scores.std():.4f}")

print("\nWhen to use TimeSeriesSplit:")
print("  - Financial data (stock prices, credit performance over time)")
print("  - Sensor data (IoT, medical vitals)")
print("  - Any data with temporal dependence")
print("  - NEVER shuffle temporal data for k-fold — it leaks the future!")

# ── Checkpoint 9 ─────────────────────────────────────────────────────
assert len(ts_scores) == 5, "Should have 5 time-series CV scores"
# INTERPRETATION: If time-series CV gives substantially different scores
# than standard k-fold, your data has temporal dependence — standard
# k-fold is overly optimistic because it trains on future data.  For
# credit scoring, the relevant question is: "can we predict tomorrow's
# defaults from today's data?" not "can we interpolate between random
# time points?"
print("\n✓ Checkpoint 9 passed — time-series CV demonstrated\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 10: GroupKFold for grouped observations
# ══════════════════════════════════════════════════════════════════════
# When observations are grouped (e.g., multiple admissions per patient,
# multiple transactions per customer), standard k-fold can put the same
# group in both train and test — leaking group-level information.
# GroupKFold ensures all observations from one group stay together.

print(f"\n=== GroupKFold ===")

# Create synthetic groups (simulate customer_id for credit data)
# In practice, this comes from your data's natural grouping.
n_samples = X_train.shape[0]
n_groups = n_samples // 5  # ~5 observations per group
groups = np.repeat(np.arange(n_groups), 5)[:n_samples]
rng.shuffle(groups)

group_cv = GroupKFold(n_splits=5)
group_scores = cross_val_score(
    Ridge(alpha=1.0), X_train, y_train, cv=group_cv, groups=groups, scoring="r2"
)

print(f"Standard k-fold: R² = {kfold_scores.mean():.4f} ± {kfold_scores.std():.4f}")
print(f"GroupKFold:       R² = {group_scores.mean():.4f} ± {group_scores.std():.4f}")

# Verify group integrity — no group should appear in both train and test
for fold, (train_idx, test_idx) in enumerate(group_cv.split(X_train, groups=groups)):
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    assert len(overlap) == 0, f"Fold {fold}: groups overlap! {overlap}"

print("  ✓ All folds verified: no group appears in both train and test")

print("\nWhen to use GroupKFold:")
print("  - Multiple observations per patient/customer/device")
print("  - Repeated measures design (same subject measured over time)")
print("  - Hierarchical data (students within schools)")
print("  - Without grouping, the model learns individual identity, not patterns")

# ── Checkpoint 10 ────────────────────────────────────────────────────
assert len(group_scores) == 5, "Should have 5 GroupKFold scores"
# INTERPRETATION: GroupKFold typically gives LOWER scores than standard
# k-fold because the model cannot exploit group-level identity.  This
# is the honest estimate — in production, you will predict on NEW
# patients/customers, not ones you have already seen.  If GroupKFold
# scores are much lower, your model relies on memorising individuals
# rather than learning generalisable patterns.
print("\n✓ Checkpoint 10 passed — GroupKFold group integrity verified\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 11: Learning curve analysis
# ══════════════════════════════════════════════════════════════════════
# Learning curves show how performance changes with training set size.
# They diagnose whether you need MORE DATA or a BETTER MODEL.
#   Converging curves far apart → high bias (underfit) → need better model
#   Converging curves close → low bias, low variance → good model, more
#     data will not help
#   Curves not converging → high variance (overfit) → need more data

print(f"\n=== Learning Curve Analysis ===")

models_for_lc = {
    "OLS (no regularisation)": LinearRegression(),
    "Ridge (α=1)": Ridge(alpha=1.0),
    "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=10_000),
}

train_sizes_frac = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

for name, model_lc in models_for_lc.items():
    train_sizes, train_scores, test_scores = learning_curve(
        model_lc,
        X_train,
        y_train,
        train_sizes=train_sizes_frac,
        cv=5,
        scoring="r2",
        n_jobs=-1,
    )

    print(f"\n--- {name} ---")
    print(f"{'Train size':>12} {'Train R²':>10} {'Test R²':>10} {'Gap':>10}")
    print("─" * 44)
    for size, tr_score, te_score in zip(
        train_sizes, train_scores.mean(axis=1), test_scores.mean(axis=1)
    ):
        gap = tr_score - te_score
        print(f"{size:>12} {tr_score:>10.4f} {te_score:>10.4f} {gap:>10.4f}")

# ── Checkpoint 11 ────────────────────────────────────────────────────
assert len(train_sizes) == len(train_sizes_frac), "Should have all training sizes"
# INTERPRETATION: Learning curves are the most underused diagnostic tool.
# If your test curve is still rising at full training size, MORE DATA will
# help.  If it plateaued, you need a BETTER MODEL (more features, non-linear
# model, less regularisation).  The gap between train and test curves is
# the variance — regularisation shrinks this gap.
print("\n✓ Checkpoint 11 passed — learning curve analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 12: Summary — compare all CV strategies
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("CROSS-VALIDATION STRATEGY COMPARISON")
print(f"{'=' * 70}")

cv_summary = {
    "Standard k-fold": {
        "R²_mean": kfold_scores.mean(),
        "R²_std": kfold_scores.std(),
        "when": "i.i.d. data, no groups, no temporal",
    },
    "Nested CV": {
        "R²_mean": nested_mean,
        "R²_std": nested_std,
        "when": "Hyperparameter selection + performance estimate",
    },
    "Time-series": {
        "R²_mean": ts_scores.mean(),
        "R²_std": ts_scores.std(),
        "when": "Temporal data, prediction into the future",
    },
    "GroupKFold": {
        "R²_mean": group_scores.mean(),
        "R²_std": group_scores.std(),
        "when": "Grouped observations (patients, customers)",
    },
}

print(f"\n{'Strategy':<20} {'R²':>12} {'±':>4} {'When to use'}")
print("─" * 72)
for name, info in cv_summary.items():
    print(
        f"{name:<20} {info['R²_mean']:>12.4f} ± {info['R²_std']:.4f}  "
        f"{info['when']}"
    )

print(
    """
Decision Guide:
  1. Always start with standard k-fold as baseline
  2. If data is temporal → use TimeSeriesSplit
  3. If data has groups → use GroupKFold
  4. If reporting final performance → use nested CV
  5. For production: use the strategy that matches your deployment scenario
"""
)

# ── Checkpoint 12 ────────────────────────────────────────────────────
assert len(cv_summary) == 4, "Should compare 4 CV strategies"
# INTERPRETATION: The "right" CV strategy depends on your data and
# deployment scenario.  Using standard k-fold on temporal data gives
# an optimistic estimate because it trains on future data.  Using
# GroupKFold on i.i.d. data is unnecessarily conservative.  Match
# the CV strategy to the real-world prediction task.
print("\n✓ Checkpoint 12 passed — CV strategy comparison complete\n")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ Bias-Variance: complexity ↑ → Bias² ↓, Variance ↑
  ✓ Ridge (L2): uniform shrinkage, never exactly zero, Gaussian prior
  ✓ Lasso (L1): sparse selection, some β_i exactly = 0, Laplace prior
  ✓ ElasticNet: handles correlated features, blends both penalties
  ✓ Bayesian view: choosing α = specifying prior beliefs about β
  ✓ Regularisation path: coefficient trajectories as α varies
  ✓ Nested CV: unbiased model selection without information leakage
  ✓ Time-series CV: walk-forward validation for temporal data
  ✓ GroupKFold: grouped observations stay together
  ✓ Learning curves: diagnose data hunger vs model weakness

  KEY INSIGHT: Regularisation is not just about preventing overfitting.
  It encodes your prior beliefs about the world.  L2 says "effects are
  spread across many features."  L1 says "only a few features matter."
  Cross-validation tells you which belief matches reality.

  CV STRATEGY SELECTION:
    i.i.d. data → k-fold
    temporal    → TimeSeriesSplit
    grouped     → GroupKFold
    reporting   → nested CV

  NEXT: Exercise 3 trains the complete supervised model zoo — SVM,
  KNN, Naive Bayes, Decision Trees, and Random Forests — on e-commerce
  churn data.  You'll compute Gini impurity from scratch and build a
  decision boundary visualisation.
"""
)

print(
    "\n✓ Exercise 2 complete — bias-variance, L1/L2 geometry, Bayesian "
    "interpretation, nested/time-series/group CV"
)
