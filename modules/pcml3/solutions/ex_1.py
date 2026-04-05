# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT3 — Exercise 1: Bias-Variance and Regularisation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand the bias-variance decomposition through visual
#   experiments, derive L1/L2 geometry, connect L2 to Gaussian priors,
#   and show regularisation effects on model coefficients.
#
# TASKS:
#   1. Demonstrate underfitting vs overfitting with polynomial models
#   2. Visualise bias-variance decomposition across model complexity
#   3. L2 regularisation (Ridge): geometry and coefficient shrinkage
#   4. L1 regularisation (Lasso): sparsity and feature selection
#   5. ElasticNet: combining L1 and L2
#   6. Bayesian interpretation: L2 = Gaussian prior on weights
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

print(f"=== Singapore Credit Data ===")
print(f"Shape: {credit.shape}")
print(f"Default rate: {credit['default'].mean():.2%}")

# For regression demonstration, we'll predict credit utilisation ratio
# (a continuous variable, suitable for linear models)
target_col = (
    "credit_utilisation" if "credit_utilisation" in credit.columns else "annual_income"
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
print(f"{'Degree':>8} {'Train MSE':>12} {'Test MSE':>12} {'Diagnosis':>15}")
print("─" * 52)

degree_results = {}
for degree in [1, 2, 4, 9, 15, 20]:
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

    degree_results[degree] = {"train_mse": train_mse, "test_mse": test_mse}
    print(f"{degree:>8} {train_mse:>12.4f} {test_mse:>12.4f} {diagnosis:>15}")

print("\nKey insight:")
print("  Underfitting: high train AND test error (high bias)")
print("  Good fit: low train error, test error close to train")
print("  Overfitting: very low train error, high test error (high variance)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Bias-variance decomposition
# ══════════════════════════════════════════════════════════════════════
# Decompose expected test error = Bias² + Variance + Noise
# Measured empirically across bootstrap samples.


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
    noise = 0.04  # Known noise variance (sigma=0.2)
    expected_error = bias_sq + variance + noise

    return {"bias_sq": bias_sq, "variance": variance, "expected_error": expected_error}


print(f"\n=== Bias-Variance Decomposition ===")
print(f"{'Degree':>8} {'Bias²':>10} {'Variance':>10} {'Expected':>12} {'Dominant':>12}")
print("─" * 58)

for degree in [1, 3, 6, 10, 15]:
    bv = bias_variance_decomposition(degree, n_bootstrap=30)
    dominant = "Bias" if bv["bias_sq"] > bv["variance"] else "Variance"
    print(
        f"{degree:>8} {bv['bias_sq']:>10.4f} {bv['variance']:>10.4f} "
        f"{bv['expected_error']:>12.4f} {dominant:>12}"
    )

print("\nBias-Variance Trade-off:")
print("  As degree ↑: Bias² ↓, Variance ↑")
print("  Sweet spot = where Bias² ≈ Variance")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: L2 Regularisation (Ridge) — geometry and shrinkage
# ══════════════════════════════════════════════════════════════════════
# Ridge objective: min ||y - Xβ||² + α||β||²
#
# Geometry: the L2 penalty defines a sphere in coefficient space.
# The solution is where the MSE ellipsoid first touches the sphere.
# Result: all coefficients shrink uniformly toward zero, none exactly zero.
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
    f"{'Alpha':>10} {'Train MSE':>12} {'Test MSE':>12} {'||β||₂':>10} {'Zero coefs':>12}"
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
    f"\nBest Ridge α = {best_alpha_ridge[0]} (test MSE={best_alpha_ridge[1]['test_mse']:.4f})"
)

print("\nL2 Key Properties:")
print("  1. Shrinks all coefficients toward zero (never exactly zero)")
print("  2. Equivalent to MAP estimate with Gaussian prior: β ~ N(0, σ²/α)")
print("  3. Ridge closed-form: β = (X'X + αI)⁻¹X'y")
print("  4. Always invertible — regularisation fixes multicollinearity")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: L1 Regularisation (Lasso) — sparsity and feature selection
# ══════════════════════════════════════════════════════════════════════
# Lasso objective: min ||y - Xβ||² + α||β||₁
#
# Geometry: the L1 penalty defines a diamond (hypercube corners in high-d).
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
    f"{'Alpha':>10} {'Train MSE':>12} {'Test MSE':>12} {'||β||₁':>10} {'Zero coefs':>12}"
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

print("\nL1 Key Properties:")
print("  1. Produces SPARSE solutions — some β_i exactly = 0")
print("  2. Acts as built-in feature selection")
print("  3. Equivalent to MAP estimate with Laplace prior: β ~ Laplace(0, 1/α)")
print("  4. No closed-form solution — requires coordinate descent or ADMM")


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
        f"{l1_ratio:>10.1f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} {r['n_zero']:>12}"
    )

print("\nElasticNet Benefits:")
print("  - Groups correlated features together (unlike pure Lasso)")
print("  - More stable than Lasso when features are correlated")
print("  - l1_ratio is a hyperparameter to tune")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Bayesian Interpretation — L2 as Gaussian Prior
# ══════════════════════════════════════════════════════════════════════
# MAP (Maximum A Posteriori) estimate:
# β_MAP = argmax P(β|y) = argmax P(y|β) P(β)
#
# If P(y|β) = N(Xβ, σ²I)  → log-likelihood = -||y-Xβ||²/(2σ²)
# If P(β) = N(0, τ²I)      → log-prior = -||β||²/(2τ²)
#
# Then:
# β_MAP = argmin ||y-Xβ||² / σ² + ||β||² / τ²
#       = argmin ||y-Xβ||² + (σ²/τ²) ||β||²
#
# This is exactly Ridge with α = σ²/τ²
# → Stronger prior (smaller τ²) = stronger regularisation (larger α)

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
  ┌─────────────────────────────────────────────────────────┐
  │ α (Ridge) │  τ (prior std) │  Belief                  │
  │───────────┼────────────────┼──────────────────────────│
  │  large    │  small τ       │  Strong: β ≈ 0           │
  │  small    │  large τ       │  Weak: β can be anything │
  │  0        │  τ → ∞         │  Flat prior (OLS)        │
  └─────────────────────────────────────────────────────────┘

Why this matters:
  1. Choosing α is equivalent to specifying prior beliefs about β
  2. Cross-validation finds α that balances data evidence vs prior
  3. Bayesian ML = probabilistic regularisation selection
"""
)

# Verify: Ridge with α = σ²/τ² matches normal equations
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
print(
    f"{'Feature':<30} {'OLS':>10} {'Ridge α={:.3f}'.format(alpha_best):>16} {'Ridge α={:.0f}'.format(alpha_over):>16}"
)
print("─" * 75)
ols = LinearRegression().fit(X_train, y_train)
for i, name in enumerate(feature_names[:10]):
    print(
        f"{name:<30} {ols.coef_[i]:>10.4f} {ridge_best.coef_[i]:>16.4f} {ridge_over.coef_[i]:>16.4f}"
    )

# Final visualisation
viz = ModelVisualizer()

# Regularisation path: coefficient trajectories as α varies
coef_matrix = np.array([ridge_results[a]["coef"] for a in alphas])
fig = viz.training_history(
    {f"||β||₂": [ridge_results[a]["coef_norm"] for a in alphas]},
    x_label="Regularisation Strength (α)",
)
fig.update_layout(title="Ridge: Coefficient Norm vs Regularisation Strength")
fig.write_html("ex1_ridge_path.html")

# Lasso sparsity
fig_sparse = viz.training_history(
    {
        f"Non-zero coefs": [
            len(feature_names) - lasso_results[a]["n_zero"] for a in alphas
        ]
    },
    x_label="Regularisation Strength (α)",
)
fig_sparse.update_layout(title="Lasso: Sparsity vs Regularisation Strength")
fig_sparse.write_html("ex1_lasso_sparsity.html")

print("\nSaved: ex1_ridge_path.html, ex1_lasso_sparsity.html")
print(
    "\n✓ Exercise 1 complete — bias-variance, L1/L2 geometry, Bayesian interpretation"
)
print("  Key takeaways:")
print("  1. Complexity ↑ → Bias² ↓, Variance ↑ — choose sweet spot via CV")
print("  2. Ridge: uniform shrinkage, Lasso: sparse selection")
print("  3. Regularisation = Bayesian prior. Choosing α = specifying beliefs.")
