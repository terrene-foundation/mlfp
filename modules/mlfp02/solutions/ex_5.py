# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5: Linear Regression
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Derive and implement OLS using matrix algebra (β = (X'X)⁻¹X'y)
#   - Interpret regression coefficients with ceteris paribus logic
#   - Test coefficient significance using t-statistics and p-values
#   - Evaluate model fit with R², adjusted R², and the F-statistic
#   - Extend a model with polynomial and interaction terms, then
#     cross-validate with a train/test split to detect overfitting
#
# PREREQUISITES: Complete Exercises 2-3 — you should understand MLE,
#   hypothesis testing, t-statistics, and p-value interpretation.
#
# ESTIMATED TIME: 75 minutes
#
# TASKS:
#   1. Load HDB resale data and engineer features
#   2. Build simple OLS regression from scratch: β = (X'X)⁻¹X'y
#   3. Interpret coefficients: direction, magnitude, significance
#   4. Compute t-statistics and p-values for each coefficient
#   5. Compute R², adjusted R², and F-statistic
#   6. Add polynomial features (floor_area²) and interaction terms
#   7. Compare models: simple vs enriched using adjusted R²
#   8. Cross-validate with train/test split
#
# DATASET: HDB resale flat transactions (Singapore)
#   Source: data.gov.sg — public housing resale records, 2020+
#   Target: resale_price (SGD); Features: area, storey, lease, flat type, town
#
# THEORY:
#   OLS minimises Sum((yᵢ - ŷᵢ)²). The closed-form solution is:
#     β = (X'X)⁻¹X'y
#   Each βⱼ represents the expected change in y for a one-unit change
#   in xⱼ, holding all other predictors constant (ceteris paribus).
#   t-statistic = βⱼ / SE(βⱼ), testing H₀: βⱼ = 0.
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
print("  MLFP02 Exercise 5: Linear Regression")
print("=" * 60)
print(f"\n  Data loaded: hdb_resale.parquet")
print(f"  Shape: {hdb.shape}")
print(f"  Columns: {hdb.columns}")
print(hdb.head(3))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Feature Engineering
# ══════════════════════════════════════════════════════════════════════
# We need numeric features for regression. Categorical variables (flat_type,
# town) require dummy encoding — one binary column per category minus a
# base category to avoid the dummy variable trap (perfect multicollinearity).

# Parse transaction date and compute derived features
hdb = hdb.with_columns(
    pl.col("month").str.to_date("%Y-%m").alias("transaction_date"),
)

# Filter to recent data for a cleaner analysis
hdb_recent = hdb.filter(
    pl.col("transaction_date") >= pl.date(2020, 1, 1)
)

# Engineer numeric features
hdb_recent = hdb_recent.with_columns(
    # Storey midpoint: parse "07 TO 09" -> (7+9)/2 = 8
    (
        (
            pl.col("storey_range").str.extract(r"(\d+)", 1).cast(pl.Float64)
            + pl.col("storey_range").str.extract(r"TO (\d+)", 1).cast(pl.Float64)
        )
        / 2
    ).alias("storey_midpoint"),
    # Remaining lease (approximate: HDB leases are 99 years from commencement)
    (99 - (pl.col("transaction_date").dt.year() - pl.col("lease_commence_date")))
    .cast(pl.Float64)
    .alias("remaining_lease_years"),
)

print(f"\n=== Filtered Dataset (2020+) ===")
print(f"Shape: {hdb_recent.shape}")
print(f"Flat types: {sorted(hdb_recent['flat_type'].unique().to_list())}")
print(f"Towns: {hdb_recent['town'].n_unique()} unique towns")

# --- Dummy variable encoding for flat_type ---
# THEORY: For k categories, we create k-1 dummy columns.
# The omitted category is the "base" — its effect is absorbed into β₀.
# Each dummy coefficient represents the price difference relative to the base.

# Choose a base category: "3 ROOM" (common, middle-range — intuitive base)
flat_types = sorted(hdb_recent["flat_type"].unique().to_list())
base_flat_type = "3 ROOM"
dummy_flat_types = [ft for ft in flat_types if ft != base_flat_type]

print(f"\nFlat type encoding (base = '{base_flat_type}'):")
for ft in dummy_flat_types:
    count = hdb_recent.filter(pl.col("flat_type") == ft).height
    print(f"  flat_type_{ft}: {count:,} transactions")

# Create dummy columns
for ft in dummy_flat_types:
    col_name = f"flat_{ft.lower().replace(' ', '_')}"
    hdb_recent = hdb_recent.with_columns(
        (pl.col("flat_type") == ft).cast(pl.Float64).alias(col_name)
    )

# --- Dummy variable encoding for town (top 5 towns by volume as dummies) ---
# THEORY: With many categories, including all dummies can overfit.
# We select the top towns and group the rest into the base category.
town_counts = (
    hdb_recent.group_by("town")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
)
top_towns = town_counts.head(5)["town"].to_list()
base_town_label = "OTHER"

print(f"\nTown encoding (base = '{base_town_label}'):")
for town in top_towns:
    count = hdb_recent.filter(pl.col("town") == town).height
    print(f"  town_{town}: {count:,} transactions")

for town in top_towns:
    col_name = f"town_{town.lower().replace(' ', '_')}"
    hdb_recent = hdb_recent.with_columns(
        (pl.col("town") == town).cast(pl.Float64).alias(col_name)
    )

# Drop rows with any null in our features
feature_cols_numeric = ["floor_area_sqm", "storey_midpoint", "remaining_lease_years"]
dummy_flat_cols = [f"flat_{ft.lower().replace(' ', '_')}" for ft in dummy_flat_types]
dummy_town_cols = [f"town_{t.lower().replace(' ', '_')}" for t in top_towns]
all_feature_cols = feature_cols_numeric + dummy_flat_cols + dummy_town_cols

hdb_clean = hdb_recent.drop_nulls(subset=feature_cols_numeric + ["resale_price"])

print(f"\nClean dataset: {hdb_clean.shape[0]:,} rows, {len(all_feature_cols)} features")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert hdb_clean.height > 0, "Clean dataset should not be empty"
assert len(all_feature_cols) > 3, "Should have at least 3 features (numeric + dummies)"
assert "floor_area_sqm" in all_feature_cols, "floor_area_sqm must be a feature"
print("\n✓ Checkpoint 1 passed — features engineered and dataset cleaned\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build OLS regression from scratch
# ══════════════════════════════════════════════════════════════════════
# THEORY (OLS closed-form solution):
#   Given y = Xβ + ε, the OLS estimator minimises ||y - Xβ||²
#   Taking the derivative and setting to zero:
#     ∂/∂β [y - Xβ]'[y - Xβ] = -2X'y + 2X'Xβ = 0
#   Solving: β̂ = (X'X)⁻¹X'y
#
#   The first column of X is all 1s (intercept term β₀).

# Extract feature matrix and target vector
X_data = hdb_clean.select(all_feature_cols).to_numpy().astype(np.float64)
y = hdb_clean["resale_price"].to_numpy().astype(np.float64)

n, k = X_data.shape  # n observations, k features

# Add intercept column (column of 1s)
X = np.column_stack([np.ones(n), X_data])
p = X.shape[1]  # Total parameters = k features + 1 intercept

print(f"\n=== OLS Regression (from scratch) ===")
print(f"Design matrix X: {X.shape} (n={n:,}, p={p})")
print(f"Target vector y: {y.shape}")

# β̂ = (X'X)⁻¹X'y
# THEORY: X'X is the Gram matrix (p×p). Its inverse exists when columns
# of X are linearly independent (no perfect multicollinearity).
XtX = X.T @ X
Xty = X.T @ y
beta_hat = np.linalg.solve(XtX, Xty)  # More numerically stable than inv(XtX) @ Xty

# Predicted values and residuals
y_hat = X @ beta_hat
residuals = y - y_hat

# Coefficient names
coeff_names = ["intercept"] + all_feature_cols

print(f"\nOLS Coefficients:")
print(f"{'Feature':<30} {'Coefficient':>15} {'Interpretation'}")
print("─" * 85)
for name, coeff in zip(coeff_names, beta_hat):
    if name == "intercept":
        print(f"{name:<30} {coeff:>15,.0f}   Base price (3-ROOM, OTHER town, all numerics=0)")
    elif name.startswith("flat_"):
        sign = "premium" if coeff > 0 else "discount"
        print(f"{name:<30} {coeff:>15,.0f}   ${abs(coeff):,.0f} {sign} vs 3-ROOM")
    elif name.startswith("town_"):
        sign = "premium" if coeff > 0 else "discount"
        print(f"{name:<30} {coeff:>15,.0f}   ${abs(coeff):,.0f} {sign} vs OTHER towns")
    elif name == "floor_area_sqm":
        print(f"{name:<30} {coeff:>15,.0f}   Each sqm adds ${coeff:,.0f} to price")
    elif name == "storey_midpoint":
        print(f"{name:<30} {coeff:>15,.0f}   Each floor up adds ${coeff:,.0f}")
    elif name == "remaining_lease_years":
        print(f"{name:<30} {coeff:>15,.0f}   Each extra year of lease adds ${coeff:,.0f}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(beta_hat) == p, f"Should have {p} coefficients, got {len(beta_hat)}"
assert y_hat.shape == y.shape, "Predicted values should have same shape as y"
assert residuals.shape == y.shape, "Residuals should have same shape as y"
# Floor area should have positive coefficient (larger flat → higher price)
area_idx = coeff_names.index("floor_area_sqm")
assert beta_hat[area_idx] > 0, "floor_area_sqm coefficient should be positive"
print("\n✓ Checkpoint 2 passed — OLS coefficients computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Interpret coefficients — direction, magnitude, significance
# ══════════════════════════════════════════════════════════════════════
# THEORY: Coefficient interpretation (ceteris paribus):
#   β₁ = 5000 for floor_area_sqm means:
#   "Holding flat type, town, storey, and lease constant, each additional
#    square metre is associated with a $5,000 increase in resale price."

print(f"\n=== Coefficient Interpretation (ceteris paribus) ===")
for name, coeff in zip(coeff_names, beta_hat):
    if name == "intercept":
        continue
    direction = "positive" if coeff > 0 else "negative"
    print(f"\n{name}:")
    print(f"  Direction: {direction} (β = {coeff:,.2f})")
    print(f"  Magnitude: |β| = ${abs(coeff):,.2f}")
    if name in feature_cols_numeric:
        print(f"  Meaning: A one-unit increase in {name} is associated with")
        print(f"           a ${coeff:,.0f} {'increase' if coeff > 0 else 'decrease'} in price,")
        print(f"           holding all other variables constant.")
    elif name.startswith("flat_"):
        ft_label = name.replace("flat_", "").upper().replace("_", " ")
        print(f"  Meaning: {ft_label} flats sell for ${abs(coeff):,.0f}")
        print(f"           {'more' if coeff > 0 else 'less'} than 3-ROOM (base), ceteris paribus.")
    elif name.startswith("town_"):
        town_label = name.replace("town_", "").upper().replace("_", " ")
        print(f"  Meaning: Properties in {town_label} sell for ${abs(coeff):,.0f}")
        print(f"           {'more' if coeff > 0 else 'less'} than OTHER towns (base), ceteris paribus.")
# INTERPRETATION: "Ceteris paribus" is Latin for "all other things being equal."
# This is the key advantage of multivariate regression over simple comparisons —
# we can isolate the effect of each predictor while controlling for the others.


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compute t-statistics and p-values
# ══════════════════════════════════════════════════════════════════════
# THEORY:
#   t = β̂ⱼ / SE(β̂ⱼ)
#   where SE(β̂ⱼ) = √(σ̂² * (X'X)⁻¹ⱼⱼ)
#   and σ̂² = RSS / (n - p) = Sum(eᵢ²) / (n - p)
#
#   Cutoffs (two-tailed):
#     |t| > 1.645 → significant at 90% (*)
#     |t| > 1.960 → significant at 95% (**)
#     |t| > 2.576 → significant at 99% (***)

# Residual standard error
RSS = np.sum(residuals**2)
sigma_sq_hat = RSS / (n - p)  # Unbiased estimate of error variance
sigma_hat = np.sqrt(sigma_sq_hat)

# Variance-covariance matrix of β̂
# THEORY: Var(β̂) = σ² * (X'X)⁻¹
XtX_inv = np.linalg.inv(XtX)
var_beta = sigma_sq_hat * XtX_inv

# Standard errors = sqrt of diagonal of variance-covariance matrix
se_beta = np.sqrt(np.diag(var_beta))

# t-statistics
t_stats = beta_hat / se_beta

# p-values (two-tailed, t-distribution with n-p degrees of freedom)
df = n - p
p_values = 2 * stats.t.sf(np.abs(t_stats), df=df)

print(f"\n=== T-Statistics and Significance ===")
print(f"Residual SE (σ̂): ${sigma_hat:,.0f}")
print(f"Degrees of freedom: {df:,} (n={n:,}, p={p})")
print()
print(f"{'Feature':<30} {'β̂':>12} {'SE(β̂)':>12} {'t-stat':>10} {'p-value':>12} {'Sig':>6}")
print("─" * 88)
for i, name in enumerate(coeff_names):
    # Significance stars
    if p_values[i] < 0.001:
        sig = "***"
    elif p_values[i] < 0.01:
        sig = "**"
    elif p_values[i] < 0.05:
        sig = "*"
    elif p_values[i] < 0.10:
        sig = "."
    else:
        sig = ""
    print(
        f"{name:<30} {beta_hat[i]:>12,.2f} {se_beta[i]:>12,.2f} "
        f"{t_stats[i]:>10.3f} {p_values[i]:>12.2e} {sig:>6}"
    )

print(f"\nSignificance codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1")
print(f"\nInterpretation: A large |t| and small p-value means we can reject")
print(f"H₀: βⱼ = 0 — the feature has a statistically significant relationship")
print(f"with resale price after controlling for all other features.")
# INTERPRETATION: The t-statistic here is the SAME concept from Exercise 3
# (hypothesis testing). We're testing H₀: β=0 for each coefficient.
# The regression t-test and the two-sample t-test are the same mathematics
# applied in different contexts — t = estimate / standard_error.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert sigma_hat > 0, "Residual standard error must be positive"
assert df == n - p, f"Degrees of freedom should be n-p = {n}-{p} = {n-p}"
assert all(0 <= pv <= 1 for pv in p_values), "All p-values must be between 0 and 1"
# floor_area should be highly significant
assert p_values[area_idx] < 0.001, "floor_area_sqm should be highly significant"
print("\n✓ Checkpoint 3 passed — t-statistics and p-values computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compute R², adjusted R², and F-statistic
# ══════════════════════════════════════════════════════════════════════
# THEORY:
#   R² = 1 - SS_res / SS_tot
#   Adjusted R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)
#   F = (SS_reg / k) / (SS_res / (n - k - 1))
#   Under H₀ (all βⱼ=0): F ~ F(k, n-k-1)

y_mean = y.mean()
SS_tot = np.sum((y - y_mean) ** 2)
SS_res = np.sum(residuals ** 2)
SS_reg = SS_tot - SS_res

# R-squared
R_sq = 1 - SS_res / SS_tot

# Adjusted R-squared
R_sq_adj = 1 - (1 - R_sq) * (n - 1) / (n - p)

# F-statistic
# k = number of predictors (excluding intercept)
k_predictors = p - 1
F_stat = (SS_reg / k_predictors) / (SS_res / (n - p))
F_p_value = stats.f.sf(F_stat, dfn=k_predictors, dfd=n - p)

print(f"\n=== Model Fit Statistics ===")
print(f"SS_total:    {SS_tot:,.0f}")
print(f"SS_residual: {SS_res:,.0f}")
print(f"SS_regression: {SS_reg:,.0f}")
print()
print(f"R²:          {R_sq:.4f} ({R_sq:.1%} of variance explained)")
print(f"Adjusted R²: {R_sq_adj:.4f}")
print()
print(f"F-statistic: {F_stat:,.2f} (df1={k_predictors}, df2={n - p})")
print(f"F p-value:   {F_p_value:.2e}")
if F_p_value < 0.001:
    print(f"Interpretation: The model is overwhelmingly better than predicting")
    print(f"the mean price for every transaction (intercept-only model).")
    print(f"At least one predictor has a non-zero relationship with price.")
else:
    print(f"Interpretation: Insufficient evidence that the model improves on")
    print(f"the intercept-only baseline.")
# INTERPRETATION: R² measures how much price variation the model explains.
# The F-statistic tests whether the whole model is useful (H₀: all β=0).
# Adjusted R² penalises for adding predictors — unlike R², it can decrease
# when a new predictor adds more noise than signal.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert 0 <= R_sq <= 1, f"R² must be between 0 and 1, got {R_sq:.4f}"
assert R_sq_adj <= R_sq, "Adjusted R² must be <= R²"
assert F_stat > 0, "F-statistic must be positive"
assert F_p_value < 0.001, "Model with area/storey/type should be highly significant"
print("\n✓ Checkpoint 4 passed — R², adjusted R², F-statistic computed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Add polynomial and interaction terms
# ══════════════════════════════════════════════════════════════════════
# THEORY (non-linearity in linear regression):
#   Linear regression is "linear in parameters" — the features themselves
#   can be non-linear transformations of the original variables.
#
#   Polynomial: x² captures diminishing/accelerating returns.
#   Interaction: x₁ * x₂ captures effect modification.

# Add polynomial term: floor_area²
hdb_clean = hdb_clean.with_columns(
    (pl.col("floor_area_sqm") ** 2).alias("floor_area_sq"),
)

# Add interaction terms
hdb_clean = hdb_clean.with_columns(
    (pl.col("floor_area_sqm") * pl.col("storey_midpoint")).alias("area_x_storey"),
    (pl.col("floor_area_sqm") * pl.col("remaining_lease_years")).alias("area_x_lease"),
)

# Build enriched feature set
enriched_feature_cols = all_feature_cols + ["floor_area_sq", "area_x_storey", "area_x_lease"]
X_enriched_data = hdb_clean.select(enriched_feature_cols).to_numpy().astype(np.float64)
y_enriched = hdb_clean["resale_price"].to_numpy().astype(np.float64)

n_e = X_enriched_data.shape[0]
X_enriched = np.column_stack([np.ones(n_e), X_enriched_data])
p_enriched = X_enriched.shape[1]

# OLS on enriched model
XtX_e = X_enriched.T @ X_enriched
Xty_e = X_enriched.T @ y_enriched
beta_enriched = np.linalg.solve(XtX_e, Xty_e)

# Predictions and residuals
y_hat_enriched = X_enriched @ beta_enriched
residuals_enriched = y_enriched - y_hat_enriched

# Fit statistics for enriched model
SS_tot_e = np.sum((y_enriched - y_enriched.mean()) ** 2)
SS_res_e = np.sum(residuals_enriched ** 2)
SS_reg_e = SS_tot_e - SS_res_e

R_sq_e = 1 - SS_res_e / SS_tot_e
R_sq_adj_e = 1 - (1 - R_sq_e) * (n_e - 1) / (n_e - p_enriched)

k_e = p_enriched - 1
F_stat_e = (SS_reg_e / k_e) / (SS_res_e / (n_e - p_enriched))
F_p_e = stats.f.sf(F_stat_e, dfn=k_e, dfd=n_e - p_enriched)

# T-statistics for enriched model
sigma_sq_e = SS_res_e / (n_e - p_enriched)
XtX_inv_e = np.linalg.inv(XtX_e)
se_beta_e = np.sqrt(sigma_sq_e * np.diag(XtX_inv_e))
t_stats_e = beta_enriched / se_beta_e
p_values_e = 2 * stats.t.sf(np.abs(t_stats_e), df=n_e - p_enriched)

enriched_coeff_names = ["intercept"] + enriched_feature_cols

print(f"\n=== Enriched Model (with polynomial + interaction terms) ===")
print(f"Features: {p_enriched - 1} (was {p - 1})")
print()
print(f"{'Feature':<30} {'β̂':>15} {'t-stat':>10} {'p-value':>12} {'Sig':>6}")
print("─" * 78)
for i, name in enumerate(enriched_coeff_names):
    sig = "***" if p_values_e[i] < 0.001 else "**" if p_values_e[i] < 0.01 else "*" if p_values_e[i] < 0.05 else "." if p_values_e[i] < 0.1 else ""
    print(
        f"{name:<30} {beta_enriched[i]:>15,.2f} "
        f"{t_stats_e[i]:>10.3f} {p_values_e[i]:>12.2e} {sig:>6}"
    )

# Interpret the polynomial term
beta_area = beta_enriched[enriched_coeff_names.index("floor_area_sqm")]
beta_area_sq = beta_enriched[enriched_coeff_names.index("floor_area_sq")]
print(f"\n--- Polynomial Interpretation ---")
print(f"floor_area_sqm coefficient: {beta_area:,.2f}")
print(f"floor_area_sq coefficient:  {beta_area_sq:,.4f}")
if beta_area > 0 and beta_area_sq < 0:
    # Vertex of the parabola: x* = -β₁/(2β₂)
    turning_point = -beta_area / (2 * beta_area_sq)
    print(f"Concave relationship: price increases with area but at a decreasing rate.")
    print(f"Marginal return of area reaches zero at {turning_point:,.0f} sqm.")
elif beta_area > 0 and beta_area_sq > 0:
    print(f"Convex relationship: price increases with area at an accelerating rate.")
    print(f"Larger flats command disproportionately higher prices per sqm.")
else:
    print(f"The relationship between area and price has a complex shape.")
# INTERPRETATION: A concave (β₁>0, β₂<0) relationship means each extra sqm
# adds less price than the previous sqm — diminishing returns to space. This
# is economically intuitive: going from 80→90 sqm may add $5K but going
# from 130→140 sqm adds less because demand for very large flats is smaller.

# Interpret interaction terms
beta_area_x_storey = beta_enriched[enriched_coeff_names.index("area_x_storey")]
print(f"\n--- Interaction Interpretation ---")
print(f"area_x_storey coefficient: {beta_area_x_storey:,.4f}")
if beta_area_x_storey > 0:
    print(f"Positive interaction: the storey premium is larger for bigger flats.")
    print(f"A 10-sqm larger flat on a higher floor gains an extra")
    print(f"  ${10 * beta_area_x_storey:,.0f} per additional floor compared to a smaller flat.")
else:
    print(f"Negative interaction: the storey premium is smaller for bigger flats.")


# ══════════════════════════════════════════════════════════════════════
# TASK 7: Compare models — simple vs enriched
# ══════════════════════════════════════════════════════════════════════
# THEORY: R² always increases when you add predictors (even noise).
# Use adjusted R² to compare models with different numbers of predictors.
# Partial F-test: does the enriched model significantly improve on simple?

# Partial F-test: enriched model vs simple model
SS_res_simple = np.sum((y_enriched - (np.column_stack([np.ones(n_e), hdb_clean.select(all_feature_cols).to_numpy().astype(np.float64)]) @ np.linalg.solve(
    np.column_stack([np.ones(n_e), hdb_clean.select(all_feature_cols).to_numpy().astype(np.float64)]).T @ np.column_stack([np.ones(n_e), hdb_clean.select(all_feature_cols).to_numpy().astype(np.float64)]),
    np.column_stack([np.ones(n_e), hdb_clean.select(all_feature_cols).to_numpy().astype(np.float64)]).T @ y_enriched
))) ** 2)
R_sq_simple = 1 - SS_res_simple / SS_tot_e
R_sq_adj_simple = 1 - (1 - R_sq_simple) * (n_e - 1) / (n_e - (len(all_feature_cols) + 1))

extra_params = p_enriched - (len(all_feature_cols) + 1)  # 3 new terms
partial_F = ((SS_res_simple - SS_res_e) / extra_params) / (SS_res_e / (n_e - p_enriched))
partial_F_p = stats.f.sf(partial_F, dfn=extra_params, dfd=n_e - p_enriched)

print(f"\n=== Model Comparison ===")
print(f"{'Metric':<25} {'Simple':>15} {'Enriched':>15}")
print("─" * 58)
print(f"{'Predictors':<25} {len(all_feature_cols):>15} {len(enriched_feature_cols):>15}")
print(f"{'R²':<25} {R_sq_simple:>15.4f} {R_sq_e:>15.4f}")
print(f"{'Adjusted R²':<25} {R_sq_adj_simple:>15.4f} {R_sq_adj_e:>15.4f}")
print(f"{'Residual SE':<25} ${np.sqrt(SS_res_simple / (n_e - len(all_feature_cols) - 1)):>14,.0f} ${np.sqrt(sigma_sq_e):>14,.0f}")

print(f"\nPartial F-test (enriched vs simple):")
print(f"  F = {partial_F:,.2f} (df1={extra_params}, df2={n_e - p_enriched})")
print(f"  p = {partial_F_p:.2e}")
if partial_F_p < 0.001:
    print(f"  The polynomial and interaction terms significantly improve the model.")
    print(f"  Adjusted R² increased from {R_sq_adj_simple:.4f} to {R_sq_adj_e:.4f}.")
else:
    print(f"  The extra terms do not significantly improve the model.")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert R_sq_e >= R_sq_simple, "R² must increase (or stay same) when adding features"
assert 0 <= R_sq_e <= 1, "Enriched R² must be between 0 and 1"
print("\n✓ Checkpoint 5 passed — model comparison completed\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 8: Cross-validation with train/test split
# ══════════════════════════════════════════════════════════════════════
# THEORY: In-sample R² is optimistic — it evaluates the model on the
# same data it was trained on. To estimate real-world performance, we
# hold out a test set the model has never seen.
# If test R² << train R², the model memorises noise (overfitting).

rng = np.random.default_rng(seed=42)
n_total = n_e
indices = rng.permutation(n_total)
n_train = int(0.8 * n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

# Train/test split
X_train = X_enriched[train_idx]
X_test = X_enriched[test_idx]
y_train = y_enriched[train_idx]
y_test = y_enriched[test_idx]

print(f"\n=== Cross-Validation: 80/20 Train/Test Split ===")
print(f"Train: {len(train_idx):,} | Test: {len(test_idx):,}")

# Fit on training data only
beta_cv = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)

# Predict on both sets
y_hat_train = X_train @ beta_cv
y_hat_test = X_test @ beta_cv

# R² on training set
SS_res_train = np.sum((y_train - y_hat_train) ** 2)
SS_tot_train = np.sum((y_train - y_train.mean()) ** 2)
R_sq_train = 1 - SS_res_train / SS_tot_train

# R² on test set
SS_res_test = np.sum((y_test - y_hat_test) ** 2)
SS_tot_test = np.sum((y_test - y_test.mean()) ** 2)
R_sq_test = 1 - SS_res_test / SS_tot_test

# RMSE (Root Mean Squared Error) — in dollar units
RMSE_train = np.sqrt(SS_res_train / len(y_train))
RMSE_test = np.sqrt(SS_res_test / len(y_test))

# MAE (Mean Absolute Error) — more robust to outliers
MAE_train = np.mean(np.abs(y_train - y_hat_train))
MAE_test = np.mean(np.abs(y_test - y_hat_test))

print(f"\n{'Metric':<20} {'Train':>15} {'Test':>15} {'Gap':>12}")
print("─" * 65)
print(f"{'R²':<20} {R_sq_train:>15.4f} {R_sq_test:>15.4f} {R_sq_train - R_sq_test:>12.4f}")
print(f"{'RMSE':<20} ${RMSE_train:>14,.0f} ${RMSE_test:>14,.0f} ${RMSE_test - RMSE_train:>11,.0f}")
print(f"{'MAE':<20} ${MAE_train:>14,.0f} ${MAE_test:>14,.0f} ${MAE_test - MAE_train:>11,.0f}")

overfit_ratio = (R_sq_train - R_sq_test) / R_sq_train if R_sq_train > 0 else 0
print(f"\nOverfitting assessment:")
if overfit_ratio < 0.02:
    print(f"  Minimal overfitting (R² gap = {R_sq_train - R_sq_test:.4f})")
    print(f"  The model generalises well to unseen data.")
elif overfit_ratio < 0.05:
    print(f"  Moderate overfitting (R² gap = {R_sq_train - R_sq_test:.4f})")
    print(f"  Consider regularisation or removing weak predictors.")
else:
    print(f"  Significant overfitting (R² gap = {R_sq_train - R_sq_test:.4f})")
    print(f"  The model memorises training noise. Reduce complexity or add data.")
# INTERPRETATION: With n > 100K transactions and only ~15 predictors, we expect
# minimal overfitting. The train/test gap in R² is the honest measure of
# out-of-sample model quality — the number you'd report to a business stakeholder.

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert R_sq_test >= 0, "Test R² must be non-negative for a reasonable model"
assert RMSE_test > 0, "Test RMSE must be positive"
assert R_sq_train >= R_sq_test - 0.1, "Train R² should be within 10pp of test R² (no extreme overfit)"
print("\n✓ Checkpoint 6 passed — cross-validation completed\n")


# ══════════════════════════════════════════════════════════════════════
# Visualise with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# -- Plot 1: Model comparison metrics --
model_comparison = {
    "Simple Model": {
        "R_squared": R_sq_simple,
        "Adj_R_squared": R_sq_adj_simple,
        "num_features": float(len(all_feature_cols)),
    },
    "Enriched Model": {
        "R_squared": R_sq_e,
        "Adj_R_squared": R_sq_adj_e,
        "num_features": float(len(enriched_feature_cols)),
    },
}
fig_models = viz.metric_comparison(model_comparison)
fig_models.update_layout(title="Simple vs Enriched Model: R² and Adjusted R²")
fig_models.write_html("ex5_model_comparison.html")
print(f"\nSaved: ex5_model_comparison.html")

# -- Plot 2: Train vs Test performance --
cv_results = {
    "Training Set": {
        "R_squared": R_sq_train,
        "RMSE": RMSE_train,
        "MAE": MAE_train,
    },
    "Test Set": {
        "R_squared": R_sq_test,
        "RMSE": RMSE_test,
        "MAE": MAE_test,
    },
}
fig_cv = viz.metric_comparison(cv_results)
fig_cv.update_layout(title="Train vs Test: Cross-Validation Performance")
fig_cv.write_html("ex5_cross_validation.html")
print("Saved: ex5_cross_validation.html")

# -- Plot 3: Coefficient significance (absolute t-statistics) --
coeff_significance = {}
for name, t_val, p_val in zip(enriched_coeff_names[1:], t_stats_e[1:], p_values_e[1:]):
    coeff_significance[name] = {
        "abs_t_statistic": abs(float(t_val)),
        "neg_log10_p": -np.log10(max(float(p_val), 1e-300)),
    }
fig_sig = viz.metric_comparison(coeff_significance)
fig_sig.update_layout(title="Feature Significance: |t-statistic| and -log10(p)")
fig_sig.write_html("ex5_coefficient_significance.html")
print("Saved: ex5_coefficient_significance.html")

# -- Plot 4: Residual analysis --
residuals_test = y_test - y_hat_test
fig_resid = viz.scatter(
    pl.DataFrame({
        "fitted_values": y_hat_test,
        "residuals": residuals_test,
    }),
    x="fitted_values",
    y="residuals",
)
fig_resid.update_layout(
    title="Residuals vs Fitted Values (Test Set)",
    xaxis_title="Fitted Values ($)",
    yaxis_title="Residuals ($)",
)
fig_resid.write_html("ex5_residuals_vs_fitted.html")
print("Saved: ex5_residuals_vs_fitted.html")

print(f"\n=== Summary ===")
print(f"Built OLS regression from scratch using matrix algebra: β = (X'X)⁻¹X'y")
print(f"Interpreted {len(enriched_feature_cols)} coefficients (numeric, polynomial, interaction, categorical)")
print(f"Tested significance via t-statistics (H₀: β = 0)")
print(f"Model explains {R_sq_e:.1%} of price variance (adjusted R² = {R_sq_adj_e:.4f})")
print(f"F-test confirms the model is significantly better than intercept-only")
print(f"Cross-validation: train R² = {R_sq_train:.4f}, test R² = {R_sq_test:.4f}")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print(f"""
  ✓ OLS closed form: β = (X'X)⁻¹X'y via np.linalg.solve
  ✓ Ceteris paribus: each coefficient isolates one predictor's effect
  ✓ Dummy encoding: k categories → k-1 dummies + base category
  ✓ t-statistic: β/SE(β) tests H₀: β=0 — same math as Exercise 3
  ✓ SE from Hessian: Var(β̂) = σ̂²(X'X)⁻¹, SE = sqrt of diagonal
  ✓ R²: proportion of variance explained (ranges 0→1)
  ✓ Adjusted R²: penalises for predictors — can decrease with noise features
  ✓ F-statistic: tests the whole model vs intercept-only
  ✓ Polynomial/interaction: non-linear effects within a linear framework
  ✓ Partial F-test: nested model comparison (enriched vs simple)
  ✓ Train/test split: honest out-of-sample estimate of R² and RMSE

  NEXT: In Exercise 6 you'll move from continuous outcomes to binary
  classification. Logistic regression replaces OLS with MLE (Bernoulli
  likelihood), uses the sigmoid instead of a linear prediction, and
  interprets coefficients as odds ratios rather than dollar amounts.
  You'll also run one-way ANOVA with Tukey HSD post-hoc tests.
""")

print("\n✓ Exercise 5 complete — linear regression from scratch with full inference")
