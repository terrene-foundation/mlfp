# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.4: Model Enrichment and Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Extend models with polynomial and interaction terms
#   - Apply dummy variable encoding with a base category
#   - Cross-validate with train/test split and compute out-of-sample R-squared
#   - Compare model complexity: simple vs enriched vs categorical
#
# PREREQUISITES: Exercise 5.1-5.3 (OLS, diagnostics, WLS)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load data and fit baseline OLS
#   2. Add polynomial and interaction terms
#   3. Dummy variable encoding for flat type
#   4. Train/test split: out-of-sample evaluation
#   5. Model comparison and business interpretation
#
# THEORY:
#   Linear regression is "linear in parameters" — you can add x^2,
#   x1*x2, or dummy variables and still use OLS. The question is
#   whether the added complexity improves out-of-sample prediction
#   or just overfits the training data.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go
from scipy import stats

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    TARGET,
    BASE_FLAT_TYPE,
    OUTPUT_DIR,
    load_hdb_clean,
    build_design_matrix,
    fit_ols,
    format_p_value,
    print_coef_table,
    save_actual_vs_predicted,
    save_residual_diagnostics,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Data and Fit Baseline
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.4: Model Enrichment and Evaluation")
print("=" * 70)

hdb_clean = load_hdb_clean()
X, y, feature_names = build_design_matrix(hdb_clean)
n_obs, k = X.shape
X_raw = X[:, 1:]  # Without intercept

fit_baseline = fit_ols(X, y)
r_squared = fit_baseline["R2"]
adj_r_squared = fit_baseline["adj_R2"]
SSR = fit_baseline["SSR"]
SST = fit_baseline["SST"]

print(f"\n  Baseline OLS: R-squared={r_squared:.6f}, Adj R-squared={adj_r_squared:.6f}")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Beyond Straight Lines
# ════════════════════════════════════════════════════════════════════════
# "Linear regression" is linear in PARAMETERS, not in features.
# You can transform features any way you like — square them, multiply
# them, take logs — and still use the normal equation. The key
# question: does the extra complexity capture real patterns or just
# noise?
#
# Analogy: Imagine fitting a price model for HDB flats. A straight
# line says "each extra sqm adds $X." But reality might be: small
# flats have a premium per sqm (scarcity), large flats have a
# premium per sqm (luxury). A quadratic term captures this curve.
# An interaction term says "the storey premium is bigger for large
# flats" — a penthouse effect.
#
# WHY THIS MATTERS: A property developer evaluating whether to build
# 100 small flats or 50 large ones needs to know whether the
# price-per-sqm curve is linear, convex, or concave. The polynomial
# and interaction terms answer this directly.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Polynomial and Interaction Terms
# ════════════════════════════════════════════════════════════════════════
# Non-linearity: floor_area^2 captures diminishing/increasing returns
# Interactions: storey * area captures "premium for high-floor large flats"

print(f"\n=== Polynomial and Interaction Terms ===")

area = X_raw[:, 0]
storey = X_raw[:, 1]
lease = X_raw[:, 2]

# TODO: Build enriched design matrix with polynomial and interaction terms
# Hint: np.column_stack with:
#   ones, area, storey, lease, area**2, storey*area, lease*area
X_enriched = ____

enriched_names = [
    "intercept",
    "area",
    "storey",
    "lease",
    "area_sq",
    "storey_x_area",
    "lease_x_area",
]
k_enriched = X_enriched.shape[1]

# TODO: Fit enriched model using np.linalg.lstsq
# Hint: np.linalg.lstsq(X_enriched, y, rcond=None)[0]
beta_enriched = ____

y_hat_enriched = X_enriched @ beta_enriched
resid_enriched = y - y_hat_enriched
ssr_enriched = float(np.sum(resid_enriched**2))
r2_enriched = 1 - ssr_enriched / SST
adj_r2_enriched = 1 - (1 - r2_enriched) * (n_obs - 1) / (n_obs - k_enriched)

# F-test: enriched vs simple model
f_improvement = ((SSR - ssr_enriched) / (k_enriched - k)) / (
    ssr_enriched / (n_obs - k_enriched)
)
f_p_improvement = stats.f.sf(f_improvement, dfn=k_enriched - k, dfd=n_obs - k_enriched)

print(f"{'Feature':<20} {'Coefficient':>14}")
print("-" * 38)
for name, coef in zip(enriched_names, beta_enriched):
    print(f"{name:<20} {coef:>14,.4f}")

print(f"\nSimple model:   R-squared={r_squared:.6f}, Adj R-squared={adj_r_squared:.6f}")
print(
    f"Enriched model: R-squared={r2_enriched:.6f}, Adj R-squared={adj_r2_enriched:.6f}"
)
print(
    f"F-test (enriched vs simple): F={f_improvement:.2f}, "
    f"p {format_p_value(f_p_improvement)}"
)
print(
    f"Enriched model is "
    f"{'significantly better' if f_p_improvement < 0.05 else 'NOT significantly better'}"
)

# INTERPRETATION: The area^2 term captures non-linearity — perhaps
# price per sqm increases for very large flats (premium penthouses)
# or decreases (diminishing returns). The interaction storey*area
# captures whether the storey premium is larger for bigger flats.

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert (
    r2_enriched >= r_squared - 0.001
), "Adding features should not decrease R-squared substantially"
print("\n--- Checkpoint 2 passed --- enriched model built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Dummy Variable Encoding
# ════════════════════════════════════════════════════════════════════════
# Categorical variables -> binary dummies. Drop one category to avoid
# the dummy variable trap (perfect multicollinearity with intercept).

print(f"\n=== Dummy Variable Encoding ===")

flat_types_in_data = sorted(hdb_clean["flat_type"].unique().to_list())
print(f"Flat types: {flat_types_in_data}")

# BASE_FLAT_TYPE (3 ROOM) is the base: a familiar reference point that
# makes each coefficient read as "premium over a 3-room flat". It is NOT
# the most common type (4 ROOM is) — any category works as the base; the
# choice only changes what the coefficients are measured against.
type_counts = hdb_clean["flat_type"].value_counts().sort("count", descending=True)
print(f"Most common type: {type_counts['flat_type'][0]} ({type_counts['count'][0]:,} sales)")
# TODO: Create list of dummy categories (all flat types except base)
# Hint: [ft for ft in flat_types_in_data if ft != BASE_FLAT_TYPE]
dummy_categories = ____

# Build dummy columns
dummy_arrays = []
for ft in dummy_categories:
    dummy = (hdb_clean["flat_type"].to_numpy() == ft).astype(np.float64)
    dummy_arrays.append(dummy)

# TODO: Build design matrix with dummies
# Hint: np.column_stack with ones, X_raw, and np.column_stack(dummy_arrays)
X_with_dummies = ____

dummy_names = (
    ["intercept"]
    + list(NUMERIC_FEATURES)
    + [f"flat_{ft.replace(' ', '_')}" for ft in dummy_categories]
)
k_dummy = X_with_dummies.shape[1]

# Fit model with dummies
beta_dummy = np.linalg.lstsq(X_with_dummies, y, rcond=None)[0]
y_hat_dummy = X_with_dummies @ beta_dummy
ssr_dummy = float(np.sum((y - y_hat_dummy) ** 2))
r2_dummy = 1 - ssr_dummy / SST
adj_r2_dummy = 1 - (1 - r2_dummy) * (n_obs - 1) / (n_obs - k_dummy)

print(f"\nBase category: {BASE_FLAT_TYPE}")
print(f"\n{'Feature':<30} {'Coefficient':>14}")
print("-" * 48)
for name, coef in zip(dummy_names, beta_dummy):
    print(f"{name:<30} {coef:>14,.0f}")

print(
    f"\nModel with dummies: R-squared={r2_dummy:.6f}, Adj R-squared={adj_r2_dummy:.6f}"
)
print(f"Improvement over simple: Delta R-squared={r2_dummy - r_squared:+.6f}")

# INTERPRETATION: Each dummy coefficient represents the price premium
# (or discount) relative to the base category (3 ROOM). For example,
# if the 5 ROOM coefficient is +$150K, then 5-room flats sell for
# $150K more than 3-room flats, all else equal.

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert r2_dummy > r_squared, "Adding flat type should improve R-squared"
assert len(dummy_categories) == len(flat_types_in_data) - 1, "Should drop one category"
print("\n--- Checkpoint 3 passed --- dummy encoding completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train/Test Split: Out-of-Sample Evaluation
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Train/Test Split ===")

rng = np.random.default_rng(seed=42)
n_total_obs = X_with_dummies.shape[0]
indices = rng.permutation(n_total_obs)
split_point = int(0.8 * n_total_obs)

train_idx = indices[:split_point]
test_idx = indices[split_point:]

X_train = X_with_dummies[train_idx]
y_train = y[train_idx]
X_test = X_with_dummies[test_idx]
y_test = y[test_idx]

# TODO: Fit OLS on the training set
# Hint: np.linalg.lstsq(X_train, y_train, rcond=None)[0]
beta_train = ____

# Evaluate on both
y_train_pred = X_train @ beta_train
y_test_pred = X_test @ beta_train

r2_train = 1 - float(np.sum((y_train - y_train_pred) ** 2)) / float(
    np.sum((y_train - y_train.mean()) ** 2)
)
r2_test = 1 - float(np.sum((y_test - y_test_pred) ** 2)) / float(
    np.sum((y_test - y_test.mean()) ** 2)
)
rmse_train = float(np.sqrt(np.mean((y_train - y_train_pred) ** 2)))
rmse_test = float(np.sqrt(np.mean((y_test - y_test_pred) ** 2)))
mae_train = float(np.mean(np.abs(y_train - y_train_pred)))
mae_test = float(np.mean(np.abs(y_test - y_test_pred)))

print(f"Train: n={len(train_idx):,}")
print(f"Test:  n={len(test_idx):,}")
print(f"\n{'Metric':<12} {'Train':>14} {'Test':>14} {'Delta':>10}")
print("-" * 54)
print(
    f"{'R-squared':<12} {r2_train:>14.6f} {r2_test:>14.6f} {r2_test - r2_train:>+10.6f}"
)
print(
    f"{'RMSE':<12} ${rmse_train:>12,.0f} ${rmse_test:>12,.0f} "
    f"${rmse_test - rmse_train:>+8,.0f}"
)
print(
    f"{'MAE':<12} ${mae_train:>12,.0f} ${mae_test:>12,.0f} "
    f"${mae_test - mae_train:>+8,.0f}"
)

gap = abs(r2_train - r2_test)
print(f"\nTrain-test R-squared gap: {gap:.4f}")
if gap < 0.02:
    print("Minimal overfitting — model generalises well")
elif gap < 0.05:
    print("Slight overfitting — consider regularisation")
else:
    print("OVERFITTING — model is too complex for the data")

# INTERPRETATION: If train R-squared >> test R-squared, the model
# memorises training data instead of learning generalisable patterns.
# The train-test gap tells you whether your model complexity is
# appropriate.

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert r2_test > 0, "Out-of-sample R-squared must be positive"
assert (
    r2_train >= r2_test - 0.05
), "Train R-squared should be >= test R-squared (approx)"
print("\n--- Checkpoint 4 passed --- out-of-sample evaluation completed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Model Comparison Summary
# ════════════════════════════════════════════════════════════════════════

print(f"\n=== Model Comparison Summary ===")
print(f"{'Model':<30} {'R-sq':>10} {'Adj R-sq':>10} {'k':>4}")
print("-" * 58)
print(f"{'Simple (3 features)':<30} {r_squared:>10.6f} {adj_r_squared:>10.6f} {k:>4}")
print(
    f"{'Enriched (poly+interact)':<30} {r2_enriched:>10.6f} {adj_r2_enriched:>10.6f} "
    f"{k_enriched:>4}"
)
print(
    f"{'With flat type dummies':<30} {r2_dummy:>10.6f} {adj_r2_dummy:>10.6f} {k_dummy:>4}"
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
print("\n--- Checkpoint 5 passed --- model comparison complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Test Set Actual vs Predicted
# ════════════════════════════════════════════════════════════════════════

path = save_actual_vs_predicted(
    y_test,
    y_test_pred,
    title="Full Model: Actual vs Predicted (Test Set)",
    filename="04_test_actual_vs_predicted.html",
)
print(f"Saved: {path}")

# --- Polynomial fit curve: price vs floor area with linear + quadratic ---
# Both curves hold storey and lease at their medians, so the only
# difference between them is the shape in area — a like-for-like view.
area_sorted_idx = np.argsort(area)
area_sorted = area[area_sorted_idx]
med_storey = float(np.median(storey))
med_lease = float(np.median(lease))
# Linear model: intercept + beta_area * area + storey/lease at medians
beta_baseline = fit_baseline["beta"]
# TODO: Linear-model curve with storey and lease held at their medians
# Hint: beta_baseline[0] + beta_baseline[1] * area_sorted + (storey and
#       lease coefficients times med_storey / med_lease)
y_linear = ____
# Enriched model at the same storey/lease medians
y_poly = (
    beta_enriched[0]
    + beta_enriched[1] * area_sorted
    + beta_enriched[2] * med_storey
    + beta_enriched[3] * med_lease
    + beta_enriched[4] * area_sorted**2
    + beta_enriched[5] * med_storey * area_sorted
    + beta_enriched[6] * med_lease * area_sorted
)

sample = min(3000, n_obs)
fig_poly = go.Figure()
fig_poly.add_trace(
    go.Scatter(
        x=area[:sample].tolist(),
        y=y[:sample].tolist(),
        mode="markers",
        marker={"size": 2, "opacity": 0.2, "color": "#94A3B8"},
        name="Data",
    )
)
fig_poly.add_trace(
    go.Scatter(
        x=area_sorted.tolist(),
        y=y_linear.tolist(),
        mode="lines",
        line={"color": "#2563EB", "width": 2},
        name="Linear",
    )
)
fig_poly.add_trace(
    go.Scatter(
        x=area_sorted.tolist(),
        y=y_poly.tolist(),
        mode="lines",
        line={"color": "#DC2626", "width": 2, "dash": "dash"},
        name="Polynomial + Interactions",
    )
)
fig_poly.update_layout(
    title="Price vs Floor Area — Does a Curve Fit Better Than a Line?",
    xaxis_title="Floor Area (sqm)",
    yaxis_title="Predicted Price (SGD)",
    height=450,
)
path_poly = OUTPUT_DIR / "04_polynomial_fit_curves.html"
fig_poly.write_html(str(path_poly))
print(f"Saved: {path_poly}")

# --- Train/test comparison bar chart: R-squared, RMSE, MAE ---
metrics = ["R-squared", "RMSE ($K)", "MAE ($K)"]
train_vals = [r2_train, rmse_train / 1000, mae_train / 1000]
# TODO: Test-set values in the same order and units as train_vals
test_vals = ____

fig_tt = go.Figure()
fig_tt.add_trace(
    go.Bar(
        x=metrics,
        y=train_vals,
        name="Train",
        marker_color="#2563EB",
    )
)
fig_tt.add_trace(
    go.Bar(
        x=metrics,
        y=test_vals,
        name="Test",
        marker_color="#DC2626",
    )
)
fig_tt.update_layout(
    title="Train vs Test Performance — Is the Model Overfitting?",
    yaxis_title="Value",
    barmode="group",
    height=400,
)
path_tt = OUTPUT_DIR / "04_train_test_comparison.html"
fig_tt.write_html(str(path_tt))
print(f"Saved: {path_tt}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Property Developer Unit-Mix Decision
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore developer is choosing the unit mix for a site,
# using HDB resale prices as a demand proxy. Option A: 100 small units
# (67 sqm, 3-room-like). Option B: 50 large units (110 sqm, 5-room-like).
# Construction cost is similar per sqm.
#
# The dummy-encoded model prices each unit type at the median storey
# and lease: base price + area effect + the flat-type premium. The
# out-of-sample RMSE says how far a single unit's price can miss.

dummy_index = {name: i for i, name in enumerate(dummy_names)}


def price_unit(area_sqm: float, flat_type: str) -> float:
    """Predicted price from the dummy model at median storey and lease."""
    x_new = np.zeros(k_dummy)
    x_new[0] = 1.0
    x_new[1:4] = [area_sqm, med_storey, med_lease]
    dummy_name = f"flat_{flat_type.replace(' ', '_')}"
    if dummy_name in dummy_index:  # the base category has no dummy
        x_new[dummy_index[dummy_name]] = 1.0
    # TODO: Predicted price = design row times the dummy-model coefficients
    # Hint: matrix product of x_new and beta_dummy, as a float
    return float(____)


print(f"\n--- Business Application: Developer Unit-Mix Decision ---")
unit_a = price_unit(67.0, "3 ROOM")
unit_b = price_unit(110.0, "5 ROOM")
revenue_a = 100 * unit_a
revenue_b = 50 * unit_b
print(f"  5-ROOM premium over 3-ROOM at equal area: ${beta_dummy[dummy_index['flat_5_ROOM']]:,.0f}")
print(f"  Option A: 100 x ${unit_a:,.0f} = ${revenue_a:,.0f}  (6,700 sqm)")
print(f"  Option B:  50 x ${unit_b:,.0f} = ${revenue_b:,.0f}  (5,500 sqm)")
print(
    f"  Revenue per sqm built: A ${revenue_a / 6_700:,.0f}  vs  B ${revenue_b / 5_500:,.0f}"
)
better = "A (many small units)" if revenue_a / 6_700 > revenue_b / 5_500 else "B (fewer large units)"
print(f"  On this model, option {better} earns more per sqm built.")

print(f"\n  Model reliability:")
print(f"  Train-test R-squared gap: {gap:.4f}")
print(f"  Out-of-sample RMSE: ${rmse_test:,.0f} per unit")
print(
    f"  The model explains only {r2_test:.0%} of out-of-sample price variation —\n"
    f"  town, MRT access and condition are missing, so treat the comparison as\n"
    f"  a first screen, not a valuation."
)

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (5.4)")
print("=" * 70)
print(
    """
  - Polynomial terms (area^2) and interactions (storey x area)
  - F-test for nested model comparison (simple vs enriched)
  - Dummy encoding with base category to avoid the dummy trap
  - Train/test split: out-of-sample R-squared, RMSE, MAE
  - Model complexity trade-off: more features != better prediction
  - Business reasoning: pricing a unit-mix decision with dummy coefficients
"""
)

print("=" * 70)
print("  EXERCISE 5 COMPLETE — Linear Regression")
print("=" * 70)
print(
    """
  FULL EXERCISE SUMMARY:
  - 5.1: OLS from scratch, coefficient interpretation, significance
  - 5.2: Diagnostics — VIF, residual normality, Breusch-Pagan
  - 5.3: Weighted Least Squares for heteroscedastic data
  - 5.4: Model enrichment, dummy encoding, train/test evaluation

  NEXT: In Exercise 6 you'll build logistic regression for binary
  classification. You'll implement the sigmoid function, maximise
  the Bernoulli log-likelihood, interpret coefficients as odds ratios,
  and perform ANOVA for multi-group comparison.
"""
)
