# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 2.3: Lasso (L1) and ElasticNet
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Fit Lasso at many α values and read the sparsity pattern
#   - Explain L1 geometry: diamond constraint, corner solutions, exact zeros
#   - Use Lasso as built-in feature selection
#   - Blend L1 and L2 with ElasticNet for correlated features
#   - Draw the regularisation path and spot where L1 drops features
#   - Apply sparse selection to an insurance fraud scorecard
#
# PREREQUISITES:
#   - 02_ridge_regression.py (L2 geometry and Bayesian view)
#
# ESTIMATED TIME: ~40 minutes
#
# TASKS (5-phase R10):
#   1. Theory — the L1 diamond and why it produces zeros
#   2. Build — Lasso + ElasticNet fits across α and l1_ratio
#   3. Train — sparsity trajectory (α chosen by CV) + ElasticNet sweep
#   4. Visualise — coefficient paths and non-zero count vs α
#   5. Apply — feature selection for an insurer's fraud scorecard
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score

from shared.mlfp03.ex_2 import (
    LASSO_ALPHAS,
    SEED,
    load_credit_data,
    print_header,
    save_html_plot,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — L1 Regularisation
# ════════════════════════════════════════════════════════════════════════
# Lasso replaces Ridge's squared-norm penalty with an absolute-value
# penalty. In sklearn's scaling:
#
#     min_β  (1 / 2n) · ||y - Xβ||²  +  α · ||β||₁
#                                       ─────────
#                                       Sum of |β_i|
#
# The 1/(2n) in front of the squared error is why Lasso's α lives on a
# much smaller scale than Ridge's — this file uses its own LASSO_ALPHAS.
#
# GEOMETRY: ||β||₁ ≤ c traces a DIAMOND (a square rotated 45°). The
# MSE level sets are ellipses — they tend to first touch the diamond
# at a CORNER, and the corners of the diamond are precisely the points
# where some coordinate is zero. Result: Lasso zeroes out features
# exactly, performing built-in feature selection.
#
# NO CLOSED FORM: Because |x| is not differentiable at zero, there's no
# (X'X + αI)⁻¹ formula. sklearn uses coordinate descent instead, which
# is why you'll see a generous max_iter.
#
# BAYESIAN VIEW: L1 ⇔ Laplace prior P(β) ∝ exp(-|β|/b). The sharp peak
# at zero is what causes coordinates to snap to exactly zero in the MAP
# estimate.
#
# ELASTICNET: Sometimes you want "some sparsity" AND "keep correlated
# features together". sklearn's ElasticNet objective is
#
#     (1 / 2n)·||y - Xβ||²  +  α·r·||β||₁  +  ½·α·(1 - r)·||β||²
#
# where r = l1_ratio ∈ [0,1]: r=1 is Lasso, r→0 approaches Ridge. (In the
# lecture notation "α·L1 + (1-α)·L2", the lecture's α is sklearn's
# l1_ratio and the lecture's λ is sklearn's alpha.)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD Lasso + ElasticNet fits
# ════════════════════════════════════════════════════════════════════════

print_header("Lasso & ElasticNet on Singapore Credit Data")

X_train, y_train, X_test, y_test, feature_names = load_credit_data()
n_features = len(feature_names)
print(f"Train: {X_train.shape}  Test: {X_test.shape}  Features: {n_features}")
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN Lasso across the α sweep; choose α by CV
# ════════════════════════════════════════════════════════════════════════
# Watch the non-zero column: as α grows, more coefficients are driven
# to exactly zero. That's L1 doing feature selection for you.

lasso_results: dict[float, dict[str, object]] = {}
for alpha in LASSO_ALPHAS:
    lasso = Lasso(alpha=alpha, max_iter=50_000)
    lasso.fit(X_train, y_train)
    cv_mse = -cross_val_score(
        Lasso(alpha=alpha, max_iter=50_000),
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
    ).mean()
    lasso_results[alpha] = {
        "train_mse": float(mean_squared_error(y_train, lasso.predict(X_train))),
        "cv_mse": float(cv_mse),
        "test_mse": float(mean_squared_error(y_test, lasso.predict(X_test))),
        "l1_norm": float(np.sum(np.abs(lasso.coef_))),
        "n_nonzero": int(np.sum(np.abs(lasso.coef_) > 1e-10)),
        "coef": lasso.coef_.copy(),
    }

print(
    f"\n{'alpha':>8} {'train MSE':>10} {'CV MSE':>8} {'test MSE':>9} "
    f"{'||β||₁':>8} {'non-zero':>9}"
)
print("-" * 58)
for alpha, r in lasso_results.items():
    print(
        f"{alpha:>8.3f} {r['train_mse']:>10.4f} {r['cv_mse']:>8.4f} "
        f"{r['test_mse']:>9.4f} {r['l1_norm']:>8.4f} {r['n_nonzero']:>6}/{n_features}"
    )

best_alpha_lasso = min(lasso_results, key=lambda a: lasso_results[a]["cv_mse"])
best_lasso_row = lasso_results[best_alpha_lasso]
print(
    f"\nCV-chosen Lasso α = {best_alpha_lasso}  "
    f"(test MSE = {best_lasso_row['test_mse']:.4f}, "
    f"keeps {best_lasso_row['n_nonzero']} of {n_features} features)"
)

# Show the surviving features at the CV-chosen α
kept = sorted(
    [
        (name, coef)
        for name, coef in zip(feature_names, best_lasso_row["coef"])
        if abs(coef) > 1e-10
    ],
    key=lambda x: abs(x[1]),
    reverse=True,
)
print(f"\nLasso-selected features ({len(kept)}):")
for name, coef in kept[:10]:
    print(f"  {name:<34} β = {coef:+.4f}")


# ── Checkpoint 1 ───────────────────────────────────────────────────────
smallest, largest = LASSO_ALPHAS[0], LASSO_ALPHAS[-1]
assert (
    lasso_results[largest]["n_nonzero"] < lasso_results[smallest]["n_nonzero"]
), "Higher-α Lasso must keep FEWER coefficients"
assert best_alpha_lasso in LASSO_ALPHAS, "Best α must come from the sweep"
print("\n[ok] Checkpoint 1 passed — Lasso sparsity increases with α")
print(
    f"  Non-zero coefficients fall from {lasso_results[smallest]['n_nonzero']} "
    f"(α={smallest}) to {lasso_results[largest]['n_nonzero']} (α={largest})."
)
# INTERPRETATION: Compare Lasso's surviving feature list against what
# a domain expert would pick. Strong signals survive the L1 penalty
# longest; weak or redundant features are zeroed first.


# ════════════════════════════════════════════════════════════════════════
# TASK 3b — TRAIN ElasticNet across l1_ratio
# ════════════════════════════════════════════════════════════════════════
# At a fixed α we vary the mix between L1 and L2. Low l1_ratio behaves
# like Ridge (few exact zeros); l1_ratio near 1 behaves like Lasso.

EN_ALPHA = 0.1
print_header(f"ElasticNet — Mixing L1 and L2 (α fixed at {EN_ALPHA})")
en_results: dict[float, dict[str, object]] = {}
for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    en = ElasticNet(alpha=EN_ALPHA, l1_ratio=l1_ratio, max_iter=50_000)
    en.fit(X_train, y_train)
    en_results[l1_ratio] = {
        "train_mse": float(mean_squared_error(y_train, en.predict(X_train))),
        "test_mse": float(mean_squared_error(y_test, en.predict(X_test))),
        "n_nonzero": int(np.sum(np.abs(en.coef_) > 1e-10)),
        "coef": en.coef_.copy(),
    }

print(f"\n{'l1_ratio':>10} {'train MSE':>12} {'test MSE':>12} {'non-zero':>9}")
print("-" * 47)
for l1_ratio, r in en_results.items():
    print(
        f"{l1_ratio:>10.1f} {r['train_mse']:>12.4f} {r['test_mse']:>12.4f} "
        f"{r['n_nonzero']:>9}"
    )

# Correlated-group test: the features most correlated with `age` in this
# sample (age, years employed, months employed, credit history length all
# move together). Pure Lasso tends to keep ONE of a correlated group;
# ElasticNet with a low l1_ratio tends to keep the group with shared weights.
corr = np.nan_to_num(np.corrcoef(X_train, rowvar=False))
anchor = feature_names.index("age")
group = [j for j in range(n_features) if abs(corr[anchor, j]) > 0.9]
lasso_same_alpha = Lasso(alpha=EN_ALPHA, max_iter=50_000).fit(X_train, y_train)
lasso_group_kept = int(np.sum(np.abs(lasso_same_alpha.coef_[group]) > 1e-10))
en_group_kept = int(np.sum(np.abs(en_results[0.1]["coef"][group]) > 1e-10))
print(f"\nCorrelated group (|r| > 0.9 with age): {[feature_names[j] for j in group]}")
print(f"  Lasso (α={EN_ALPHA})                 keeps {lasso_group_kept} of {len(group)}")
print(f"  ElasticNet (α={EN_ALPHA}, l1_ratio=0.1) keeps {en_group_kept} of {len(group)}")


# ── Checkpoint 2 ───────────────────────────────────────────────────────
assert (
    en_results[0.9]["n_nonzero"] <= en_results[0.1]["n_nonzero"]
), "l1_ratio=0.9 should keep no more coefficients than l1_ratio=0.1"
assert len(group) >= 2, "Expected at least one correlated pair with age"
print("\n[ok] Checkpoint 2 passed — l1_ratio drives sparsity")
print(
    "  "
    + (
        "ElasticNet kept more of the correlated group than Lasso — the "
        "grouping effect."
        if en_group_kept > lasso_group_kept
        else "On this sample ElasticNet did not keep more of the group than "
        "Lasso — try a lower l1_ratio."
    )
)
# INTERPRETATION: With CORRELATED features, pure Lasso picks one member of
# the group somewhat arbitrarily (a different resample may pick a
# different one). The L2 part of ElasticNet spreads weight across the
# group, which gives more stable feature lists between refits.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the regularisation path
# ════════════════════════════════════════════════════════════════════════
# Two figures, both with the REAL α values on a log x-axis:
#   1. Every coefficient against α — lines hitting zero one by one
#   2. Number of non-zero coefficients against α — a staircase to zero

print_header("Regularisation Path Visualisation")

coef_matrix = np.array([lasso_results[a]["coef"] for a in LASSO_ALPHAS])
fig_path = go.Figure()
for j, name in enumerate(feature_names):
    fig_path.add_trace(
        go.Scatter(x=LASSO_ALPHAS, y=coef_matrix[:, j], mode="lines+markers", name=name)
    )
fig_path.add_vline(x=best_alpha_lasso, line_dash="dot", annotation_text="CV-chosen α")
fig_path.update_layout(
    title="Lasso coefficient paths — features drop to exactly zero",
    xaxis_title="Regularisation strength α (log scale)",
    yaxis_title="Coefficient (standardised units)",
    xaxis_type="log",
)
path_coef = save_html_plot(fig_path, "lasso_coef_path.html")

fig_sparse = go.Figure(
    go.Scatter(
        x=LASSO_ALPHAS,
        y=[lasso_results[a]["n_nonzero"] for a in LASSO_ALPHAS],
        mode="lines+markers",
        line_shape="hv",
        name="non-zero coefficients",
    )
)
fig_sparse.update_layout(
    title="Lasso: number of features kept vs α",
    xaxis_title="Regularisation strength α (log scale)",
    yaxis_title="Non-zero coefficients",
    xaxis_type="log",
)
path_sparse = save_html_plot(fig_sparse, "lasso_sparsity_path.html")

print(f"\nSaved: {path_coef.name}")
print(f"Saved: {path_sparse.name}")


# ── Checkpoint 3 ───────────────────────────────────────────────────────
assert coef_matrix.shape == (
    len(LASSO_ALPHAS),
    n_features,
), "Coefficient matrix should be (n_alphas, n_features)"
print("\n[ok] Checkpoint 3 passed — regularisation path matrix shape correct")
# INTERPRETATION: Each step down in the staircase is a feature being
# zeroed out. This is the L1 diamond geometry in action — corners of the
# diamond are points where some coordinate is exactly zero.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: feature selection for an insurer's fraud scorecard
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore insurer's fraud team maintains a
# claim-scoring model with ~240 candidate features. Its internal model
# governance requires a sign-off memo explaining every feature that goes
# into production.
#
# WHY LASSO IS THE RIGHT TOOL:
#   - Lasso picks a small, defensible subset automatically instead of the
#     team running 240 univariate tests.
#   - Lasso's exact zeros mean the production scorecard can DROP features
#     entirely from the data pipeline — fewer upstream dependencies,
#     fewer broken-ingest incidents.
#   - Sign-off memos scale linearly with feature count.
#
# WHY ELASTICNET FOR A PRODUCTION REFRESH:
#   - Correlated features (e.g. "claim amount" vs "claim amount relative
#     to sum insured") make pure Lasso's choice unstable between refits.
#     ElasticNet keeps correlated groups together with shared smaller
#     weights, so the memo set is more stable year on year.
#
# Rather than quote invented savings, size the decision with OUR model:
# how much predictive accuracy does sparsity cost, per feature dropped?

print_header("Sparse Scorecard Trade-off — measured on the credit data")
dense_alpha = LASSO_ALPHAS[0]
dense = lasso_results[dense_alpha]
print(
    f"""
Model                         | Features | Test MSE
------------------------------|----------|---------
Near-OLS Lasso (α={dense_alpha:<5})       | {dense['n_nonzero']:>8} | {dense['test_mse']:.4f}
CV-chosen Lasso (α={best_alpha_lasso:<5})      | {best_lasso_row['n_nonzero']:>8} | {best_lasso_row['test_mse']:.4f}

The CV-chosen model uses {best_lasso_row['n_nonzero']} of {dense['n_nonzero']} features and its test MSE
is {best_lasso_row['test_mse'] - dense['test_mse']:+.4f} relative to the near-OLS model
(negative = better). Each feature removed is one fewer memo to write and
one fewer data feed to maintain. Plug in your own cost per memo to price
the governance saving.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print(
    """
======================================================================
  WHAT YOU'VE MASTERED
======================================================================

  [x] Lasso objective: (1/2n)·||y-Xβ||² + α·||β||₁
  [x] Diamond geometry → corner solutions → exact zeros
  [x] Laplace prior as the Bayesian twin of L1
  [x] ElasticNet (α, l1_ratio) for correlated-feature stability
  [x] Regularisation path as a diagnostic visual
  [x] Using Lasso for governance-friendly feature selection

  KEY INSIGHT: Use Lasso when you believe "only a handful of features
  matter". Use Ridge when you believe "everything matters a little".
  Use ElasticNet when you're not sure AND correlated groups exist.

  NEXT: 04_cross_validation.py — we've been choosing α with a single CV
  loop. Now we formalise it with nested CV, stratified CV, time-series CV,
  and GroupKFold.
"""
)
