# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.2: XGBoost on Singapore Credit Data
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Screen a dataset for target leakage BEFORE modelling (Lesson 3.1)
#   - Train an XGBoost classifier on a real imbalanced credit dataset
#   - Read XGBoost hyperparameters in terms of the theory from 4.1
#     (learning_rate ↔ η, max_depth ↔ tree size, reg_lambda ↔ λ)
#   - Use AUC-PR as the primary metric for 12%-positive data
#   - Extract and rank gain-based feature importances (and know which
#     "gain" XGBoost reports)
#   - Explain why XGBoost is the default choice for tabular data
#
# PREREQUISITES: Exercise 4.1 (boosting theory, split-gain formula).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — hyperparameters as theory dials
#   2. Build — leakage screen, then XGBoost with course-standard defaults
#   3. Train — fit on Singapore credit data, time the training
#   4. Visualise — feature-importance bar chart + top-15 table
#   5. Apply — a bank's credit-risk team sanity-checks what the model uses
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv

from shared.mlfp03.ex_4 import (
    CREDIT_NON_FEATURE_COLUMNS,
    OUTPUT_DIR,
    TARGET_COLUMN,
    evaluate_classifier,
    load_credit_data,
    make_xgboost,
    prepare_credit_split,
    print_metrics,
    screen_single_feature_leakage,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — XGBoost hyperparameters as theory dials
# ════════════════════════════════════════════════════════════════════════
# Every XGBoost hyperparameter corresponds directly to a lever in the
# theory from 4.1:
#
#     learning_rate (η)   → additive-model step size. Smaller = slower
#                           convergence, more robust. Production: 0.01-0.05.
#     max_depth           → tree size. Deeper trees capture interactions
#                           but overfit. Production: 4-8.
#     n_estimators        → number of rounds. Pair with early_stopping.
#     reg_lambda (λ)      → L2 on leaf weights. Shrinks leaf predictions
#                           toward zero when H (Hessian sum) is small.
#     gamma (γ)           → minimum gain to accept a split. Structural
#                           pruning: refuses to split small segments.
#     subsample           → fraction of rows used per tree (stochastic
#                           boosting). Adds variance reduction on top of
#                           the bias reduction boosting gives you.
#     colsample_bytree    → fraction of columns sampled per tree. Another
#                           stochastic regulariser.
#
# The XGBoost defaults (learning_rate=0.3, max_depth=6) are aggressive —
# they're tuned to win Kaggle competitions where speed of convergence
# matters. For credit scoring you typically move to learning_rate=0.05
# and rely on early stopping for the final round count.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the XGBoost classifier
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  XGBoost on Singapore Credit Scoring")
print("=" * 70)

# --- 2a. Leakage screen: what could a model "cheat" with? ---------------
# Before any model sees this data, score every numeric column ON ITS OWN
# against the target. Real credit features rarely exceed AUC ~0.75 alone;
# a column that separates defaulters almost perfectly is almost always
# recorded after the outcome.
raw = load_credit_data()
screen = screen_single_feature_leakage(raw)
print("\n  --- Single-feature AUC screen (top 8) ---")
print(screen.head(8))

flagged = screen.filter(pl.col("suspicious"))["feature"].to_list()
print(f"\n  Flagged as suspicious (AUC >= 0.95): {flagged}")
print(
    raw.group_by("future_default_indicator")
    .agg(
        pl.len().alias("rows"),
        pl.col(TARGET_COLUMN).mean().alias("default_rate"),
    )
    .sort("future_default_indicator")
)
# INTERPRETATION: `future_default_indicator` is ~0.93 default rate when 1
# and ~0.002 when 0 — it is the outcome in disguise, filled in AFTER the
# loan was observed. It is not available when a new application arrives,
# so prepare_credit_split() drops it, together with the row ID
# (CREDIT_NON_FEATURE_COLUMNS).

data = prepare_credit_split()
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
feature_names = data["feature_names"]

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  Features: {len(feature_names)}")
print(f"  Default rate (imbalance): {data['default_rate']:.2%}")
print(f"  Excluded before modelling: {list(CREDIT_NON_FEATURE_COLUMNS)}")

# ── Checkpoint 0 ────────────────────────────────────────────────────────
assert "future_default_indicator" in flagged, "The leakage screen must flag the leak"
assert not set(CREDIT_NON_FEATURE_COLUMNS) & set(feature_names), "Leak/ID columns must be excluded"
print("\n[ok] Checkpoint 0 passed — leak found by EDA and excluded from the features\n")

# Course-standard XGBoost (see shared.mlfp03.ex_4.make_xgboost)
model = make_xgboost(n_estimators=500, learning_rate=0.1, max_depth=6)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN on Singapore credit data
# ════════════════════════════════════════════════════════════════════════

print("\n  Training XGBoost (500 rounds, η=0.1, depth=6)...")
t0 = time.perf_counter()
model.fit(X_train, y_train)  # the test set is NOT shown to the model in any form
train_time = time.perf_counter() - t0

y_proba = model.predict_proba(X_test)[:, 1]
metrics = evaluate_classifier(y_test, y_proba)

print_metrics("XGBoost", metrics, train_time=train_time)


print(
    f"  AUC-PR {metrics['auc_pr']:.4f} vs random-ranking AUC-PR "
    f"{data['default_rate']:.4f} (= the default rate): "
    f"{metrics['auc_pr'] / data['default_rate']:.1f}x better than chance"
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert metrics["auc_roc"] > 0.7, "XGBoost should beat 0.7 AUC-ROC on credit data"
assert (
    metrics["auc_pr"] > 2 * data["default_rate"]
), "XGBoost AUC-PR should be at least twice the random baseline (the default rate)"
# INTERPRETATION: AUC-PR is the metric that matters here. A random
# ranking scores AUC-PR ≈ the default rate (~0.13) and AUC-ROC ≈ 0.5;
# AUC-ROC can look respectable while precision on the rare defaulters
# stays low. Without the leak column the honest numbers are modest —
# that is what real credit data looks like.
print("\n[ok] Checkpoint 1 passed — XGBoost trained and evaluated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE feature importance
# ════════════════════════════════════════════════════════════════════════
# XGBClassifier.feature_importances_ defaults to importance_type="gain":
# the AVERAGE split gain (the 4.1 formula) over every split that used the
# feature, normalised to sum to 1. "total_gain" (average × number of
# splits) is the alternative; a feature used rarely but decisively ranks
# higher on average gain than on total gain.

importances = model.feature_importances_
ranked = sorted(
    zip(feature_names, importances),
    key=lambda pair: pair[1],
    reverse=True,
)
top_15 = ranked[:15]

print("  --- Top-15 Features by XGBoost Average-Gain Importance ---")
print(f"  {'Rank':>4}  {'Feature':<30}  {'Avg gain':>10}")
print("  " + "─" * 50)
for rank, (name, importance) in enumerate(top_15, start=1):
    print(f"  {rank:>4}  {name:<30}  {importance:>10.4f}")

# Bar chart — saved as HTML so students can scroll/hover
names = [name for name, _ in top_15][::-1]
values = [float(v) for _, v in top_15][::-1]

fig = go.Figure(
    go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=values, colorscale="Blues"),
    )
)
fig.update_layout(
    title="XGBoost Feature Importance — Singapore Credit Default",
    xaxis_title="Average split gain per use (normalised to sum to 1)",
    yaxis_title="",
    height=520,
)
viz_path = OUTPUT_DIR / "ex4_02_xgboost_feature_importance.html"
fig.write_html(viz_path)
print(f"\n  Saved: {viz_path}")


top5_share = float(sum(v for _, v in ranked[:5]))
print(f"\n  Share of importance in the top 5 features: {top5_share:.1%}")
idx_1 = feature_names.index(ranked[0][0])
idx_2 = feature_names.index(ranked[1][0])
corr_12 = float(np.corrcoef(X_train[:, idx_1], X_train[:, idx_2])[0, 1])
print(
    f"  Correlation between #1 ({ranked[0][0]}) and #2 ({ranked[1][0]}): "
    f"{corr_12:+.3f}"
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(importances) == len(
    feature_names
), "importance vector must match feature count"
assert sum(importances) > 0, "at least one feature must have positive importance"
# INTERPRETATION: When two top features are almost perfectly correlated
# (near-duplicates such as the same quantity measured in months and in
# years), the trees pick either one at each split and the credit is
# SPLIT between them — neither ranking alone tells you how much the
# underlying quantity matters. Group near-duplicates before you read an
# importance chart, or use SHAP (Exercise 6).
print("\n[ok] Checkpoint 2 passed — feature importance extracted and visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Credit-Risk Team Sanity-Checks What The Model Uses
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank's credit-risk team
# must be able to explain automated declines. The MAS FEAT principles
# (Fairness, Ethics, Accountability, Transparency — non-binding guidance
# for AI in finance) call for exactly this kind of transparency.
#
# The importance ranking is the team's first, population-level check:
# "is the model looking at things a credit officer would recognise?"
# For an individual decline, they need per-applicant explanations (SHAP,
# Exercise 6). Two red flags to look for on every re-train:
#   1. A column that should not exist at decision time ranks at the top —
#      exactly what the leak column did before Checkpoint 0 removed it.
#   2. A protected attribute (gender, race) or an obvious proxy for one
#      ranks highly — that triggers a fairness review (Exercise 6.5).

protected = {"gender", "race", "nationality"}
top_10_names = [name for name, _ in ranked[:10]]
print("\n  Pre-release sanity check:")
print(f"    Top 3 features: {top_10_names[:3]}")
print(f"    Protected attributes in the top 10: {sorted(protected & set(top_10_names)) or 'none'}")
# A five-minute check like this before every release is cheap; shipping a
# model that relies on a leaked or protected column is not.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Found the planted leak with a single-feature AUC screen and kept it
      (and the row ID) out of every model
  [x] Trained XGBoost on real Singapore credit data (AUC-PR={metrics['auc_pr']:.4f})
  [x] Connected every hyperparameter back to the theory in 4.1
  [x] Used AUC-PR as the primary metric for 12%-positive imbalanced data
  [x] Ranked features by average-gain importance and saw near-duplicate
      features split the credit between them
  [x] Turned the ranking into a pre-release sanity check for leaked and
      protected columns

  KEY INSIGHT: XGBoost is the default choice for tabular credit/fraud/
  risk data because (a) it handles mixed feature types, (b) the split-
  gain formula gives you structural regularisation for free, and
  (c) gain importance is a quick first look at what the model uses —
  a starting point for explanation, not the whole story. Compare against
  LightGBM for speed and CatBoost for categorical-heavy data.

  Next: 03_lightgbm_catboost.py — the same data, same metric, two
  alternative libraries, and the decision tree for choosing one.
"""
)
