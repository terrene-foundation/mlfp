# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.3: LightGBM and CatBoost — Same Task, Faster Trees
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Train LightGBM and CatBoost on the same Singapore credit data
#   - Explain LightGBM's histogram-based split finding (why it's fast on
#     large data) and GOSS (gradient-based one-side sampling)
#   - Explain CatBoost's ordered boosting (why it's robust to target
#     leakage on categorical features)
#   - Compare the three libraries on AUC-PR, log loss, and train time
#   - Let CatBoost treat categoricals natively instead of ordinal codes
#   - Decide which library to use based on data shape and constraints
#
# PREREQUISITES: Exercise 4.2 (XGBoost baseline on the same dataset).
#
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Theory — how LightGBM and CatBoost differ from XGBoost
#   2. Build — LightGBM + CatBoost with matched hyperparameters
#   3. Train — fit all, record train time and AUC-PR; CatBoost native cats
#   4. Visualise — side-by-side bar chart with XGBoost as baseline
#   5. Apply — a grocery chain picks a library for fraud detection
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import plotly.graph_objects as go
from dotenv import load_dotenv

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    as_catboost_categoricals,
    categorical_feature_indices,
    evaluate_classifier,
    make_catboost,
    make_lightgbm,
    make_xgboost,
    prepare_credit_split,
    print_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Three Libraries, One Gradient-Boosting Recipe
# ════════════════════════════════════════════════════════════════════════
# All three libraries build additive trees on pseudo-residuals — the
# theory from 4.1 applies identically. They differ in HOW they find the
# best split at each round:
#
# XGBoost: originally "exact" split finding (sort every feature, try
#   every threshold, O(n·d·log n) per round). Since XGBoost 2.0 the
#   default tree_method is "hist" — the same histogram idea as LightGBM;
#   "exact" is still available as an option.
#
# LightGBM: histogram-based split finding. Buckets each feature into
#   ~256 bins, evaluates splits between bins only. O(n·d) per round.
#   Adds two more tricks:
#     - GOSS (gradient-based one-side sampling): keep all rows with
#       large gradients (hard examples), randomly sample the easy ones.
#     - EFB (exclusive feature bundling): bundle sparse features that
#       never overlap into a single "super-feature" — further speedup
#       on wide sparse data.
#   LightGBM also grows trees leaf-wise (best-first) instead of
#   level-wise, which usually gives lower loss for the same leaf count
#   but can overfit if max_depth is unset.
#
# CatBoost: ordered boosting. The problem CatBoost solves is target
#   leakage on categorical features — "mean-target encoding" (replace
#   each category with its average target) leaks the target into the
#   training set, inflating training accuracy. CatBoost fixes this by
#   computing target statistics only from rows that come BEFORE the
#   current row in a random permutation, then re-permuting each round.
#   Result: best out-of-the-box performance on data with many high-
#   cardinality categoricals, and the least tuning effort of the three.
#
# Practical decision tree:
#   ≫ >1M rows, mostly numeric        → LightGBM (speed wins)
#   ≫ High-cardinality categoricals   → CatBoost (native categoricals)
#   ≫ Medium data, mixed types        → XGBoost (stable default)
#
# We train all three with matched hyperparameters on the SAME
# ordinal-encoded matrix so the comparison is about the LIBRARY, not the
# hyperparameters. Then we give CatBoost the categoricals as categories.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD all three models with matched defaults
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  LightGBM + CatBoost vs XGBoost — Singapore Credit Scoring")
print("=" * 70)

data = prepare_credit_split()
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  Default rate: {data['default_rate']:.2%}")
cat_idx = categorical_feature_indices(data["feature_names"])
print(f"  Categorical columns: {[data['feature_names'][i] for i in cat_idx]}")

models = {
    # TODO: course-standard factories make_xgboost / make_lightgbm /
    # make_catboost with n_estimators=500 (iterations=500 for CatBoost),
    # learning_rate=0.1, max_depth=6 (depth=6 for CatBoost).
    "XGBoost": ____,
    "LightGBM": ____,
    "CatBoost": ____,
}


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN each library and record time + metrics
# ════════════════════════════════════════════════════════════════════════

results: dict[str, dict] = {}
print("\n  --- Training ---")
for name, model in models.items():
    t0 = time.perf_counter()
    # TODO: fit on the training data only (never pass the test set to fit).
    ____
    train_time = time.perf_counter() - t0

    # TODO: Get class-1 probabilities and evaluate with evaluate_classifier.
    y_proba = ____
    metrics = ____
    results[name] = {"metrics": metrics, "train_time": train_time, "model": model}
    print_metrics(name, metrics, train_time=train_time)


# CatBoost again, this time told which columns are categories. Its
# ordered target statistics replace the arbitrary ordinal codes.
cb_native = make_catboost(iterations=500, learning_rate=0.1, depth=6)
t0 = time.perf_counter()
# TODO: fit cb_native on as_catboost_categoricals(X_train, cat_idx) and
# tell CatBoost which columns are categories (cat_features=cat_idx).
____
native_time = time.perf_counter() - t0
native_metrics = evaluate_classifier(
    y_test, cb_native.predict_proba(as_catboost_categoricals(X_test, cat_idx))[:, 1]
)
results["CatBoost (native cats)"] = {
    "metrics": native_metrics,
    "train_time": native_time,
    "model": cb_native,
}
print_metrics("CatBoost (native cats)", native_metrics, train_time=native_time)

auc_pr_spread = max(r["metrics"]["auc_pr"] for r in results.values()) - min(
    r["metrics"]["auc_pr"] for r in results.values()
)
print(
    f"\n  On {X_train.shape[0]:,} training rows the AUC-PR spread across "
    f"libraries is {auc_pr_spread:.4f}; train times range "
    f"{min(r['train_time'] for r in results.values()):.1f}s - "
    f"{max(r['train_time'] for r in results.values()):.1f}s."
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
for name, r in results.items():
    assert r["metrics"]["auc_roc"] > 0.7, f"{name} AUC-ROC should exceed 0.7"
    assert (
        r["metrics"]["auc_pr"] > 2 * data["default_rate"]
    ), f"{name} AUC-PR should be at least twice the random baseline"
# INTERPRETATION: With matched settings on the same features, the three
# libraries usually land within a few hundredths of AUC-PR of each other
# — compare the spread above with the gaps between rows before declaring
# a winner. Train time is where they differ most. Native categoricals
# matter most with MANY high-cardinality categories; this dataset's nine
# categoricals have at most six levels each, so expect little change.
print("\n[ok] Checkpoint 1 passed — all three libraries trained on credit data\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the comparison
# ════════════════════════════════════════════════════════════════════════
# One chart with two stacked panels: AUC-PR (higher is better) and train
# time (lower is better). Side-by-side bars so the speed/quality trade-
# off jumps out visually.

names = list(results.keys())
auc_pr_values = [results[n]["metrics"]["auc_pr"] for n in names]
auc_roc_values = [results[n]["metrics"]["auc_roc"] for n in names]
log_loss_values = [results[n]["metrics"]["log_loss"] for n in names]
time_values = [results[n]["train_time"] for n in names]

fig = go.Figure()
# offsetgroup keeps the two bar series side by side even though they use
# different y-axes (without it they would be drawn on top of each other).
fig.add_trace(
    go.Bar(
        name="AUC-PR (higher=better)",
        x=names,
        y=auc_pr_values,
        yaxis="y1",
        offsetgroup=0,
    )
)
fig.add_trace(
    go.Bar(
        name="Train time (s, lower=better)",
        x=names,
        y=time_values,
        yaxis="y2",
        offsetgroup=1,
    )
)
fig.update_layout(
    title="Boosting Library Comparison — Singapore Credit Default",
    barmode="group",
    yaxis=dict(title="AUC-PR", range=[0, max(auc_pr_values) * 1.25]),
    yaxis2=dict(title="Train time (s)", overlaying="y", side="right"),
)
viz_path = OUTPUT_DIR / "ex4_03_library_comparison.html"
fig.write_html(viz_path)
print(f"  Saved: {viz_path}")

# Console table so the reader sees numbers even without opening the HTML
print("\n  --- Library Comparison Table ---")
print(
    f"  {'Library':<24} {'AUC-ROC':>10} {'AUC-PR':>10} {'Log Loss':>10} {'Time (s)':>10}"
)
print("  " + "─" * 70)
for n, auc_pr, auc_roc, ll, t in zip(
    names, auc_pr_values, auc_roc_values, log_loss_values, time_values
):
    print(f"  {n:<24} {auc_roc:>10.4f} {auc_pr:>10.4f} {ll:>10.4f} {t:>10.2f}")


# ── Checkpoint 2 ────────────────────────────────────────────────────────
best_name = max(results.items(), key=lambda kv: kv[1]["metrics"]["auc_pr"])[0]
fastest_name = min(results.items(), key=lambda kv: kv[1]["train_time"])[0]
assert best_name in results, "best_name must be a trained library"
assert fastest_name in results, "fastest_name must be a trained library"
print(f"  Best by AUC-PR: {best_name} | fastest: {fastest_name}")
# INTERPRETATION: Report BOTH numbers. They are often different libraries.
# Knowing both lets you choose: is a small AUC-PR gain worth a multiple of
# the training time? That depends on how often you re-train and how much
# each missed default costs.
print("\n[ok] Checkpoint 2 passed — comparison table computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: A Grocery Chain Picks A Fraud-Detection Library
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative — all figures are teaching assumptions): a
# regional grocery chain scores every card-present transaction across its
# stores in <50ms and flags suspects for a manager callback.
#
# Data shape:
#   - Rows: millions of transactions per day; nightly re-train on the
#     last 30 days (on the order of 100M rows)
#   - Features: mostly numeric (amount, velocity, time-of-day) plus
#     categoricals (store, SKU bundle, payment network, card BIN)
#   - Cardinality: store ~hundreds of levels; SKU bundle and card BIN
#     ~ten thousand levels each
#   - Class balance: fraud is a fraction of a percent (severely imbalanced)
#
# LIBRARY CHOICE: CatBoost
#   - High-cardinality categoricals (SKU bundle, card BIN) would need
#     ordinal or target encoding for XGBoost/LightGBM; ordinal codes
#     impose an order those categories do not have.
#   - CatBoost's ordered target statistics handle them natively without
#     target leakage — the cat_features run above is the same mechanism.
#   - Its extra training time (measure it on YOUR data, as Task 3 did) is
#     acceptable if the nightly window has room.
#
# WHY NOT LIGHTGBM: it has its own native categorical handling
# (categorical_feature=), but with ~10K-level columns you must manage
# overfitting on rare levels yourself; CatBoost's ordered statistics are
# designed for exactly this case.
#
# WHY NOT XGBOOST: enable_categorical=True exists, but you would still
# own the high-cardinality tuning; for this data shape CatBoost needs the
# least custom encoding code.
#
# The general rule: decide the library from the DATA SHAPE (rows,
# categorical cardinality, re-train window), then confirm with a
# measured comparison like Task 3 — not from benchmark folklore.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Trained XGBoost, LightGBM, and CatBoost on the same credit data
      with matched hyperparameters
  [x] Read each library's design in terms of split-finding strategy
      (histogram split finding, GOSS, ordered boosting)
  [x] Let CatBoost treat the categoricals natively via cat_features
  [x] Compared AUC-PR and train time side-by-side
  [x] Identified {best_name} as the best on this dataset by AUC-PR
  [x] Identified {fastest_name} as the fastest to train
  [x] Matched library choice to data shape using the grocery-fraud scenario

  KEY INSIGHT: On ordinary tabular data the three libraries reach similar
  accuracy; they differ in speed, categorical handling and tuning effort.
  LightGBM is usually fastest on large numeric data; CatBoost needs the
  least encoding work on high-cardinality categoricals; XGBoost is a
  stable default. Measure on your data before committing.

  Next: 04_boosting_tuning.py — hyperparameter sweeps, learning-rate
  sensitivity, and early stopping on the same dataset.
"""
)
