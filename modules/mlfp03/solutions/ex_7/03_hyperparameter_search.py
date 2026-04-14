# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 7.3: Bayesian Hyperparameter Search
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Declare a search space with ParamDistribution (int/float, log-scale)
#   - Configure a Bayesian search run with SearchConfig
#   - Execute the search with kailash-ml's HyperparameterSearch engine
#   - Compare Bayesian vs grid-search cost per evaluation
#   - Feed the best_params back into the final model fit
#
# PREREQUISITES: 01_workflow_builder.py, 02_dataflow_persistence.py
# ESTIMATED TIME: ~45 min
#
# 5-PHASE R10:
#   1. Theory     — why Bayesian search beats grid search
#   2. Build      — SearchSpace + SearchConfig for LightGBM
#   3. Train      — .search() against the credit-default training split
#   4. Visualise  — top-5 trials leaderboard + final model AUC-PR
#   5. Apply      — Singapore credit default lift worth S$4M+/yr
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score

from kailash_ml.engines.hyperparameter_search import (
    HyperparameterSearch,
    ParamDistribution,
    SearchConfig,
    SearchSpace,
)

from shared.mlfp03.ex_7 import (
    RANDOM_SEED,
    SG_BANK_PORTFOLIO,
    compute_classification_metrics,
    headline_roi_text,
    prepare_credit_split,
    print_metric_block,
    scale_pos_weight_for,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Bayesian vs grid search
# ════════════════════════════════════════════════════════════════════════
# Grid search is the "brute-force every combination" approach. A 5x5x5
# grid is 125 trials even at 3 hyperparameters. Every extra dimension
# multiplies the cost. At 5 hyperparameters you're at 3,125 trials and
# you still haven't explored any fractional learning rates.
#
# Bayesian optimisation is smarter:
#   1. Fit a cheap surrogate (Gaussian process / tree-based regressor)
#      to the (hyperparameters -> score) pairs seen so far.
#   2. Compute an "acquisition function" that balances EXPLORATION
#      (try a region with high variance) and EXPLOITATION (try a region
#      with a high mean).
#   3. Evaluate the hyperparameter with the highest acquisition score.
#   4. Repeat.
#
# In practice, 20 Bayesian trials match or beat 200 random trials on
# most tabular problems. kailash-ml's HyperparameterSearch engine
# wraps this algorithm behind a 5-line interface: SearchSpace,
# SearchConfig, .search().


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the search space and configuration
# ════════════════════════════════════════════════════════════════════════

search_space = SearchSpace(
    params=[
        ParamDistribution("n_estimators", "int", low=100, high=1000),
        ParamDistribution("learning_rate", "float", low=0.01, high=0.3, log=True),
        ParamDistribution("max_depth", "int", low=3, high=10),
        ParamDistribution("num_leaves", "int", low=15, high=127),
        ParamDistribution("min_child_samples", "int", low=5, high=50),
    ]
)

search_config = SearchConfig(
    n_trials=20,
    metric="average_precision",
    direction="maximize",
    cv_folds=5,
    random_state=RANDOM_SEED,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN the search run
# ════════════════════════════════════════════════════════════════════════

split = prepare_credit_split()
pos_weight = scale_pos_weight_for(split.y_train)

print("\n" + "=" * 70)
print("  Bayesian Hyperparameter Search — 20 trials, 5-fold CV, AUC-PR")
print("=" * 70)

searcher = HyperparameterSearch(search_space, search_config)
best_params, best_score, all_trials = searcher.search(
    estimator_class=lgb.LGBMClassifier,
    X=split.X_train,
    y=split.y_train,
    fixed_params={
        "random_state": RANDOM_SEED,
        "verbose": -1,
        "scale_pos_weight": pos_weight,
    },
)

print(f"\n  Best AUC-PR: {best_score:.4f}")
print(f"  Best params: {best_params}")


# Train the final model with the winning hyperparameters
best_model = lgb.LGBMClassifier(
    **best_params,
    random_state=RANDOM_SEED,
    verbose=-1,
    scale_pos_weight=pos_weight,
)
best_model.fit(split.X_train, split.y_train)
y_pred = best_model.predict(split.X_test)
y_proba = best_model.predict_proba(split.X_test)[:, 1]
final_metrics = compute_classification_metrics(split.y_test, y_pred, y_proba)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert best_score > 0.0, "Task 3: search should yield a positive AUC-PR"
assert best_params is not None, "Task 3: search should return a params dict"
assert final_metrics["auc_pr"] > 0.0, "Task 3: final fit should evaluate"
print("\n[ok] Checkpoint passed — Bayesian search complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the leaderboard + final model metrics
# ════════════════════════════════════════════════════════════════════════

top_k = sorted(all_trials, key=lambda t: t["score"], reverse=True)[:5]
print("Top 5 trials (score = 5-fold CV AUC-PR):")
for i, trial in enumerate(top_k, 1):
    print(f"  {i}. score={trial['score']:.4f}")
    for k, v in trial["params"].items():
        print(f"       {k}: {v}")

print_metric_block("Final model on held-out test set", final_metrics)


# Baseline (grid) comparison — 4 manual points, same training split
baseline_grid = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 5},
    {"n_estimators": 500, "learning_rate": 0.10, "max_depth": 6},
    {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 7},
    {"n_estimators": 700, "learning_rate": 0.03, "max_depth": 8},
]
best_grid = -1.0
for params in baseline_grid:
    est = lgb.LGBMClassifier(
        **params,
        random_state=RANDOM_SEED,
        verbose=-1,
        scale_pos_weight=pos_weight,
    )
    cv = cross_val_score(
        est, split.X_train, split.y_train, cv=5, scoring="average_precision"
    )
    best_grid = max(best_grid, float(cv.mean()))

lift = best_score - best_grid
print(
    f"\nGrid-search best   : {best_grid:.4f}"
    f"\nBayesian-search best: {best_score:.4f}"
    f"\nLift                : +{lift:.4f} AUC-PR points"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Lift the S$48B portfolio's model quality
# ════════════════════════════════════════════════════════════════════════
# The hand-tuned production model used by a typical SG bank credit team
# is equivalent to the 4-point grid baseline above. A 20-trial Bayesian
# search, running overnight on a single CPU, typically lifts AUC-PR by
# 2-4 points on this dataset. Every AUC-PR point at the bank's default
# threshold translates to ~140 additional defaults CAUGHT per year.
#
# At a blended S$18,000 principal and 65% LGD, every caught default
# saves ~S$11,700 — so a 4-point lift saves ~560 * S$11,700 = S$6.5M.
# The `headline_roi_text()` table uses the conservative S$4M figure.
print("\n" + "=" * 70)
print("  APPLY: Bayesian Search Lift = Real Defaults Caught")
print("=" * 70)
print(headline_roi_text())
print(
    "\n  The `Loss avoided` line above is UNLOCKED by this file."
    "\n  Without Bayesian search, the bank ships the grid baseline."
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Declared a 5-dimensional hyperparameter SearchSpace
  [x] Configured Bayesian search with 5-fold CV on AUC-PR
  [x] Executed 20 trials via kailash-ml's HyperparameterSearch engine
  [x] Compared the Bayesian winner against a 4-point grid baseline
  [x] Tied AUC-PR lift to the portfolio's annual dollar savings

  KEY INSIGHT: Bayesian search is NOT exotic — it is a 5-line API call
  in kailash-ml that routinely finds better hyperparameters than a
  grid five times its size.

  Next: 04_model_registry.py — promote the winning model through the
  ModelRegistry lifecycle so production can safely pick it up.

  Portfolio anchor: S${SG_BANK_PORTFOLIO['portfolio_sgd']/1e9:.0f}B  |  Lift: +{lift:.4f} AUC-PR
"""
)
