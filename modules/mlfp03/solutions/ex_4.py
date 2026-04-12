# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4: Gradient Boosting Deep Dive
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain how boosting reduces bias by correcting residuals sequentially
#   - Compare XGBoost, LightGBM, and CatBoost on the same dataset
#   - Tune key hyperparameters (learning rate, depth, regularisation)
#   - Choose AUC-PR over AUC-ROC for imbalanced classification tasks
#   - Interpret calibration plots to assess probability reliability
#
# PREREQUISITES:
#   - MLFP03 Exercise 3 (model zoo, Random Forest — boosting extends bagging)
#   - MLFP03 Exercise 2 (regularisation — L2 on leaf weights in XGBoost)
#
# ESTIMATED TIME: 60-90 minutes
#
# TASKS:
#   1. Load and prepare credit scoring data with kailash_ml.interop
#   2. Train XGBoost, LightGBM, CatBoost with default params
#   3. Compare learning curves and convergence
#   4. Hyperparameter sensitivity analysis (learning rate, depth, regularisation)
#   5. Evaluate with proper metrics (AUC-PR, calibration, log loss)
#   6. Visualise with ModelVisualizer
#
# DATASET: Singapore credit scoring (from MLFP02)
#   Target: default (binary — 12% positive rate — imbalanced)
#   Rows: ~5,000 credit applications | Features: financial + behavioural
#   Key challenge: 12% default rate makes accuracy misleading — use AUC-PR
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader


# ── Data Loading & Preparation ────────────────────────────────────────

loader = MLFPDataLoader()
credit = loader.load("mlfp02", "sg_credit_scoring.parquet")

print(f"=== Singapore Credit Data ===")
print(f"Shape: {credit.shape}")
print(f"Default rate: {credit['default'].mean():.2%}")

# Preprocess with PreprocessingPipeline
pipeline = PreprocessingPipeline()
result = pipeline.setup(
    data=credit,
    target="default",
    train_size=0.8,
    seed=42,
    normalize=False,  # Tree models don't need normalisation
    categorical_encoding="ordinal",
    imputation_strategy="median",
)

print(f"\nTask type: {result.task_type}")
print(f"Train: {result.train_data.shape}, Test: {result.test_data.shape}")

# Convert to sklearn format
X_train, y_train, col_info = to_sklearn_input(
    result.train_data,
    feature_columns=[c for c in result.train_data.columns if c != "default"],
    target_column="default",
)

X_test, y_test, _ = to_sklearn_input(
    result.test_data,
    feature_columns=[c for c in result.test_data.columns if c != "default"],
    target_column="default",
)

feature_names = col_info["feature_columns"]
print(f"Features: {len(feature_names)}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_train.shape[0] > 0, "Training set empty"
assert len(feature_names) > 0, "No features"
assert y_train.mean() < 0.5, "Default rate should be minority class (< 50%)"
print("\n✓ Checkpoint 1 passed — credit data prepared for gradient boosting\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Train all three boosting algorithms with defaults
# ══════════════════════════════════════════════════════════════════════

models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbose=-1,
    ),
    "CatBoost": cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")

    if name == "CatBoost":
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
    else:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "auc_roc": roc_auc_score(y_test, y_proba),
        "auc_pr": average_precision_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
    }

    print(f"  AUC-ROC: {results[name]['auc_roc']:.4f}")
    print(f"  AUC-PR:  {results[name]['auc_pr']:.4f}")
    print(f"  Log Loss: {results[name]['log_loss']:.4f}")
    print(f"  Brier:   {results[name]['brier']:.4f}")
    # INTERPRETATION: With 12% default rate, AUC-ROC can be misleadingly high.
    # A model that never predicts default gets AUC-ROC ~0.5 but looks fine.
    # AUC-PR rewards finding the rare positives — the metric that matters here.


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compare learning curves
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Learning curves showing bias-variance trade-off
print("\n=== Learning Curves ===")
for name, r in results.items():
    fig = viz.learning_curve(
        r["model"],
        X_train,
        y_train,
        cv=5,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
    )
    fig.update_layout(title=f"Learning Curve: {name}")
    fig.write_html(f"ex1_learning_curve_{name.lower()}.html")
    print(f"  Saved: ex1_learning_curve_{name.lower()}.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Hyperparameter sensitivity (learning rate)
# ══════════════════════════════════════════════════════════════════════

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5]
lr_results = {name: {} for name in ["XGBoost", "LightGBM"]}

for lr in learning_rates:
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=lr,
        max_depth=6,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    lr_results["XGBoost"][lr] = roc_auc_score(y_test, y_proba)

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=lr,
        max_depth=6,
        random_state=42,
        verbose=-1,
    )
    lgb_model.fit(X_train, y_train)
    y_proba = lgb_model.predict_proba(X_test)[:, 1]
    lr_results["LightGBM"][lr] = roc_auc_score(y_test, y_proba)

print("\n=== Learning Rate Sensitivity ===")
print(f"{'LR':>6} {'XGBoost':>10} {'LightGBM':>10}")
print("─" * 30)
for lr in learning_rates:
    print(
        f"{lr:>6.2f} {lr_results['XGBoost'][lr]:>10.4f} {lr_results['LightGBM'][lr]:>10.4f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Comprehensive evaluation — AUC-PR vs AUC-ROC
# ══════════════════════════════════════════════════════════════════════
# With 12% default rate, AUC-ROC can be misleading.
# AUC-PR is more informative for imbalanced classification.

print(f"\n=== Model Comparison ===")
print(f"{'Model':<12} {'AUC-ROC':>10} {'AUC-PR':>10} {'Log Loss':>10} {'Brier':>10}")
print("─" * 55)
for name, r in results.items():
    print(
        f"{name:<12} {r['auc_roc']:>10.4f} {r['auc_pr']:>10.4f} "
        f"{r['log_loss']:>10.4f} {r['brier']:>10.4f}"
    )

# Best model
best_model = max(results.items(), key=lambda x: x[1]["auc_pr"])
print(f"\nBest by AUC-PR: {best_model[0]}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
for name, r in results.items():
    assert r["auc_roc"] > 0.5, f"{name} AUC-ROC should exceed random baseline"
    assert r["auc_pr"] > 0, f"{name} AUC-PR should be positive"
    assert r["model"] is not None, f"{name} model should exist"
# INTERPRETATION: If LightGBM trains fastest with similar accuracy to XGBoost,
# use LightGBM in production. Speed matters when you retrain frequently.
# CatBoost's native categorical support is valuable when you have many
# high-cardinality categoricals — no ordinal encoding needed.
print("\n✓ Checkpoint 2 passed — all three boosting models trained and compared\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Calibration analysis
# ══════════════════════════════════════════════════════════════════════

for name, r in results.items():
    fig = viz.calibration_curve(y_test, r["y_proba"])
    fig.update_layout(title=f"Calibration: {name}")
    fig.write_html(f"ex1_calibration_{name.lower()}.html")

print("\nCalibration plots saved.")
print("→ A model with AUC=0.95 but poor calibration is NOT production-ready.")
print("  Calibration = 'when the model says 20% chance, it happens 20% of the time'")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
for name, r in results.items():
    proba = r["y_proba"]
    assert proba.min() >= 0.0, f"{name} probabilities must be >= 0"
    assert proba.max() <= 1.0, f"{name} probabilities must be <= 1"
# INTERPRETATION: Calibration matters in credit: if the model says 15% default
# probability, the bank needs that to be a real probability, not just a ranking
# score. Poorly calibrated models lead to wrong loan pricing and reserve calculations.
print("\n✓ Checkpoint 3 passed — calibration analysis complete\n")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Final comparison visualisation
# ══════════════════════════════════════════════════════════════════════

metric_comparison = {
    name: {
        "AUC_ROC": r["auc_roc"],
        "AUC_PR": r["auc_pr"],
        "Log_Loss": r["log_loss"],
        "Brier_Score": r["brier"],
    }
    for name, r in results.items()
}

fig = viz.metric_comparison(metric_comparison)
fig.update_layout(title="Gradient Boosting Comparison: Singapore Credit Scoring")
fig.write_html("ex1_model_comparison.html")
print("Saved: ex1_model_comparison.html")

# Feature importance for best model
fig_fi = viz.feature_importance(
    best_model[1]["model"],
    feature_names,
    top_n=15,
)
fig_fi.update_layout(title=f"Feature Importance: {best_model[0]}")
fig_fi.write_html("ex1_feature_importance.html")
print("Saved: ex1_feature_importance.html")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(f"""
  ✓ Boosting: sequential ensemble that reduces bias (vs bagging's variance)
  ✓ XGBoost: 2nd-order Taylor expansion, L2 leaf regularisation (lambda)
  ✓ LightGBM: GOSS sampling, histogram splits, leaf-wise growth — fastest
  ✓ CatBoost: ordered boosting, native categoricals — least leakage-prone
  ✓ AUC-PR vs AUC-ROC: AUC-PR is the right metric for imbalanced data

  BEST MODEL (AUC-PR): {best_model[0]}

  KEY INSIGHT: Gradient boosting is the dominant algorithm for tabular data.
  The three libraries trade off: XGBoost (stable, well-documented), LightGBM
  (fastest), CatBoost (best with categoricals). In production, start with
  LightGBM for speed, then compare with XGBoost.

  NEXT: Exercise 5 tackles class imbalance and calibration head-on — SMOTE,
  cost-sensitive learning, focal loss, and probability calibration. The 12%
  default rate you saw here will be a problem for naive training.
""")

print("\n✓ Exercise 4 complete — gradient boosting comparison on credit data")
