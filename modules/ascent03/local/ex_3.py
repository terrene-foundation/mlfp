# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT03 — Exercise 3: Class Imbalance and Calibration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compare SMOTE, cost-sensitive learning, Focal Loss, and
#   threshold optimisation for handling class imbalance. Calibrate after.
#
# TASKS:
#   1. Establish baseline (no imbalance handling)
#   2. SMOTE oversampling — why it often fails in practice
#   3. Cost-sensitive learning (sample weights)
#   4. Focal Loss (derive γ parameter effect)
#   5. Threshold optimisation from cost matrix
#   6. Post-hoc calibration (Platt scaling, isotonic regression)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

from kailash_ml import PreprocessingPipeline, ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

pipeline = PreprocessingPipeline()
result = pipeline.setup(
    credit, target="default", seed=42, normalize=False, categorical_encoding="ordinal"
)

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

pos_rate = y_train.mean()
print(
    f"Default rate: {pos_rate:.2%} (imbalance ratio: {(1 - pos_rate) / pos_rate:.0f}:1)"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Baseline — no imbalance handling
# ══════════════════════════════════════════════════════════════════════

# TODO: Create and fit a baseline LGBMClassifier with n_estimators=300
baseline = (
    ____  # Hint: lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
)
baseline.fit(X_train, y_train)

# TODO: Get predicted probabilities for the positive class
y_proba_base = ____  # Hint: baseline.predict_proba(X_test)[:, 1]

print(f"\n=== Baseline (no correction) ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_base):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_proba_base):.4f}")
print(f"Brier:   {brier_score_loss(y_test, y_proba_base):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: SMOTE — and why it often fails
# ══════════════════════════════════════════════════════════════════════
# SMOTE creates synthetic minority examples by interpolating between
# nearest neighbours. Problems:
# 1. Lipschitz violation: interpolation assumes smooth decision boundary
# 2. Noisy minority: amplifies noise in the minority class
# 3. High-dimensional collapse: in high dimensions, nearest neighbours
#    are nearly equidistant, making interpolation meaningless

from imblearn.over_sampling import SMOTE

# TODO: Create a SMOTE instance and resample the training data
smote = ____  # Hint: SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f"\n=== SMOTE ===")
print(f"Before SMOTE: {len(y_train):,} (pos={y_train.sum():.0f})")
print(f"After SMOTE:  {len(y_smote):,} (pos={y_smote.sum():.0f})")

smote_model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
smote_model.fit(X_smote, y_smote)

# TODO: Get predicted probabilities from the SMOTE-trained model
y_proba_smote = ____  # Hint: smote_model.predict_proba(X_test)[:, 1]

print(f"AUC-ROC: {roc_auc_score(y_test, y_proba_smote):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_proba_smote):.4f}")
print(f"Brier:   {brier_score_loss(y_test, y_proba_smote):.4f}")

print("\nSMOTE Failure Taxonomy:")
print("  1. Lipschitz: interpolated samples may cross decision boundary")
print("  2. Noise: noisy minority examples get amplified")
print("  3. Dimensionality: with 45 features, NN distances converge")
print(f"  → 92% citation rate in papers, ~6% production deployment")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Cost-sensitive learning (sample weights)
# ══════════════════════════════════════════════════════════════════════
# Weight minority class higher in the loss function.
# LightGBM supports scale_pos_weight and sample_weight.

# Method A: scale_pos_weight
# TODO: Compute scale_pos_weight as ratio of negative to positive rate
scale_weight = ____  # Hint: (1 - pos_rate) / pos_rate

# TODO: Create LGBMClassifier with scale_pos_weight set
cost_model_a = ____  # Hint: lgb.LGBMClassifier(n_estimators=300, scale_pos_weight=scale_weight, random_state=42, verbose=-1)
cost_model_a.fit(X_train, y_train)
y_proba_cost_a = cost_model_a.predict_proba(X_test)[:, 1]

# Method B: custom sample weights (from cost matrix)
# Cost matrix: FP costs $100 (wasted investigation), FN costs $10,000 (undetected default)
cost_fn = 10_000
cost_fp = 100

# TODO: Create sample_weights array: cost_fn for positives, cost_fp for negatives
sample_weights = ____  # Hint: np.where(y_train == 1, cost_fn, cost_fp)

cost_model_b = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)

# TODO: Fit cost_model_b with the sample_weight argument
cost_model_b.fit(
    X_train, y_train, sample_weight=____
)  # Hint: sample_weight=sample_weights
y_proba_cost_b = cost_model_b.predict_proba(X_test)[:, 1]

print(f"\n=== Cost-Sensitive Learning ===")
print(f"Method A (scale_pos_weight={scale_weight:.1f}):")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_proba_cost_a):.4f}")
print(f"  AUC-PR:  {average_precision_score(y_test, y_proba_cost_a):.4f}")
print(f"Method B (cost matrix: FN=${cost_fn:,}, FP=${cost_fp:,}):")
print(f"  AUC-ROC: {roc_auc_score(y_test, y_proba_cost_b):.4f}")
print(f"  AUC-PR:  {average_precision_score(y_test, y_proba_cost_b):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Focal Loss
# ══════════════════════════════════════════════════════════════════════
# Focal Loss: FL(p) = -α(1-p)^γ log(p)
# γ > 0 down-weights easy examples (well-classified)
# γ = 0 reduces to standard cross-entropy
# γ = 2 is the original setting (Lin et al., 2017)


def focal_loss_lgb(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Custom focal loss for LightGBM."""
    p = 1.0 / (1.0 + np.exp(-y_pred))  # sigmoid
    grad = alpha * (
        -((1 - p) ** gamma)
        * (gamma * p * np.log(np.clip(p, 1e-8, 1)) + (1 - p))
        * y_true
        + p**gamma
        * (gamma * (1 - p) * np.log(np.clip(1 - p, 1e-8, 1)) + p)
        * (1 - y_true)
    )
    hess = np.abs(grad) * (1 - np.abs(grad))
    hess = np.clip(hess, 1e-8, None)
    return grad, hess


# TODO: Create LGBMClassifier with a custom focal loss objective (gamma=2.0)
#       Pass objective=lambda y_true, y_pred: focal_loss_lgb(y_true, y_pred, gamma=2.0)
focal_model = ____  # Hint: lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1, objective=lambda y_true, y_pred: focal_loss_lgb(y_true, y_pred, gamma=2.0))
focal_model.fit(X_train, y_train)
y_raw_focal = focal_model.predict_proba(X_test)[:, 1]

# Note: custom objective outputs are not calibrated probabilities
# Need post-hoc calibration
print(f"\n=== Focal Loss (γ=2.0) ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_raw_focal):.4f}")
print(f"AUC-PR:  {average_precision_score(y_test, y_raw_focal):.4f}")
print(
    f"Brier:   {brier_score_loss(y_test, y_raw_focal):.4f} (uncalibrated — expected to be poor)"
)

# Compare γ values
for gamma in [0.0, 0.5, 1.0, 2.0, 5.0]:
    m = lgb.LGBMClassifier(
        n_estimators=300,
        random_state=42,
        verbose=-1,
        objective=lambda y_true, y_pred, g=gamma: focal_loss_lgb(
            y_true, y_pred, gamma=g
        ),
    )
    m.fit(X_train, y_train)
    y_p = m.predict_proba(X_test)[:, 1]
    print(f"  γ={gamma:.1f}: AUC-PR={average_precision_score(y_test, y_p):.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Threshold optimisation from cost matrix
# ══════════════════════════════════════════════════════════════════════
# Optimal threshold: t* = cost_FP / (cost_FP + cost_FN)
# This minimises expected total cost

# Use best model so far
best_proba = y_proba_cost_a

# TODO: Derive the optimal threshold from the cost matrix
optimal_threshold = ____  # Hint: cost_fp / (cost_fp + cost_fn)
print(f"\n=== Threshold Optimisation ===")
print(f"Cost matrix: FP=${cost_fp:,}, FN=${cost_fn:,}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Default threshold (0.5) would miss many defaults!")

# Evaluate at different thresholds
thresholds = np.arange(0.01, 0.50, 0.01)
best_cost = float("inf")
best_t = 0.5

for t in thresholds:
    y_pred_t = (best_proba >= t).astype(int)
    fp = ((y_pred_t == 1) & (y_test == 0)).sum()
    fn = ((y_pred_t == 0) & (y_test == 1)).sum()
    total_cost = fp * cost_fp + fn * cost_fn
    if total_cost < best_cost:
        best_cost = total_cost
        best_t = t

print(f"Empirically optimal threshold: {best_t:.3f}")
print(f"Minimum total cost: ${best_cost:,.0f}")

# Compare with default threshold
y_pred_default = (best_proba >= 0.5).astype(int)
fp_d = ((y_pred_default == 1) & (y_test == 0)).sum()
fn_d = ((y_pred_default == 0) & (y_test == 1)).sum()
cost_default = fp_d * cost_fp + fn_d * cost_fn
print(f"Cost at threshold=0.5: ${cost_default:,.0f}")
print(
    f"Savings from optimisation: ${cost_default - best_cost:,.0f} ({(cost_default - best_cost) / cost_default:.1%})"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Post-hoc calibration
# ══════════════════════════════════════════════════════════════════════

# TODO: Create a Platt-scaled (sigmoid) calibrated version of cost_model_a using cv=5
platt_model = ____  # Hint: CalibratedClassifierCV(cost_model_a, method="sigmoid", cv=5)
platt_model.fit(X_train, y_train)
y_proba_platt = platt_model.predict_proba(X_test)[:, 1]

# TODO: Create an isotonic regression calibrated version of cost_model_a using cv=5
iso_model = ____  # Hint: CalibratedClassifierCV(cost_model_a, method="isotonic", cv=5)
iso_model.fit(X_train, y_train)
y_proba_iso = iso_model.predict_proba(X_test)[:, 1]

print(f"\n=== Calibration Comparison ===")
print(f"{'Method':<20} {'Brier':>8} {'AUC-PR':>8}")
print("─" * 40)
print(
    f"{'Uncalibrated':<20} {brier_score_loss(y_test, y_proba_cost_a):>8.4f} {average_precision_score(y_test, y_proba_cost_a):>8.4f}"
)
print(
    f"{'Platt Scaling':<20} {brier_score_loss(y_test, y_proba_platt):>8.4f} {average_precision_score(y_test, y_proba_platt):>8.4f}"
)
print(
    f"{'Isotonic':<20} {brier_score_loss(y_test, y_proba_iso):>8.4f} {average_precision_score(y_test, y_proba_iso):>8.4f}"
)

# Visualise calibration curves
viz = ModelVisualizer()

for name, proba in [
    ("Uncalibrated", y_proba_cost_a),
    ("Platt", y_proba_platt),
    ("Isotonic", y_proba_iso),
]:
    fig = viz.calibration_curve(y_test, proba)
    fig.update_layout(title=f"Calibration: {name}")
    fig.write_html(f"ex2_calibration_{name.lower()}.html")

# Final comparison
all_results = {
    "Baseline": {
        "AUC_PR": average_precision_score(y_test, y_proba_base),
        "Brier": brier_score_loss(y_test, y_proba_base),
    },
    "SMOTE": {
        "AUC_PR": average_precision_score(y_test, y_proba_smote),
        "Brier": brier_score_loss(y_test, y_proba_smote),
    },
    "Cost-Sensitive": {
        "AUC_PR": average_precision_score(y_test, y_proba_cost_a),
        "Brier": brier_score_loss(y_test, y_proba_cost_a),
    },
    "Focal(γ=2)": {
        "AUC_PR": average_precision_score(y_test, y_raw_focal),
        "Brier": brier_score_loss(y_test, y_raw_focal),
    },
    "Cost+Platt": {
        "AUC_PR": average_precision_score(y_test, y_proba_platt),
        "Brier": brier_score_loss(y_test, y_proba_platt),
    },
}

fig = viz.metric_comparison(all_results)
fig.update_layout(title="Class Imbalance Methods Comparison")
fig.write_html("ex2_imbalance_comparison.html")
print("\nSaved: ex2_imbalance_comparison.html")

print("\n✓ Exercise 3 complete — class imbalance handling + calibration")
