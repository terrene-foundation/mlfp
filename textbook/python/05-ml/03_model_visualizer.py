# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / ModelVisualizer
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Create ML visualizations — confusion matrix, ROC curve,
#            precision-recall curve, feature importance, learning curve,
#            residual analysis, calibration curve, metric comparison,
#            and training history.  All methods return plotly Figures.
# LEVEL: Basic
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: ModelVisualizer — confusion_matrix(), roc_curve(),
#            precision_recall_curve(), feature_importance(),
#            residuals(), calibration_curve(), metric_comparison(),
#            training_history()
#
# Run: uv run python textbook/python/05-ml/03_model_visualizer.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from kailash_ml.engines.model_visualizer import ModelVisualizer

# ── 1. Instantiate ModelVisualizer ──────────────────────────────────
# ModelVisualizer is stateless — no connection or initialization needed.
# It requires plotly to be installed.

viz = ModelVisualizer()
assert isinstance(viz, ModelVisualizer)

# ── 2. Train a classifier for demonstrations ────────────────────────

X_cls, y_cls = make_classification(
    n_samples=200, n_features=5, n_informative=3, random_state=42
)
feature_names = [f"feat_{i}" for i in range(5)]
X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # Probability of positive class

# ── 3. Confusion matrix ─────────────────────────────────────────────
# confusion_matrix(y_true, y_pred, labels=None) -> plotly Figure

fig_cm = viz.confusion_matrix(y_test, y_pred)
assert fig_cm is not None
assert hasattr(fig_cm, "update_layout"), "Returns a plotly Figure"
assert fig_cm.layout.title.text == "Confusion Matrix"

# With custom labels
fig_cm_labeled = viz.confusion_matrix(y_test, y_pred, labels=["No", "Yes"])
assert fig_cm_labeled is not None

# ── 4. ROC curve ─────────────────────────────────────────────────────
# roc_curve(y_true, y_scores, pos_label=1) -> plotly Figure

fig_roc = viz.roc_curve(y_test, y_proba)
assert fig_roc is not None
assert "ROC Curve" in fig_roc.layout.title.text
# The title includes the AUC value
assert "AUC" in fig_roc.layout.title.text

# ── 5. Precision-recall curve ────────────────────────────────────────
# precision_recall_curve(y_true, y_scores, pos_label=1) -> plotly Figure

fig_pr = viz.precision_recall_curve(y_test, y_proba)
assert fig_pr is not None
assert "Precision-Recall" in fig_pr.layout.title.text
assert "AP" in fig_pr.layout.title.text  # Average Precision

# ── 6. Feature importance ───────────────────────────────────────────
# feature_importance(model, feature_names, top_n=20) -> plotly Figure
# Uses model.feature_importances_ for tree-based models.

fig_fi = viz.feature_importance(clf, feature_names)
assert fig_fi is not None
assert "Feature Importance" in fig_fi.layout.title.text

# Limit to top 3 features
fig_fi_top3 = viz.feature_importance(clf, feature_names, top_n=3)
assert fig_fi_top3 is not None

# Fallback: models with coef_ attribute (logistic regression)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42, max_iter=200)
lr.fit(X_train, y_train)
fig_fi_lr = viz.feature_importance(lr, feature_names)
assert fig_fi_lr is not None

# ── 7. Learning curve ───────────────────────────────────────────────
# learning_curve(model, X, y, cv=5, train_sizes=None) -> plotly Figure
# Shows training vs validation score across training set sizes.

fig_lc = viz.learning_curve(
    RandomForestClassifier(n_estimators=10, random_state=42),
    X_train,
    y_train,
    cv=3,
    train_sizes=[0.3, 0.6, 1.0],
)
assert fig_lc is not None
assert fig_lc.layout.title.text == "Learning Curve"

# ── 8. Residuals (regression) ───────────────────────────────────────
# residuals(y_true, y_pred) -> plotly Figure with two panels:
# predicted-vs-actual scatter + residual distribution histogram

X_reg, y_reg = make_regression(n_samples=100, n_features=3, noise=10.0, random_state=42)
reg = RandomForestRegressor(n_estimators=20, random_state=42)
reg.fit(X_reg[:80], y_reg[:80])
y_reg_pred = reg.predict(X_reg[80:])

fig_res = viz.residuals(y_reg[80:], y_reg_pred)
assert fig_res is not None
assert fig_res.layout.title.text == "Residual Analysis"

# ── 9. Calibration curve ────────────────────────────────────────────
# calibration_curve(y_true, y_proba, n_bins=10) -> plotly Figure

fig_cal = viz.calibration_curve(y_test, y_proba, n_bins=5)
assert fig_cal is not None
assert fig_cal.layout.title.text == "Calibration Curve"

# ── 10. Metric comparison ───────────────────────────────────────────
# metric_comparison(results) -> plotly Figure
# Compares metrics across multiple models as grouped bars.

comparison_results = {
    "RandomForest": {"accuracy": 0.95, "f1": 0.93, "precision": 0.94},
    "LogisticRegression": {"accuracy": 0.88, "f1": 0.85, "precision": 0.87},
    "GradientBoosting": {"accuracy": 0.97, "f1": 0.96, "precision": 0.95},
}

fig_mc = viz.metric_comparison(comparison_results)
assert fig_mc is not None
assert fig_mc.layout.title.text == "Model Comparison"
assert fig_mc.layout.barmode == "group"

# ── 11. Training history ────────────────────────────────────────────
# training_history(metrics, x_label="Epoch") -> plotly Figure
# Plots loss curves or other per-epoch metrics.

history = {
    "train_loss": [0.9, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08],
    "val_loss": [1.0, 0.8, 0.65, 0.55, 0.5, 0.48, 0.47, 0.46],
}

fig_th = viz.training_history(history)
assert fig_th is not None
assert fig_th.layout.title.text == "Training History"
assert fig_th.layout.xaxis.title.text == "Epoch"

# Custom x_label
fig_th_step = viz.training_history(history, x_label="Step")
assert fig_th_step.layout.xaxis.title.text == "Step"

# ── 12. Edge case: feature importance with mismatched names ─────────

try:
    viz.feature_importance(clf, ["a", "b"])  # Only 2 names for 5 features
    assert False, "Should raise ValueError for mismatched feature_names"
except ValueError:
    pass  # Expected: length mismatch

# ── 13. Edge case: feature importance without attributes ─────────────
# Model with no feature_importances_ or coef_ and no X/y fallback


class BareModel:
    """Model with no importance attributes."""

    def predict(self, X):
        return np.zeros(len(X))


try:
    viz.feature_importance(BareModel(), ["a", "b", "c"])
    assert False, "Should raise ValueError without importance attributes"
except ValueError:
    pass  # Expected: no feature_importances_ or coef_

# ── 14. Edge case: empty metric_comparison ──────────────────────────

try:
    viz.metric_comparison({})
    assert False, "Should raise ValueError for empty results"
except ValueError:
    pass  # Expected: must contain at least one model

# ── 15. Edge case: empty training_history ────────────────────────────

try:
    viz.training_history({})
    assert False, "Should raise ValueError for empty metrics"
except ValueError:
    pass  # Expected: must contain at least one series

# ── 16. Feature importance fallback to permutation importance ────────
# When model has no feature_importances_ or coef_, but X and y provided

fig_perm = viz.feature_importance(
    BareModel(),
    ["a", "b", "c"],
    X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
    y=np.array([0, 0, 1, 1]),
)
assert fig_perm is not None

print("PASS: 05-ml/03_model_visualizer")
