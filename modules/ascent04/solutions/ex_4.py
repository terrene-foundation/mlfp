# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT04 — Exercise 4: Anomaly Detection and Ensembles
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Dimensionality reduction on fraud data, anomaly detection
#   with multiple detectors, and ensemble scoring with EnsembleEngine.blend().
#
# TASKS:
#   1. Load fraud data and reduce dimensionality with UMAP
#   2. Compare UMAP vs t-SNE embeddings
#   3. Isolation Forest anomaly scoring
#   4. LOF and additional anomaly detectors
#   5. Ensemble anomaly scores with EnsembleEngine
#   6. Evaluate and visualise detection performance
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml import ModelVisualizer, EnsembleEngine
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader

try:
    import umap
except ImportError:
    umap = None


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
fraud = loader.load("ascent04", "credit_card_fraud.parquet")

print(f"=== Credit Card Fraud Data ===")
print(f"Shape: {fraud.shape}")
print(f"Fraud rate: {fraud['is_fraud'].mean():.4%}")

feature_cols = [c for c in fraud.columns if c not in ("is_fraud", "transaction_id")]

X, y, col_info = to_sklearn_input(
    fraud.drop_nulls(),
    feature_columns=feature_cols,
    target_column="is_fraud",
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Dimensionality reduction with UMAP
# ══════════════════════════════════════════════════════════════════════

# Sample for visualisation (UMAP on full 284K is slow)
rng = np.random.default_rng(42)
sample_size = min(20_000, len(y))
idx = rng.choice(len(y), sample_size, replace=False)
X_sample = X_scaled[idx]
y_sample = y[idx]

if umap is not None:
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_umap = reducer.fit_transform(X_sample)
    print(f"\nUMAP embedding: {embedding_umap.shape}")
else:
    from sklearn.decomposition import PCA

    embedding_umap = PCA(n_components=2, random_state=42).fit_transform(X_sample)
    print("\nUMAP not installed, using PCA fallback")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compare with t-SNE
# ══════════════════════════════════════════════════════════════════════

from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
embedding_tsne = tsne.fit_transform(X_sample[:5000])  # t-SNE is O(n²)

print(f"t-SNE embedding: {embedding_tsne.shape}")
print("\nUMAP vs t-SNE:")
print("  UMAP: preserves global structure, faster, supports transform()")
print("  t-SNE: better local structure, O(n²), no out-of-sample transform")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Isolation Forest
# ══════════════════════════════════════════════════════════════════════
# Theory: anomalies are isolated in fewer random partitions.
# Average path length in random trees is shorter for outliers.

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.002,  # Expected fraud rate
    random_state=42,
    n_jobs=-1,
)
iso_scores = -iso_forest.fit(X_scaled).score_samples(
    X_scaled
)  # Higher = more anomalous
iso_labels = iso_forest.predict(X_scaled)  # -1 = anomaly, 1 = normal

iso_auc = roc_auc_score(y, iso_scores)
iso_ap = average_precision_score(y, iso_scores)

print(f"\n=== Isolation Forest ===")
print(f"AUC-ROC: {iso_auc:.4f}")
print(f"Average Precision: {iso_ap:.4f}")
print(f"Predicted anomalies: {(iso_labels == -1).sum():,} / {len(iso_labels):,}")
print(f"True frauds: {y.sum():.0f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Local Outlier Factor
# ══════════════════════════════════════════════════════════════════════

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.002, novelty=False)
lof_labels = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_  # Higher = more anomalous

lof_auc = roc_auc_score(y, lof_scores)
lof_ap = average_precision_score(y, lof_scores)

print(f"\n=== Local Outlier Factor ===")
print(f"AUC-ROC: {lof_auc:.4f}")
print(f"Average Precision: {lof_ap:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Ensemble anomaly scores with EnsembleEngine.blend()
# ══════════════════════════════════════════════════════════════════════
# EnsembleEngine.blend() averages predictions from multiple estimators.
# For supervised tasks, blend() also supports stack() and bag().
# For anomaly detection, we normalise scores to [0,1] and blend.


# Normalise scores to [0, 1] for blending
def normalise_scores(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)


iso_norm = normalise_scores(iso_scores)
lof_norm = normalise_scores(lof_scores)

# Method A: Manual weighted average (AUC-weighted)
w_iso = iso_auc / (iso_auc + lof_auc)
w_lof = lof_auc / (iso_auc + lof_auc)
ensemble_scores_manual = w_iso * iso_norm + w_lof * lof_norm

# Method B: EnsembleEngine.blend() on supervised anomaly wrappers
# EnsembleEngine.blend() works with sklearn-compatible estimators.
# We wrap the detectors via a scoring API and blend predictions.
from kailash_ml import EnsembleEngine

# Build sklearn-style wrapper for Isolation Forest (predict_proba interface)
from sklearn.base import BaseEstimator, ClassifierMixin


class AnomalyScorer(BaseEstimator, ClassifierMixin):
    """Wraps an anomaly detector to expose predict_proba for EnsembleEngine."""

    def __init__(self, detector, scores: np.ndarray):
        self.detector = detector
        self._scores = scores
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        # For blending we return precomputed scores as class-1 probability
        norm = normalise_scores(self._scores[: len(X)])
        return np.column_stack([1 - norm, norm])

    def predict(self, X):
        return (self._scores[: len(X)] > np.median(self._scores)).astype(int)


iso_scorer = AnomalyScorer(iso_forest, iso_scores)
lof_scorer = AnomalyScorer(lof, lof_scores)

engine = EnsembleEngine()
blended_proba = engine.blend(
    estimators=[iso_scorer, lof_scorer],
    X=X_scaled,
    weights=[w_iso, w_lof],
)
ensemble_scores = blended_proba[:, 1]

ensemble_auc = roc_auc_score(y, ensemble_scores)
ensemble_ap = average_precision_score(y, ensemble_scores)

print(f"\n=== Ensemble via EnsembleEngine.blend() ===")
print(f"Weights: IF={w_iso:.3f}, LOF={w_lof:.3f}")
print(f"AUC-ROC: {ensemble_auc:.4f}")
print(f"Average Precision: {ensemble_ap:.4f}")

# Improvement over individual detectors
print(f"\nImprovement over best individual:")
best_individual = max(iso_auc, lof_auc)
print(f"  AUC-ROC: {ensemble_auc - best_individual:+.4f}")

print("\nEnsembleEngine methods:")
print("  blend()  — weighted average of predictions (soft voting)")
print("  stack()  — meta-learner trained on base model outputs")
print("  bag()    — bootstrap aggregation (bagging)")
print("  boost()  — sequential boosting on residuals")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Evaluate and visualise
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# ROC curves
for name, scores in [
    ("IsolationForest", iso_scores),
    ("LOF", lof_scores),
    ("Ensemble", ensemble_scores),
]:
    fig = viz.roc_curve(y, scores)
    fig.update_layout(title=f"ROC: {name}")
    fig.write_html(f"ex2_roc_{name.lower()}.html")

# Comparison
comparison = {
    "Isolation Forest": {"AUC_ROC": iso_auc, "Avg_Precision": iso_ap},
    "LOF": {"AUC_ROC": lof_auc, "Avg_Precision": lof_ap},
    "Ensemble": {"AUC_ROC": ensemble_auc, "Avg_Precision": ensemble_ap},
}
fig = viz.metric_comparison(comparison)
fig.update_layout(title="Anomaly Detection Comparison")
fig.write_html("ex2_anomaly_comparison.html")
print("\nSaved: ex2_anomaly_comparison.html")

# Precision-recall at different thresholds
precision, recall, thresholds = precision_recall_curve(y, ensemble_scores)
print(
    f"\nAt recall=0.80: precision={precision[np.searchsorted(-recall[::-1], -0.80)]:.4f}"
)

print("\n✓ Exercise 4 complete — UMAP + anomaly detection ensemble")
