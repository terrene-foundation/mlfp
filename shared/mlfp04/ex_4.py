# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 4 — Anomaly Detection and Ensembles.

Contains: data loading (e-commerce customers + rare-return anomaly label),
feature engineering, score normalisation helpers, metric reporting,
visualisation shortcuts.

Technique-specific code (Z-score thresholding, Isolation Forest fit, LOF
neighbour count, EnsembleEngine blend weights) does NOT belong here — it
lives in the per-technique files in `modules/mlfp04/solutions/ex_4/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
np.random.seed(42)

OUTPUT_DIR = Path("outputs") / "ex4_anomaly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ANOMALY_QUANTILE = 0.99
FEATURE_BLOCKLIST = {
    "is_fraud",
    "customer_id",
    "ltv_tier",
    "product_categories",
    "review_text",
    "region",
    "device_type",
    "payment_method",
    "loyalty_member",
    "churned",
}


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — E-commerce customer data with rare-return anomaly label
# ════════════════════════════════════════════════════════════════════════
# The dataset ships with the mlfp03 module. We reuse it here because the
# anomaly story lives in the top 1% of return rates — a natural rare-event
# signal for unsupervised anomaly detection methods to find.


def load_anomaly_frame(quantile: float = ANOMALY_QUANTILE) -> pl.DataFrame:
    """Load e-commerce customers and attach a 1% rare-return anomaly label.

    Returns a polars DataFrame with an additional `is_fraud` column
    (1 where num_returns is in the top (1-quantile) percentile, else 0).
    """
    loader = MLFPDataLoader()
    raw = loader.load("mlfp03", "ecommerce_customers.parquet")
    threshold = raw["num_returns"].quantile(quantile)
    return raw.with_columns(
        (pl.col("num_returns") >= threshold).cast(pl.Int64).alias("is_fraud")
    )


def build_features(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drop nulls, pick numeric features, standardise and return (X, y, cols).

    Returns standardised X (float64), y (int), and the feature column names.
    The returned X is suitable for sklearn-style anomaly detectors.
    """
    feature_cols = [c for c in frame.columns if c not in FEATURE_BLOCKLIST]
    X, y, _col_info = to_sklearn_input(
        frame.drop_nulls(),
        feature_columns=feature_cols,
        target_column="is_fraud",
    )
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float64)
    return X_scaled, y.astype(int), feature_cols


def load_dataset() -> tuple[np.ndarray, np.ndarray, list[str], pl.DataFrame]:
    """One-call helper: load the frame, build features, return everything."""
    frame = load_anomaly_frame()
    X, y, cols = build_features(frame)
    return X, y, cols, frame


# ════════════════════════════════════════════════════════════════════════
# SCORE HELPERS
# ════════════════════════════════════════════════════════════════════════
# Anomaly detectors emit scores on wildly different scales. Normalising to
# [0, 1] (or to a rank) is what makes blending across methods possible.


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise an anomaly score array to [0, 1]."""
    scores = np.asarray(scores, dtype=np.float64)
    span = scores.max() - scores.min()
    return (scores - scores.min()) / (span + 1e-10)


def rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Convert an anomaly score array to percentile ranks in [0, 1]."""
    from scipy.stats import rankdata

    return rankdata(np.asarray(scores, dtype=np.float64)) / len(scores)


def score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Return AUC-ROC and average precision (AUC-PR) for an anomaly score."""
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "avg_precision": float(average_precision_score(y_true, scores)),
    }


def print_metrics(
    name: str, y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Compute metrics, print them on one line, and return the dict."""
    m = score_metrics(y_true, scores)
    print(f"  {name:<24} AUC-ROC={m['auc_roc']:.4f}  " f"AP={m['avg_precision']:.4f}")
    return m


def precision_at_recall(
    y_true: np.ndarray, scores: np.ndarray, target_recall: float
) -> tuple[float, float]:
    """Return (precision, threshold) at the TIGHTEST point where recall >= target.

    sklearn returns precisions/recalls ordered by ascending threshold, so
    recall decreases as threshold increases. We want the highest threshold
    that still meets the recall target — i.e. the last index where recall
    is still >= the target, which gives the maximum precision for that
    recall level.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    # Drop the sentinel last point (precision=1.0, recall=0.0, no threshold)
    ps = precisions[:-1]
    rs = recalls[:-1]
    ts = thresholds
    mask = rs >= target_recall
    if not mask.any():
        return float(ps[0]), float(ts[0])
    # The tightest threshold satisfying the recall target is the largest
    # index where mask is True (thresholds are ascending).
    idx = int(np.where(mask)[0][-1])
    return float(ps[idx]), float(ts[idx])


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ════════════════════════════════════════════════════════════════════════


def write_comparison_chart(
    comparison: dict[str, dict[str, float]], filename: str
) -> Path:
    """Render a kailash-ml ModelVisualizer metric_comparison chart to HTML."""
    from kailash_ml import ModelVisualizer

    viz = ModelVisualizer()
    fig = viz.metric_comparison(comparison)
    fig.update_layout(title="Anomaly Detection Method Comparison")
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


def write_roc_chart(
    y_true: np.ndarray, scores: np.ndarray, name: str, filename: str
) -> Path:
    """Render a ROC curve for a single detector."""
    from kailash_ml import ModelVisualizer

    viz = ModelVisualizer()
    fig = viz.roc_curve(y_true, scores)
    fig.update_layout(title=f"ROC — {name}")
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


def write_monitoring_chart(anomaly_rates: list[float], filename: str) -> Path:
    """Render an anomaly-rate-over-time chart for production monitoring."""
    from kailash_ml import ModelVisualizer

    viz = ModelVisualizer()
    fig = viz.training_history(
        {"Anomaly Rate %": [r * 100 for r in anomaly_rates]},
        x_label="Time Window",
    )
    fig.update_layout(title="Anomaly Rate Over Time (Production Monitoring)")
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    return path


# ════════════════════════════════════════════════════════════════════════
# ENSEMBLE ADAPTER
# ════════════════════════════════════════════════════════════════════════
# kailash-ml EnsembleEngine.blend() expects estimator-shaped objects with
# predict_proba. Each detector in this exercise has already produced a
# score vector, so we wrap those vectors in a minimal estimator.


class AnomalyScoreEstimator:
    """Minimal sklearn-shaped wrapper exposing precomputed scores.

    EnsembleEngine.blend() calls predict_proba(X) on every estimator and
    averages the resulting class-1 probabilities. This adapter normalises
    the underlying scores to [0, 1] and returns them as P(anomaly).
    """

    def __init__(self, scores: np.ndarray):
        self._scores = np.asarray(scores, dtype=np.float64)
        self._norm = normalise_scores(self._scores)
        self.classes_ = np.array([0, 1])

    def fit(self, X: Any, y: Any = None) -> "AnomalyScoreEstimator":
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else self._norm.shape[0]
        norm = self._norm[:n]
        return np.column_stack([1.0 - norm, norm])

    def predict(self, X: Any) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else self._scores.shape[0]
        threshold = float(np.median(self._scores))
        return (self._scores[:n] > threshold).astype(int)
