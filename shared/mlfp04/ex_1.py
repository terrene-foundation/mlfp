# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 1 — Clustering Zoo.

Contains: customer-feature loading & standardisation, metric helpers,
subsampling utilities, and output-directory management.

Technique-specific code (K-means elbow, linkage methods, DBSCAN epsilon
search, HDBSCAN, spectral Laplacian, AutoMLEngine) does NOT belong
here — it lives in the per-technique files under modules/mlfp04/solutions/ex_1/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from kailash_ml.interop import to_sklearn_input

from shared.data_loader import MLFPDataLoader
from shared.kailash_helpers import setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()

OUTPUT_DIR = Path("outputs") / "ex1_clustering"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Shared random state so every technique file is reproducible
RANDOM_STATE: int = 42


# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — Singapore e-commerce customers
# ════════════════════════════════════════════════════════════════════════


def load_customers() -> tuple[pl.DataFrame, list[str]]:
    """Load the e-commerce customer dataset and return (df, numeric_feature_cols).

    The dataset (from MLFP03) is ~6K rows of Singapore e-commerce customers
    with recency, frequency, monetary, basket-size, and channel features.
    Clustering is unsupervised segmentation: no labels, just behaviour.
    """
    loader = MLFPDataLoader()
    customers = loader.load("mlfp03", "ecommerce_customers.parquet")
    numeric_types = (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    feature_cols = [
        c
        for c, d in zip(customers.columns, customers.dtypes)
        if d in numeric_types and c not in ("customer_id",)
    ]
    return customers.drop_nulls(subset=feature_cols), feature_cols


def standardise(
    df: pl.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, StandardScaler]:
    """Return (X_scaled, scaler). Zero mean, unit variance — mandatory for
    all distance-based clustering."""
    X, _, _ = to_sklearn_input(df, feature_columns=feature_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


# ════════════════════════════════════════════════════════════════════════
# SUBSAMPLING — spectral / hierarchical are O(n^2) or worse
# ════════════════════════════════════════════════════════════════════════


def subsample(
    X: np.ndarray, n: int, seed: int = RANDOM_STATE
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_sub, idx) where idx are the original row indices chosen."""
    rng = np.random.default_rng(seed)
    n = min(n, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    return X[idx], idx


# ════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════


def score_partition(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Compute silhouette, Calinski-Harabasz, Davies-Bouldin.

    Points with label == -1 (DBSCAN noise) are excluded. Returns NaN
    fields if fewer than 2 valid clusters remain.
    """
    valid = labels != -1
    labs = labels[valid]
    data = X[valid]
    if data.shape[0] < 2 or len(set(labs.tolist())) < 2:
        return {
            "n_clusters": len(set(labs.tolist())),
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    return {
        "n_clusters": len(set(labs.tolist())),
        "silhouette": float(silhouette_score(data, labs)),
        "calinski_harabasz": float(calinski_harabasz_score(data, labs)),
        "davies_bouldin": float(davies_bouldin_score(data, labs)),
    }


def agreement(labels_a: np.ndarray, labels_b: np.ndarray) -> dict[str, float]:
    """ARI and NMI between two label vectors, ignoring noise (-1)."""
    valid = (labels_a >= 0) & (labels_b >= 0)
    if valid.sum() < 2:
        return {"ari": float("nan"), "nmi": float("nan")}
    return {
        "ari": float(adjusted_rand_score(labels_a[valid], labels_b[valid])),
        "nmi": float(normalized_mutual_info_score(labels_a[valid], labels_b[valid])),
    }


def print_metric_row(name: str, m: dict[str, Any]) -> None:
    """One-line summary of a partition's metrics."""
    print(
        f"  {name:<14} K={m['n_clusters']:>3}  "
        f"sil={m['silhouette']:>7.4f}  "
        f"CH={m['calinski_harabasz']:>9.0f}  "
        f"DB={m['davies_bouldin']:>7.4f}"
    )


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION OUTPUT PATH
# ════════════════════════════════════════════════════════════════════════


def out_path(filename: str) -> Path:
    """Return a path under OUTPUT_DIR for a visualisation artefact."""
    return OUTPUT_DIR / filename
