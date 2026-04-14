# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 3 — Dimensionality Reduction.

Contains: data loading, scaling, common output directory, KMeans-based
silhouette evaluation in the embedding space. Technique-specific code
(PCA/KPCA/t-SNE/UMAP algorithms and their plots) lives in the per-
technique files, NOT here.

    from shared.mlfp04.ex_3 import (
        OUTPUT_DIR, load_customer_matrix, evaluate_embedding_silhouette,
    )
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# OUTPUT + REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "ex3_dimreduce"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — E-commerce customers (reused from MLFP03)
# ════════════════════════════════════════════════════════════════════════


def load_customer_matrix() -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Load e-commerce customers, standardise numeric features.

    Returns:
        X          : (n_samples, n_features) standardised float matrix
        feature_cols: list of feature column names in order
        df_raw     : the raw polars DataFrame before scaling
    """
    loader = MLFPDataLoader()
    customers = loader.load("mlfp03", "ecommerce_customers.parquet")

    feature_cols = [
        c
        for c, d in zip(customers.columns, customers.dtypes)
        if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and c not in ("customer_id",)
    ]

    df_clean = customers.drop_nulls(subset=feature_cols)
    X_raw, _, _ = to_sklearn_input(df_clean, feature_columns=feature_cols)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, feature_cols, df_clean


# ════════════════════════════════════════════════════════════════════════
# EMBEDDING-SPACE CLUSTER QUALITY
# ════════════════════════════════════════════════════════════════════════


def evaluate_embedding_silhouette(
    embedding: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> float:
    """Fit KMeans in the embedding space and return the silhouette score.

    This is the standard "does the reducer preserve structure?" probe used
    across all five technique files. Returns -1.0 when only one cluster is
    found (e.g. collapsed embedding).
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5)
    labels = km.fit_predict(embedding)
    if len(set(labels)) < 2:
        return -1.0
    return float(silhouette_score(embedding, labels))


# ════════════════════════════════════════════════════════════════════════
# SUBSAMPLING — used by KPCA / t-SNE / UMAP / Isomap for kernel-cost paths
# ════════════════════════════════════════════════════════════════════════


def subsample_indices(
    n_samples: int, n_target: int, random_state: int = RANDOM_STATE
) -> np.ndarray:
    """Deterministic subsample indices for expensive O(n^2) methods."""
    rng = np.random.default_rng(random_state)
    return rng.choice(n_samples, min(n_target, n_samples), replace=False)
