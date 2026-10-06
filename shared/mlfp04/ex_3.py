# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for MLFP04 Exercise 3 — Dimensionality Reduction.

Contains: data loading, scaling, common output directory, embedding
quality metrics (neighbourhood preservation via trustworthiness / kNN
overlap, plus KMeans silhouette as a "clusterability" probe), and
subsampling helpers. Technique-specific code (PCA/KPCA/t-SNE/UMAP
algorithms and their plots) lives in the per-technique files, NOT here.

    from shared.mlfp04.ex_3 import (
        OUTPUT_DIR, load_customer_matrix, evaluate_embedding,
    )
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kailash_ml import ExperimentTracker
from kailash_ml.interop import to_sklearn_input

from shared import MLFPDataLoader

# ════════════════════════════════════════════════════════════════════════
# OUTPUT + REPRODUCIBILITY
# ════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("outputs") / "ex3_dimreduce"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 4
DEFAULT_QUALITY_NEIGHBOURS = 10
QUALITY_MAX_ROWS = 3000  # trustworthiness is O(n^2) memory — cap the rows

# Behavioural features only. `churned` is an OUTCOME label, not behaviour:
# it is kept on the raw frame for post-hoc profiling but never enters the
# reducer (a standardised 0/1 column would dominate Euclidean structure).
# satisfaction_score (1-5) and num_returns (0-6) are deliberately kept as
# ordinal counts, standardised like the continuous columns.
CUSTOMER_FEATURES = [
    "total_revenue",
    "order_count",
    "avg_order_value",
    "days_since_last_order",
    "customer_tenure_days",
    "satisfaction_score",
    "num_returns",
]

# ════════════════════════════════════════════════════════════════════════
# DATA LOADING — E-commerce customers (reused from MLFP03)
# ════════════════════════════════════════════════════════════════════════


def load_customer_matrix() -> tuple[np.ndarray, list[str], pl.DataFrame]:
    """Load e-commerce customers, standardise the behavioural features.

    Returns:
        X          : (n_samples, n_features) standardised float matrix
                     (50,000 x 7 on the shipped dataset)
        feature_cols: list of feature column names in order
        df_raw     : the raw polars DataFrame before scaling (still holds
                     `churned` and the categorical columns for profiling)
    """
    loader = MLFPDataLoader()
    customers = loader.load("mlfp03", "ecommerce_customers.parquet")

    feature_cols = list(CUSTOMER_FEATURES)

    df_clean = customers.drop_nulls(subset=feature_cols)
    X_raw, _, _ = to_sklearn_input(df_clean, feature_columns=feature_cols)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    return X, feature_cols, df_clean


# ════════════════════════════════════════════════════════════════════════
# EMBEDDING QUALITY — neighbourhood preservation + clusterability
# ════════════════════════════════════════════════════════════════════════
# "Does the reducer preserve structure?" is answered by comparing each
# point's neighbours BEFORE and AFTER the reduction:
#   - trustworthiness (sklearn): penalises points that become neighbours in
#     the embedding although they were far apart in the original space.
#     1.0 = no false neighbours; ~0.5 = random layout.
#   - kNN overlap: average fraction of each point's k original neighbours
#     that are still among its k neighbours in the embedding.
#
# K-means silhouette IN THE EMBEDDING is a different question — "how
# blob-like is the picture?" t-SNE and UMAP deliberately pull points into
# tight blobs, so they inflate silhouette by construction. Treat it as a
# clusterability probe, never as proof that structure was preserved.


def knn_overlap(
    X_high: np.ndarray,
    embedding: np.ndarray,
    n_neighbors: int = DEFAULT_QUALITY_NEIGHBOURS,
) -> float:
    """Mean fraction of original k-NN that survive in the embedding."""
    nn_high = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_high)
    nn_low = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embedding)
    idx_high = nn_high.kneighbors(X_high, return_distance=False)[:, 1:]
    idx_low = nn_low.kneighbors(embedding, return_distance=False)[:, 1:]
    shared = [len(set(a) & set(b)) for a, b in zip(idx_high, idx_low)]
    return float(np.mean(shared) / n_neighbors)


def evaluate_embedding(
    X_high: np.ndarray,
    embedding: np.ndarray,
    n_neighbors: int = DEFAULT_QUALITY_NEIGHBOURS,
    max_rows: int = QUALITY_MAX_ROWS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Score an embedding: trustworthiness, kNN overlap, silhouette.

    `X_high` and `embedding` must be row-aligned. At most `max_rows` rows
    are scored (deterministic subsample) because trustworthiness needs the
    full pairwise distance matrix.
    """
    X_high = np.asarray(X_high)
    embedding = np.asarray(embedding)
    if X_high.shape[0] != embedding.shape[0]:
        raise ValueError(
            f"X_high has {X_high.shape[0]} rows but embedding has "
            f"{embedding.shape[0]} — they must be row-aligned"
        )
    if X_high.shape[0] > max_rows:
        rows = subsample_indices(X_high.shape[0], max_rows, random_state)
        X_high, embedding = X_high[rows], embedding[rows]
    return {
        "trustworthiness": float(
            trustworthiness(X_high, embedding, n_neighbors=n_neighbors)
        ),
        "knn_overlap": knn_overlap(X_high, embedding, n_neighbors),
        "silhouette": evaluate_embedding_silhouette(embedding),
    }


def evaluate_embedding_silhouette(
    embedding: np.ndarray,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> float:
    """Fit KMeans in the embedding space and return the silhouette score.

    A CLUSTERABILITY probe ("how separable are K-means blobs in this
    picture?"), not a structure-preservation metric — see the note above.
    Returns -1.0 when only one cluster is found (e.g. collapsed embedding).
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


def holdout_indices(
    n_samples: int,
    exclude: np.ndarray,
    n_target: int,
    random_state: int = RANDOM_STATE,
) -> np.ndarray:
    """Deterministic subsample of rows NOT in `exclude`.

    Used for genuine out-of-sample transforms: the rows a reducer is
    applied to must not include the rows it was fitted on.
    """
    remaining = np.setdiff1d(np.arange(n_samples), exclude)
    rng = np.random.default_rng(random_state)
    return rng.choice(remaining, min(n_target, len(remaining)), replace=False)


# ════════════════════════════════════════════════════════════════════════
# KAILASH-ML EXPERIMENT TRACKER — shared by every dim-reduction technique
# ════════════════════════════════════════════════════════════════════════
# Every M4 ex_3 lesson logs its sweep + final-fit metrics to a single
# SQLite store so students can compare PCA / Kernel-PCA / t-SNE / UMAP
# embedding-quality runs after the lesson group ends. Mirrors the ex_1
# clustering-zoo pattern; separate DB so dim-reduction has its own
# leaderboard distinct from clustering.

DIMREDUCE_DB = "sqlite:///mlfp04_ex3_dimreduction.db"
EXPERIMENT_NAME = "m4_dimreduction_zoo"


async def _setup_engines_async() -> tuple[ExperimentTracker, str]:
    """Open the dim-reduction ExperimentTracker."""
    tracker = await ExperimentTracker.create(store_url=DIMREDUCE_DB)
    return tracker, EXPERIMENT_NAME


def setup_engines() -> tuple[ExperimentTracker, str]:
    """Sync wrapper. Returns (tracker, experiment_name)."""
    return asyncio.run(_setup_engines_async())


def teardown_engines(tracker: ExperimentTracker) -> None:
    """Drain the aiosqlite worker threads before the script returns.

    kailash's AsyncSQLitePool spawns NON-DAEMON aiosqlite worker threads on
    first pool use. Python 3.13's ``Py_FinalizeEx`` joins non-daemon threads
    BEFORE running ``atexit`` handlers, so an atexit-based close runs too
    late — the interpreter hangs forever in ``wait_for_thread_shutdown``
    waiting on workers stuck in ``queue.get()``.

    Solutions MUST call ``teardown_engines(tracker)`` after the REFLECTION
    block. See ``rules/patterns.md`` § "Async Resource Cleanup".
    """
    asyncio.run(tracker.close())


async def _track_run_async(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Log one lesson's run: scalar metrics + optional per-step series."""
    async with tracker.track(experiment=exp_name, run_name=run_name) as run:
        await run.log_params({k: str(v) for k, v in params.items()})
        for name, value in scalar_metrics.items():
            await run.log_metric(name, float(value))
        if series_metrics:
            for name, values in series_metrics.items():
                for step, value in enumerate(values, start=1):
                    await run.log_metric(name, float(value), step=step)


def track_run(
    tracker: ExperimentTracker,
    exp_name: str,
    run_name: str,
    params: dict[str, Any],
    scalar_metrics: dict[str, float],
    series_metrics: dict[str, list[float]] | None = None,
) -> None:
    """Sync wrapper for logging a single technique's run."""
    asyncio.run(
        _track_run_async(
            tracker, exp_name, run_name, params, scalar_metrics, series_metrics
        )
    )
