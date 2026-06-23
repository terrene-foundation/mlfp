# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 1: Customer Segmentation by Clustering (Reference)

Reference implementation. Withheld from students. Verified to pass grader.py.
Framework-first: clustering runs through kailash-ml ClusteringEngine.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

SEED = 20260401
FEATURES = [
    "recency_days",
    "frequency",
    "monetary_sgd",
    "tenure_months",
    "avg_basket_sgd",
]


def make_customers() -> pl.DataFrame:
    """Deterministic loyalty cohort with four planted spending personas.

    The four persona centroids and per-feature spreads are fixed; the planted
    persona id is NOT returned (this is unsupervised discovery).
    """
    rng = np.random.default_rng(SEED)
    centers = np.array(
        [
            [5.0, 50.0, 2000.0, 60.0, 180.0],  # champions
            [40.0, 10.0, 400.0, 12.0, 60.0],  # new low-value
            [90.0, 3.0, 150.0, 48.0, 45.0],  # dormant at-risk
            [15.0, 25.0, 1200.0, 36.0, 220.0],  # loyal big-basket
        ]
    )
    spreads = np.array([3.0, 3.0, 120.0, 5.0, 18.0])
    sizes = [320, 300, 280, 300]
    blocks = []
    for c, n in zip(centers, sizes):
        blocks.append(c + rng.normal(0, 1, (n, 5)) * spreads)
    X = np.vstack(blocks)
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    return pl.DataFrame({col: X[:, j] for j, col in enumerate(FEATURES)})


def solve() -> dict:
    """Recover the planted personas with the kailash-ml ClusteringEngine."""
    df = make_customers()

    # Standardise to z-scores — load-bearing: raw scales differ by 40x.
    zdf = df.select(
        [((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in FEATURES]
    )

    engine = ClusteringEngine()

    # Objective K selection via the silhouette sweep.
    sweep = engine.sweep_k(zdf, range(2, 9), algorithm="kmeans", criterion="silhouette")
    best_k = int(sweep.optimal_k)

    # Final fit at the recovered K.
    fit = engine.fit(zdf, algorithm="kmeans", n_clusters=best_k)
    labels = [int(v) for v in fit.labels]

    return {
        "labels": labels,
        "n_clusters": best_k,
        "silhouette": float(fit.silhouette_score),
    }


if __name__ == "__main__":
    out = solve()
    sizes = np.bincount(np.array(out["labels"]))
    print(f"n_clusters : {out['n_clusters']}")
    print(f"silhouette : {out['silhouette']:.4f}")
    print(f"sizes      : {sizes.tolist()}")
