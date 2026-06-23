# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 2: Dim Reduction & Anomaly Detection (Reference)

Reference implementation. Withheld from students. Verified to pass grader.py.
Framework-first: PCA via DimReductionEngine, detection via AnomalyDetectionEngine.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine
from kailash_ml.engines.dim_reduction import DimReductionEngine

SEED = 20260402
N_NORMAL = 975
N_ANOM = 25
D = 24
K_LATENT = 3
CONTAMINATION = 0.025


def make_sensor_matrix() -> pl.DataFrame:
    """Deterministic 24-channel telemetry on a 3-factor manifold + 25 outliers.

    The planted anomaly flag is NOT returned (unsupervised detection).
    """
    rng = np.random.default_rng(SEED)
    Z = rng.normal(0, 1, (N_NORMAL, K_LATENT))
    W = rng.normal(0, 1, (K_LATENT, D)) * 3.5
    X_normal = Z @ W + rng.normal(0, 0.5, (N_NORMAL, D))
    X_anom = rng.normal(12.0, 4.0, (N_ANOM, D)) * rng.choice([-1, 1], (N_ANOM, D))
    X = np.vstack([X_normal, X_anom])
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    cols = [f"f{i:02d}" for i in range(D)]
    return pl.DataFrame({c: X[:, j] for j, c in enumerate(cols)})


def solve() -> dict:
    """Compress with PCA and flag off-manifold rows — kailash-ml engines."""
    df = make_sensor_matrix()
    dre = DimReductionEngine()

    # Intrinsic dimensionality: smallest k reaching >=90% cumulative variance.
    full = dre.reduce(df, algorithm="pca", n_components=df.width)
    evr = np.asarray(full.explained_variance_ratio)
    cum = np.cumsum(evr)
    n_components_90 = int(np.searchsorted(cum, 0.90) + 1)

    # Compress at that rank and read reconstruction error.
    reduced = dre.reduce(df, algorithm="pca", n_components=n_components_90)
    reconstruction_error = float(reduced.reconstruction_error)

    # Anomaly detection over the raw channels.
    ade = AnomalyDetectionEngine()
    res = ade.detect(df, algorithm="isolation_forest", contamination=CONTAMINATION)
    scores = [float(s) for s in res.scores]

    # Engine labels: map to 1 = anomaly, 0 = normal (engine flags anomalies as -1).
    raw_labels = np.asarray(res.labels)
    anomaly_labels = [int(v) for v in (raw_labels == -1).astype(int)]
    n_anomalies = int(sum(anomaly_labels))

    return {
        "n_components_90": n_components_90,
        "reconstruction_error": reconstruction_error,
        "anomaly_scores": scores,
        "anomaly_labels": anomaly_labels,
        "n_anomalies": n_anomalies,
    }


if __name__ == "__main__":
    out = solve()
    print(f"n_components_90       : {out['n_components_90']}")
    print(f"reconstruction_error : {out['reconstruction_error']:.4f}")
    print(f"n_anomalies          : {out['n_anomalies']}")
    print(
        f"score range          : "
        f"{min(out['anomaly_scores']):.3f}..{max(out['anomaly_scores']):.3f}"
    )
