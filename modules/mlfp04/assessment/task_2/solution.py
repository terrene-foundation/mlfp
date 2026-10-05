# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 2: Reduction, Embeddings and Anomaly Screening (Reference)

Instructor-only reference. Withheld from students.

Decisions:
  * the fields are in incompatible units (S$, counts, ratios), so every
    reduction and detector works on z-scored fields — without it PCA's first
    component is just the largest S$ column;
  * the 2-D map uses t-SNE (local neighbourhoods), not PCA, because the
    review team reads it for "who sits next to whom";
  * no single detector covers all three anomaly kinds: Isolation Forest
    misses stitched identities, LOF with a small neighbourhood is masked by a
    tight ring of near-duplicates, Mahalanobis distance catches broken
    correlations. Scores are combined by taking each row's worst rank.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from kailash_ml.engines.anomaly_detection import AnomalyDetectionEngine
from kailash_ml.engines.dim_reduction import DimReductionEngine
from shared import MLFPDataLoader


def load_applications(n: int = 3000, seed: int = 0) -> pl.DataFrame:
    """A development sample in the grader's format (no anomalies injected)."""
    from _applications import FIELDS

    raw = MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")
    sample = raw.sample(n, seed=seed).select(FIELDS).with_columns(pl.col(FIELDS).cast(pl.Float64))
    return sample.insert_column(0, pl.Series("application_id", [f"A{i:07d}" for i in range(n)]))


def _fields(applications: pl.DataFrame) -> list[str]:
    return [c for c in applications.columns if c != "application_id" and applications[c].dtype.is_numeric()]


def _standardised(applications: pl.DataFrame) -> pl.DataFrame:
    cols = _fields(applications)
    return applications.select([((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in cols])


def component_profile(applications: pl.DataFrame) -> dict:
    z = _standardised(applications)
    res = DimReductionEngine().reduce(z, algorithm="pca", n_components=z.width)
    evr = [float(v) for v in res.explained_variance_ratio]
    n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
    # PC1 direction: project the standardised fields on the PC1 scores and
    # normalise to unit length (the engine returns scores, not loadings)
    scores = np.asarray(res.transformed)[:, 0]
    direction = z.to_numpy().T @ scores
    direction /= np.linalg.norm(direction)
    return {
        "n_components_90": n90,
        "explained_variance_ratio": evr,
        "pc1_loadings": {c: float(v) for c, v in zip(z.columns, direction)},
    }


def embed_2d(applications: pl.DataFrame) -> np.ndarray:
    z = _standardised(applications)
    res = DimReductionEngine().reduce(z, algorithm="tsne", n_components=2, seed=7)
    return np.asarray(res.transformed, dtype=float)


def _rank(s: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(s)) / (len(s) - 1)


def anomaly_scores(applications: pl.DataFrame) -> list[float]:
    z = _standardised(applications)
    Z = z.to_numpy()
    engine = AnomalyDetectionEngine()
    iso = np.asarray(engine.detect(z, algorithm="isolation_forest", contamination=0.03, seed=7).scores)
    lof = np.asarray(engine.detect(z, algorithm="lof", contamination=0.03, n_neighbors=50).scores)
    cov_inv = np.linalg.pinv(np.cov(Z.T))
    maha = np.einsum("ij,jk,ik->i", Z, cov_inv, Z)
    combined = np.maximum.reduce([_rank(iso), _rank(lof), _rank(maha)])
    return [float(v) for v in combined]


if __name__ == "__main__":
    apps = load_applications()
    prof = component_profile(apps)
    print(prof["n_components_90"], prof["pc1_loadings"])
