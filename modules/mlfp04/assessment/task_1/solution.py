# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 1: Customer Segments and Mixture Models (Reference)

Instructor-only reference. Withheld from students.

Segmentation decisions:
  * customer_id and signup_channel are identifiers / metadata, not behaviour;
  * counts and money are multiplicative (log-normal), so they are logged;
  * after logging, every column is z-scored so no unit dominates distance;
  * K is chosen by silhouette over K-means fits through ClusteringEngine,
    ignoring tiny segments (bulk accounts) when counting personas.
EM is written from scratch with several restarts (the likelihood surface has
local optima) and a convergence test on the log-likelihood.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.special import logsumexp

from kailash_ml.engines.clustering import ClusteringEngine

BEHAVIOUR_COLS = [
    "recency_days",
    "orders_12m",
    "spend_12m_sgd",
    "tenure_months",
    "avg_basket_sgd",
    "pct_discount_orders",
    "app_sessions_30d",
]
MULTIPLICATIVE = ["orders_12m", "spend_12m_sgd", "avg_basket_sgd", "app_sessions_30d"]


def load_dev_customers() -> pl.DataFrame:
    return pl.read_parquet(Path(__file__).with_name("dev_customers.parquet"))


def _model_space(customers: pl.DataFrame) -> pl.DataFrame:
    logged = customers.select(
        [
            pl.col(c).cast(pl.Float64).log1p().alias(c) if c in MULTIPLICATIVE else pl.col(c).cast(pl.Float64)
            for c in BEHAVIOUR_COLS
        ]
    )
    # z-score in the log space: logging pulls the bulk accounts back towards
    # their persona, so they no longer dominate the mean and variance.
    return logged.select([((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) for c in BEHAVIOUR_COLS])


def segment_customers(customers: pl.DataFrame) -> dict:
    space = _model_space(customers)
    engine = ClusteringEngine()
    sweep = engine.sweep_k(space, range(2, 10), algorithm="kmeans", criterion="silhouette")
    fit = engine.fit(space, algorithm="kmeans", n_clusters=sweep.optimal_k)
    labels = [int(v) for v in fit.labels]

    profiles = (
        customers.with_columns(pl.Series("segment", labels))
        .group_by("segment")
        .agg(pl.len().alias("n_customers"), *[pl.col(c).cast(pl.Float64).median() for c in BEHAVIOUR_COLS])
        .sort("segment")
    )
    big = profiles.filter(pl.col("n_customers") >= 0.03 * customers.height)
    at_risk = int(big.sort("recency_days", descending=True)["segment"][0])
    return {"labels": labels, "profiles": profiles, "at_risk_segment": at_risk}


def _log_gauss(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    sol = np.linalg.solve(L, (X - mean).T)
    return -0.5 * (sol**2).sum(axis=0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)


def _log_joint(X, w, means, covs) -> np.ndarray:
    return np.column_stack([np.log(w[j]) + _log_gauss(X, means[j], covs[j]) for j in range(len(w))])


def fit_mixture(X: np.ndarray, k: int, seed: int = 0) -> dict:
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    rng = np.random.default_rng(seed)
    reg = 1e-6 * np.eye(d)
    best = None
    for _ in range(10):
        means = X[rng.choice(n, k, replace=False)]
        covs = np.array([np.cov(X.T) + reg] * k)
        w = np.full(k, 1.0 / k)
        prev = -np.inf
        for _ in range(2000):
            lj = _log_joint(X, w, means, covs)
            ll = logsumexp(lj, axis=1)
            R = np.exp(lj - ll[:, None])
            nk = R.sum(axis=0)
            w = nk / n
            means = (R.T @ X) / nk[:, None]
            covs = np.array([((R[:, j, None] * (X - means[j])).T @ (X - means[j])) / nk[j] + reg for j in range(k)])
            if ll.sum() - prev < 1e-9 * n:
                break
            prev = ll.sum()
        lj = _log_joint(X, w, means, covs)
        total = float(logsumexp(lj, axis=1).sum())
        if best is None or total > best[0]:
            best = (total, w, means, covs)
    total, w, means, covs = best
    lj = _log_joint(X, w, means, covs)
    ll = logsumexp(lj, axis=1)
    return {
        "weights": w,
        "means": means,
        "covariances": covs,
        "responsibilities": np.exp(lj - ll[:, None]),
        "log_likelihood": float(ll.sum()),
    }


if __name__ == "__main__":
    out = segment_customers(load_dev_customers())
    print(out["profiles"])
    print("at-risk segment:", out["at_risk_segment"])
