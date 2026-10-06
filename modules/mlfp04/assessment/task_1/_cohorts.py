# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Instructor-only data builders for MLFP04 Task 1 (never shipped to students).

``make_cohort`` simulates a loyalty-programme customer table with planted
personas; ``make_mixture`` simulates an overlapping Gaussian mixture. The
grader calls both with a fresh secret seed. ``python _cohorts.py`` rebuilds
the development file shipped with the task (labels stripped).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

BEHAVIOUR_COLS = [
    "recency_days",
    "orders_12m",
    "spend_12m_sgd",
    "tenure_months",
    "avg_basket_sgd",
    "pct_discount_orders",
    "app_sessions_30d",
]
CHANNELS = ["app", "web", "store", "partner"]
DEV_SEED = 4104


def make_cohort(seed: int) -> tuple[pl.DataFrame, np.ndarray, int, int, np.ndarray]:
    """Return (customers, persona, k, at_risk_persona, is_bulk_account).

    Personas live on a latent unit cube; counts and money are log-normal in
    natural units (multiplicative spread), so a few personas sit in a long
    right tail. About 1% of rows are corporate bulk accounts whose spend and
    basket size are inflated 10-30x — they belong to a persona but distort
    any scale estimated with means and variances.
    """
    rng = np.random.default_rng(seed)
    k = int(rng.integers(3, 7))
    d = len(BEHAVIOUR_COLS)
    while True:
        centres = rng.uniform(0, 1, (k, d))
        gaps = [np.linalg.norm(centres[i] - centres[j]) for i in range(k) for j in range(i)]
        if min(gaps) > 0.6:
            break
    sizes = rng.multinomial(int(rng.integers(1400, 2200)), rng.dirichlet(np.full(k, 8.0)))
    latent = np.vstack([centres[i] + rng.normal(0, 0.07, (n, d)) for i, n in enumerate(sizes)])
    persona = np.repeat(np.arange(k), sizes)
    latent = np.clip(latent, -0.2, 1.2)

    X = np.empty_like(latent)
    X[:, 0] = 3 + latent[:, 0] * 200                                   # recency_days
    X[:, 1] = np.exp(latent[:, 1] * np.log(80))                        # orders_12m
    X[:, 2] = np.exp(np.log(50) + latent[:, 2] * np.log(200))          # spend_12m_sgd
    X[:, 3] = 1 + latent[:, 3] * 96                                    # tenure_months
    X[:, 4] = np.exp(np.log(15) + latent[:, 4] * np.log(30))           # avg_basket_sgd
    X[:, 5] = np.clip(0.02 + latent[:, 5] * 0.7, 0, 1)                 # pct_discount_orders
    X[:, 6] = np.maximum(0.0, np.exp(latent[:, 6] * np.log(60)) - 1)       # app_sessions_30d

    n = len(persona)
    bulk = np.zeros(n, dtype=bool)
    bulk[rng.choice(n, max(1, n // 100), replace=False)] = True
    X[bulk, 2] *= rng.uniform(10, 30, bulk.sum())
    X[bulk, 4] *= rng.uniform(10, 30, bulk.sum())

    order = rng.permutation(n)
    X, persona, bulk = X[order], persona[order], bulk[order]
    at_risk = int(np.argmax(centres[:, 0]))  # longest time since last order

    frame = pl.DataFrame({c: np.round(X[:, j], 2) for j, c in enumerate(BEHAVIOUR_COLS)})
    frame = frame.with_columns(
        pl.Series("orders_12m", np.maximum(1, np.round(X[:, 1])).astype(np.int64)),
        pl.Series("app_sessions_30d", np.round(X[:, 6]).astype(np.int64)),
        pl.Series("recency_days", np.round(X[:, 0]).astype(np.int64)),
        pl.Series("tenure_months", np.round(X[:, 3]).astype(np.int64)),
        pl.Series("pct_discount_orders", np.round(X[:, 5], 3)),
    )
    ids = rng.choice(9_000_000, n, replace=False) + 1_000_000
    frame = frame.select(
        pl.Series("customer_id", ids.astype(np.int64)),
        pl.Series("signup_channel", rng.choice(CHANNELS, n)),
        *BEHAVIOUR_COLS,
    )
    return frame, persona, k, at_risk, bulk


def make_mixture(seed: int) -> tuple[np.ndarray, int]:
    """Overlapping full-covariance Gaussian mixture in 2-4 dimensions."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 5))
    d = int(rng.integers(2, 5))
    n = int(rng.integers(600, 1100))
    means = rng.normal(0, 2.6, (k, d))
    covs = []
    for _ in range(k):
        A = rng.normal(0, 1, (d, d))
        covs.append(A @ A.T / d + 0.3 * np.eye(d))
    w = rng.dirichlet(np.full(k, 4.0))
    z = rng.choice(k, n, p=w)
    X = np.array([rng.multivariate_normal(means[j], covs[j]) for j in z])
    return X, k


if __name__ == "__main__":
    frame, *_ = make_cohort(DEV_SEED)
    out = Path(__file__).with_name("dev_customers.parquet")
    frame.write_parquet(out)
    print(f"wrote {out} {frame.shape}")
