# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Instructor-only data builder for MLFP04 Task 5 (never shipped to students).

Customers belong to five latent behaviour segments arranged so that churn
risk is NOT monotone in any direction of the raw fields: the two high-churn
segments sit at opposite corners of a latent plane, the low-churn segments
at the other two corners and the centre. The plane is mixed into eight
observed fields by a random rotation, so no single field reveals it.
``python _churn.py`` rebuilds the development file shipped with the task.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

FIELDS = [
    "visits_30d", "avg_session_min", "spend_90d_sgd", "support_tickets_90d",
    "days_since_last_order", "pct_promo_orders", "delivery_delay_days", "app_rating_given",
]
TARGET = "churned"
DEV_SEED = 4504
_SCALE = np.array([4.0, 6.0, 400.0, 1.0, 15.0, 0.1, 1.5, 0.6])
_OFFSET = np.array([10.0, 20.0, 900.0, 2.0, 40.0, 0.3, 3.0, 3.5])


def make_customers(seed: int, n: int = 4500) -> tuple[pl.DataFrame, np.ndarray]:
    """Returns (frame with customer_id + FIELDS + churned, true churn probability)."""
    rng = np.random.default_rng(seed)
    d = len(FIELDS)
    corners = np.array([[3.2, 3.2], [-3.2, -3.2], [3.2, -3.2], [-3.2, 3.2], [0.0, 0.0]])
    effect = np.array([1.0, 1.0, -2.0, -2.0, -1.0])
    perm = rng.permutation(len(corners))
    corners, effect = corners[perm], effect[perm]
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    seg = rng.choice(len(corners), n, p=rng.dirichlet(np.full(len(corners), 8.0)))
    latent = np.hstack([corners[seg] + rng.normal(0, 0.6, (n, 2)), rng.normal(0, 0.6, (n, d - 2))])
    X = _OFFSET + (latent @ Q.T) * _SCALE
    p = 1 / (1 + np.exp(-(effect[seg] - 0.3)))
    y = (rng.random(n) < p).astype(np.int64)
    ids = rng.choice(9_000_000, n, replace=False) + 1_000_000
    frame = pl.DataFrame({"customer_id": ids.astype(np.int64), **{f: np.round(X[:, j], 3) for j, f in enumerate(FIELDS)},
                          TARGET: y})
    return frame, p


if __name__ == "__main__":
    frame, _ = make_customers(DEV_SEED, n=3000)
    out = Path(__file__).with_name("dev_churn.parquet")
    frame.write_parquet(out)
    print(f"wrote {out} {frame.shape}, churn rate {frame[TARGET].mean():.3f}")
