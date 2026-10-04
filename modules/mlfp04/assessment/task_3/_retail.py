# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Instructor-only data builders for MLFP04 Task 3 (never shipped to students).

``make_baskets`` simulates till records from a neighbourhood mini-mart with
planted co-purchase bundles, hand-keyed product names and double scans.
``make_ratings`` simulates product ratings from a low-rank taste model with
user and item biases, re-ratings, and a per-user hold-out.
``python _retail.py`` rebuilds the development files shipped with the task.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

CATALOGUE = [
    "bread", "butter", "milk", "eggs", "rice", "noodles", "soy sauce", "cooking oil",
    "chicken", "fish", "coffee", "tea", "sugar", "condensed milk", "biscuits", "chips",
    "soft drink", "beer", "wine", "tissue", "shampoo", "soap", "detergent", "toothpaste",
    "bananas", "kaya", "tofu", "kangkong", "chilli sauce", "instant noodles", "yoghurt",
    "orange juice", "bak kwa", "curry paste", "frozen dumplings", "mineral water",
]
DEV_SEED = 4304


def _messy(name: str, rng: np.random.Generator) -> str:
    u = rng.random()
    if u < 0.10:
        name = name.upper()
    elif u < 0.20:
        name = name.title()
    if rng.random() < 0.10:
        name = " " + name
    if rng.random() < 0.10:
        name = name + "  "
    return name


def make_baskets(seed: int) -> pl.DataFrame:
    """Long-format till lines: basket_id, item (as keyed at the till)."""
    rng = np.random.default_rng(seed)
    items = list(rng.choice(CATALOGUE, int(rng.integers(24, 32)), replace=False))
    bundles = []
    for _ in range(int(rng.integers(7, 11))):
        size = int(rng.choice([2, 3, 3, 4]))
        bundles.append((list(rng.choice(items, size, replace=False)), float(rng.uniform(0.04, 0.16))))
    n = int(rng.integers(2000, 3200))
    rows = []
    bid = rng.choice(900_000, n, replace=False) + 100_000
    for b in range(n):
        basket: set[str] = set()
        for members, p in bundles:
            if rng.random() < p:
                basket |= {m for m in members if rng.random() < 0.85}
        k = min(int(rng.poisson(2)), 5)
        if k:
            basket |= set(rng.choice(items, k, replace=False))
        if not basket:
            basket = {str(rng.choice(items))}
        for it in basket:
            rows.append((int(bid[b]), _messy(str(it), rng)))
            if rng.random() < 0.05:  # scanned twice
                rows.append((int(bid[b]), _messy(str(it), rng)))
    order = rng.permutation(len(rows))
    return pl.DataFrame([rows[i] for i in order], schema=["basket_id", "item"], orient="row")


def make_ratings(seed: int, n_cold: int = 12) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Returns (history, holdout, cold_user_ids).

    history: user_id, item_id, rating, rated_at — all visible ratings,
    including superseded earlier ratings of the same product (re-ratings).
    holdout: user_id, item_id, rating — each user's latest opinion on items
    hidden from history (25% per user; ALL ratings of ``n_cold`` new users).
    """
    rng = np.random.default_rng(seed)
    nu, ni, f = int(rng.integers(350, 450)), int(rng.integers(120, 180)), 5
    U, V = rng.normal(0, 1, (nu, f)), rng.normal(0, 1, (ni, f))
    bu, bi = rng.normal(0, 0.35, nu), rng.normal(0, 0.45, ni)
    R = 3.2 + bu[:, None] + bi[None, :] + 0.9 * U @ V.T / np.sqrt(f) + rng.normal(0, 0.35, (nu, ni))
    R = np.clip(np.round(R * 2) / 2, 1, 5)
    users = rng.choice(90_000, nu, replace=False) + 10_000
    items = np.array([f"SKU-{v:05d}" for v in rng.choice(100_000, ni, replace=False)])
    u, i = np.nonzero(rng.random((nu, ni)) < 0.25)
    test = np.zeros(len(u), dtype=bool)
    cold = rng.choice(nu, n_cold, replace=False)
    for uu in range(nu):
        idx = np.flatnonzero(u == uu)
        rng.shuffle(idx)
        test[idx if uu in cold else idx[: max(1, len(idx) // 4)]] = True
    start = datetime(2025, 1, 1)
    when = [start + timedelta(minutes=int(m)) for m in rng.integers(0, 500_000, len(u))]
    hist = pl.DataFrame({"user_id": users[u[~test]], "item_id": items[i[~test]], "rating": R[u[~test], i[~test]],
                         "rated_at": [w for w, t in zip(when, test) if not t]})
    # re-ratings: an earlier, different opinion that the latest rating replaced
    re_idx = rng.choice(hist.height, hist.height // 20, replace=False)
    earlier = hist[re_idx].with_columns(
        pl.Series("rating", np.clip(np.round(rng.uniform(1, 5, len(re_idx)) * 2) / 2, 1, 5)),
        (pl.col("rated_at") - pl.duration(days=int(rng.integers(30, 200)))).alias("rated_at"),
    )
    history = pl.concat([hist, earlier]).sample(fraction=1.0, shuffle=True, seed=int(rng.integers(1 << 31)))
    holdout = pl.DataFrame({"user_id": users[u[test]], "item_id": items[i[test]], "rating": R[u[test], i[test]]})
    return history, holdout, users[cold]


if __name__ == "__main__":
    here = Path(__file__).parent
    make_baskets(DEV_SEED).write_parquet(here / "dev_baskets.parquet")
    hist, _, _ = make_ratings(DEV_SEED)
    hist.write_parquet(here / "dev_ratings.parquet")
    print("wrote dev_baskets.parquet, dev_ratings.parquet")
