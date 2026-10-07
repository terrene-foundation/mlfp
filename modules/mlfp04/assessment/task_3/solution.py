# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 3: Baskets and Recommendations (Reference)

Instructor-only reference. Withheld from students.

Rules: till names are normalised (trim + lowercase) and double scans
collapse to one line per basket before counting; itemsets are enumerated
level by level (Apriori pruning) with basket sets as python sets of ids.
Recommender: only each customer's LATEST rating of a product counts; a
biased matrix factorisation is fitted by alternating least squares, and
unknown customers / products fall back to the bias terms.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent


def load_dev_baskets() -> pl.DataFrame:
    return pl.read_parquet(HERE / "dev_baskets.parquet")


def load_dev_ratings() -> pl.DataFrame:
    return pl.read_parquet(HERE / "dev_ratings.parquet")


def mine_rules(baskets: pl.DataFrame, min_support: float, min_confidence: float, max_len: int = 3) -> pl.DataFrame:
    clean = baskets.select(pl.col("basket_id"), pl.col("item").str.strip_chars().str.to_lowercase()).unique()
    n = clean["basket_id"].n_unique()
    holders = {r["item"]: set(r["basket_id"]) for r in clean.group_by("item").agg(pl.col("basket_id")).iter_rows(named=True)}

    support: dict[frozenset, float] = {}
    level = {}
    for it, bs in holders.items():
        if len(bs) / n >= min_support:
            level[frozenset([it])] = bs
    size = 1
    while level:
        for s, bs in level.items():
            support[s] = len(bs) / n
        if size == max_len:
            break
        nxt = {}
        keys = list(level)
        singles = sorted({i for s in keys for i in s})
        for s in keys:
            for it in singles:
                if it in s:
                    continue
                cand = s | {it}
                if cand in nxt or any(frozenset(sub) not in level for sub in combinations(cand, size)):
                    continue
                bs = level[s] & holders[it]
                if len(bs) / n >= min_support:
                    nxt[cand] = bs
        level, size = nxt, size + 1

    rows = []
    for s, sup in support.items():
        if len(s) < 2:
            continue
        for r in range(1, len(s)):
            for ante in combinations(sorted(s), r):
                a = frozenset(ante)
                cons = s - a
                conf = sup / support[a]
                if conf >= min_confidence:
                    rows.append((sorted(a), sorted(cons), sup, conf, conf / support[cons]))
    schema = {"antecedent": pl.List(pl.String), "consequent": pl.List(pl.String),
              "support": pl.Float64, "confidence": pl.Float64, "lift": pl.Float64}
    return pl.DataFrame(rows, schema=schema, orient="row").sort("lift", descending=True)


def fit_recommender(history: pl.DataFrame, factors: int = 6, reg: float = 3.0, sweeps: int = 15):
    latest = history.sort("rated_at").unique(["user_id", "item_id"], keep="last")
    users = latest["user_id"].unique().to_list()
    items = latest["item_id"].unique().to_list()
    uix = {u: k for k, u in enumerate(users)}
    iix = {i: k for k, i in enumerate(items)}
    u = np.array([uix[v] for v in latest["user_id"].to_list()])
    i = np.array([iix[v] for v in latest["item_id"].to_list()])
    r = latest["rating"].to_numpy().astype(float)
    nu, ni = len(users), len(items)

    mu = r.mean()
    bu, bi = np.zeros(nu), np.zeros(ni)
    for _ in range(10):
        bi = np.bincount(i, r - mu - bu[u], ni) / (np.bincount(i, minlength=ni) + 5.0)
        bu = np.bincount(u, r - mu - bi[i], nu) / (np.bincount(u, minlength=nu) + 5.0)
    res = r - mu - bu[u] - bi[i]
    rng = np.random.default_rng(0)
    P, Q = rng.normal(0, 0.1, (nu, factors)), rng.normal(0, 0.1, (ni, factors))
    by_u = [np.flatnonzero(u == k) for k in range(nu)]
    by_i = [np.flatnonzero(i == k) for k in range(ni)]
    eye = reg * np.eye(factors)
    for _ in range(sweeps):
        for k, idx in enumerate(by_u):
            A = Q[i[idx]]
            P[k] = np.linalg.solve(A.T @ A + eye, A.T @ res[idx])
        for k, idx in enumerate(by_i):
            A = P[u[idx]]
            Q[k] = np.linalg.solve(A.T @ A + eye, A.T @ res[idx])

    def predict(pairs: pl.DataFrame) -> np.ndarray:
        pu = np.array([uix.get(v, -1) for v in pairs["user_id"].to_list()])
        pi = np.array([iix.get(v, -1) for v in pairs["item_id"].to_list()])
        out = np.full(len(pu), mu)
        ku, ki = pu >= 0, pi >= 0
        out[ku] += bu[pu[ku]]
        out[ki] += bi[pi[ki]]
        both = ku & ki
        out[both] += (P[pu[both]] * Q[pi[both]]).sum(1)
        return np.clip(out, 1.0, 5.0)

    return predict


if __name__ == "__main__":
    print(mine_rules(load_dev_baskets(), 0.03, 0.5).head(10))
