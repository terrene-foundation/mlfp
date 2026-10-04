# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 3: Regression, ANOVA & Logistic Inference
(Reference Solution — withheld from students; verified to pass grader.py)
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable

import numpy as np
import polars as pl
from scipy import stats

from shared import MLFPDataLoader

BASE_FLAT_TYPE = "3 ROOM"
PRICE_BOUNDS = (100_000, 2_000_000)
NUMERIC = ["floor_area_sqm", "remaining_lease_years", "storey_mid"]
LOGIT_FEATURES = [
    "credit_utilization",
    "num_late_payments",
    "previous_defaults",
    "debt_to_income",
    "num_hard_inquiries",
]


def load_hdb() -> pl.DataFrame:
    return MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")


def load_credit() -> pl.DataFrame:
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


# ── Part A: linear regression ─────────────────────────────────────────


def derive(raw: pl.DataFrame) -> pl.DataFrame:
    """Add the model columns (sale year, remaining lease, storey midpoint)."""
    sale_year = pl.col("month").str.slice(0, 4).cast(pl.Int64)
    storey = pl.col("storey_range").str.to_uppercase().str.strip_chars()

    def bound(i: int) -> pl.Expr:  # data-entry typo: letter O typed for digit 0
        return (
            storey.str.extract(r"^([0-9O]+) TO ([0-9O]+)$", i)
            .str.replace_all("O", "0")
            .cast(pl.Float64)
        )

    return raw.with_columns(
        sale_year.alias("sale_year"),
        (99 - (sale_year - pl.col("lease_commence_date"))).cast(pl.Float64).alias("remaining_lease_years"),
        (
            (bound(1) + bound(2)) / 2
        ).alias("storey_mid"),
    )


def prepare(raw: pl.DataFrame) -> pl.DataFrame:
    """Derive model columns and drop recording errors."""
    return derive(raw).filter(
        pl.col("resale_price").is_between(*PRICE_BOUNDS)
        & (pl.col("lease_commence_date") <= pl.col("sale_year"))
        & pl.col("storey_mid").is_not_null()
    )


def _design(df: pl.DataFrame, levels: list[str], interact: bool = False) -> tuple[np.ndarray, list[str]]:
    cols = [np.ones(df.height)]
    names = ["intercept"]
    for c in NUMERIC:
        cols.append(df[c].to_numpy().astype(np.float64))
        names.append(c)
    ft = df["flat_type"].to_numpy()
    area = df["floor_area_sqm"].to_numpy().astype(np.float64)
    for lv in levels:
        d = (ft == lv).astype(np.float64)
        cols.append(d)
        names.append(f"flat_type={lv}")
        if interact:
            cols.append(d * area)
            names.append(f"floor_area_sqm:flat_type={lv}")
    return np.column_stack(cols), names


def _ols(X: np.ndarray, y: np.ndarray) -> dict:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = rss / (n - p)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    return {"beta": beta, "se": se, "p": 2 * stats.t.sf(np.abs(t), n - p), "rss": rss, "tss": tss, "n": n, "k": p}


def fit_price_model(transactions: pl.DataFrame) -> dict:
    df = prepare(transactions)
    levels = sorted(set(df["flat_type"].unique().to_list()) - {BASE_FLAT_TYPE})
    y = df["resale_price"].to_numpy().astype(np.float64)
    X, names = _design(df, levels)
    m = _ols(X, y)
    n, p = m["n"], m["k"]
    r2 = 1 - m["rss"] / m["tss"]
    Xf, _ = _design(df, levels, interact=True)
    mf = _ols(Xf, y)
    q = Xf.shape[1] - p
    f_int = ((m["rss"] - mf["rss"]) / q) / (mf["rss"] / (n - Xf.shape[1]))
    beta = m["beta"]

    def predict(new_rows: pl.DataFrame) -> np.ndarray:
        Xn, _ = _design(derive(new_rows), levels)
        return Xn @ beta

    return {
        "n_used": int(n),
        "coefficients": dict(zip(names, map(float, beta))),
        "std_errors": dict(zip(names, map(float, m["se"]))),
        "p_values": dict(zip(names, map(float, m["p"]))),
        "r_squared": float(r2),
        "adj_r_squared": float(1 - (1 - r2) * (n - 1) / (n - p)),
        "f_statistic": float((r2 / (p - 1)) / ((1 - r2) / (n - p))),
        "interaction_f": float(f_int),
        "interaction_p": float(stats.f.sf(f_int, q, n - Xf.shape[1])),
        "predict": predict,
    }


# ── Part B: one-way ANOVA + Tukey HSD ─────────────────────────────────


def anova_flat_types(transactions: pl.DataFrame, flat_types: list[str]) -> dict:
    df = transactions.filter(pl.col("flat_type").is_in(flat_types)).with_columns(
        (pl.col("resale_price") / pl.col("floor_area_sqm")).alias("psm")
    )
    groups = [df.filter(pl.col("flat_type") == f)["psm"].to_numpy() for f in flat_types]
    f, p = stats.f_oneway(*groups)
    allv = np.concatenate(groups)
    ss_between = sum(g.size * (g.mean() - allv.mean()) ** 2 for g in groups)
    ss_total = ((allv - allv.mean()) ** 2).sum()
    res = stats.tukey_hsd(*groups)
    tukey = {}
    for i, j in combinations(range(len(flat_types)), 2):
        a, b = sorted([flat_types[i], flat_types[j]])
        pv = float(res.pvalue[i, j])
        tukey[f"{a}|{b}"] = {"p_adj": pv, "significant": pv < 0.05}
    return {"f_stat": float(f), "p_value": float(p), "eta_squared": float(ss_between / ss_total), "tukey": tukey}


# ── Part C: logistic regression by maximum likelihood ─────────────────


def _irls(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        H = X.T @ (X * w[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            break
    mu = 1 / (1 + np.exp(-(X @ beta)))
    cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    return beta, cov


def fit_default_model(train: pl.DataFrame) -> dict:
    Z = train.select(LOGIT_FEATURES).to_numpy().astype(np.float64)
    mean, sd = Z.mean(axis=0), Z.std(axis=0)
    X = np.column_stack([np.ones(len(Z)), (Z - mean) / sd])
    y = train["default"].to_numpy().astype(np.float64)
    beta, cov = _irls(X, y)
    b, V = beta[1:], cov[1:, 1:]
    order = np.argsort(-np.abs(b))
    i, j = int(order[0]), int(order[1])
    z = (b[i] - b[j]) / np.sqrt(V[i, i] + V[j, j] - 2 * V[i, j])
    p_diff = float(2 * stats.norm.sf(abs(z)))

    def predict_proba(rows: pl.DataFrame) -> np.ndarray:
        Zn = (rows.select(LOGIT_FEATURES).to_numpy().astype(np.float64) - mean) / sd
        return 1 / (1 + np.exp(-(beta[0] + Zn @ b)))

    return {
        "odds_ratios": {f: float(np.exp(v)) for f, v in zip(LOGIT_FEATURES, b)},
        "std_errors": {f: float(np.sqrt(V[k, k])) for k, f in enumerate(LOGIT_FEATURES)},
        "top_two": [LOGIT_FEATURES[i], LOGIT_FEATURES[j]],
        "top_two_p": p_diff,
        "top_two_differ": p_diff < 0.05,
        "predict_proba": predict_proba,
    }


if __name__ == "__main__":
    hdb = load_hdb()
    train = hdb.filter(pl.col("month") < "2023-01")
    m = fit_price_model(train)
    print({k: v for k, v in m.items() if k != "predict"})
    test = prepare(hdb.filter(pl.col("month") >= "2023-01"))
    pred = m["predict"](test)
    y = test["resale_price"].to_numpy()
    print("out-of-time R2", 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    print(anova_flat_types(prepare(hdb).sample(600, seed=3), ["3 ROOM", "4 ROOM", "5 ROOM"]))
    credit = load_credit()
    r = fit_default_model(credit.head(70_000))
    print({k: v for k, v in r.items() if k != "predict_proba"})
