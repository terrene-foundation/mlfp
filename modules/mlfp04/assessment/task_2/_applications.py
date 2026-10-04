# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Instructor-only data builders for MLFP04 Task 2 (never shipped to students).

Normal rows are real applications from the course's Singapore credit-scoring
dataset; anomalies are injected with known types (the benchmark recipe used
in Exercise 4), so labels come from the injection, never from a threshold on
an input column.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from shared import MLFPDataLoader

FIELDS = [
    "age", "employment_years", "months_employed", "credit_utilization",
    "avg_balance_utilization", "num_credit_lines", "credit_age_years",
    "num_hard_inquiries", "payment_history_score", "num_late_payments",
    "revolving_balance", "installment_balance", "loan_amount_sgd",
    "monthly_installment", "num_dependents", "debt_to_income",
    "savings_balance", "checking_balance", "previous_defaults",
    "property_value_sgd",
]
TYPES = ("global", "dependency", "clustered")


def _pool() -> pl.DataFrame:
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet").select(FIELDS)


def sample_applications(rng: np.random.Generator, n: int) -> pl.DataFrame:
    """Secret random sample of real applications, columns in a random order,
    with a fresh application_id."""
    base = _pool().sample(n, seed=int(rng.integers(1 << 31))).with_columns(pl.col(FIELDS).cast(pl.Float64))
    cols = list(rng.permutation(FIELDS))
    ids = [f"A{v:07d}" for v in rng.choice(10_000_000, n, replace=False)]
    return base.select(cols).insert_column(0, pl.Series("application_id", ids))


def with_anomalies(rng: np.random.Generator, n: int = 3000, n_global: int = 30,
                   n_dependency: int = 30, n_clustered: int = 20) -> tuple[pl.DataFrame, np.ndarray]:
    """Real applications plus injected anomalies; returns (frame, type per row)."""
    base = _pool().sample(n, seed=int(rng.integers(1 << 31)))
    B = base.to_numpy().astype(float)
    mu, sd = B.mean(0), B.std(0)
    p = B.shape[1]
    G = B[rng.choice(n, n_global, replace=False)].copy()
    for r in range(n_global):
        j = int(rng.integers(p))
        G[r, j] = mu[j] + rng.uniform(5.0, 8.0) * sd[j]
    D = np.column_stack([B[rng.integers(n, size=n_dependency), j] for j in range(p)])
    centre = B[int(rng.integers(n))].copy()
    shifted = rng.choice(p, 3, replace=False)
    centre[shifted] += 3.0 * sd[shifted]
    C = centre + rng.normal(0.0, 0.05, (n_clustered, p)) * sd
    X = np.vstack([B, G, D, C])
    types = np.array(["normal"] * n + ["global"] * n_global + ["dependency"] * n_dependency + ["clustered"] * n_clustered)
    order = rng.permutation(len(X))
    X, types = X[order], types[order]
    cols = list(rng.permutation(FIELDS))
    frame = pl.DataFrame({c: X[:, FIELDS.index(c)] for c in cols})
    ids = [f"A{v:07d}" for v in rng.choice(10_000_000, len(X), replace=False)]
    return frame.insert_column(0, pl.Series("application_id", ids)), types
