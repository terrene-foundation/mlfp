# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 2: Reduction, Embeddings and Anomaly Screening

Implement the three functions below; problem.md defines their outputs and how
they are accepted. Submit this file to the portal for grading; the grader
runs your functions on samples and screening batches you have not seen.
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


def load_applications(n: int = 3000, seed: int = 0) -> pl.DataFrame:
    """A development sample in the grader's format (no anomalies injected)."""
    raw = MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")
    sample = raw.sample(n, seed=seed).select(FIELDS).with_columns(pl.col(FIELDS).cast(pl.Float64))
    return sample.insert_column(0, pl.Series("application_id", [f"A{i:07d}" for i in range(n)]))


def component_profile(applications: pl.DataFrame) -> dict:
    raise NotImplementedError


def embed_2d(applications: pl.DataFrame) -> np.ndarray:
    raise NotImplementedError


def anomaly_scores(applications: pl.DataFrame) -> list[float]:
    raise NotImplementedError


if __name__ == "__main__":
    apps = load_applications()
    print(apps.describe())
