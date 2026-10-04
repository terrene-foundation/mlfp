# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader-held credit applications for the MLFP03 assessment (instructors only).

NOT shipped to students: ``scripts/build-student-repo.sh`` excludes ``_*.py``
and ``*solution*`` from ``assessment/``.

Every task develops on ``sg_credit_scoring.parquet`` (100,000 labelled
applications, 12.9% default). Graders hand a submission a secret sample of
those rows, then score it on **fresh applications** drawn here from the same
generating process as the file (``scripts/generate_datasets.py``,
``make_sg_credit_scoring``) with a grader-chosen seed and new IDs. The fresh
rows exist nowhere a student can see, so their outcomes cannot be looked up —
only predicted. Because the process is known, the grader also knows each fresh
applicant's TRUE default probability, which makes calibration and expected-cost
checks almost noise-free.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import polars as pl

from shared import MLFPDataLoader

ID_COLUMN = "customer_id"
TARGET = "default"
LEAK_COLUMN = "future_default_indicator"
TRUE_PROBABILITY = "true_default_probability"
PROTECTED = ("age", "gender", "race")


@lru_cache(maxsize=1)
def credit_file() -> pl.DataFrame:
    """The development file, exactly as students have it."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def sample_history(n: int, seed: int) -> pl.DataFrame:
    """A secret random sample of ``n`` labelled rows of the development file."""
    df = credit_file()
    idx = np.sort(np.random.default_rng(seed).choice(df.height, size=n, replace=False))
    return df[idx.tolist()]


def fresh_applications(n: int, seed: int) -> pl.DataFrame:
    """Draw ``n`` new labelled applications from the documented process.

    Returns the file's 36 columns plus ``true_default_probability`` (the
    generating probability of ``default``). Use :func:`as_submitted` before a
    submission sees them.
    """
    rng = np.random.default_rng(seed)
    ids = [f"APP-{seed % 100_000:05d}-{i:06d}" for i in range(n)]

    age = rng.integers(21, 70, n)
    income_full = np.clip(30000 + (age - 21) * 2500 + rng.normal(0, 15000, n), 24000, 350000).astype(int)
    income = income_full.astype(float)
    income[rng.choice(n, size=int(n * 0.30), replace=False)] = np.nan
    employment_years = np.clip((age - 22) * 0.7 + rng.normal(0, 3, n), 0, 40).astype(int)
    months_employed = employment_years * 12 + rng.integers(0, 12, n)
    credit_utilization = np.round(np.clip(rng.beta(2, 5, n), 0, 1), 4)
    num_credit_lines = np.clip(rng.poisson(3.5, n), 0, 15).astype(int)
    payment_history_score = np.clip(650 + rng.normal(0, 80, n) - credit_utilization * 150, 300, 850).astype(int)
    loan_amount = np.round(np.clip(income_full * rng.uniform(0.5, 4.0, n), 5000, 800000), -2).astype(int)
    loan_purpose = rng.choice(["home", "car", "education", "personal", "business", "renovation"],
                              size=n, p=[0.35, 0.20, 0.10, 0.20, 0.10, 0.05])  # fmt: skip
    marital = rng.choice(["single", "married", "divorced", "widowed"], size=n, p=[0.35, 0.50, 0.12, 0.03])
    education = rng.choice(["primary", "secondary", "diploma", "degree", "postgraduate"],
                           size=n, p=[0.05, 0.20, 0.30, 0.35, 0.10])  # fmt: skip
    housing = rng.choice(["HDB 2-3 room", "HDB 4-5 room", "private condo", "landed", "rental"],
                         size=n, p=[0.15, 0.40, 0.25, 0.12, 0.08])  # fmt: skip
    num_dependents = np.clip(rng.poisson(1.2, n), 0, 6).astype(int)
    debt_to_income = np.round(np.clip((loan_amount / 12) / (income_full / 12 + 1) + rng.normal(0, 0.1, n), 0, 5), 4)
    savings = np.round(np.clip(income_full * rng.uniform(0, 2.0, n), 0, 500000), 2)
    checking = np.round(np.clip(income_full * rng.uniform(0, 0.5, n), 0, 100000), 2)
    previous_defaults = np.clip(rng.poisson(0.15, n), 0, 5).astype(int)
    property_value = np.where(
        loan_purpose == "home",
        np.round(np.clip(income_full * rng.uniform(5, 15, n), 200000, 3000000), -3),
        0,
    ).astype(float)
    monthly_installment = np.round(loan_amount / np.clip(rng.uniform(12, 360, n), 12, 360), 2)
    num_late_payments = np.clip(rng.poisson(0.8, n), 0, 20).astype(int)
    avg_balance_utilization = np.round(np.clip(credit_utilization * rng.uniform(0.8, 1.2, n), 0, 1), 4)
    credit_age_years = np.clip(employment_years + rng.integers(0, 5, n), 0, 40).astype(int)
    num_hard_inquiries = np.clip(rng.poisson(1.2, n), 0, 10).astype(int)
    revolving_balance = np.round(np.clip(income_full * credit_utilization * 0.3 + rng.normal(0, 2000, n), 0, 50000), 2)
    installment_balance = np.round(np.clip(loan_amount * 0.7 + rng.normal(0, 5000, n), 0, 800000), 2)
    gender = rng.choice(["M", "F", "U"], size=n, p=[0.48, 0.48, 0.04])
    race = rng.choice(["Chinese", "Malay", "Indian", "Others"], size=n, p=[0.74, 0.13, 0.09, 0.04])
    nationality = rng.choice(["Singaporean", "PR", "EP Holder", "S Pass"], size=n, p=[0.65, 0.18, 0.10, 0.07])
    channel = rng.choice(["branch", "online", "mobile", "broker"], size=n, p=[0.20, 0.40, 0.30, 0.10])
    region = rng.choice(["Central", "East", "West", "North", "North-East"],
                        size=n, p=[0.30, 0.20, 0.20, 0.15, 0.15])  # fmt: skip

    log_odds = (
        -2.0
        + 2.5 * credit_utilization
        - 0.3 * (payment_history_score - 650) / 80
        + 0.5 * (loan_amount / income_full - 2.0)
        - 0.1 * employment_years
        + 0.3 * previous_defaults
        + 0.2 * num_late_payments
        - 0.1 * (savings / (income_full + 1))
        + rng.normal(0, 0.3, n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    default = (rng.random(n) < prob).astype(int)
    leak = default.copy()
    flip = rng.random(n) < 0.01
    leak[flip] = 1 - leak[flip]
    ltv = np.where(property_value > 0, np.round(loan_amount / np.clip(property_value, 1, None), 4), np.nan)
    coe = rng.choice([0, 1], size=n, p=[0.70, 0.30]).astype(int)
    cpf = np.round(np.clip(income_full * 0.2 + rng.normal(0, 500, n), 0, 3700), 2)

    frame = pl.DataFrame(
        {
            "customer_id": ids,
            "age": age,
            "gender": gender,
            "race": race,
            "nationality": nationality,
            "region": region,
            "income_sgd": pl.Series(income).fill_nan(None).cast(pl.Int64),
            "employment_years": employment_years,
            "months_employed": months_employed,
            "credit_utilization": credit_utilization,
            "avg_balance_utilization": avg_balance_utilization,
            "num_credit_lines": num_credit_lines,
            "credit_age_years": credit_age_years,
            "num_hard_inquiries": num_hard_inquiries,
            "payment_history_score": payment_history_score,
            "num_late_payments": num_late_payments,
            "revolving_balance": revolving_balance,
            "installment_balance": installment_balance,
            "loan_amount_sgd": loan_amount,
            "loan_purpose": loan_purpose,
            "monthly_installment": monthly_installment,
            "marital_status": marital,
            "education": education,
            "housing_type": housing,
            "num_dependents": num_dependents,
            "debt_to_income": debt_to_income,
            "savings_balance": savings,
            "checking_balance": checking,
            "previous_defaults": previous_defaults,
            "property_value_sgd": property_value,
            "loan_to_value": pl.Series(ltv).fill_nan(None),
            "coe_vehicle_owner": coe,
            "cpf_monthly_contribution": cpf,
            "application_channel": channel,
            "future_default_indicator": leak,
            "default": default,
        }
    )
    schema = credit_file().schema
    frame = frame.cast({c: schema[c] for c in frame.columns})
    return frame.with_columns(pl.Series(TRUE_PROBABILITY, prob))


def as_submitted(df: pl.DataFrame) -> pl.DataFrame:
    """Applications as they arrive for scoring: no outcome, no grader-only
    column, and the post-outcome field still empty."""
    out = df.drop([c for c in (TARGET, TRUE_PROBABILITY) if c in df.columns])
    if LEAK_COLUMN in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Int64).alias(LEAK_COLUMN))
    return out


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the rank-sum identity (ties get average ranks)."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(int)
    r = rankdata(np.asarray(score, dtype=float))
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
