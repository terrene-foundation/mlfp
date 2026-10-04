# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 1: Application-Time Model Inputs (Reference Solution)

Instructors only. Graded by grader.py on rows and applications the student
never sees.

Decisions this reference makes (one defensible route, not the only one):

1. Split first. Every rule below is learned from the training rows only
   (``is_holdout == False``); hold-out rows are transformed, never learned from.
2. Leak screen on the training rows: a field that separates defaulters from
   non-defaulters almost perfectly on its own (single-column AUC > 0.9) cannot
   be known when an application arrives. Here that is
   ``future_default_indicator``. ``customer_id`` is a key, never an input.
3. Missing income (about 30% of rows) is imputed from age with a straight line
   fitted on the training rows (income rises with age in this book); an
   ``income_reported`` flag keeps the fact that it was missing.
4. Three affordability features: instalment burden (monthly instalment as a
   share of monthly income), savings cover (months of instalments the savings
   would pay) and loan-to-income.
5. Selection: kailash-ml ``FeatureEngineer`` ranks every candidate by tree
   importance on the training rows; the top ranks fill the 12-input budget and
   the two affordability measures the committee asked for are always kept.
6. ``PreprocessingPipeline`` learns the remaining imputation and z-score scaling
   from the training rows; ``transform`` only applies it.
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import polars as pl
from scipy.stats import rankdata

from kailash_ml import PreprocessingPipeline
from kailash_ml.engines.feature_engineer import (
    FeatureEngineer,
    GeneratedColumn,
    GeneratedFeatures,
)
from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

ID = "customer_id"
TARGET = "default"
MAX_INPUTS = 12
REQUIRED = ["instalment_burden", "savings_cover"]
ENGINEERED = REQUIRED + ["loan_to_income", "income_reported"]


def load_history() -> pl.DataFrame:
    """The labelled development file (for local runs only)."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def _single_column_auc(y: np.ndarray, x: np.ndarray) -> float:
    ranks = rankdata(x)  # ties share their average rank
    n1 = y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def _leak_screen(train: pl.DataFrame, numeric: list[str]) -> list[str]:
    """Fields that predict the outcome almost perfectly on their own."""
    y = train[TARGET].to_numpy()
    leaks = []
    for col in numeric:
        x = train[col].cast(pl.Float64)
        x = x.fill_null(x.median()).to_numpy()
        if np.std(x) == 0:
            continue
        a = _single_column_auc(y, x)
        if max(a, 1 - a) > 0.9:
            leaks.append(col)
    return leaks


def _engineer(df: pl.DataFrame, slope: float, intercept: float) -> pl.DataFrame:
    income = pl.col("income_sgd").cast(pl.Float64)
    filled = income.fill_null(pl.lit(intercept) + pl.lit(slope) * pl.col("age").cast(pl.Float64))
    return df.with_columns(
        income.is_not_null().cast(pl.Int64).alias("income_reported"),
        (pl.col("monthly_installment") / (filled / 12.0)).alias("instalment_burden"),
        (pl.col("savings_balance") / pl.col("monthly_installment")).alias("savings_cover"),
        (pl.col("loan_amount_sgd") / filled).alias("loan_to_income"),
        filled.alias("income_sgd"),
    )


def build_model_inputs(history: pl.DataFrame, is_holdout: pl.Series) -> dict:
    """Learn every rule from the training rows; apply it to all rows."""
    train = history.filter(~is_holdout)

    numeric = [c for c, t in train.schema.items() if t.is_numeric() and c != TARGET]
    leaks = _leak_screen(train, numeric)
    candidates = [c for c in numeric if c not in leaks]

    known = train.filter(pl.col("income_sgd").is_not_null())
    slope, intercept = np.polyfit(
        known["age"].to_numpy().astype(float), known["income_sgd"].to_numpy().astype(float), 1
    )
    slope, intercept = float(slope), float(intercept)
    frame = _engineer(train, slope, intercept)

    generated = GeneratedFeatures(
        original_columns=candidates,
        generated_columns=[
            GeneratedColumn("instalment_burden", ["monthly_installment", "income_sgd"], "interaction", "float64"),
            GeneratedColumn("savings_cover", ["savings_balance", "monthly_installment"], "interaction", "float64"),
            GeneratedColumn("loan_to_income", ["loan_amount_sgd", "income_sgd"], "interaction", "float64"),
            GeneratedColumn("income_reported", ["income_sgd"], "binning", "int64"),
        ],
        total_candidates=len(candidates) + len(ENGINEERED),
        data=frame,
    )
    ranked = FeatureEngineer(max_features=MAX_INPUTS).select(
        frame.select(candidates + ENGINEERED + [TARGET]), generated, TARGET, method="importance"
    )
    selected = list(REQUIRED)
    for name in ranked.selected_columns:
        if len(selected) >= MAX_INPUTS:
            break
        if name not in selected:
            selected.append(name)

    pipeline = PreprocessingPipeline()
    pipeline.setup(
        frame.select(selected + [TARGET]),
        target=TARGET,
        normalize=True,
        imputation_strategy="median",
        seed=42,
    )

    def transform(applications: pl.DataFrame) -> pl.DataFrame:
        """Apply the learned rules to applications (no refitting)."""
        engineered = _engineer(applications, slope, intercept)
        inputs = pipeline.transform(engineered.select(selected))
        return pl.concat([applications.select(ID), inputs.select(selected)], how="horizontal")

    return {
        "selected": selected,
        "inputs": transform(history.drop(TARGET)),
        "transform": transform,
    }


if __name__ == "__main__":
    data = load_history().head(10_000)
    holdout = pl.Series("is_holdout", np.random.default_rng(0).random(data.height) < 0.25)
    out = build_model_inputs(data, holdout)
    print(f"model inputs ({len(out['selected'])}): {out['selected']}")
    print(out["inputs"].head())
