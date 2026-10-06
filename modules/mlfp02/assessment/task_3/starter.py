# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 3: Regression, ANOVA & Logistic Inference

Implement the three functions below; problem.md defines their outputs and how
they are accepted. The grader fits your models on secret samples and scores
them on data you never see.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader


def load_hdb() -> pl.DataFrame:
    """Raw HDB resale transactions — for development only."""
    return MLFPDataLoader().load("mlfp01", "hdb_resale.parquet")


def load_credit() -> pl.DataFrame:
    """Credit-scoring book — for development only."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def fit_price_model(transactions: pl.DataFrame) -> dict:
    raise NotImplementedError


def anova_flat_types(transactions: pl.DataFrame, flat_types: list[str]) -> dict:
    raise NotImplementedError


def fit_default_model(train: pl.DataFrame) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    hdb = load_hdb()
    print(hdb.head())
    print(hdb["storey_range"].unique().sort().to_list())
    credit = load_credit()
    print(credit.select("default").mean())
