# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 1: Application-Time Model Inputs

Implement build_model_inputs(); problem.md defines what it returns and how it
is accepted. The grader calls it on a secret sample with a secret hold-out
flag, and scores your transform on applications you have never seen.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from shared import MLFPDataLoader


def load_history() -> pl.DataFrame:
    """The labelled credit applications — for development only."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def build_model_inputs(history: pl.DataFrame, is_holdout: pl.Series) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    history = load_history().head(10_000)
    is_holdout = pl.Series("is_holdout", np.random.default_rng(0).random(history.height) < 0.25)
    print(history.head())
    print(history.null_count())
    result = build_model_inputs(history, is_holdout)
    print(result["selected"])
    print(result["inputs"].head())
