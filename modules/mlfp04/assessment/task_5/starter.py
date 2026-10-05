# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 5: From Discovered Segments to a Neural Network

Implement the three functions below; problem.md defines their outputs and how
they are accepted. Submit this file to the portal for grading; the grader
runs your functions on networks and customer populations you have not seen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

TARGET = "churned"


def load_dev_churn() -> pl.DataFrame:
    """One customer table for development. The grader uses different ones."""
    return pl.read_parquet(Path(__file__).with_name("dev_churn.parquet"))


def loss_and_gradients(params: dict, X: np.ndarray, y: np.ndarray, l2: float = 0.0) -> tuple[float, dict]:
    raise NotImplementedError


def discover_features(customers: pl.DataFrame) -> pl.DataFrame:
    raise NotImplementedError


def fit_and_predict(train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    df = load_dev_churn()
    print(df.describe())
    # Hold out your own development test set; the grader holds out its own.
    train, test = df.head(2200), df.tail(800)
    print(train.height, test.height, train[TARGET].mean())
