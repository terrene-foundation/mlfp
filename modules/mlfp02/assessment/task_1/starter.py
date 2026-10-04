# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 1: Bayesian Updating & Likelihood Estimation

Implement the four functions below. problem.md defines what each must return
and how it is accepted. The grader calls them on data you have not seen, so
compute everything from the arguments.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from shared import MLFPDataLoader


def load_orders() -> pl.DataFrame:
    """The experiment log, all arms — for development only."""
    return MLFPDataLoader().load("mlfp02", "experiment_data.parquet")


def conversion_posterior(
    orders: pl.DataFrame, arm: str, prior: tuple[float, float]
) -> dict[str, float]:
    raise NotImplementedError


def prob_beats_control(
    orders: pl.DataFrame, treatment: str, prior: tuple[float, float]
) -> float:
    raise NotImplementedError


def fit_order_value(values: np.ndarray) -> dict:
    raise NotImplementedError


def map_gamma(values: np.ndarray) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    orders = load_orders()
    print(orders.head())
    # Try your functions here, e.g. on the treatment_a arm with prior (2, 20)
    # and on a few hundred positive order values.
