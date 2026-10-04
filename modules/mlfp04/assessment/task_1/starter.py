# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 1: Customer Segments and Mixture Models

Implement the two functions below; problem.md defines their outputs and how
they are accepted. Submit this file to the portal for grading; the grader
runs your functions on cohorts and mixtures you have not seen.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def load_dev_customers() -> pl.DataFrame:
    """One CRM export for development. The grader uses different cohorts."""
    return pl.read_parquet(Path(__file__).with_name("dev_customers.parquet"))


def segment_customers(customers: pl.DataFrame) -> dict:
    raise NotImplementedError


def fit_mixture(X: np.ndarray, k: int, seed: int = 0) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    customers = load_dev_customers()
    print(customers.describe())
    # A development mixture you can check your EM on (any seed works).
    rng = np.random.default_rng(0)
    X_dev = np.vstack([rng.normal([0, 0], 1.0, (300, 2)), rng.normal([2.5, 1.0], 0.7, (200, 2))])
    print(X_dev.shape)
