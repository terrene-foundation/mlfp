# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP04 — Assessment Task 3: Baskets and Recommendations

Implement the two functions below; problem.md defines their outputs and how
they are accepted. Submit this file to the portal for grading; the grader
runs your functions on till exports and rating histories you have not seen.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

HERE = Path(__file__).parent


def load_dev_baskets() -> pl.DataFrame:
    """One till export for development. The grader uses different exports."""
    return pl.read_parquet(HERE / "dev_baskets.parquet")


def load_dev_ratings() -> pl.DataFrame:
    """One rating history for development. The grader uses different histories."""
    return pl.read_parquet(HERE / "dev_ratings.parquet")


def mine_rules(baskets: pl.DataFrame, min_support: float, min_confidence: float, max_len: int = 3) -> pl.DataFrame:
    raise NotImplementedError


def fit_recommender(history: pl.DataFrame) -> Callable[[pl.DataFrame], np.ndarray]:
    raise NotImplementedError


if __name__ == "__main__":
    baskets = load_dev_baskets()
    print(baskets.head(), baskets["item"].n_unique())
    ratings = load_dev_ratings()
    print(ratings.head(), ratings.height)
