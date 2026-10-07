# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 2: Experiment Read-out

Implement the five functions below; problem.md defines their outputs and how
they are accepted. The grader calls them on experiments you have not seen.
"""
from __future__ import annotations

import polars as pl

from shared import MLFPDataLoader

# The experiment's designed traffic split (from the experiment brief).
DESIGN = {"control": 0.40, "treatment_a": 0.35, "treatment_b": 0.15, "variant_c": 0.10}


def load_orders() -> pl.DataFrame:
    """The experiment log, all arms — for development only."""
    return MLFPDataLoader().load("mlfp02", "experiment_data.parquet")


def srm_check(orders: pl.DataFrame, design: dict[str, float]) -> dict:
    raise NotImplementedError


def sample_size_per_arm(
    baseline: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    raise NotImplementedError


def analyse_ab(orders: pl.DataFrame, treatment: str, design: dict[str, float]) -> dict:
    raise NotImplementedError


def segment_tests(orders: pl.DataFrame, treatment: str) -> dict:
    raise NotImplementedError


def log_to_tracker(results: dict, store_url: str) -> str:
    raise NotImplementedError


if __name__ == "__main__":
    orders = load_orders()
    print(orders.group_by("experiment_group").len())
    # Develop on a random sample first — the full log is 500,000 rows.
    sample = orders.sample(20_000, seed=1)
    print(sample.head())
