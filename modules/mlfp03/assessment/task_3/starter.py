# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 3: Decisions Priced in Dollars

Implement build_decision_model(); problem.md defines what it returns and how
it is accepted. The grader calls it on a secret sample with secret costs and
scores it on applications you have never seen.
"""
from __future__ import annotations

import tempfile

import numpy as np
import polars as pl

from shared import MLFPDataLoader


def load_history() -> pl.DataFrame:
    """The labelled credit applications — for development only."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def build_decision_model(history: pl.DataFrame, costs: dict, registry_dir: str) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    history = load_history()
    sample = history.sample(8_000, seed=1)
    costs = {"missed_default": 10_000.0, "declined_good": 1_500.0}
    with tempfile.TemporaryDirectory() as registry_dir:
        result = build_decision_model(sample, costs, registry_dir)
        unseen = history.sample(4_000, seed=2)
        p = np.asarray(result["predict_proba"](unseen.drop("default")))
        print("mean predicted:", p.mean(), " observed default rate:", unseen["default"].mean())
        print("approval rate:", np.asarray(result["decide"](unseen.drop("default"))).mean())
        print("estimated cost per application:", result["expected_cost"])
