# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 2: Model Selection You Can Defend

Implement select_and_fit(); problem.md defines what it returns and how it is
accepted. The grader calls it on a secret sample of the credit file and scores
your model on applications you have never seen.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from shared import MLFPDataLoader


def load_history() -> pl.DataFrame:
    """The labelled credit applications — for development only."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def select_and_fit(train: pl.DataFrame) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    history = load_history()
    train = history.sample(4_000, seed=1)
    print(train.describe())
    result = select_and_fit(train)
    for family, score in sorted(result["cv_auc"].items(), key=lambda kv: -kv[1]):
        print(f"{family:<22} {score:.4f}")
    print("chosen:", result["chosen"], " estimated AUC:", result["estimated_auc"])
    unseen = history.sample(2_000, seed=2).drop("default")
    print(np.asarray(result["predict_proba"](unseen))[:10])
