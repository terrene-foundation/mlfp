# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 4: The Release Review

Implement explain(), fairness_report() and drift_alerts(); problem.md defines
what each returns and how it is accepted. The grader hands your functions
scoring functions, audit tables, references and batches you have never seen,
and scores them against ground truth it holds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from shared import MLFPDataLoader


def load_history() -> pl.DataFrame:
    """The labelled credit applications — for development only."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


def explain(predict_proba, background: pl.DataFrame, applications: pl.DataFrame) -> pl.DataFrame:
    """Per-field attributions for each applicant, against the background book."""
    raise NotImplementedError


def fairness_report(audit: pl.DataFrame, group_column: str) -> pl.DataFrame:
    """Per-group approval shares, four-fifths ratio and equalised-odds rates."""
    raise NotImplementedError


def drift_alerts(reference: pl.DataFrame, batch: pl.DataFrame, features: list[str]) -> list[str]:
    """Names of the monitored fields whose distribution moved in this batch."""
    raise NotImplementedError


if __name__ == "__main__":
    history = load_history()
    sample = history.sample(4_000, seed=1)
    fields = ["credit_utilization", "payment_history_score", "num_late_payments"]

    def toy_score(apps: pl.DataFrame) -> np.ndarray:
        z = -2 + 2.5 * apps["credit_utilization"].to_numpy() + 0.2 * apps["num_late_payments"].to_numpy()
        return 1 / (1 + np.exp(-z))

    apps = sample.select(["customer_id"] + fields)
    print(explain(toy_score, apps.head(200), apps.tail(5)))

    audit = sample.with_columns((pl.col("credit_utilization") < 0.4).alias("approved"))
    print(fairness_report(audit, "race"))

    print(drift_alerts(sample.head(2_000), sample.tail(2_000), fields))
