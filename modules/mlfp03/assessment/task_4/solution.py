# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 4: Release Review (Reference Solution)

Instructors only. Graded by grader.py on models, decisions and batches the
student never sees.

Decisions this reference makes (one defensible route, not the only one):

1. Explanations: the committee's three properties (attributions add up to
   "this applicant minus the average applicant", an unused field gets
   nothing, credit is split by Shapley's rule) define interventional Shapley
   values against the background book. ``shap`` computes them exactly for a
   black-box scoring function: an ``Independent`` masker over the background
   and the ``exact`` algorithm (the feature count is small). Probabilities are
   explained directly, so the attributions are in probability units.
2. Fairness: per group, the approval share, its ratio to the best-treated
   group (the four-fifths rule), and the approval shares among applicants who
   repaid and among those who defaulted (the two halves of equalised odds).
   Unrecorded group membership is a group of its own, never dropped.
3. Drift: kailash-ml ``DriftMonitor`` on its own SQLite file. The batch is
   tested on every listed field at once, so a 0.05 per-field significance
   level would raise false alarms on stable data. Both significance tests the
   monitor runs — KS for continuous fields and chi-squared for count fields
   (``DriftThresholds`` sets each) — are Bonferroni-tightened to
   0.01 / number of fields; PSI keeps the usual 0.2 "material shift" bar.
"""
from __future__ import annotations

import asyncio
import tempfile
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl
import shap

from kailash.db import ConnectionManager
from kailash_ml import DriftMonitor
from kailash_ml.engines.drift_monitor import DriftThresholds
from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

ID = "customer_id"
UNRECORDED = "unrecorded"


def load_history() -> pl.DataFrame:
    """The labelled development file (for local runs only)."""
    return MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet")


# --------------------------------------------------------------------------
# 1. Explanations
# --------------------------------------------------------------------------
def explain(predict_proba: Callable, background: pl.DataFrame, applications: pl.DataFrame) -> pl.DataFrame:
    features = [c for c in background.columns if c != ID]
    schema = {c: pl.Float64 for c in features}

    def f(X: np.ndarray) -> np.ndarray:
        return np.asarray(predict_proba(pl.DataFrame(X, schema=schema, orient="row")), dtype=float)

    bg = background.select(features).cast(pl.Float64).to_numpy()
    X = applications.select(features).cast(pl.Float64).to_numpy()
    explainer = shap.Explainer(f, shap.maskers.Independent(bg, max_samples=bg.shape[0]), algorithm="exact")
    values = explainer(X).values
    return pl.concat(
        [applications.select(ID), pl.DataFrame(values, schema=schema, orient="row")],
        how="horizontal",
    )


# --------------------------------------------------------------------------
# 2. Fairness
# --------------------------------------------------------------------------
def fairness_report(audit: pl.DataFrame, group_column: str) -> pl.DataFrame:
    df = audit.with_columns(
        pl.col(group_column).cast(pl.Utf8).fill_null(UNRECORDED).alias("group"),
        pl.col("approved").cast(pl.Float64).alias("_a"),
        pl.col("default").cast(pl.Int64).alias("_y"),
    )
    report = df.group_by("group").agg(
        pl.len().alias("applicants"),
        pl.col("_a").mean().alias("approval_rate"),
        pl.col("_a").filter(pl.col("_y") == 0).mean().alias("good_approval_rate"),
        pl.col("_a").filter(pl.col("_y") == 1).mean().alias("default_approval_rate"),
    )
    best = report["approval_rate"].max()
    return (
        report.with_columns((pl.col("approval_rate") / best).alias("approval_ratio"))
        .with_columns((pl.col("approval_ratio") >= 0.8).alias("passes_four_fifths"))
        .select(
            "group", "applicants", "approval_rate", "approval_ratio",
            "good_approval_rate", "default_approval_rate", "passes_four_fifths",
        )  # fmt: skip
        .sort("group")
    )


# --------------------------------------------------------------------------
# 3. Drift
# --------------------------------------------------------------------------
async def _drift(reference: pl.DataFrame, batch: pl.DataFrame, features: list[str]) -> list[str]:
    with tempfile.TemporaryDirectory() as d:
        conn = ConnectionManager(f"sqlite:///{(Path(d) / 'drift.db').as_posix()}")
        await conn.initialize()
        try:
            alpha = 0.01 / len(features)  # Bonferroni: one test per field per batch
            monitor = DriftMonitor(
                conn,
                tenant_id="credit_desk",
                thresholds=DriftThresholds(psi=0.2, ks_pvalue=alpha, chi2_pvalue=alpha),
            )
            await monitor.set_reference_data("credit_default", reference, features)
            report = await monitor.check_drift("credit_default", batch)
        finally:
            await conn.close()
    return sorted(r.feature_name for r in report.feature_results if r.drift_detected)


def drift_alerts(reference: pl.DataFrame, batch: pl.DataFrame, features: list[str]) -> list[str]:
    return asyncio.run(_drift(reference, batch, list(features)))


if __name__ == "__main__":
    data = load_history()
    feats = ["credit_utilization", "payment_history_score", "num_late_payments"]
    sample = data.sample(4_000, seed=1)

    def toy(apps: pl.DataFrame) -> np.ndarray:
        z = -2 + 2.5 * apps["credit_utilization"].to_numpy() + 0.2 * apps["num_late_payments"].to_numpy()
        return 1 / (1 + np.exp(-z))

    print(explain(toy, sample.select([ID] + feats).head(50), sample.select([ID] + feats).tail(3)))
    audit = sample.with_columns((pl.col("credit_utilization") < 0.4).alias("approved"))
    print(fairness_report(audit, "race"))
    print(drift_alerts(sample.head(2_000), sample.tail(2_000), feats))
