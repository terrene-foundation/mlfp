#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP03 Assessment Task 2 — The Model Zoo.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader runs the submission's solve(), then independently re-trains one
reference model through the same TrainingPipeline to tie the reported table to
reality. A stub returning a fabricated table fails the re-derivation check.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
EXPECTED_COLUMNS = ["model", "accuracy", "f1", "auc"]
REQUIRED_MODELS = {
    "logistic_regression",
    "naive_bayes",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "lightgbm",
}
ENSEMBLES = {"random_forest", "extra_trees", "lightgbm", "gradient_boosting"}
BASE_FEATURES = [
    "satisfaction_score",
    "avg_order_value",
    "num_returns",
    "order_count",
    "loyalty_int",
    "total_revenue",
    "days_since_last_order",
    "customer_tenure_days",
]


def _model_frame() -> pl.DataFrame:
    df = MLFPDataLoader().load("mlfp03", "ecommerce_customers.parquet")
    df = df.sort("customer_id").head(N_ROWS)
    rng = np.random.default_rng(SEED)

    def z(col: str) -> np.ndarray:
        a = df[col].to_numpy().astype(float)
        return (a - a.mean()) / (a.std() + 1e-9)

    loyal = df["loyalty_member"].cast(pl.Int64).to_numpy().astype(float)
    sat_high = (df["satisfaction_score"] >= 4).cast(pl.Int64).to_numpy().astype(float)
    logit = (
        1.0 * z("satisfaction_score")
        + 0.9 * loyal
        + 0.8 * z("avg_order_value")
        - 0.7 * z("num_returns")
        + 0.5 * z("order_count")
        + 1.4 * (loyal * sat_high)
        + rng.normal(0.0, 1.3, size=df.height)
    )
    df = df.with_columns(
        [
            pl.col("loyalty_member").cast(pl.Int64).alias("loyalty_int"),
            pl.Series(TARGET, (logit > 2.0).astype(np.int64)),
            pl.int_range(0, df.height, dtype=pl.Int64).alias("row_id"),
        ]
    )
    return df.select(BASE_FEATURES + ["row_id", TARGET])


async def _reference_rf_auc() -> float:
    """Independently re-train random_forest via the same pipeline."""
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry, TrainingPipeline
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
    from kailash_ml.types import FeatureField, FeatureSchema

    frame = _model_frame()
    schema = FeatureSchema(
        name="premium_zoo",
        features=[FeatureField(name=f, dtype="float64") for f in BASE_FEATURES],
        entity_id_column="row_id",
    )
    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=ModelRegistry(conn))
        result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=ModelSpec(
                model_class="sklearn.ensemble.RandomForestClassifier",
                framework="sklearn",
                hyperparameters={"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
            ),
            eval_spec=EvalSpec(
                metrics=["accuracy", "f1", "auc"],
                split_strategy="holdout",
                test_size=0.25,
            ),
            experiment_name="ref_rf",
        )
        return float(result.metrics["auc"])
    finally:
        await conn.close()


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t2", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score
    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score
    try:
        r = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    c["returns_dataframe"] = isinstance(r, pl.DataFrame)
    if not c["returns_dataframe"]:
        return _finalize(score)

    c["columns_exact"] = r.columns == EXPECTED_COLUMNS
    if not c["columns_exact"]:
        return _finalize(score)

    names = r["model"].to_list()
    c["at_least_six_models"] = r.height >= 6
    c["no_duplicate_models"] = len(set(names)) == len(names)
    c["required_models_present"] = REQUIRED_MODELS.issubset(set(names))

    # metric ranges
    try:
        for col in ("accuracy", "f1", "auc"):
            vals = r[col].to_numpy().astype(float)
            assert np.all((vals >= 0.0) & (vals <= 1.0))
        c["metrics_in_range"] = True
    except Exception:
        c["metrics_in_range"] = False

    aucs = r["auc"].to_numpy().astype(float)
    f1s = r["f1"].to_numpy().astype(float)
    has_rows = aucs.size > 0 and f1s.size > 0
    c["all_models_beat_floor"] = bool(has_rows and np.all(aucs > 0.82))
    c["best_auc_above_target"] = bool(has_rows and np.max(aucs) >= 0.88)
    c["best_f1_above_target"] = bool(has_rows and np.max(f1s) >= 0.80)

    # sorted by auc descending
    c["sorted_by_auc_desc"] = bool(has_rows and np.all(np.diff(aucs) <= 1e-9))

    # an ensemble in the top-3 by auc
    top3 = set(r.sort("auc", descending=True).head(3)["model"].to_list())
    c["ensemble_in_top3"] = len(ENSEMBLES.intersection(top3)) >= 1

    # anti-stub: reported random_forest auc must match an independent re-train
    try:
        reported = r.filter(pl.col("model") == "random_forest")["auc"]
        ref_auc = asyncio.run(_reference_rf_auc())
        c["random_forest_auc_matches"] = (
            reported.len() == 1 and abs(float(reported[0]) - ref_auc) < 0.02
        )
    except Exception:
        c["random_forest_auc_matches"] = False

    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["max"] > 0 and score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
