#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP03 Assessment Task 3 — Evaluation, Imbalance & Interpretability.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader independently re-trains the baseline and class-balanced models and
recomputes per-class recall via km.diagnose, then verifies the submission's
reported minority-recall lift matches reality (defeats hardcoded dicts).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import pickle
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
TOP_K = 6
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
REQUIRED_KEYS = {
    "baseline_minority_recall",
    "balanced_minority_recall",
    "baseline_recall_macro",
    "balanced_recall_macro",
    "baseline_accuracy",
    "balanced_accuracy",
    "roc_auc",
    "top_features",
    "n_features",
}


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


async def _reference_recalls() -> dict:
    """Independently train baseline + balanced; return per-class recalls."""
    from kailash.db import ConnectionManager
    from kailash_ml import ModelRegistry, TrainingPipeline, diagnose
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
    from kailash_ml.types import FeatureField, FeatureSchema

    frame = _model_frame()
    schema = FeatureSchema(
        name="premium_eval",
        features=[FeatureField(name=f, dtype="float64") for f in BASE_FEATURES],
        entity_id_column="row_id",
    )
    n = frame.height
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    test = frame[idx[int(n * 0.75):].tolist()]
    x_test, y_test = test.select(BASE_FEATURES).to_numpy(), test[TARGET].to_numpy()
    ev = EvalSpec(metrics=["accuracy", "f1", "auc"], split_strategy="holdout", test_size=0.25)

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    try:
        reg = ModelRegistry(conn)
        pipe = TrainingPipeline(feature_store=None, registry=reg)
        base = await pipe.train(
            data=frame, schema=schema,
            model_spec=ModelSpec(
                model_class="sklearn.ensemble.RandomForestClassifier", framework="sklearn",
                hyperparameters={"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
            ), eval_spec=ev, experiment_name="b",
        )
        bal = await pipe.train(
            data=frame, schema=schema,
            model_spec=ModelSpec(
                model_class="sklearn.ensemble.RandomForestClassifier", framework="sklearn",
                hyperparameters={
                    "n_estimators": 150, "random_state": SEED, "n_jobs": -1,
                    "class_weight": "balanced",
                },
            ), eval_spec=ev, experiment_name="bl",
        )
        bm = pickle.loads(await reg.load_artifact(base.model_version.name, base.model_version.version))
        blm = pickle.loads(await reg.load_artifact(bal.model_version.name, bal.model_version.version))
        brep = diagnose(bm, kind="classical_classifier", data=(x_test, y_test), show=False)
        blrep = diagnose(blm, kind="classical_classifier", data=(x_test, y_test), show=False)
        return {
            "base_minor": float(brep.per_class["1.0"]["recall"]),
            "bal_minor": float(blrep.per_class["1.0"]["recall"]),
            "auc": float(bal.metrics["auc"]),
        }
    finally:
        await conn.close()


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t3", path)
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
    c["returns_dict"] = isinstance(r, dict) and REQUIRED_KEYS.issubset(r.keys())
    if not c["returns_dict"]:
        return _finalize(score)

    def fget(k: float) -> float:
        try:
            return float(r[k])
        except Exception:
            return float("nan")

    base_min = fget("baseline_minority_recall")
    bal_min = fget("balanced_minority_recall")
    base_macro = fget("baseline_recall_macro")
    bal_macro = fget("balanced_recall_macro")
    base_acc = fget("baseline_accuracy")
    bal_acc = fget("balanced_accuracy")
    auc = fget("roc_auc")
    top = r.get("top_features")

    finite = [base_min, bal_min, base_macro, bal_macro, base_acc, bal_acc, auc]
    c["values_finite_in_range"] = all(
        np.isfinite(v) and 0.0 <= v <= 1.0 for v in finite
    )

    # 1. imbalance handling lifts minority recall
    c["balanced_lifts_minority_recall"] = bool(
        np.isfinite(base_min) and np.isfinite(bal_min) and bal_min > base_min + 0.02
    )
    # 2. balanced minority recall clears an honest floor
    c["balanced_minority_recall_floor"] = bool(bal_min >= 0.68)
    # 3. baseline minority recall sits in the un-handled band
    c["baseline_minority_recall_band"] = bool(0.55 <= base_min <= 0.67)
    # 4. macro recall improves under balancing
    c["macro_recall_improves"] = bool(bal_macro > base_macro)
    # 5. the accuracy / recall tradeoff: balanced accuracy drops
    c["accuracy_tradeoff"] = bool(bal_acc < base_acc + 1e-9)
    # 6. held-out AUC above floor
    c["roc_auc_floor"] = bool(auc >= 0.85)

    # 7. interpretability output shape
    c["top_features_shape"] = (
        isinstance(top, list)
        and len(top) == TOP_K
        and set(top).issubset(set(BASE_FEATURES))
    )
    # 8. SHAP surfaced the dominant driver
    c["top_feature_is_driver"] = bool(
        isinstance(top, list)
        and len(top) >= 3
        and "satisfaction_score" in set(top[:3])
    )
    # 9. n_features correct
    c["n_features_correct"] = r.get("n_features") == len(BASE_FEATURES)

    # 10/11. anti-stub: re-derive recalls and match reported values
    try:
        ref = asyncio.run(_reference_recalls())
        c["baseline_recall_matches"] = abs(base_min - ref["base_minor"]) < 0.03
        c["balanced_recall_matches"] = abs(bal_min - ref["bal_minor"]) < 0.03
    except Exception:
        c["baseline_recall_matches"] = False
        c["balanced_recall_matches"] = False

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
