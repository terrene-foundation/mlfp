# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 3: Evaluation, Class Imbalance & Interpretability
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.

Trains a baseline and a class-balanced RandomForest through the kailash-ml
``TrainingPipeline``, evaluates per-class behaviour with ``km.diagnose`` (so the
minority-class recall is exposed), and explains the balanced model globally with
``ModelExplainer`` (SHAP). ``solve()`` wraps the async work in ``asyncio.run``.
"""
from __future__ import annotations

import asyncio
import pickle
import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
TOP_K = 6
SHAP_BACKGROUND = 64
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


def _holdout_test(frame: pl.DataFrame) -> pl.DataFrame:
    """Reproduce TrainingPipeline's deterministic holdout split (test portion)."""
    n = frame.height
    idx = np.arange(n)
    np.random.RandomState(42).shuffle(idx)
    split_idx = int(n * 0.75)
    return frame[idx[split_idx:].tolist()]


async def _run() -> dict:
    from kailash.db import ConnectionManager
    from kailash_ml import ModelExplainer, ModelRegistry, TrainingPipeline, diagnose
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
    from kailash_ml.types import FeatureField, FeatureSchema

    frame = _model_frame()
    schema = FeatureSchema(
        name="premium_eval",
        features=[FeatureField(name=f, dtype="float64") for f in BASE_FEATURES],
        entity_id_column="row_id",
    )
    eval_spec = EvalSpec(
        metrics=["accuracy", "f1", "auc"], split_strategy="holdout", test_size=0.25
    )
    test = _holdout_test(frame)
    x_test = test.select(BASE_FEATURES).to_numpy()
    y_test = test[TARGET].to_numpy()

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    try:
        registry = ModelRegistry(conn)
        pipeline = TrainingPipeline(feature_store=None, registry=registry)

        baseline_spec = ModelSpec(
            model_class="sklearn.ensemble.RandomForestClassifier",
            framework="sklearn",
            hyperparameters={"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
        )
        balanced_spec = ModelSpec(
            model_class="sklearn.ensemble.RandomForestClassifier",
            framework="sklearn",
            hyperparameters={
                "n_estimators": 150,
                "random_state": SEED,
                "n_jobs": -1,
                "class_weight": "balanced",
            },
        )
        base_res = await pipeline.train(
            data=frame, schema=schema, model_spec=baseline_spec,
            eval_spec=eval_spec, experiment_name="baseline",
        )
        bal_res = await pipeline.train(
            data=frame, schema=schema, model_spec=balanced_spec,
            eval_spec=eval_spec, experiment_name="balanced",
        )

        base_model = pickle.loads(
            await registry.load_artifact(
                base_res.model_version.name, base_res.model_version.version
            )
        )
        bal_model = pickle.loads(
            await registry.load_artifact(
                bal_res.model_version.name, bal_res.model_version.version
            )
        )

        # Per-class evaluation — km.diagnose exposes minority-class recall.
        base_rep = diagnose(
            base_model, kind="classical_classifier", data=(x_test, y_test), show=False
        )
        bal_rep = diagnose(
            bal_model, kind="classical_classifier", data=(x_test, y_test), show=False
        )

        # Interpretability — global SHAP importance for the balanced model.
        explainer = ModelExplainer(
            model=bal_model,
            X=frame.select(BASE_FEATURES).head(SHAP_BACKGROUND),
            feature_names=BASE_FEATURES,
        )
        global_report = explainer.explain_global(max_display=TOP_K)
        importance = global_report["feature_importance"]
        top_features = list(importance.keys())[:TOP_K]

        return {
            "baseline_minority_recall": float(base_rep.per_class["1.0"]["recall"]),
            "balanced_minority_recall": float(bal_rep.per_class["1.0"]["recall"]),
            "baseline_recall_macro": float(base_rep.metrics["recall_macro"]),
            "balanced_recall_macro": float(bal_rep.metrics["recall_macro"]),
            "baseline_accuracy": float(base_rep.metrics["accuracy"]),
            "balanced_accuracy": float(bal_rep.metrics["accuracy"]),
            "roc_auc": float(bal_res.metrics["auc"]),
            "top_features": top_features,
            "n_features": len(BASE_FEATURES),
        }
    finally:
        await conn.close()


def solve() -> dict:
    """Evaluate imbalance handling + interpret the balanced model.

    Returns a dict with keys:
      baseline_minority_recall, balanced_minority_recall (minority = positive
      class, ~25%), baseline_recall_macro, balanced_recall_macro,
      baseline_accuracy, balanced_accuracy, roc_auc, top_features (top-6 by SHAP
      global importance), n_features.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    out = solve()
    print(
        f"minority recall  baseline={out['baseline_minority_recall']:.4f} "
        f"-> balanced={out['balanced_minority_recall']:.4f}"
    )
    print(
        f"accuracy         baseline={out['baseline_accuracy']:.4f} "
        f"-> balanced={out['balanced_accuracy']:.4f}  (tradeoff)"
    )
    print(f"recall_macro     baseline={out['baseline_recall_macro']:.4f} "
          f"-> balanced={out['balanced_recall_macro']:.4f}")
    print(f"roc_auc          {out['roc_auc']:.4f}")
    print(f"top_features     {out['top_features']}")
