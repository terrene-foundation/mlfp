# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 2: The Model Zoo (Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.

Trains six classifiers on identical data through the kailash-ml
``TrainingPipeline`` (one engine ``train()`` call per algorithm — no raw
``.fit()`` in user code) and returns a fair, sorted comparison table.
``solve()`` wraps the async pipeline in ``asyncio.run`` so the grader can call
it synchronously.
"""
from __future__ import annotations

import asyncio
import warnings

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
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

# Six required algorithms (model_class, framework, hyperparameters). All seeds
# fixed for a deterministic, reproducible comparison.
MODEL_ZOO: dict[str, tuple[str, str, dict]] = {
    "logistic_regression": (
        "sklearn.linear_model.LogisticRegression",
        "sklearn",
        {"max_iter": 2000, "random_state": SEED},
    ),
    "naive_bayes": ("sklearn.naive_bayes.GaussianNB", "sklearn", {}),
    "decision_tree": (
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn",
        {"max_depth": 6, "random_state": SEED},
    ),
    "random_forest": (
        "sklearn.ensemble.RandomForestClassifier",
        "sklearn",
        {"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
    ),
    "extra_trees": (
        "sklearn.ensemble.ExtraTreesClassifier",
        "sklearn",
        {"n_estimators": 150, "random_state": SEED, "n_jobs": -1},
    ),
    "lightgbm": (
        "lightgbm.LGBMClassifier",
        "lightgbm",
        {"n_estimators": 200, "random_state": SEED, "verbose": -1},
    ),
}


def _model_frame() -> pl.DataFrame:
    """Load N_ROWS, derive ``premium_response``, return the model-ready frame.

    The frame contains exactly the 8 base features, ``row_id`` (entity id), and
    the target — the column layout TrainingPipeline's target-detection expects.
    """
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


async def _run_zoo() -> list[dict]:
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
    eval_spec = EvalSpec(
        metrics=["accuracy", "f1", "auc"], split_strategy="holdout", test_size=0.25
    )

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()
    try:
        pipeline = TrainingPipeline(feature_store=None, registry=ModelRegistry(conn))
        rows: list[dict] = []
        for name, (model_class, framework, hp) in MODEL_ZOO.items():
            result = await pipeline.train(
                data=frame,
                schema=schema,
                model_spec=ModelSpec(
                    model_class=model_class, framework=framework, hyperparameters=hp
                ),
                eval_spec=eval_spec,
                experiment_name=f"zoo_{name}",
            )
            rows.append(
                {
                    "model": name,
                    "accuracy": float(result.metrics["accuracy"]),
                    "f1": float(result.metrics["f1"]),
                    "auc": float(result.metrics["auc"]),
                }
            )
        return rows
    finally:
        await conn.close()


def solve() -> pl.DataFrame:
    """Train the six-model zoo and return a comparison table.

    Returns a Polars DataFrame with columns ``[model, accuracy, f1, auc]``,
    one row per algorithm, sorted by ``auc`` descending.
    """
    rows = asyncio.run(_run_zoo())
    return pl.DataFrame(rows).sort("auc", descending=True)


if __name__ == "__main__":
    table = solve()
    print(table)
    best = table.row(0, named=True)
    print(f"\nBest model: {best['model']}  AUC={best['auc']:.4f}  F1={best['f1']:.4f}")
