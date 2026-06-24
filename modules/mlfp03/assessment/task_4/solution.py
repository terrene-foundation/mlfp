# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP03 — Assessment Task 4: Production Pipeline — Registry, Drift, Deploy
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.

Trains a LightGBM model through the kailash-ml ``TrainingPipeline``, registers
it and promotes staging -> production in the ``ModelRegistry``, then arms a
``DriftMonitor`` against the training distribution and checks two incoming
batches: a clean same-distribution slice (no alarm) and an economic-downturn
shifted slice (drift fires). ``solve()`` wraps the async work in ``asyncio.run``.

NOTE ON TWO DATABASES: we give the ModelRegistry and the DriftMonitor separate
SQLite files — the realistic production posture (a model registry and a
monitoring store are distinct systems with independent lifecycles). Using fresh,
separate files per store also sidesteps the "stale .db" gotcha where a database
created by an older kailash-ml version carries a pre-migration schema. (On
kailash-ml 2.2.2 a single shared connection works — _kml_drift_reports is
created and written by the DriftMonitor engine with a consistent ``id`` schema.)
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np
import polars as pl

from shared import MLFPDataLoader

warnings.filterwarnings("ignore")

N_ROWS = 10_000
SEED = 42
TARGET = "premium_response"
REFERENCE_ROWS = 7_500
PSI_THRESHOLD = 0.2
KS_THRESHOLD = 0.05
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


def _shift_slice(clean: pl.DataFrame) -> pl.DataFrame:
    """Economic-downturn shift: spend collapses, recency stretches, mood drops."""
    return clean.with_columns(
        [
            (pl.col("avg_order_value") * 0.6).alias("avg_order_value"),
            (pl.col("total_revenue") * 0.6).alias("total_revenue"),
            (pl.col("days_since_last_order") * 1.5 + 60).alias("days_since_last_order"),
            (pl.col("satisfaction_score") - 1).alias("satisfaction_score"),
        ]
    )


async def _run() -> dict:
    from kailash.db import ConnectionManager
    from kailash_ml import DriftMonitor, ModelRegistry, TrainingPipeline
    from kailash_ml.engines.training_pipeline import EvalSpec, ModelSpec
    from kailash_ml.types import FeatureField, FeatureSchema

    frame = _model_frame()
    schema = FeatureSchema(
        name="premium_prod",
        features=[FeatureField(name=f, dtype="float64") for f in BASE_FEATURES],
        entity_id_column="row_id",
    )
    reference = frame.select(BASE_FEATURES).head(REFERENCE_ROWS)
    clean = frame.select(BASE_FEATURES).tail(frame.height - REFERENCE_ROWS)
    shifted = _shift_slice(clean)

    tmp = Path(tempfile.gettempdir())
    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    registry_db = tmp / f"mlfp03_t4_registry_{uid}.db"
    drift_db = tmp / f"mlfp03_t4_drift_{uid}.db"
    reg_conn = ConnectionManager(f"sqlite:///{registry_db.resolve().as_posix()}")
    drift_conn = ConnectionManager(f"sqlite:///{drift_db.resolve().as_posix()}")
    await reg_conn.initialize()
    await drift_conn.initialize()
    try:
        registry = ModelRegistry(reg_conn)
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        result = await pipeline.train(
            data=frame,
            schema=schema,
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier",
                framework="lightgbm",
                hyperparameters={
                    "n_estimators": 200,
                    "random_state": SEED,
                    "verbose": -1,
                },
            ),
            eval_spec=EvalSpec(
                metrics=["accuracy", "f1", "auc"],
                split_strategy="holdout",
                test_size=0.25,
            ),
            experiment_name="premium_prod",
        )
        version = result.model_version.version

        # Promote staging -> production with an audit reason.
        model_name = result.model_version.name
        await registry.promote_model(
            model_name,
            version,
            "production",
            reason=f"AUC gate passed: auc={result.metrics['auc']:.4f}",
        )
        promoted = await registry.get_model(model_name, stage="production")

        # Arm drift monitoring against the training distribution.
        monitor = DriftMonitor(
            drift_conn,
            tenant_id="_single",
            psi_threshold=PSI_THRESHOLD,
            ks_threshold=KS_THRESHOLD,
        )
        await monitor.set_reference_data(model_name, reference, BASE_FEATURES)
        clean_report = await monitor.check_drift(model_name, clean)
        shift_report = await monitor.check_drift(model_name, shifted)

        return {
            "registered_version": int(version),
            "production_stage": str(promoted.stage),
            "reference_auc": float(result.metrics["auc"]),
            "clean_drift_detected": bool(clean_report.overall_drift_detected),
            "shift_drift_detected": bool(shift_report.overall_drift_detected),
            "n_drifted_features_clean": int(
                sum(1 for f in clean_report.feature_results if f.drift_detected)
            ),
            "n_drifted_features_shift": int(
                sum(1 for f in shift_report.feature_results if f.drift_detected)
            ),
            "shift_severity": str(shift_report.overall_severity),
        }
    finally:
        await reg_conn.close()
        await drift_conn.close()
        registry_db.unlink(missing_ok=True)
        drift_db.unlink(missing_ok=True)


def solve() -> dict:
    """Train, register, promote to production, and run drift detection.

    Returns a dict with keys: registered_version, production_stage,
    reference_auc, clean_drift_detected, shift_drift_detected,
    n_drifted_features_clean, n_drifted_features_shift, shift_severity.
    """
    return asyncio.run(_run())


if __name__ == "__main__":
    out = solve()
    print(
        f"registered version : {out['registered_version']} ({out['production_stage']})"
    )
    print(f"reference AUC      : {out['reference_auc']:.4f}")
    print(
        f"drift  clean={out['clean_drift_detected']} "
        f"({out['n_drifted_features_clean']} feats)  "
        f"shift={out['shift_drift_detected']} "
        f"({out['n_drifted_features_shift']} feats, {out['shift_severity']})"
    )
