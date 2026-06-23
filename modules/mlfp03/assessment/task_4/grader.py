#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP03 Assessment Task 4 — Production Pipeline.

Usage:
    python grader.py starter.py
    python grader.py solution.py

The grader runs the submission's solve(), then independently re-trains the model
(for the AUC tie) and re-runs DriftMonitor on the same clean / shifted slices to
confirm the reported drift outcomes are real (defeats hardcoded dicts).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
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
REQUIRED_KEYS = {
    "registered_version",
    "production_stage",
    "reference_auc",
    "clean_drift_detected",
    "shift_drift_detected",
    "n_drifted_features_clean",
    "n_drifted_features_shift",
    "shift_severity",
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


def _shift_slice(clean: pl.DataFrame) -> pl.DataFrame:
    return clean.with_columns(
        [
            (pl.col("avg_order_value") * 0.6).alias("avg_order_value"),
            (pl.col("total_revenue") * 0.6).alias("total_revenue"),
            (pl.col("days_since_last_order") * 1.5 + 60).alias("days_since_last_order"),
            (pl.col("satisfaction_score") - 1).alias("satisfaction_score"),
        ]
    )


async def _reference() -> dict:
    """Independently train + re-run drift; return AUC and drift outcomes."""
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
    uid = f"grader_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    reg_db = tmp / f"mlfp03_t4g_reg_{uid}.db"
    drift_db = tmp / f"mlfp03_t4g_drift_{uid}.db"
    reg_conn = ConnectionManager(f"sqlite:///{reg_db.resolve().as_posix()}")
    drift_conn = ConnectionManager(f"sqlite:///{drift_db.resolve().as_posix()}")
    await reg_conn.initialize()
    await drift_conn.initialize()
    try:
        registry = ModelRegistry(reg_conn)
        pipeline = TrainingPipeline(feature_store=None, registry=registry)
        result = await pipeline.train(
            data=frame, schema=schema,
            model_spec=ModelSpec(
                model_class="lightgbm.LGBMClassifier", framework="lightgbm",
                hyperparameters={"n_estimators": 200, "random_state": SEED, "verbose": -1},
            ),
            eval_spec=EvalSpec(metrics=["accuracy", "f1", "auc"], split_strategy="holdout", test_size=0.25),
            experiment_name="premium_prod",
        )
        name = result.model_version.name
        monitor = DriftMonitor(
            drift_conn, tenant_id="_single",
            psi_threshold=PSI_THRESHOLD, ks_threshold=KS_THRESHOLD,
        )
        await monitor.set_reference_data(name, reference, BASE_FEATURES)
        cr = await monitor.check_drift(name, clean)
        sr = await monitor.check_drift(name, shifted)
        return {
            "auc": float(result.metrics["auc"]),
            "clean_drift": bool(cr.overall_drift_detected),
            "shift_drift": bool(sr.overall_drift_detected),
            "n_shift": int(sum(1 for f in sr.feature_results if f.drift_detected)),
            "n_clean": int(sum(1 for f in cr.feature_results if f.drift_detected)),
        }
    finally:
        await reg_conn.close()
        await drift_conn.close()
        reg_db.unlink(missing_ok=True)
        drift_db.unlink(missing_ok=True)


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_t4", path)
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

    # registry lifecycle
    c["registered_version_valid"] = (
        isinstance(r["registered_version"], int) and r["registered_version"] >= 1
    )
    c["promoted_to_production"] = r["production_stage"] == "production"

    # reference auc
    try:
        auc = float(r["reference_auc"])
    except Exception:
        auc = float("nan")
    c["reference_auc_floor"] = bool(np.isfinite(auc) and auc >= 0.85)

    # drift outcomes (as reported)
    c["clean_no_drift"] = r["clean_drift_detected"] is False
    c["shift_drift_detected"] = r["shift_drift_detected"] is True
    try:
        n_clean = int(r["n_drifted_features_clean"])
        n_shift = int(r["n_drifted_features_shift"])
    except Exception:
        n_clean, n_shift = -1, -1
    c["clean_zero_drifted_features"] = n_clean == 0
    c["shift_multiple_drifted_features"] = n_shift >= 3
    c["shift_severity_signals_drift"] = str(r.get("shift_severity", "")).lower() not in (
        "",
        "none",
    )

    # anti-stub: independently re-derive and confirm the reported outcomes
    try:
        ref = asyncio.run(_reference())
        c["auc_matches_reference"] = abs(auc - ref["auc"]) < 0.02
        c["drift_outcomes_match_reference"] = (
            r["clean_drift_detected"] == ref["clean_drift"]
            and r["shift_drift_detected"] == ref["shift_drift"]
            and ref["clean_drift"] is False
            and ref["shift_drift"] is True
            and n_shift == ref["n_shift"]
        )
    except Exception:
        c["auc_matches_reference"] = False
        c["drift_outcomes_match_reference"] = False

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
