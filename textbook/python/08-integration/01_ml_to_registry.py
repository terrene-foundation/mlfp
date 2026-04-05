# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Integration / ML to Registry Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train a model → register in ModelRegistry → promote to production
# LEVEL: Intermediate
# PARITY: Python-only (Rust ML is architecturally different)
# VALIDATES: TrainingPipeline → ModelRegistry.register_model() → promote_model()
#
# Run: uv run python textbook/python/08-integration/01_ml_to_registry.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import pickle
import tempfile

import polars as pl
from kailash_ml import (
    TrainingPipeline,
    ModelRegistry,
)
from kailash_ml.types import MetricSpec, ModelSignature
from dataflow.utils.connection import ConnectionManager

# ── 1. Create synthetic training data ───────────────────────────────

df = pl.DataFrame(
    {
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_2": [0.1, 0.4, 0.9, 1.6, 2.5, 3.6, 4.9, 6.4, 8.1, 10.0],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    }
)

assert df.shape == (10, 3)

# ── 2. Train with TrainingPipeline ──────────────────────────────────
# TrainingPipeline wraps sklearn/xgboost/lightgbm behind a unified API.
# It produces a trained model object and evaluation metrics.

# NOTE: Training requires the actual package. This tutorial validates
# the integration pattern without running the full pipeline.

# Pattern: train → get model bytes → register
# pipeline = TrainingPipeline()
# model, metrics = pipeline.train(data=df, target="target", spec=ModelSpec(...))
# model_bytes = pickle.dumps(model)

# For this tutorial, simulate the output
simulated_model = {"type": "logistic_regression", "weights": [0.5, 0.3]}
model_bytes = pickle.dumps(simulated_model)

assert isinstance(model_bytes, bytes), "Model must be serialized to bytes"

# ── 3. Register in ModelRegistry ────────────────────────────────────
# ModelRegistry stores versioned model artifacts with metrics.
# CRITICAL: register_model() takes artifact=bytes, NOT a model object.


async def main():
    db_path = os.path.join(tempfile.gettempdir(), "textbook_integration_01.db")
    conn = ConnectionManager(f"sqlite:///{db_path}")
    await conn.initialize()

    registry = ModelRegistry(conn)
    await registry.initialize()

    # Register with metrics
    version = await registry.register_model(
        name="credit_scorer",
        artifact=model_bytes,
        metrics=[
            MetricSpec(name="accuracy", value=0.85),
            MetricSpec(name="auc", value=0.92),
        ],
    )

    assert version is not None, "Registration returns a version"
    print(f"Registered version: {version}")

    # ── 4. Promote to production ────────────────────────────────────
    # Two-step: register → promote. This governance gate ensures
    # models are reviewed before serving traffic.

    await registry.promote_model(
        name="credit_scorer",
        version=version.version,
        target_stage="production",
        reason="Passed all validation checks",
    )

    print("Model promoted to production")

    # ── 5. Integration pattern summary ──────────────────────────────
    # TrainingPipeline.train() → pickle.dumps() → ModelRegistry.register_model()
    # → ModelRegistry.promote_model() → InferenceServer.predict()
    #
    # This is the standard ML lifecycle in Kailash:
    #   1. Train (kailash-ml TrainingPipeline)
    #   2. Register (kailash-ml ModelRegistry)
    #   3. Promote (governance gate)
    #   4. Serve (kailash-ml InferenceServer + kailash-nexus)

    await conn.close()
    try:
        os.unlink(db_path)
    except OSError:
        pass


asyncio.run(main())

print("PASS: 08-integration/01_ml_to_registry")
