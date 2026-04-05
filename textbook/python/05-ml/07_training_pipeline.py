# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / TrainingPipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train models using TrainingPipeline with ModelSpec, EvalSpec,
#            and FeatureSchema.  Pipeline validates data, splits, trains,
#            evaluates, and registers models in ModelRegistry.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: TrainingPipeline, ModelSpec, EvalSpec, TrainingResult —
#            train(), evaluate(), retrain()
#
# Run: uv run python textbook/python/05-ml/07_training_pipeline.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import tempfile

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema
from kailash_ml.engines.feature_store import FeatureStore
from kailash_ml.engines.model_registry import LocalFileArtifactStore, ModelRegistry
from kailash_ml.engines.training_pipeline import (
    EvalSpec,
    ModelSpec,
    TrainingPipeline,
    TrainingResult,
)


async def main() -> None:
    # ── 1. Set up infrastructure ────────────────────────────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_store = LocalFileArtifactStore(artifact_dir)
        registry = ModelRegistry(conn, artifact_store)
        fs = FeatureStore(conn)
        await fs.initialize()

        pipeline = TrainingPipeline(feature_store=fs, registry=registry)
        assert isinstance(pipeline, TrainingPipeline)

        # ── 2. Create synthetic classification data ─────────────────────

        n_samples = 200
        df = pl.DataFrame(
            {
                "entity_id": [f"e{i}" for i in range(n_samples)],
                "feature_a": [float(i % 10) for i in range(n_samples)],
                "feature_b": [float(i * 0.5) for i in range(n_samples)],
                "feature_c": [float(i % 7) for i in range(n_samples)],
                "target": [i % 2 for i in range(n_samples)],
            }
        )

        # ── 3. Define FeatureSchema ─────────────────────────────────────
        # Target column is NOT in the schema features list.
        # TrainingPipeline auto-detects it as the column not in features
        # or entity_id.

        schema = FeatureSchema(
            name="binary_classification",
            features=[
                FeatureField(name="feature_a", dtype="float64"),
                FeatureField(name="feature_b", dtype="float64"),
                FeatureField(name="feature_c", dtype="float64"),
            ],
            entity_id_column="entity_id",
        )

        # ── 4. Define ModelSpec — what to train ─────────────────────────

        model_spec = ModelSpec(
            model_class="sklearn.ensemble.RandomForestClassifier",
            hyperparameters={"n_estimators": 20, "random_state": 42},
            framework="sklearn",
        )

        assert model_spec.framework == "sklearn"

        # ModelSpec.instantiate() creates the model instance
        model_instance = model_spec.instantiate()
        assert hasattr(model_instance, "fit")

        # ── 5. Define EvalSpec — how to evaluate ────────────────────────

        eval_spec = EvalSpec(
            metrics=["accuracy", "f1"],
            split_strategy="holdout",
            test_size=0.2,
        )

        assert eval_spec.split_strategy == "holdout"
        assert eval_spec.test_size == 0.2

        # ── 6. Train the model ──────────────────────────────────────────

        result = await pipeline.train(
            data=df,
            schema=schema,
            model_spec=model_spec,
            eval_spec=eval_spec,
            experiment_name="tutorial_experiment",
        )

        assert isinstance(result, TrainingResult)
        assert "accuracy" in result.metrics
        assert "f1" in result.metrics
        assert result.training_time_seconds > 0
        assert result.data_shape == (200, 5)
        assert result.threshold_met is True  # No thresholds set
        assert result.registered is True  # Model was registered

        # Model version was created
        assert result.model_version is not None
        assert result.model_version.stage == "staging"
        assert result.model_version.name == "tutorial_experiment"

        # ── 7. Train with minimum thresholds ────────────────────────────

        strict_eval = EvalSpec(
            metrics=["accuracy"],
            split_strategy="holdout",
            test_size=0.2,
            min_threshold={"accuracy": 0.99},  # Very high threshold
        )

        strict_result = await pipeline.train(
            data=df,
            schema=schema,
            model_spec=model_spec,
            eval_spec=strict_eval,
            experiment_name="strict_experiment",
        )

        # If threshold not met, model is not registered
        if not strict_result.threshold_met:
            assert strict_result.registered is False
            assert strict_result.model_version is None

        # ── 8. Different split strategies ───────────────────────────────

        # K-fold split
        kfold_eval = EvalSpec(
            metrics=["accuracy"],
            split_strategy="kfold",
            n_splits=5,
        )

        kfold_result = await pipeline.train(
            data=df,
            schema=schema,
            model_spec=model_spec,
            eval_spec=kfold_eval,
            experiment_name="kfold_experiment",
        )
        assert isinstance(kfold_result, TrainingResult)

        # Walk-forward split (time-series aware — no shuffle)
        wf_eval = EvalSpec(
            metrics=["accuracy"],
            split_strategy="walk_forward",
            test_size=0.2,
        )

        wf_result = await pipeline.train(
            data=df,
            schema=schema,
            model_spec=model_spec,
            eval_spec=wf_eval,
            experiment_name="walk_forward_experiment",
        )
        assert isinstance(wf_result, TrainingResult)

        # ── 9. Evaluate a registered model on new data ──────────────────

        new_data = pl.DataFrame(
            {
                "entity_id": [f"n{i}" for i in range(50)],
                "feature_a": [float(i % 10) for i in range(50)],
                "feature_b": [float(i * 0.3) for i in range(50)],
                "feature_c": [float(i % 5) for i in range(50)],
                "target": [i % 2 for i in range(50)],
            }
        )

        eval_metrics = await pipeline.evaluate(
            model_name="tutorial_experiment",
            version=result.model_version.version,
            data=new_data,
            schema=schema,
            eval_spec=EvalSpec(metrics=["accuracy"]),
        )

        assert isinstance(eval_metrics, dict)
        assert "accuracy" in eval_metrics

        # ── 10. Retrain with new data ───────────────────────────────────

        retrain_result = await pipeline.retrain(
            model_name="tutorial_experiment",
            schema=schema,
            model_spec=model_spec,
            eval_spec=eval_spec,
            data=new_data,
        )
        assert isinstance(retrain_result, TrainingResult)

        # ── 11. Serialization round-trips ───────────────────────────────

        # ModelSpec
        ms_dict = model_spec.to_dict()
        ms_restored = ModelSpec.from_dict(ms_dict)
        assert ms_restored.model_class == model_spec.model_class
        assert ms_restored.framework == "sklearn"

        # EvalSpec
        es_dict = eval_spec.to_dict()
        es_restored = EvalSpec.from_dict(es_dict)
        assert es_restored.metrics == eval_spec.metrics
        assert es_restored.test_size == eval_spec.test_size

        # TrainingResult
        tr_dict = result.to_dict()
        tr_restored = TrainingResult.from_dict(tr_dict)
        assert tr_restored.metrics == result.metrics
        assert tr_restored.registered == result.registered

        # ── 12. Edge case: missing columns ──────────────────────────────

        bad_data = pl.DataFrame(
            {
                "entity_id": ["x"],
                "feature_a": [1.0],
                # Missing feature_b, feature_c, target
            }
        )

        try:
            await pipeline.train(
                data=bad_data,
                schema=schema,
                model_spec=model_spec,
                eval_spec=eval_spec,
                experiment_name="should_fail",
            )
            assert False, "Should raise ValueError for missing columns"
        except ValueError:
            pass  # Expected

        # ── 13. Edge case: invalid model class ──────────────────────────

        try:
            bad_spec = ModelSpec(model_class="os.system")
            bad_spec.instantiate()
            assert False, "Should reject non-allowlisted model class"
        except ValueError:
            pass  # Expected: not in ALLOWED_MODEL_PREFIXES

        # ── 14. Edge case: invalid split strategy ───────────────────────

        try:
            await pipeline.train(
                data=df,
                schema=schema,
                model_spec=model_spec,
                eval_spec=EvalSpec(split_strategy="invalid"),
                experiment_name="bad_split",
            )
            assert False, "Should raise ValueError for invalid split"
        except ValueError:
            pass  # Expected

    # ── 15. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/07_training_pipeline")


asyncio.run(main())
