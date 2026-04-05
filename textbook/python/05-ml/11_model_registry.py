# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / ModelRegistry
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Register versioned models with artifact=bytes, promote
#            through staging->shadow->production->archived lifecycle,
#            compare versions, and export/import MLflow format.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: ModelRegistry, ModelVersion, LocalFileArtifactStore,
#            MetricSpec, ModelSignature — register_model(),
#            promote_model(), get_model(), compare(), list_models(),
#            load_artifact(), export_mlflow(), import_mlflow()
#
# Run: uv run python textbook/python/05-ml/11_model_registry.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
import tempfile

from sklearn.ensemble import RandomForestClassifier

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema, MetricSpec, ModelSignature
from kailash_ml.engines.model_registry import (
    LocalFileArtifactStore,
    ModelNotFoundError,
    ModelRegistry,
    ModelVersion,
)


async def main() -> None:
    # ── 1. Set up ConnectionManager + ModelRegistry ─────────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_store = LocalFileArtifactStore(artifact_dir)
        registry = ModelRegistry(conn, artifact_store)
        assert isinstance(registry, ModelRegistry)

        # ── 2. Train and serialize a model ──────────────────────────────
        # register_model() takes artifact=bytes, NOT the model object.

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [0, 1, 0, 1],
        )
        artifact_bytes = pickle.dumps(model)
        assert isinstance(artifact_bytes, bytes)

        # ── 3. Define metrics and signature ─────────────────────────────

        metrics = [
            MetricSpec(name="accuracy", value=0.92),
            MetricSpec(name="f1", value=0.89),
        ]

        signature = ModelSignature(
            input_schema=FeatureSchema(
                name="churn_model_input",
                features=[
                    FeatureField(name="feature_a", dtype="float64"),
                    FeatureField(name="feature_b", dtype="float64"),
                ],
                entity_id_column="entity_id",
            ),
            output_columns=["prediction"],
            output_dtypes=["float64"],
            model_type="classifier",
        )

        # ── 4. register_model() — version 1 at STAGING ──────────────────

        mv1 = await registry.register_model(
            "churn_model",
            artifact_bytes,
            metrics=metrics,
            signature=signature,
        )

        assert isinstance(mv1, ModelVersion)
        assert mv1.name == "churn_model"
        assert mv1.version == 1
        assert mv1.stage == "staging"
        assert len(mv1.metrics) == 2
        assert mv1.metrics[0].name == "accuracy"
        assert mv1.metrics[0].value == 0.92
        assert mv1.signature is not None
        assert mv1.model_uuid != ""
        assert mv1.created_at != ""

        # ── 5. register_model() — version 2 auto-increments ─────────────

        better_model = RandomForestClassifier(n_estimators=50, random_state=42)
        better_model.fit(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [0, 1, 0, 1],
        )

        mv2 = await registry.register_model(
            "churn_model",
            pickle.dumps(better_model),
            metrics=[MetricSpec(name="accuracy", value=0.95)],
            signature=signature,
        )

        assert mv2.version == 2
        assert mv2.stage == "staging"

        # ── 6. get_model() — various lookups ─────────────────────────────

        # Latest version (default)
        latest = await registry.get_model("churn_model")
        assert latest.version == 2

        # Specific version
        v1 = await registry.get_model("churn_model", version=1)
        assert v1.version == 1

        # By stage
        staging = await registry.get_model("churn_model", stage="staging")
        assert staging.stage == "staging"

        # ── 7. list_models() ─────────────────────────────────────────────

        models = await registry.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1
        model_entry = models[0]
        assert model_entry["name"] == "churn_model"
        assert model_entry["latest_version"] == 2

        # ── 8. get_model_versions() ──────────────────────────────────────

        versions = await registry.get_model_versions("churn_model")
        assert len(versions) == 2
        # Newest first
        assert versions[0].version == 2
        assert versions[1].version == 1

        # ── 9. promote_model() — staging -> shadow ───────────────────────
        # Valid transitions:
        #   staging  -> shadow, production, archived
        #   shadow   -> production, archived, staging
        #   production -> archived, shadow
        #   archived -> staging

        promoted = await registry.promote_model(
            "churn_model",
            version=1,
            target_stage="shadow",
            reason="Shadow testing before production",
        )

        assert promoted.stage == "shadow"

        # ── 10. promote_model() — shadow -> production ───────────────────

        prod = await registry.promote_model(
            "churn_model",
            version=1,
            target_stage="production",
            reason="Shadow tests passed",
        )

        assert prod.stage == "production"

        # Retrieve by stage
        in_prod = await registry.get_model("churn_model", stage="production")
        assert in_prod.version == 1

        # ── 11. Promote v2 to production — auto-archives v1 ──────────────

        prod_v2 = await registry.promote_model(
            "churn_model",
            version=2,
            target_stage="production",
            reason="v2 outperforms v1",
        )

        assert prod_v2.stage == "production"

        # v1 is now archived (auto-demoted)
        v1_after = await registry.get_model("churn_model", version=1)
        assert v1_after.stage == "archived"

        # ── 12. compare() — compare two versions ────────────────────────

        comparison = await registry.compare("churn_model", 1, 2)
        assert isinstance(comparison, dict)
        assert comparison["name"] == "churn_model"
        assert comparison["version_a"] == 1
        assert comparison["version_b"] == 2
        assert "metrics_a" in comparison
        assert "metrics_b" in comparison
        assert "deltas" in comparison
        assert "better_version" in comparison

        # ── 13. load_artifact() — retrieve model bytes ───────────────────

        loaded_bytes = await registry.load_artifact("churn_model", 1)
        assert isinstance(loaded_bytes, bytes)
        loaded_model = pickle.loads(loaded_bytes)
        assert hasattr(loaded_model, "predict")

        # ── 14. MLflow export/import ─────────────────────────────────────

        with tempfile.TemporaryDirectory() as export_dir:
            export_path = await registry.export_mlflow("churn_model", 1, export_dir)
            assert export_path.exists()
            assert (export_path / "MLmodel").exists()
            assert (export_path / "model.pkl").exists()

            # Import back
            imported_mv = await registry.import_mlflow(export_path)
            assert isinstance(imported_mv, ModelVersion)
            assert imported_mv.stage == "staging"

        # ── 15. Serialization round-trips ────────────────────────────────

        # ModelVersion
        mv_dict = mv1.to_dict()
        mv_restored = ModelVersion.from_dict(mv_dict)
        assert mv_restored.name == mv1.name
        assert mv_restored.version == mv1.version
        assert mv_restored.stage == mv1.stage
        assert len(mv_restored.metrics) == len(mv1.metrics)

        # MetricSpec
        ms_dict = mv1.metrics[0].to_dict()
        ms_restored = MetricSpec.from_dict(ms_dict)
        assert ms_restored.name == "accuracy"
        assert ms_restored.value == 0.92

        # ModelSignature
        sig_dict = signature.to_dict()
        sig_restored = ModelSignature.from_dict(sig_dict)
        assert sig_restored.model_type == "classifier"
        assert len(sig_restored.input_schema.features) == 2

        # ── 16. Edge case: invalid stage transition ──────────────────────

        try:
            await registry.promote_model(
                "churn_model",
                version=2,
                target_stage="staging",  # production -> staging is invalid
            )
            assert False, "Should reject invalid transition"
        except ValueError as e:
            assert "invalid transition" in str(e).lower()

        # ── 17. Edge case: model not found ───────────────────────────────

        try:
            await registry.get_model("nonexistent_model")
            assert False, "Should raise ModelNotFoundError"
        except ModelNotFoundError:
            pass  # Expected

        try:
            await registry.get_model("churn_model", version=999)
            assert False, "Should raise ModelNotFoundError"
        except ModelNotFoundError:
            pass  # Expected

        # ── 18. Edge case: invalid stage name ────────────────────────────

        try:
            await registry.promote_model(
                "churn_model", version=2, target_stage="invalid_stage"
            )
            assert False, "Should raise ValueError"
        except ValueError:
            pass  # Expected

        # ── 19. Edge case: MetricSpec rejects NaN/Inf ────────────────────

        try:
            MetricSpec(name="bad", value=float("nan"))
            assert False, "Should reject NaN"
        except ValueError:
            pass  # Expected

        try:
            MetricSpec(name="bad", value=float("inf"))
            assert False, "Should reject Inf"
        except ValueError:
            pass  # Expected

    # ── 20. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/11_model_registry")


asyncio.run(main())
