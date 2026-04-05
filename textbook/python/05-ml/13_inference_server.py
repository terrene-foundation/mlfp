# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / InferenceServer
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Serve model predictions via InferenceServer.  Covers
#            single-record predict(), batch predict_batch(), cache
#            warming, and the PredictionResult type.
# LEVEL: Intermediate
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: InferenceServer, PredictionResult — predict(),
#            predict_batch(), warm_cache(), get_metrics(), get_model_info()
#
# Run: uv run python textbook/python/05-ml/13_inference_server.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
import tempfile

from sklearn.ensemble import RandomForestClassifier

from kailash.db.connection import ConnectionManager
from kailash_ml import FeatureField, FeatureSchema, MetricSpec, ModelSignature
from kailash_ml.engines.inference_server import InferenceServer, PredictionResult
from kailash_ml.engines.model_registry import LocalFileArtifactStore, ModelRegistry


async def main() -> None:
    # ── 1. Set up registry with a trained model ─────────────────────────

    conn = ConnectionManager("sqlite:///:memory:")
    await conn.initialize()

    with tempfile.TemporaryDirectory() as artifact_dir:
        artifact_store = LocalFileArtifactStore(artifact_dir)
        registry = ModelRegistry(conn, artifact_store)

        # Train and register a classifier
        model = RandomForestClassifier(n_estimators=20, random_state=42)
        model.fit(
            [
                [1.0, 2.0, 0.5],
                [3.0, 4.0, 1.5],
                [5.0, 6.0, 2.5],
                [7.0, 8.0, 3.5],
                [2.0, 1.0, 0.2],
                [4.0, 3.0, 1.2],
                [6.0, 5.0, 2.2],
                [8.0, 7.0, 3.2],
            ],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )

        signature = ModelSignature(
            input_schema=FeatureSchema(
                name="inference_input",
                features=[
                    FeatureField(name="feat_a", dtype="float64"),
                    FeatureField(name="feat_b", dtype="float64"),
                    FeatureField(name="feat_c", dtype="float64"),
                ],
                entity_id_column="entity_id",
            ),
            output_columns=["prediction"],
            output_dtypes=["float64"],
            model_type="classifier",
        )

        mv = await registry.register_model(
            "serving_model",
            pickle.dumps(model),
            metrics=[
                MetricSpec(name="accuracy", value=0.95),
                MetricSpec(name="f1", value=0.93),
            ],
            signature=signature,
        )

        # ── 2. Create InferenceServer ───────────────────────────────────

        server = InferenceServer(registry, cache_size=5)
        assert isinstance(server, InferenceServer)

        # ── 3. predict() — single-record prediction ─────────────────────

        result = await server.predict(
            "serving_model",
            features={"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
        )

        assert isinstance(result, PredictionResult)
        assert result.prediction in (0, 1)
        assert result.model_name == "serving_model"
        assert result.model_version == mv.version
        assert result.inference_time_ms >= 0
        assert result.inference_path in ("native", "onnx")

        # Probabilities for classification models
        assert result.probabilities is not None
        assert len(result.probabilities) == 2  # Binary classifier
        assert abs(sum(result.probabilities) - 1.0) < 0.01

        # ── 4. predict() — with specific version ────────────────────────

        versioned_result = await server.predict(
            "serving_model",
            features={"feat_a": 5.0, "feat_b": 6.0, "feat_c": 2.5},
            version=1,
        )

        assert versioned_result.model_version == 1

        # ── 5. predict_batch() — multiple records at once ────────────────

        records = [
            {"feat_a": 1.0, "feat_b": 2.0, "feat_c": 0.5},
            {"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
            {"feat_a": 5.0, "feat_b": 6.0, "feat_c": 2.5},
            {"feat_a": 7.0, "feat_b": 8.0, "feat_c": 3.5},
        ]

        batch_results = await server.predict_batch("serving_model", records)

        assert isinstance(batch_results, list)
        assert len(batch_results) == 4
        for r in batch_results:
            assert isinstance(r, PredictionResult)
            assert r.prediction in (0, 1)
            assert r.model_name == "serving_model"

        # ── 6. predict_batch() — empty list ──────────────────────────────

        empty_results = await server.predict_batch("serving_model", [])
        assert empty_results == []

        # ── 7. warm_cache() — pre-load models ───────────────────────────

        await server.warm_cache(["serving_model"])

        # Subsequent predictions use cache (faster)
        cached_result = await server.predict(
            "serving_model",
            features={"feat_a": 2.0, "feat_b": 3.0, "feat_c": 1.0},
        )
        assert isinstance(cached_result, PredictionResult)

        # ── 8. get_metrics() — MLToolProtocol ────────────────────────────

        metrics_info = await server.get_metrics("serving_model")
        assert isinstance(metrics_info, dict)
        assert "metrics" in metrics_info
        assert metrics_info["metrics"]["accuracy"] == 0.95
        assert metrics_info["metrics"]["f1"] == 0.93
        assert "version" in metrics_info

        # With specific version
        versioned_metrics = await server.get_metrics("serving_model", version="1")
        assert versioned_metrics["version"] == 1

        # ── 9. get_model_info() — MLToolProtocol ────────────────────────

        model_info = await server.get_model_info("serving_model")
        assert isinstance(model_info, dict)
        assert model_info["name"] == "serving_model"
        assert isinstance(model_info["versions"], list)
        assert 1 in model_info["versions"]
        assert model_info["signature"] is not None

        # ── 10. Register a second version and verify latest-serving ──────

        model_v2 = RandomForestClassifier(n_estimators=50, random_state=42)
        model_v2.fit(
            [
                [1.0, 2.0, 0.5],
                [3.0, 4.0, 1.5],
                [5.0, 6.0, 2.5],
                [7.0, 8.0, 3.5],
                [2.0, 1.0, 0.2],
                [4.0, 3.0, 1.2],
                [6.0, 5.0, 2.2],
                [8.0, 7.0, 3.2],
            ],
            [0, 1, 0, 1, 0, 1, 0, 1],
        )

        mv2 = await registry.register_model(
            "serving_model",
            pickle.dumps(model_v2),
            metrics=[MetricSpec(name="accuracy", value=0.97)],
            signature=signature,
        )
        assert mv2.version == 2

        # predict() without version uses latest
        latest_result = await server.predict(
            "serving_model",
            features={"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
        )
        assert latest_result.model_version == 2

        # Can still target v1 explicitly
        v1_result = await server.predict(
            "serving_model",
            features={"feat_a": 3.0, "feat_b": 4.0, "feat_c": 1.5},
            version=1,
        )
        assert v1_result.model_version == 1

        # ── 11. Serialization round-trip ─────────────────────────────────

        pr_dict = result.to_dict()
        pr_restored = PredictionResult.from_dict(pr_dict)
        assert pr_restored.prediction == result.prediction
        assert pr_restored.model_name == result.model_name
        assert pr_restored.model_version == result.model_version
        assert pr_restored.inference_path == result.inference_path

    # ── 12. Clean up ─────────────────────────────────────────────────────

    await conn.close()

    print("PASS: 05-ml/13_inference_server")


asyncio.run(main())
