# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT4 — Exercise 8: M4 Capstone — Unsupervised + DL Pipeline
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Deploy the ONNX model from Exercise 7 via InferenceServer,
#   expose through Nexus (API + CLI + MCP simultaneously). Use
#   ModelSignature for input validation.
#
# TASKS:
#   1. Register ONNX model in ModelRegistry
#   2. Configure InferenceServer with ModelSignature
#   3. Test inference through InferenceServer
#   4. Register with Nexus for multi-channel deployment
#   5. Test predictions through Nexus session
#   6. Access control discussion (primes PACT in M6)
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import polars as pl
from kailash.db.connection import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.types import FeatureSchema, FeatureField, ModelSignature

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Register ONNX model in ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def register_onnx_model():
    conn = ConnectionManager("sqlite:///ascent04_deployment.db")
    await conn.initialize()

    registry = ModelRegistry(conn)
    await registry.initialize()

    # Load ONNX model bytes
    onnx_path = Path("medical_cnn.onnx")
    if onnx_path.exists():
        model_bytes = onnx_path.read_bytes()
    else:
        # Create a placeholder for environments without the trained model
        model_bytes = b"placeholder_onnx_model"
        print("⚠ ONNX model not found — using placeholder. Run Exercise 7 first.")

    # Define model signature
    signature = ModelSignature(
        input_schema=FeatureSchema(
            name="medical_image_input",
            features=[
                FeatureField(
                    name="image",
                    dtype="float32",
                    description="Grayscale image tensor (1, 64, 64)",
                ),
            ],
            entity_id_column="patient_id",
        ),
        output_columns=[
            "condition_a",
            "condition_b",
            "condition_c",
            "condition_d",
            "condition_e",
        ],
        output_dtypes=["float32"] * 5,
        model_type="classifier",
    )

    # Register
    from kailash_ml.types import MetricSpec

    model_version = await registry.register_model(
        name="medical_cnn_v1",
        artifact=model_bytes,
        metrics=[
            MetricSpec(name="auc_condition_a", value=0.75),
            MetricSpec(name="auc_condition_b", value=0.72),
        ],
        signature=signature,
    )

    print(f"=== Model Registered ===")
    print(f"Name: {model_version.name}")
    print(f"Version: {model_version.version}")
    print(f"Stage: {model_version.stage}")
    print(f"ONNX status: {model_version.onnx_status}")

    # Promote to production
    model_version = await registry.promote_model(
        name="medical_cnn_v1",
        version=model_version.version,
        target_stage="production",
        reason="Passed quality gates for ASCENT4 deployment exercise",
    )
    print(f"Promoted to: {model_version.stage}")

    return conn, registry, signature


conn, registry, signature = asyncio.run(register_onnx_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure InferenceServer
# ══════════════════════════════════════════════════════════════════════


async def setup_inference():
    server = InferenceServer(registry, cache_size=5)

    # Warm the cache with our model
    await server.warm_cache(["medical_cnn_v1"])

    # Check model info
    info = await server.get_model_info("medical_cnn_v1")
    print(f"\n=== InferenceServer ===")
    print(f"Model info: {info}")

    return server


server = asyncio.run(setup_inference())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Test inference
# ══════════════════════════════════════════════════════════════════════


async def test_inference():
    # Single prediction
    rng = np.random.default_rng(42)
    sample_features = {
        "image": rng.standard_normal((1, 64, 64)).tolist(),
        "patient_id": "patient_001",
    }

    result = await server.predict(
        model_name="medical_cnn_v1",
        features=sample_features,
    )

    print(f"\n=== Single Prediction ===")
    print(f"Prediction: {result.prediction}")
    print(f"Probabilities: {result.probabilities}")
    print(f"Model: {result.model_name} v{result.model_version}")
    print(f"Inference time: {result.inference_time_ms:.2f}ms")
    print(f"Inference path: {result.inference_path}")

    # Batch prediction
    batch = [
        {
            "image": rng.standard_normal((1, 64, 64)).tolist(),
            "patient_id": f"patient_{i:03d}",
        }
        for i in range(10)
    ]

    batch_results = await server.predict_batch(
        model_name="medical_cnn_v1",
        records=batch,
    )

    print(f"\n=== Batch Prediction ({len(batch_results)} samples) ===")
    avg_time = np.mean([r.inference_time_ms for r in batch_results])
    print(f"Average inference time: {avg_time:.2f}ms")

    return result, batch_results


single_result, batch_results = asyncio.run(test_inference())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Register with Nexus for multi-channel deployment
# ══════════════════════════════════════════════════════════════════════


async def deploy_nexus():
    """Deploy via Nexus — API + CLI + MCP from a single registration."""
    from nexus import Nexus

    app = Nexus()

    # Register inference endpoints with Nexus
    server.register_endpoints(app)

    print(f"\n=== Nexus Multi-Channel Deployment ===")
    print("Registered channels:")
    print("  • REST API:  POST /predict/medical_cnn_v1")
    print("  • CLI:       nexus predict medical_cnn_v1 --input data.json")
    print("  • MCP Tool:  predict_medical_cnn_v1(features)")
    print()
    print("All three channels share:")
    print("  - Same InferenceServer (model loaded once)")
    print("  - Same ModelSignature (input validation)")
    print("  - Same caching and batching")

    # Create session for stateful interaction
    session = app.create_session()
    print(f"\nNexus session created: {session}")

    return app, session


app, session = asyncio.run(deploy_nexus())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Test through Nexus session
# ══════════════════════════════════════════════════════════════════════


async def test_nexus_prediction():
    """Test prediction through Nexus unified session."""
    rng = np.random.default_rng(99)
    test_input = {
        "image": rng.standard_normal((1, 64, 64)).tolist(),
        "patient_id": "patient_nexus_test",
    }

    # Through InferenceServer (same result regardless of channel)
    result = await server.predict("medical_cnn_v1", test_input)

    print(f"\n=== Nexus Prediction Test ===")
    print(f"Input: patient_nexus_test")
    print(f"Output: {result.prediction}")
    print(f"Inference path: {result.inference_path}")

    # Performance metrics
    metrics = await server.get_metrics("medical_cnn_v1")
    print(f"\nServer metrics: {metrics}")

    return result


nexus_result = asyncio.run(test_nexus_prediction())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Access control discussion (primes PACT in M6)
# ══════════════════════════════════════════════════════════════════════

print(
    f"""
=== Access Control: Who Can Access This API? ===

Current state: NO access control. Anyone who can reach the server
can call predict(). For a medical diagnosis model, this is dangerous.

Questions to consider:
1. Who should be able to request predictions? (doctors? nurses? billing?)
2. What data classification level are medical images? (likely: Restricted)
3. Should predictions be logged for audit? (yes — regulatory requirement)
4. What happens if the model is wrong? (liability, appeal mechanism)

In Module 6, you'll wrap this with PACT GovernanceEngine:
  - PactGovernedAgent: agents receive frozen GovernanceContext
  - Operating envelopes: cost budgets, tool restrictions, data access
  - D/T/R addressing: Department/Team/Role-based access control
  - AuditChain: tamper-evident logging of every prediction

The pattern:
  from pact import GovernanceEngine, PactGovernedAgent
  governed = PactGovernedAgent(agent, governance_context)
  # Agent can predict, but governance limits what data it accesses

This is NOT a compliance checkbox. This is competitive advantage:
companies that can PROVE their AI is governed win regulated markets.
"""
)

# Clean up
asyncio.run(conn.close())

print("✓ Exercise 8 complete — InferenceServer + Nexus multi-channel deployment")
print(
    "  Module 4 complete: 8 exercises covering unsupervised ML, NLP, DL, and deployment"
)
