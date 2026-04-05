# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ══════════���══════════════════════════════════��══════════════════════════
# ASCENT6 — Exercise 8: Capstone — Full Platform Deployment
# ═══════════════���═══════════════════════════════════════════���════════════
# OBJECTIVE: Deploy a governed ML system combining the full Kailash
#   platform: trained model (M3) → InferenceServer → Kaizen agent (M5)
#   → PactGovernedAgent → Nexus multi-channel → PACT governance.
#
# This is the capstone exercise demonstrating the entire Kailash platform.
#
# TASKS:
#   1. Load trained model from ModelRegistry
#   2. Deploy via InferenceServer
#   3. Wrap with Kaizen agent for intelligent interaction
#   4. Apply PACT governance (operating envelopes)
#   5. Deploy through Nexus (API + CLI + MCP)
#   6. End-to-end test: governed prediction through all layers
# ══════��═══════════════════════════════��═════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.engines.drift_monitor import DriftMonitor
from kailash_ml.types import FeatureSchema, FeatureField, ModelSignature, MetricSpec

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.react import ReActAgent

from pact import GovernanceEngine, GovernanceContext, PactGovernedAgent
from pact import Address, compile_org

from nexus import Nexus

from shared.kailash_helpers import setup_environment

setup_environment()

model_name_llm = os.environ.get(
    "DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL")
)


# ══════��══════════════════════════════��═══════════════════════���════════
# TASK 1: Load trained model from ModelRegistry
# ═══════��══════════════���═══════════════════════════════════════════════


async def setup_model():
    # TODO: Create a ConnectionManager with SQLite
    # Hint: ConnectionManager("sqlite:///ascent_capstone.db")
    conn = ____
    await conn.initialize()

    # TODO: Create a ModelRegistry from the connection
    # Hint: ModelRegistry(conn)
    registry = ____

    # TODO: Define the model signature — input schema with credit features
    # Hint: ModelSignature with FeatureSchema containing FeatureField entries
    signature = ModelSignature(
        input_schema=FeatureSchema(
            name="credit_input",
            features=[
                # TODO: Add 5 FeatureField entries for credit scoring
                # Hint: FeatureField(name="annual_income", dtype="float64"), etc.
                ____,
            ],
            entity_id_column="application_id",
        ),
        output_columns=["default_probability", "decision"],
        output_dtypes=["float64", "utf8"],
        model_type="classifier",
    )

    # TODO: Register the model with the registry
    # Hint: await registry.register_model(name=..., artifact=..., metrics=[...], signature=...)
    model_version = ____

    # TODO: Promote to production
    # Hint: await registry.promote_model(name=..., version=..., target_stage="production", reason=...)
    ____

    print(f"=== Model Loaded ===")
    print(f"Name: {model_version.name} v{model_version.version}")
    print(f"Stage: production")

    return conn, registry, signature


conn, registry, signature = asyncio.run(setup_model())


# ═══════��═══════════════════���══════════════════════════════��═══════════
# TASK 2: Deploy via InferenceServer
# ═══��══════════════════════════════════════════════��═══════════════════


async def setup_inference():
    # TODO: Create InferenceServer and warm the cache
    # Hint: InferenceServer(registry, cache_size=5)
    server = ____
    # Hint: await server.warm_cache(["credit_default_production"])
    ____

    print(f"\n=== InferenceServer ===")
    info = await server.get_model_info("credit_default_production")
    print(f"Model info: {info}")

    return server


server = asyncio.run(setup_inference())


# ═══��═════════════════════════════��═════════════════════════════���══════
# TASK 3: Kaizen agent for intelligent interaction
# ══════════════��═══════════════════════════════════════════════════════


# TODO: Define a Signature for structured credit decisions
# Hint: class CreditDecisionResult(Signature): with InputField and OutputField
class CreditDecisionResult(Signature):
    """Structured credit decision with explanation."""

    application: str = InputField(description="Credit application details")
    # TODO: Add OutputFields for risk_assessment, decision, explanation, confidence
    ____


# Tool that wraps InferenceServer for the agent
async def tool_predict_default(application_data: str) -> str:
    """Get default probability from the ML model."""
    # TODO: Call server.predict() with model name and features
    # Hint: await server.predict(model_name="credit_default_production", features={...})
    result = ____
    return (
        f"Default probability: {result.prediction}\n"
        f"Model: {result.model_name} v{result.model_version}\n"
        f"Inference: {result.inference_time_ms:.1f}ms ({result.inference_path})"
    )


# ��═══════════════════════════════════════════════��═════════════════════
# TASK 4: PACT governance
# ��═════════════════════════════════════════════���═══════════════════════


async def setup_governance():
    # TODO: Define organization structure with departments, teams, roles
    # Hint: Nested dict with "organization" → "departments" → "teams" → "roles"
    org = ____

    # TODO: Compile the org and create GovernanceEngine
    # Hint: compile_org(org) then GovernanceEngine(compiled)
    compiled = ____
    engine = ____

    # TODO: Create agent address and governance context
    # Hint: Address("lending", "auto_underwriting", "credit_agent")
    agent_address = ____
    context = await engine.create_context(agent_address)

    print(f"\n=== PACT Governance ===")
    print(f"Agent: {agent_address}")
    print(f"Cost budget: ${context.max_cost_usd}")
    print(f"Data access: {context.data_access}")
    print(f"Tools: {context.tools_allowed}")

    return engine, context


governance_engine, gov_context = asyncio.run(setup_governance())


# ═════���═════════════════════════════════════════════���══════════════════
# TASK 5: Nexus multi-channel deployment
# ═══════════════════════════════════════════���══════════════════════════


async def setup_nexus():
    # TODO: Create Nexus app and register inference endpoints
    # Hint: Nexus() then server.register_endpoints(app)
    app = ____
    ____

    print(f"\n=== Nexus Multi-Channel ===")
    print(f"Channels: API + CLI + MCP")

    # TODO: Create a session
    # Hint: app.create_session()
    session = ____
    print(f"Session: {session}")

    return app, session


app, session = asyncio.run(setup_nexus())


# ═══════════════════════════════════��══════════════════════════════════
# TASK 6: End-to-end governed prediction
# ═════════��════════════════���═══════════════════════════════════════════


async def end_to_end():
    """Full stack: Application → Governed Agent → Model → Decision."""

    # TODO: Create governed agent wrapping a Delegate
    # Hint: Delegate(model=model_name_llm) then PactGovernedAgent(agent=..., governance_context=...)
    base_agent = ____
    governed_agent = ____

    application = """
    Credit Application #CAP-2026-001:
    - Applicant: Working professional, age 35
    - Annual income: $85,000 SGD
    - Total debt: $25,000 (car loan + credit card)
    - Credit utilisation: 45%
    - Late payments (12m): 1
    - Account age: 36 months
    - Purpose: Home renovation loan, $50,000
    """

    print(f"\n{'═' * 70}")
    print(f"   CAPSTONE: END-TO-END GOVERNED CREDIT DECISION")
    print(f"{'═' * 70}")

    print(f"\n1. Application received")
    print(application)

    # TODO: Check governance permission
    # Hint: await governed_agent.check_permission("access_data", "credit_applications")
    print(f"2. Governance check: agent has permission to access credit data")
    can_access = ____
    print(f"   → {'ALLOW' if can_access else 'DENY'}")

    # TODO: Get ML prediction via InferenceServer
    # Hint: await server.predict(model_name=..., features={...})
    print(f"\n3. ML model prediction via InferenceServer")
    prediction = ____
    print(f"   → Default probability: {prediction.prediction}")
    print(
        f"   → Inference: {prediction.inference_time_ms:.1f}ms via {prediction.inference_path}"
    )

    # TODO: Run governed agent with the application and prediction
    # Hint: async for event in governed_agent.run(prompt): ...
    print(
        f"\n4. Agent reasoning (governed, cost-capped at ${gov_context.max_cost_usd})"
    )
    response_text = ""
    prompt = (
        f"You are a credit underwriting agent. Based on this application and model prediction:\n"
        f"{application}\n"
        f"Model says default probability = {prediction.prediction}\n"
        f"Provide: risk assessment, decision (APPROVE/REVIEW/DENY), explanation."
    )
    ____  # TODO: Stream agent response

    print(f"\n5. Audit trail")
    print(f"   → Application: CAP-2026-001")
    print(f"   → Model: {prediction.model_name} v{prediction.model_version}")
    print(f"   → Agent: lending/auto_underwriting/credit_agent")
    print(
        f"   → Governance: cost=${gov_context.max_cost_usd}, tools={gov_context.tools_allowed}"
    )
    print(f"   → All logged to AuditChain (tamper-evident)")

    print(f"\n{'═' * 70}")
    print(f"   PLATFORM STACK")
    print(f"{'═' * 70}")
    print(
        f"""
    Layer 6: Nexus    — Multi-channel (API + CLI + MCP)
    Layer 5: PACT     — Governance (D/T/R, envelopes, audit)
    Layer 4: Kaizen   — AI agents (Delegate, ReAct, RAG)
    Layer 3: ML       — InferenceServer, DriftMonitor, ModelRegistry
    Layer 2: DataFlow — Persistence, @db.model, db.express
    Layer 1: Core SDK — WorkflowBuilder, runtime.execute(workflow.build())

    Every layer is integrated. Every action is governed.
    This is production ML.
    """
    )


asyncio.run(end_to_end())

# Clean up
asyncio.run(conn.close())

print("✓ Exercise 8 (CAPSTONE) complete — full Kailash platform deployment")
print("  Module 6 complete: alignment, governance, RL, and governed deployment")
print("  ASCENT course complete: 48 exercises across 6 modules")
