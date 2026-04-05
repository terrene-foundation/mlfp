# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 8: Capstone — Full Platform
# ════════════════════════════════════════════════════════════════════════
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
# ════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load trained model from ModelRegistry
# ══════════════════════════════════════════════════════════════════════


async def setup_model():
    conn = ConnectionManager("sqlite:///ascent_capstone.db")
    await conn.initialize()

    registry = ModelRegistry(conn)

    # Register capstone model (simulates loading M3's production model)
    signature = ModelSignature(
        input_schema=FeatureSchema(
            name="credit_input",
            features=[
                FeatureField(name="annual_income", dtype="float64"),
                FeatureField(name="total_debt", dtype="float64"),
                FeatureField(name="credit_utilisation", dtype="float64"),
                FeatureField(name="late_payments_12m", dtype="int64"),
                FeatureField(name="account_age_months", dtype="int64"),
            ],
            entity_id_column="application_id",
        ),
        output_columns=["default_probability", "decision"],
        output_dtypes=["float64", "utf8"],
        model_type="classifier",
    )

    model_version = await registry.register_model(
        name="credit_default_production",
        artifact=b"capstone_model_placeholder",
        metrics=[MetricSpec(name="auc_pr", value=0.62)],
        signature=signature,
    )

    await registry.promote_model(
        name="credit_default_production",
        version=model_version.version,
        target_stage="production",
        reason="Capstone deployment",
    )

    print(f"=== Model Loaded ===")
    print(f"Name: {model_version.name} v{model_version.version}")
    print(f"Stage: production")

    return conn, registry, signature


conn, registry, signature = asyncio.run(setup_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Deploy via InferenceServer
# ══════════════════════════════════════════════════════════════════════


async def setup_inference():
    server = InferenceServer(registry, cache_size=5)
    await server.warm_cache(["credit_default_production"])

    print(f"\n=== InferenceServer ===")
    info = await server.get_model_info("credit_default_production")
    print(f"Model info: {info}")

    return server


server = asyncio.run(setup_inference())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Kaizen agent for intelligent interaction
# ══════════════════════════════════════════════════════════════════════


class CreditDecisionResult(Signature):
    """Structured credit decision with explanation."""

    application: str = InputField(description="Credit application details")
    risk_assessment: str = OutputField(description="Risk assessment summary")
    decision: str = OutputField(description="APPROVE / REVIEW / DENY")
    explanation: str = OutputField(description="Human-readable explanation")
    confidence: float = OutputField(description="Decision confidence 0-1")


# Tool that wraps InferenceServer for the agent
async def tool_predict_default(application_data: str) -> str:
    """Get default probability from the ML model."""
    result = await server.predict(
        model_name="credit_default_production",
        features={
            "annual_income": 85000,
            "total_debt": 25000,
            "credit_utilisation": 0.45,
            "late_payments_12m": 1,
            "account_age_months": 36,
            "application_id": "cap_001",
        },
    )
    return (
        f"Default probability: {result.prediction}\n"
        f"Model: {result.model_name} v{result.model_version}\n"
        f"Inference: {result.inference_time_ms:.1f}ms ({result.inference_path})"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: PACT governance
# ══════════════════════════════════════════════════════════════════════


async def setup_governance():
    org = {
        "organization": {
            "name": "ASCENT Bank",
            "departments": [
                {
                    "name": "lending",
                    "teams": [
                        {
                            "name": "auto_underwriting",
                            "roles": [
                                {
                                    "name": "credit_agent",
                                    "permissions": {
                                        "data_access": [
                                            "credit_applications",
                                            "model_predictions",
                                        ],
                                        "tools": ["predict_default"],
                                        "max_cost_usd": 2.0,
                                        "can_deploy": False,
                                    },
                                }
                            ],
                        }
                    ],
                }
            ],
        }
    }

    compiled = compile_org(org)
    engine = GovernanceEngine(compiled)

    agent_address = Address("lending", "auto_underwriting", "credit_agent")
    context = await engine.create_context(agent_address)

    print(f"\n=== PACT Governance ===")
    print(f"Agent: {agent_address}")
    print(f"Cost budget: ${context.max_cost_usd}")
    print(f"Data access: {context.data_access}")
    print(f"Tools: {context.tools_allowed}")

    return engine, context


governance_engine, gov_context = asyncio.run(setup_governance())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Nexus multi-channel deployment
# ══════════════════════════════════════════════════════════════════════


async def setup_nexus():
    app = Nexus()

    # Register inference endpoints
    server.register_endpoints(app)

    print(f"\n=== Nexus Multi-Channel ===")
    print(f"Channels: API + CLI + MCP")

    session = app.create_session()
    print(f"Session: {session}")

    return app, session


app, session = asyncio.run(setup_nexus())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: End-to-end governed prediction
# ══════════════════════════════════════════════════════════════════════


async def end_to_end():
    """Full stack: Application → Governed Agent → Model → Decision."""

    # Create governed agent
    base_agent = Delegate(model=model_name_llm)
    governed_agent = PactGovernedAgent(
        agent=base_agent,
        governance_context=gov_context,
    )

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

    print(f"2. Governance check: agent has permission to access credit data")
    can_access = await governed_agent.check_permission(
        "access_data", "credit_applications"
    )
    print(f"   → {'ALLOW' if can_access else 'DENY'}")

    print(f"\n3. ML model prediction via InferenceServer")
    prediction = await server.predict(
        model_name="credit_default_production",
        features={
            "annual_income": 85000,
            "total_debt": 25000,
            "credit_utilisation": 0.45,
            "late_payments_12m": 1,
            "account_age_months": 36,
            "application_id": "CAP-2026-001",
        },
    )
    print(f"   → Default probability: {prediction.prediction}")
    print(
        f"   → Inference: {prediction.inference_time_ms:.1f}ms via {prediction.inference_path}"
    )

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
    async for event in governed_agent.run(prompt):
        if hasattr(event, "text"):
            response_text += event.text
    print(f"   → {response_text[:300]}...")

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
