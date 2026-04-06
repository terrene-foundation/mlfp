# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 8: Production Deployment with Nexus
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Deploy an ML model and an agent wrapper through Nexus for
#   multi-channel access (API + CLI + MCP). Add JWT/RBAC authentication,
#   middleware (logging, rate limiting), and DriftMonitor integration
#   for production health monitoring.
#
# TASKS:
#   1. Set up ML model via InferenceServer + ModelRegistry
#   2. Create a Nexus app and register inference endpoints
#   3. Add JWT authentication and RBAC middleware
#   4. Register an agent-wrapped endpoint for intelligent queries
#   5. Integrate DriftMonitor for production health monitoring
#   6. Test all three channels: API, CLI, MCP
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash.db.connection import ConnectionManager
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.engines.inference_server import InferenceServer
from kailash_ml.engines.drift_monitor import DriftMonitor, DriftSpec
from kailash_ml.types import (
    FeatureSchema,
    FeatureField,
    ModelSignature,
    MetricSpec,
)

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate

from kailash_nexus import Nexus
from kailash_nexus.auth import JWTAuth, RBACMiddleware, Role, Permission
from kailash_nexus.middleware import RequestLogger, RateLimiter

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model_name_llm = os.environ.get(
    "DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL")
)
jwt_secret = os.environ.get("NEXUS_JWT_SECRET", "dev-secret-change-in-production")


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")
# Use a slice as "production reference data" for drift monitoring
reference_data = credit.sample(n=min(5000, credit.height), seed=0)
production_data = credit.sample(n=min(1000, credit.height), seed=42)

print(f"=== Production Deployment Exercise ===")
print(f"Reference data: {reference_data.shape}")
print(f"Simulated production batch: {production_data.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Set up ML model (ModelRegistry + InferenceServer)
# ══════════════════════════════════════════════════════════════════════


async def setup_ml_stack():
    """Register and warm a model, return InferenceServer."""
    # TODO: Create ConnectionManager("sqlite:///ascent05_nexus_demo.db"),
    #   initialize it, then instantiate ModelRegistry(conn).
    conn = ____  # Hint: ConnectionManager("sqlite:///ascent05_nexus_demo.db")
    ____  # Hint: await conn.initialize()
    registry = ____  # Hint: ModelRegistry(conn)

    # TODO: Define a ModelSignature for credit scoring.
    #   input_schema: FeatureSchema named "credit_input" with five FeatureFields:
    #     annual_income (float64), total_debt (float64), credit_utilisation (float64),
    #     late_payments_12m (int64), account_age_months (int64)
    #   entity_id_column="application_id"
    #   output_columns=["default_probability", "risk_tier"]
    #   output_dtypes=["float64", "utf8"], model_type="classifier"
    signature = ModelSignature(
        input_schema=FeatureSchema(
            name=____,  # Hint: "credit_input"
            features=[
                FeatureField(name=____, dtype=____),  # Hint: "annual_income", "float64"
                FeatureField(name=____, dtype=____),  # Hint: "total_debt", "float64"
                FeatureField(
                    name=____, dtype=____
                ),  # Hint: "credit_utilisation", "float64"
                FeatureField(
                    name=____, dtype=____
                ),  # Hint: "late_payments_12m", "int64"
                FeatureField(
                    name=____, dtype=____
                ),  # Hint: "account_age_months", "int64"
            ],
            entity_id_column=____,  # Hint: "application_id"
        ),
        output_columns=____,  # Hint: ["default_probability", "risk_tier"]
        output_dtypes=____,  # Hint: ["float64", "utf8"]
        model_type=____,  # Hint: "classifier"
    )

    # TODO: Register the model "credit_default_v2" with artifact placeholder,
    #   metrics=[MetricSpec(name="auc_pr", value=0.62), MetricSpec(name="auc_roc", value=0.89)],
    #   signature=signature, tags={"framework": "lightgbm", "dataset": "sg_credit"}
    model_version = await registry.register_model(
        name=____,  # Hint: "credit_default_v2"
        artifact=____,  # Hint: b"model_weights_placeholder"
        metrics=____,  # Hint: [MetricSpec(name="auc_pr", value=0.62), MetricSpec(name="auc_roc", value=0.89)]
        signature=____,  # Hint: signature
        tags=____,  # Hint: {"framework": "lightgbm", "dataset": "sg_credit"}
    )

    # TODO: Promote the model to "production" stage.
    await registry.promote_model(
        name=____,  # Hint: "credit_default_v2"
        version=____,  # Hint: model_version.version
        target_stage=____,  # Hint: "production"
        reason=____,  # Hint: "Exercise 8 deployment"
    )

    # TODO: Create InferenceServer(registry, cache_size=5) and warm cache.
    server = ____  # Hint: InferenceServer(registry, cache_size=5)
    ____  # Hint: await server.warm_cache(["credit_default_v2"])

    print(f"=== ML Stack ===")
    print(f"Model: credit_default_v2 v{model_version.version} (production)")
    info = await server.get_model_info("credit_default_v2")
    print(f"Model info: {info}")

    return conn, registry, server


conn, registry, inference_server = asyncio.run(setup_ml_stack())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create Nexus app and register inference endpoints
# ══════════════════════════════════════════════════════════════════════
#
# Nexus is a zero-config multi-channel deployment platform.
# app.register() attaches a workflow or server to all channels.
# Channels: REST API (FastAPI), CLI (Click), MCP (tool server).


async def build_nexus_app(server: InferenceServer) -> Nexus:
    # TODO: Instantiate Nexus with title, version, and description.
    app = Nexus(
        title=____,  # Hint: "ASCENT Credit Scoring API"
        version=____,  # Hint: "2.0.0"
        description=____,  # Hint: "Production credit default prediction — kailash-ml + Nexus"
    )

    # TODO: Register inference endpoints from the InferenceServer onto the Nexus app.
    ____  # Hint: server.register_endpoints(app)

    print(f"\n=== Nexus App ===")
    print(f"Registered model endpoints:")
    for route in app.list_routes():
        print(f"  {route.method:6} {route.path}")

    return app


# ══════════════════════════════════════════════════════════════════════
# TASK 3: JWT authentication + RBAC middleware
# ══════════════════════════════════════════════════════════════════════
#
# RBAC (Role-Based Access Control) defines what each role can do.
# JWT tokens carry the user's role in the payload.
# Middleware is applied in order; each request passes through the stack.
#
# Role hierarchy for this deployment:
#   admin        -> full access (predict, monitor, manage models)
#   analyst      -> predict + view metrics (read-only)
#   external_api -> predict only (for third-party integrations)


def configure_auth(app: Nexus) -> Nexus:
    # TODO: Define three Role objects with the following permissions:
    #   admin:        [PREDICT, READ_METRICS, MANAGE_MODELS, READ_AUDIT_LOG]
    #   analyst:      [PREDICT, READ_METRICS]
    #   external_api: [PREDICT]
    roles = [
        Role(
            name=____,  # Hint: "admin"
            permissions=____,  # Hint: [Permission.PREDICT, Permission.READ_METRICS, Permission.MANAGE_MODELS, Permission.READ_AUDIT_LOG]
        ),
        Role(
            name=____,  # Hint: "analyst"
            permissions=____,  # Hint: [Permission.PREDICT, Permission.READ_METRICS]
        ),
        Role(
            name=____,  # Hint: "external_api"
            permissions=____,  # Hint: [Permission.PREDICT]
        ),
    ]

    # TODO: Create JWTAuth with secret=jwt_secret, algorithm="HS256",
    #   token_expiry_hours=24, exempt_paths=["/health", "/docs", "/openapi.json"]
    jwt_auth = ____  # Hint: JWTAuth(secret=jwt_secret, algorithm="HS256", token_expiry_hours=24, exempt_paths=["/health", "/docs", "/openapi.json"])

    # TODO: Create RBACMiddleware with roles=roles, role_claim="role", default_deny=True.
    rbac = (
        ____  # Hint: RBACMiddleware(roles=roles, role_claim="role", default_deny=True)
    )

    # TODO: Create RequestLogger with log_level="INFO", include_body=False,
    #   include_headers=["x-request-id", "x-correlation-id"]
    logger = ____  # Hint: RequestLogger(log_level="INFO", include_body=False, include_headers=["x-request-id", "x-correlation-id"])

    # TODO: Create RateLimiter with requests_per_minute=60, burst=10, key_func="user_id".
    rate_limiter = (
        ____  # Hint: RateLimiter(requests_per_minute=60, burst=10, key_func="user_id")
    )

    # TODO: Apply middleware in order: logger -> jwt_auth -> rbac -> rate_limiter
    ____  # Hint: app.add_middleware(logger)
    ____  # Hint: app.add_middleware(jwt_auth)
    ____  # Hint: app.add_middleware(rbac)
    ____  # Hint: app.add_middleware(rate_limiter)

    print(f"\n=== Auth + Middleware Stack ===")
    print(f"JWT:          HS256, 24h expiry, exempt: /health /docs")
    print(f"RBAC:         3 roles, default_deny=True (fail-closed)")
    print(f"RequestLogger: INFO, body logging disabled (PII)")
    print(f"RateLimiter:  60 req/min per user, burst=10")
    print(f"\nRoles:")
    for role in roles:
        print(f"  {role.name}: {[p.value for p in role.permissions]}")

    return app


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Agent-wrapped endpoint for intelligent queries
# ══════════════════════════════════════════════════════════════════════
#
# Beyond raw prediction, expose a natural-language endpoint backed by
# a Delegate agent. Users send a question in plain English; the agent
# calls the InferenceServer and returns an explanation.


class CreditAdvice(Signature):
    """Agent-powered credit advice with model prediction."""

    application_details: str = InputField(description="Credit application details")
    risk_assessment: str = OutputField(description="Risk assessment")
    model_prediction_summary: str = OutputField(description="Model prediction summary")
    recommendation: str = OutputField(description="APPROVE / REVIEW / DENY")
    explanation: str = OutputField(
        description="Plain-language explanation for the applicant"
    )


async def build_agent_endpoint(app: Nexus, server: InferenceServer) -> None:
    """Register a /predict/explain endpoint backed by an agent."""

    # TODO: Create a Delegate agent with max_llm_cost_usd=1.0.
    agent = ____  # Hint: Delegate(model=model_name_llm, max_llm_cost_usd=1.0)

    async def handle_explain_request(payload: dict) -> dict:
        """Agent-powered credit explanation endpoint."""
        # Get raw ML prediction first
        features = {
            "annual_income": payload.get("annual_income", 0),
            "total_debt": payload.get("total_debt", 0),
            "credit_utilisation": payload.get("credit_utilisation", 0.0),
            "late_payments_12m": payload.get("late_payments_12m", 0),
            "account_age_months": payload.get("account_age_months", 0),
            "application_id": payload.get("application_id", "unknown"),
        }

        # TODO: Call server.predict with model_name="credit_default_v2" and features.
        prediction = await server.predict(
            model_name=____,  # Hint: "credit_default_v2"
            features=____,  # Hint: features
        )

        agent_prompt = (
            f"You are a credit underwriting advisor. A model scored this application:\n"
            f"Default probability: {prediction.prediction}\n"
            f"Risk tier: {features}\n\n"
            f"Provide: risk assessment, model prediction summary, recommendation "
            f"(APPROVE/REVIEW/DENY), and a plain-language explanation."
        )

        # TODO: Stream the agent response to build response_text.
        response_text = ""
        async for event in ____:  # Hint: agent.run(agent_prompt)
            if hasattr(event, "text"):
                response_text += event.text

        return {
            "default_probability": prediction.prediction,
            "inference_time_ms": prediction.inference_time_ms,
            "model_version": prediction.model_version,
            "agent_explanation": response_text,
        }

    # TODO: Register the handler as a Nexus endpoint.
    #   method="POST", path="/predict/explain",
    #   description="Agent-powered credit decision with plain-language explanation",
    #   required_permission=Permission.PREDICT,
    #   request_schema with properties for all five feature fields + application_id
    app.add_endpoint(
        method=____,  # Hint: "POST"
        path=____,  # Hint: "/predict/explain"
        handler=____,  # Hint: handle_explain_request
        description=____,  # Hint: "Agent-powered credit decision with plain-language explanation"
        required_permission=____,  # Hint: Permission.PREDICT
        request_schema={
            "type": "object",
            "properties": {
                "annual_income": {"type": "number"},
                "total_debt": {"type": "number"},
                "credit_utilisation": {"type": "number"},
                "late_payments_12m": {"type": "integer"},
                "account_age_months": {"type": "integer"},
                "application_id": {"type": "string"},
            },
            "required": ["annual_income", "total_debt", "credit_utilisation"],
        },
    )

    print(f"\n=== Agent Endpoint ===")
    print(f"  POST /predict/explain — agent-powered explanation")
    print(f"  -> raw prediction (InferenceServer) + agent reasoning (Delegate)")
    print(f"  -> required_permission: PREDICT")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: DriftMonitor integration
# ══════════════════════════════════════════════════════════════════════
#
# DriftMonitor watches for data distribution shifts between a reference
# dataset (training distribution) and production batches.
# PSI > 0.1 = moderate drift, > 0.2 = severe drift.
#
# Production pattern:
#   - Run DriftMonitor on each batch of predictions
#   - If drift detected: alert on-call, trigger retraining review
#   - Expose drift metrics via Nexus /monitor/drift endpoint


async def setup_drift_monitoring(app: Nexus) -> DriftMonitor:
    """Set up DriftMonitor and register a health endpoint."""

    numeric_features = [
        c
        for c in reference_data.columns
        if reference_data[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and c != "default"
    ][
        :5
    ]  # Monitor top 5 numeric features for this exercise

    # TODO: Create a DriftSpec with:
    #   feature_columns=numeric_features, psi_threshold=0.1,
    #   ks_threshold=0.05, monitor_interval_hours=6
    drift_spec = DriftSpec(
        feature_columns=____,  # Hint: numeric_features
        psi_threshold=____,  # Hint: 0.1
        ks_threshold=____,  # Hint: 0.05
        monitor_interval_hours=____,  # Hint: 6
    )

    # TODO: Instantiate DriftMonitor with reference_data and spec.
    monitor = DriftMonitor(
        reference_data=____,  # Hint: reference_data.select(numeric_features)
        spec=____,  # Hint: drift_spec
    )

    print(f"\n=== DriftMonitor ===")
    print(f"Reference: {reference_data.height:,} rows")
    print(f"Production batch: {production_data.height:,} rows")
    print(f"Monitoring features: {numeric_features}")

    # TODO: Run a drift check on the production batch using monitor.check.
    drift_report = await monitor.check(
        production_data=____,  # Hint: production_data.select(numeric_features)
    )

    print(f"\nDrift Report:")
    print(f"  Overall severity: {drift_report.overall_severity}")
    print(f"  Features drifted: {drift_report.features_drifted}")
    for feature, result in drift_report.feature_results.items():
        flag = " <- ALERT" if result.is_drifted else ""
        print(f"  {feature}: PSI={result.psi:.4f}{flag}")

    # TODO: Register a GET /monitor/drift endpoint that returns drift health.
    #   required_permission=Permission.READ_METRICS
    async def drift_health_handler(payload: dict) -> dict:
        report = await monitor.check(
            production_data=production_data.select(numeric_features),
        )
        return {
            "overall_severity": report.overall_severity,
            "features_drifted": report.features_drifted,
            "alerts": [
                {"feature": f, "psi": r.psi, "drifted": r.is_drifted}
                for f, r in report.feature_results.items()
            ],
        }

    app.add_endpoint(
        method=____,  # Hint: "GET"
        path=____,  # Hint: "/monitor/drift"
        handler=____,  # Hint: drift_health_handler
        description=____,  # Hint: "Real-time drift health check for the credit model"
        required_permission=____,  # Hint: Permission.READ_METRICS
    )

    print(f"\nDrift endpoint registered: GET /monitor/drift")

    return monitor


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Assemble and test all channels
# ══════════════════════════════════════════════════════════════════════


async def assemble_and_test():
    """Wire all components together and run channel tests."""

    # TODO: Build the Nexus app, configure auth, add agent endpoint, add drift monitor.
    app = await build_nexus_app(inference_server)
    app = ____  # Hint: configure_auth(app)
    await ____  # Hint: build_agent_endpoint(app, inference_server)
    monitor = await ____  # Hint: setup_drift_monitoring(app)

    # TODO: Create a unified session using app.create_session().
    session = ____  # Hint: app.create_session()

    print(f"\n=== Full Nexus App — Route Summary ===")
    for route in app.list_routes():
        print(f"  {route.method:6} {route.path:<35} [{route.required_permission}]")

    # --- Test: API channel ---
    print(f"\n=== Channel Test: REST API ===")
    api_client = app.get_test_client(channel="api")

    # TODO: Issue JWT tokens for admin and analyst using app.auth.issue_token.
    admin_token = app.auth.issue_token(
        user_id=____,  # Hint: "test_admin"
        role=____,  # Hint: "admin"
        expires_in_hours=____,  # Hint: 1
    )
    analyst_token = app.auth.issue_token(
        user_id=____,  # Hint: "test_analyst"
        role=____,  # Hint: "analyst"
        expires_in_hours=____,  # Hint: 1
    )

    # Prediction request as admin
    pred_response = await api_client.post(
        "/models/credit_default_v2/predict",
        json={
            "annual_income": 72000,
            "total_debt": 18000,
            "credit_utilisation": 0.38,
            "late_payments_12m": 0,
            "account_age_months": 48,
            "application_id": "TEST-001",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    print(f"  POST /predict (admin): {pred_response.status_code}")
    print(f"  Response: {pred_response.json()}")

    # Attempt a manage-models action as analyst (should be denied)
    manage_response = await api_client.post(
        "/models/credit_default_v2/promote",
        json={"target_stage": "archived"},
        headers={"Authorization": f"Bearer {analyst_token}"},
    )
    print(f"  POST /promote (analyst): {manage_response.status_code} (expected 403)")

    # Drift health check
    drift_response = await api_client.get(
        "/monitor/drift",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    print(f"  GET /monitor/drift: {drift_response.status_code}")
    drift_data = drift_response.json()
    print(f"  Drift severity: {drift_data.get('overall_severity')}")

    # --- Test: CLI channel ---
    print(f"\n=== Channel Test: CLI ===")
    cli_client = app.get_test_client(channel="cli")

    cli_result = await cli_client.run_command(
        command="predict",
        args={
            "model": "credit_default_v2",
            "annual-income": 85000,
            "total-debt": 25000,
            "credit-utilisation": 0.45,
            "late-payments-12m": 1,
            "account-age-months": 36,
            "application-id": "CLI-001",
        },
        token=admin_token,
    )
    print(f"  $ kailash predict --model credit_default_v2 ...")
    print(f"  Output: {cli_result.stdout[:200]}")

    # --- Test: MCP channel ---
    print(f"\n=== Channel Test: MCP ===")
    mcp_client = app.get_test_client(channel="mcp")

    tools = await mcp_client.list_tools()
    print(f"  MCP tools exposed: {[t.name for t in tools]}")

    mcp_result = await mcp_client.call_tool(
        "predict_credit_default_v2",
        {
            "annual_income": 60000,
            "total_debt": 30000,
            "credit_utilisation": 0.70,
            "late_payments_12m": 3,
            "account_age_months": 12,
            "application_id": "MCP-001",
        },
    )
    print(f"  MCP tool call result: {mcp_result.content[:200]}")

    return app, monitor


app, monitor = asyncio.run(assemble_and_test())


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"   PRODUCTION DEPLOYMENT ARCHITECTURE")
print(f"{'=' * 70}")
print(
    """
  REST API ------+
  CLI      ------+ Nexus --- Auth (JWT + RBAC) --- Middleware stack
  MCP      ------+    |
                      +-- POST /models/*/predict  --- InferenceServer
                      +-- POST /predict/explain   --- InferenceServer + Delegate
                      +-- GET  /monitor/drift     --- DriftMonitor

  Middleware stack (in order):
    1. RequestLogger  — structured JSON logs, no PII
    2. JWTAuth        — validates Bearer token
    3. RBACMiddleware — enforces role permissions, fail-closed
    4. RateLimiter    — 60 req/min per user, burst=10

  Channel parity: same model, same auth, same middleware, three channels.
  This is the Nexus promise: deploy once, serve everywhere.

  Governance (Module 6 preview):
    -> PACT GovernanceEngine wraps the agent at /predict/explain
    -> Each credit decision is recorded in an AuditChain
    -> DriftMonitor alerts feed into the PACT policy engine
    -> Regulated industries: every prediction is auditable
"""
)

# Clean up
asyncio.run(conn.close())

print("Exercise 8 complete — production Nexus deployment with auth + drift monitoring")
