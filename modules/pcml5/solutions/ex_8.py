# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 8: Production Deployment with Nexus
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
    conn = ConnectionManager("sqlite:///ascent05_nexus_demo.db")
    await conn.initialize()

    registry = ModelRegistry(conn)

    # Model signature defines the contract for prediction requests
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
        output_columns=["default_probability", "risk_tier"],
        output_dtypes=["float64", "utf8"],
        model_type="classifier",
    )

    model_version = await registry.register_model(
        name="credit_default_v2",
        artifact=b"model_weights_placeholder",
        metrics=[
            MetricSpec(name="auc_pr", value=0.62),
            MetricSpec(name="auc_roc", value=0.89),
        ],
        signature=signature,
        tags={"framework": "lightgbm", "dataset": "sg_credit"},
    )

    await registry.promote_model(
        name="credit_default_v2",
        version=model_version.version,
        target_stage="production",
        reason="Exercise 8 deployment",
    )

    # InferenceServer wraps the registry for low-latency serving
    # cache_size caches up to N models in memory (ONNX / raw weights)
    server = InferenceServer(registry, cache_size=5)
    await server.warm_cache(["credit_default_v2"])

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
    app = Nexus(
        title="ASCENT Credit Scoring API",
        version="2.0.0",
        description="Production credit default prediction — kailash-ml + Nexus",
    )

    # Register the InferenceServer — Nexus exposes prediction endpoints
    # automatically from the model's ModelSignature
    server.register_endpoints(app)

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
#   admin        → full access (predict, monitor, manage models)
#   analyst      → predict + view metrics (read-only)
#   external_api → predict only (for third-party integrations)


def configure_auth(app: Nexus) -> Nexus:
    # Define roles and their permissions
    roles = [
        Role(
            name="admin",
            permissions=[
                Permission.PREDICT,
                Permission.READ_METRICS,
                Permission.MANAGE_MODELS,
                Permission.READ_AUDIT_LOG,
            ],
        ),
        Role(
            name="analyst",
            permissions=[Permission.PREDICT, Permission.READ_METRICS],
        ),
        Role(
            name="external_api",
            permissions=[Permission.PREDICT],
        ),
    ]

    # JWT authenticator — validates Bearer tokens on every request
    jwt_auth = JWTAuth(
        secret=jwt_secret,
        algorithm="HS256",
        token_expiry_hours=24,
        exempt_paths=["/health", "/docs", "/openapi.json"],
    )

    # RBAC middleware — enforces role permissions on protected endpoints
    rbac = RBACMiddleware(
        roles=roles,
        role_claim="role",  # JWT payload field that carries the role
        default_deny=True,  # Fail-closed: deny if role not matched
    )

    # Request logger — structured JSON logs for every request
    logger = RequestLogger(
        log_level="INFO",
        include_body=False,  # Do not log request bodies (PII risk)
        include_headers=["x-request-id", "x-correlation-id"],
    )

    # Rate limiter — prevents runaway API consumers
    rate_limiter = RateLimiter(
        requests_per_minute=60,  # Per authenticated user
        burst=10,  # Allow short bursts above the rate
        key_func="user_id",  # Rate-limit per user, not per IP
    )

    # Apply middleware stack (order matters: logger → auth → rbac → limiter)
    app.add_middleware(logger)
    app.add_middleware(jwt_auth)
    app.add_middleware(rbac)
    app.add_middleware(rate_limiter)

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

    agent = Delegate(model=model_name_llm, max_llm_cost_usd=1.0)

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

        prediction = await server.predict(
            model_name="credit_default_v2",
            features=features,
        )

        # Pass prediction context to agent for explanation
        agent_prompt = (
            f"You are a credit underwriting advisor. A model scored this application:\n"
            f"Default probability: {prediction.prediction}\n"
            f"Risk tier: {features}\n\n"
            f"Provide: risk assessment, model prediction summary, recommendation "
            f"(APPROVE/REVIEW/DENY), and a plain-language explanation."
        )

        response_text = ""
        async for event in agent.run(agent_prompt):
            if hasattr(event, "text"):
                response_text += event.text

        return {
            "default_probability": prediction.prediction,
            "inference_time_ms": prediction.inference_time_ms,
            "model_version": prediction.model_version,
            "agent_explanation": response_text,
        }

    # Register the custom handler as a Nexus endpoint
    app.add_endpoint(
        method="POST",
        path="/predict/explain",
        handler=handle_explain_request,
        description="Agent-powered credit decision with plain-language explanation",
        required_permission=Permission.PREDICT,
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
    print(f"  → raw prediction (InferenceServer) + agent reasoning (Delegate)")
    print(f"  → required_permission: PREDICT")


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

    # Numeric feature columns to monitor
    numeric_features = [
        c
        for c in reference_data.columns
        if reference_data[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
        and c != "default"
    ][
        :5
    ]  # Monitor top 5 numeric features for this exercise

    drift_spec = DriftSpec(
        feature_columns=numeric_features,
        psi_threshold=0.1,  # PSI > 0.1 triggers moderate drift alert
        ks_threshold=0.05,  # KS p-value < 0.05 triggers alert
        monitor_interval_hours=6,  # Check every 6 hours in production
    )

    monitor = DriftMonitor(
        reference_data=reference_data.select(numeric_features),
        spec=drift_spec,
    )

    # Run drift check on simulated production batch
    print(f"\n=== DriftMonitor ===")
    print(f"Reference: {reference_data.height:,} rows")
    print(f"Production batch: {production_data.height:,} rows")
    print(f"Monitoring features: {numeric_features}")

    drift_report = await monitor.check(
        production_data=production_data.select(numeric_features),
    )

    print(f"\nDrift Report:")
    print(f"  Overall severity: {drift_report.overall_severity}")
    print(f"  Features drifted: {drift_report.features_drifted}")
    for feature, result in drift_report.feature_results.items():
        flag = " ← ALERT" if result.is_drifted else ""
        print(f"  {feature}: PSI={result.psi:.4f}{flag}")

    # Register a /monitor/drift endpoint so ops can poll health
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
        method="GET",
        path="/monitor/drift",
        handler=drift_health_handler,
        description="Real-time drift health check for the credit model",
        required_permission=Permission.READ_METRICS,
    )

    print(f"\nDrift endpoint registered: GET /monitor/drift")

    return monitor


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Assemble and test all channels
# ══════════════════════════════════════════════════════════════════════


async def assemble_and_test():
    """Wire all components together and run channel tests."""

    # Build app
    app = await build_nexus_app(inference_server)

    # Add auth + middleware
    app = configure_auth(app)

    # Add agent endpoint
    await build_agent_endpoint(app, inference_server)

    # Add drift monitoring
    monitor = await setup_drift_monitoring(app)

    # Create a session (unified state across channels)
    session = app.create_session()

    print(f"\n=== Full Nexus App — Route Summary ===")
    for route in app.list_routes():
        print(f"  {route.method:6} {route.path:<35} [{route.required_permission}]")

    # --- Test: API channel ---
    print(f"\n=== Channel Test: REST API ===")
    api_client = app.get_test_client(channel="api")

    # Issue a JWT token for testing (admin role)
    admin_token = app.auth.issue_token(
        user_id="test_admin",
        role="admin",
        expires_in_hours=1,
    )
    analyst_token = app.auth.issue_token(
        user_id="test_analyst",
        role="analyst",
        expires_in_hours=1,
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

print(f"\n{'═' * 70}")
print(f"   PRODUCTION DEPLOYMENT ARCHITECTURE")
print(f"{'═' * 70}")
print(
    """
  REST API ──────┐
  CLI      ──────┤ Nexus ─── Auth (JWT + RBAC) ─── Middleware stack
  MCP      ──────┘    │
                      ├── POST /models/*/predict  ─── InferenceServer
                      ├── POST /predict/explain   ─── InferenceServer + Delegate
                      └── GET  /monitor/drift     ─── DriftMonitor

  Middleware stack (in order):
    1. RequestLogger  — structured JSON logs, no PII
    2. JWTAuth        — validates Bearer token
    3. RBACMiddleware — enforces role permissions, fail-closed
    4. RateLimiter    — 60 req/min per user, burst=10

  Channel parity: same model, same auth, same middleware, three channels.
  This is the Nexus promise: deploy once, serve everywhere.

  Governance (Module 6 preview):
    → PACT GovernanceEngine wraps the agent at /predict/explain
    → Each credit decision is recorded in an AuditChain
    → DriftMonitor alerts feed into the PACT policy engine
    → Regulated industries: every prediction is auditable
"""
)

# Clean up
asyncio.run(conn.close())

print(
    "✓ Exercise 8 complete — production Nexus deployment with auth + drift monitoring"
)
