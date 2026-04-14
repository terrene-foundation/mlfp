# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.3: Multi-Channel Serving with Nexus + RBAC/JWT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Register a handler with Nexus for API + CLI + MCP simultaneously
#   - Validate JWTs via middleware and extract an RBAC role claim
#   - Apply rate limiting as the first line of defence against abuse
#   - Visualise the middleware stack order
#   - Apply to a Singapore GovTech multi-ministry policy assistant
#
# PREREQUISITES: Exercise 8.2
# ESTIMATED TIME: ~35 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from kailash_nexus import Nexus
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.mlfp06.ex_8 import (
    CapstoneQAAgent,
    OUTPUT_DIR,
    RateLimiter,
    SimpleJWTAuth,
    handle_qa,
    run_async,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — One handler, three channels
# ════════════════════════════════════════════════════════════════════════
# Nexus turns a single handler into API + CLI + MCP simultaneously.
# Governance lives INSIDE the handler, so every channel benefits from
# the same budget/tool/clearance checks. There is no "trusted channel".


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

governance_engine = GovernanceEngine()


async def _init_engine() -> None:
    governance_engine.compile_org(write_org_yaml())


run_async(_init_engine())

base_qa = CapstoneQAAgent()

# TODO: rebuild governed_qa / governed_admin / governed_audit exactly as
#       you did in 02_governance_pipeline.py. The same envelopes apply.
governed_qa = ____
governed_admin = ____
governed_audit = ____

agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}

assert len(agents_by_role) == 3
print("\u2713 Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Register a handler with Nexus for all three channels
# ════════════════════════════════════════════════════════════════════════


async def serve_qa(question: str, role: str = "qa") -> dict:
    """Single handler Nexus exposes on API + CLI + MCP."""
    # TODO: delegate to handle_qa(question, role=role, agents_by_role=agents_by_role)
    ____


# TODO: instantiate Nexus()
app = ____

# TODO: register serve_qa on the app
____

assert app is not None
print("\u2713 Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — JWT validation and RBAC role extraction
# ════════════════════════════════════════════════════════════════════════

for token in (
    "token_viewer_001",
    "token_operator_001",
    "token_auditor_001",
    "invalid_token",
):
    # TODO: validate the token via SimpleJWTAuth.validate
    claims = ____
    status = f"role={claims['role']}" if claims else "REJECTED (401)"
    print(f"  {token[:18]:18s} -> {status}")

assert SimpleJWTAuth.validate("token_viewer_001") is not None
assert SimpleJWTAuth.validate("invalid_token") is None
print("\u2713 Checkpoint 3 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply a sliding-window rate limiter
# ════════════════════════════════════════════════════════════════════════

# TODO: build a RateLimiter with max_requests=5, window_seconds=60
limiter = ____

for i in range(7):
    allowed = limiter.allow("client_alice")
    print(f"  Request {i + 1}: {'ALLOWED' if allowed else 'RATE LIMITED (429)'}")

assert not limiter.allow("client_alice")
print("\u2713 Checkpoint 4 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: GovTech multi-ministry assistant
# ════════════════════════════════════════════════════════════════════════

middleware_stack = pl.DataFrame(
    {
        "Order": [1, 2, 3, 4, 5, 6],
        "Layer": [
            "CORS",
            "Rate Limit",
            "JWT / RBAC",
            "Request Log",
            "Handler",
            "Governance",
        ],
        "Enforces": [
            "origin whitelist",
            "per-client sliding window",
            "role claim extraction",
            "structured JSON logs",
            "serve_qa()",
            "PactGovernedAgent envelope",
        ],
    }
)
middleware_stack.write_parquet(OUTPUT_DIR / "middleware_stack.parquet")
print(middleware_stack)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Registered a single handler across API + CLI + MCP via Nexus
  [x] Validated JWTs and extracted RBAC role claims
  [x] Rate-limited per-client traffic with a sliding window
  [x] Visualised the middleware stack order

  Next: 04_drift_monitoring.py adds drift monitoring and automated
  agent tests.
"""
)
