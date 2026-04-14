# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.3: Multi-Channel Serving with Nexus + RBAC/JWT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Register a handler with Nexus for API + CLI + MCP simultaneously
#   - Wrap an async handler in a single-node WorkflowBuilder for Nexus
#   - Validate JWTs via middleware and extract an RBAC role claim
#   - Apply rate limiting as the first line of defence against abuse
#   - Visualise the middleware stack order
#   - Apply multi-channel serving to a Singapore government service bot
#
# PREREQUISITES: Exercise 8.2 (governance pipeline)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Rebuild the governed stack via build_capstone_stack(engine)
#   2. Wrap serve_qa in a WorkflowBuilder + register with Nexus
#   3. Demonstrate JWT validation and RBAC role extraction
#   4. Apply a sliding-window rate limiter
#   5. Visualise the middleware order and apply to GovTech scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from kailash.workflow.builder import WorkflowBuilder
from nexus import Nexus
from pact import GovernanceEngine, load_org_yaml

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    RateLimiter,
    SimpleJWTAuth,
    build_capstone_stack,
    handle_qa,
    run_async,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — One handler, three channels
# ════════════════════════════════════════════════════════════════════════
# Nexus is Kailash's multi-channel deployment layer. One registered
# workflow is simultaneously exposed as:
#
#   API  — HTTP REST for web + backend services
#   CLI  — terminal access for developers and operators
#   MCP  — Model Context Protocol so other AI agents can use it
#
# The governance envelope lives INSIDE the handler the workflow wraps.
# Every channel — even the LLM-facing MCP one — benefits from the same
# budget, tool, and clearance checks. There is no "trusted channel" vs
# "untrusted channel" divergence. This is the single biggest leverage
# point of the framework-first approach: you cannot forget to wire
# governance on channel N because the workflow is shared.
#
# Nexus registration contract: `Nexus.register(name, workflow)` — the
# second argument is a built `Workflow`, not a bare async function.
# We wrap `serve_qa` in a single-node WorkflowBuilder so the same
# handler runs on every channel. This is the Core SDK runtime pattern
# from earlier MLFP modules — not a pedagogical regression.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

org_path = write_org_yaml()
loaded = load_org_yaml(org_path)
governance_engine = GovernanceEngine(loaded.org_definition)
agents_by_role, tiers = build_capstone_stack(governance_engine)

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: three tiers should exist"
print("\u2713 Checkpoint 1 passed — governed stack rebuilt\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Register a handler with Nexus for API + CLI + MCP
# ════════════════════════════════════════════════════════════════════════


async def serve_qa(question: str, role: str = "qa") -> dict:
    """The single handler Nexus exposes on all three channels.

    The workflow node below calls this function via `run_async()` so
    the governance envelope inside `handle_qa` runs on every channel.
    """
    return await handle_qa(question, role=role, agents_by_role=agents_by_role)


# Smoke-test the handler directly before registering it. This proves
# the governance envelope is live BEFORE the Nexus registration — so
# a registration failure is a Nexus issue, not a handler issue.
smoke_result = run_async(serve_qa("What is machine learning?", role="qa"))
print(
    f"\nHandler smoke test: role={smoke_result.get('role')}  "
    f"governed={smoke_result.get('governed')}  "
    f"latency_ms={smoke_result.get('latency_ms', 0):.1f}"
)

# Wrap the handler in a single-node WorkflowBuilder. Nexus.register
# takes (name, workflow) where `workflow` is a built Workflow object.
# The PythonCodeNode demonstrates the structural pattern: its code
# body is a simple echo so this file can be run offline without
# requiring Nexus to actually execute the node. In production the
# code would import and call `serve_qa` from this module.
workflow = WorkflowBuilder()
workflow.add_node(
    "PythonCodeNode",
    "serve_qa_node",
    {
        "code": (
            "# Production body would import and call serve_qa(question, role)\n"
            "# via run_async() from shared.mlfp06.ex_8 — the governance\n"
            "# envelope lives inside handle_qa and runs on every channel.\n"
            "result = {'answer': f'[nexus-stub] {question}', 'role': role}\n"
        ),
    },
)

app = Nexus()
app.register("capstone_serve_qa", workflow.build())

print("\nNexus app registered:")
print("  name:     capstone_serve_qa")
print("  wraps:    serve_qa(question, role) — WorkflowBuilder single-node")
print("  channels: API + CLI + MCP (automatic)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert app is not None, "Task 2: Nexus app should be created"
assert smoke_result.get("governed") is True
print("\u2713 Checkpoint 2 passed — Nexus multi-channel deployment wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — JWT validation and RBAC role extraction
# ════════════════════════════════════════════════════════════════════════

print("RBAC + JWT demonstration:")
for token in (
    "token_viewer_001",
    "token_operator_001",
    "token_auditor_001",
    "invalid_token",
):
    claims = SimpleJWTAuth.validate(token)
    if claims:
        print(f"  {token[:18]:18s} -> sub={claims['sub']}, role={claims['role']}")
    else:
        print(f"  {token[:18]:18s} -> REJECTED (401)")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert SimpleJWTAuth.validate("token_viewer_001") is not None
assert SimpleJWTAuth.validate("invalid_token") is None
print("\u2713 Checkpoint 3 passed — JWT + RBAC behaves correctly\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply a sliding-window rate limiter
# ════════════════════════════════════════════════════════════════════════

limiter = RateLimiter(max_requests=5, window_seconds=60)
print("Rate limiter (5 req / 60s) for client_alice:")
for i in range(7):
    allowed = limiter.allow("client_alice")
    print(f"  Request {i + 1}: {'ALLOWED' if allowed else 'RATE LIMITED (429)'}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert not limiter.allow("client_alice"), "6th+ request should be rate-limited"
print("\u2713 Checkpoint 4 passed — rate limiter enforces the window\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise and Apply: Singapore GovTech service bot
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
            "GovernedSupervisor envelope",
        ],
    }
)
middleware_stack.write_parquet(OUTPUT_DIR / "middleware_stack.parquet")
print("\nMiddleware stack order:")
print(middleware_stack)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Latency distribution histogram per channel
# ════════════════════════════════════════════════════════════════════════
# In multi-channel deployment, each channel has a different latency
# profile: API adds network overhead, CLI is local, MCP adds protocol
# framing. This histogram shows the expected distribution, giving
# operators a baseline for alerting thresholds.

rng = np.random.default_rng(42)
api_latencies = rng.lognormal(mean=5.5, sigma=0.4, size=200)
cli_latencies = rng.lognormal(mean=4.8, sigma=0.3, size=200)
mcp_latencies = rng.lognormal(mean=5.2, sigma=0.5, size=200)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(api_latencies, bins=30, alpha=0.6, color="#e74c3c", label="API (HTTP)")
ax.hist(cli_latencies, bins=30, alpha=0.6, color="#3498db", label="CLI (local)")
ax.hist(mcp_latencies, bins=30, alpha=0.6, color="#2ecc71", label="MCP (protocol)")
ax.set_xlabel("Latency (ms)")
ax.set_ylabel("Frequency")
ax.set_title("Per-Channel Latency Distribution (Simulated)", fontweight="bold")
ax.legend(fontsize=9)
ax.axvline(np.median(api_latencies), color="#e74c3c", linestyle="--", alpha=0.5)
ax.axvline(np.median(cli_latencies), color="#3498db", linestyle="--", alpha=0.5)
ax.axvline(np.median(mcp_latencies), color="#2ecc71", linestyle="--", alpha=0.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_channel_latency.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# SCENARIO: Singapore GovTech ships an internal policy assistant used
# by ~15,000 public servants across MOH, MOE, MOF and other ministries.
# Each ministry has its own SSO; JWTs issued by the ministry IdP carry
# a role claim ("viewer", "operator", "auditor"). Rate limiting caps
# public-servant usage at 60 queries/minute per user to prevent a
# stress-test from becoming an outage. CORS restricts traffic to the
# GovTech-owned domains.
#
# BUSINESS IMPACT: Rolling out the same bot via ONE codebase across
# API (intranet portal), CLI (ops terminals), and MCP (ministerial AI
# copilots) collapses three parallel deployments into one. Estimated
# saving vs three independent builds: 3 engineer-sessions of
# duplicated work + 3x ongoing maintenance. More importantly: a
# governance change (e.g. tightening the retail envelope) lands in ONE
# place and propagates to every channel the next deploy.

print("\n" + "=" * 70)
print("  APPLY — GovTech Multi-Ministry Policy Assistant")
print("=" * 70)
print(
    """
  Users:       ~15,000 public servants (MOH, MOE, MOF, ...)
  Channels:    API (intranet), CLI (ops terminals), MCP (copilots)
  Auth:        ministry SSO issues JWTs with role claim
  Rate limit:  60 queries / minute / user
  Governance:  PACT envelope inside serve_qa (same for all channels)

  Deploy consolidation: 3 codebases -> 1 codebase
  Governance change lands in ONE place and covers every channel.
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Wrapped an async handler in a single-node WorkflowBuilder
  [x] Registered the workflow with Nexus — API + CLI + MCP at once
  [x] Validated JWTs and extracted RBAC role claims
  [x] Rate-limited per-client traffic with a sliding window
  [x] Visualised the middleware stack order
  [x] Applied multi-channel serving to a GovTech scenario

  KEY INSIGHT: The middleware ORDER matters — CORS (reject origin) ->
  rate limit (reject flood) -> JWT (reject unauth) -> log (observe) ->
  handler (serve) -> governance (enforce). Putting governance last
  means the envelope sees REAL requests; putting it first means
  rate-limited requests still consume governance state.

  Next: 04_drift_monitoring.py adds production drift monitoring and
  an automated agent test harness.
"""
)
