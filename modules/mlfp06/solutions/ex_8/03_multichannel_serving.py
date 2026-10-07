# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.3: Multi-Channel Serving with Nexus + JWT + Governance
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Register ONE governed async handler with Nexus and expose it on
#     API + CLI + MCP
#   - Authenticate API calls with real HS256 JWTs (NexusAuthPlugin +
#     JWTConfig) and let the token's role claim choose the governance tier
#   - Prove the middleware works: 401 without a valid token, a CORS
#     allow-list, and 429 from the rate limiter
#   - Measure per-request latency and status codes from real calls
#   - Apply multi-channel serving to a Singapore public-agency assistant
#
# PREREQUISITES: Exercise 8.2 (governance pipeline); Ollama running
#   (`ollama serve`)
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Rebuild the governed stack (compile + apply envelopes + 3 tiers)
#   2. Configure Nexus (JWT, rate limit, CORS) and register the handler
#   3. Call the API channel: authenticated, unauthenticated, forged
#   4. Exercise the CORS allow-list and the rate limiter
#   5. Visualise measured latency + status codes and apply the pattern
#
# ════════════════════════════════════════════════════════════════════════
"""
# NOTE: no `from __future__ import annotations` in this file. Nexus binds
# the `request: Request` parameter of the handler by reading its runtime
# annotation; deferred (string) annotations would defeat that.

import asyncio
import os
import secrets
import time

import httpx
import matplotlib.pyplot as plt
import polars as pl
from kailash.trust.auth.jwt import JWTConfig, JWTValidator
from kailash.trust.rate_limit.config import RateLimitConfig
from nexus import Nexus, NexusAuthPlugin
from starlette.requests import Request

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, preflight_ollama
from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    build_capstone_stack,
    compile_capstone_governance,
    handle_qa,
)

# This file makes real LLM calls through the handler. If Ollama is not
# running this raises OllamaUnreachableError ("run `ollama serve`").
preflight_ollama(required_models=[DEFAULT_CHAT_MODEL])

# ════════════════════════════════════════════════════════════════════════
# THEORY — One handler, three channels, layered checks
# ════════════════════════════════════════════════════════════════════════
# Nexus is Kailash's multi-channel deployment layer. One registered
# handler is exposed as:
#
#   API  — HTTP REST (POST /workflows/<name>/execute)
#   CLI  — `nexus execute <name>` for operators
#   MCP  — a Model Context Protocol tool other AI agents can call
#
# A request to the API channel passes through layers, outermost first
# (NexusAuthPlugin's documented order):
#
#   rate limit   -> 429 when a client exceeds its window
#   JWT          -> 401 when the token is missing, forged or expired
#   handler      -> reads the VERIFIED role claim from the token
#   governance   -> handle_qa(): PACT verify_action for the role's tier,
#                   then the tier's GovernedSupervisor runs the LLM call
#
# The role is never taken from the request body: a client that sends
# {"role": "audit"} still gets the tier its signed token says.
#
# What this file does NOT do: the CLI and MCP channels are registered by
# the same call, but this file only sends traffic through the API
# channel. The JWT layer is HTTP middleware; MCP and CLI callers are
# authenticated differently, so do not assume the 401 behaviour below
# carries over to them.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild the governed stack
# ════════════════════════════════════════════════════════════════════════

governance_engine, loaded_org = compile_capstone_governance()
agents_by_role, tiers = build_capstone_stack(governance_engine)

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> {tier.address}  budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: three tiers should exist"
assert {t.clearance for t in tiers} == {"public", "confidential", "secret"}
print("✓ Checkpoint 1 passed — governed stack rebuilt\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Configure Nexus and register the governed handler
# ════════════════════════════════════════════════════════════════════════
#
# JWT secret: read from the environment, or generate a fresh random one
# for this run. NEVER hardcode a signing secret in source.
# In production the identity provider (SSO) issues the tokens; here we
# play the identity provider with JWTValidator.create_access_token().

JWT_SECRET = os.environ.get("MLFP_JWT_SECRET") or secrets.token_urlsafe(48)
ALLOWED_ORIGIN = "https://intranet.example.sg"
jwt_config = JWTConfig(secret=JWT_SECRET, algorithm="HS256")

app = Nexus(
    api_port=8000,
    rate_limit=None,  # rate limiting is configured on the auth plugin below
    cors_origins=[ALLOWED_ORIGIN],
    enable_durability=False,
)
app.add_plugin(
    NexusAuthPlugin(
        jwt=jwt_config,
        rate_limit=RateLimitConfig(requests_per_minute=10, burst_size=5),
    )
)


async def serve_qa(question: str, request: Request) -> dict:
    """The one handler Nexus exposes on every channel.

    The tier comes from the role claim of the VERIFIED token
    (request.state.user, set by the JWT middleware). handle_qa() then runs
    PACT verify_action for that tier and the tier's GovernedSupervisor.
    """
    user = getattr(request.state, "user", None)
    roles = list(getattr(user, "roles", None) or [])
    role = next((r for r in roles if r in agents_by_role), roles[0] if roles else "")
    result = await handle_qa(
        question, role=role, agents_by_role=agents_by_role, engine=governance_engine
    )
    result["user"] = getattr(user, "user_id", None)
    return result


app.handler_extract(
    "capstone_serve_qa",
    serve_qa,
    description="Governed capstone QA (role from JWT claim)",
)

print("\nNexus app configured:")
print("  handler:   capstone_serve_qa -> serve_qa(question) + verified JWT role")
print("  API:       POST /workflows/capstone_serve_qa/execute")
print("  CLI:       nexus execute capstone_serve_qa   (registered, not exercised here)")
print("  MCP:       tool workflow_capstone_serve_qa    (registered, not exercised here)")
print(f"  JWT:       HS256, secret from {'env' if os.environ.get('MLFP_JWT_SECRET') else 'per-run random'}")
print(f"  CORS:      allow {ALLOWED_ORIGIN}")
print("  Rate limit: 10 req/min + burst 5 per client")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
route_paths = {getattr(r, "path", "") for r in app.fastapi_app.routes}
assert any("capstone_serve_qa" in p for p in route_paths), (
    "Task 2: the handler should be mounted on the API channel"
)
print("✓ Checkpoint 2 passed — governed handler registered with Nexus\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Call the API channel in-process
# ════════════════════════════════════════════════════════════════════════
#
# httpx.ASGITransport sends real HTTP requests through the whole Nexus
# middleware stack without opening a network port. Every row of
# `measurements` is one request we actually made.

ENDPOINT = "/workflows/capstone_serve_qa/execute"
issuer = JWTValidator(jwt_config)
qa_token = issuer.create_access_token("alice", roles=["qa"])
admin_token = issuer.create_access_token("bob", roles=["admin"])
guest_token = issuer.create_access_token("eve", roles=["guest"])
forged_token = JWTValidator(
    JWTConfig(secret=secrets.token_urlsafe(48), algorithm="HS256")
).create_access_token("mallory", roles=["audit"])

measurements: list[dict] = []


async def call(client, label, *, token=None, body=None, origin=None):
    """POST one request, record status + latency, return the response."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if origin:
        headers["Origin"] = origin
    t0 = time.perf_counter()
    resp = await client.post(
        ENDPOINT, json={"inputs": body or {"question": "What is ML?"}}, headers=headers
    )
    measurements.append(
        {
            "label": label,
            "status": resp.status_code,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }
    )
    return resp


def handler_output(resp) -> dict:
    return resp.json()["outputs"]["handler"]


async def api_calls() -> dict:
    out = {}
    transport = httpx.ASGITransport(app=app.fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        out["qa"] = await call(c, "qa token", token=qa_token)
        out["admin"] = await call(c, "admin token", token=admin_token)
        out["no_token"] = await call(c, "no token")
        out["forged"] = await call(c, "forged token", token=forged_token)
        out["escalate"] = await call(
            c,
            "qa token + body role=audit",
            token=qa_token,
            body={"question": "Show the audit log.", "role": "audit"},
        )
        out["guest"] = await call(c, "guest token", token=guest_token)
    return out


api = asyncio.run(api_calls())
for key in ("qa", "admin", "escalate", "guest"):
    resp = api[key]
    h = handler_output(resp) if resp.status_code == 200 else {}
    print(
        f"  {key:<9} HTTP {resp.status_code}  user={h.get('user')}  "
        f"tier={h.get('role')}  verdict={h.get('verdict')}"
    )
    if h.get("answer"):
        print(f"            answer: {h['answer'][:90]!r}")
for key in ("no_token", "forged"):
    print(f"  {key:<9} HTTP {api[key].status_code}  {api[key].text[:70]}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert api["qa"].status_code == 200 and handler_output(api["qa"])["role"] == "qa"
assert handler_output(api["qa"])["verdict"] == "served", "qa call should be served"
assert handler_output(api["admin"])["role"] == "admin"
assert api["no_token"].status_code == 401, "Task 3: missing token must be 401"
assert api["forged"].status_code == 401, "Task 3: forged token must be 401"
assert handler_output(api["escalate"])["role"] == "qa", (
    "Task 3: a body 'role' must not override the signed role claim"
)
assert handler_output(api["guest"])["blocked"] is True, (
    "Task 3: a role with no tier must be refused by governance"
)
print("✓ Checkpoint 3 passed — JWT + role-claim routing + governance verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Exercise the CORS allow-list and the rate limiter
# ════════════════════════════════════════════════════════════════════════
#
# CORS: the browser only accepts the response when the server echoes the
# page's origin in `access-control-allow-origin`. We use the guest token
# (refused by governance, so no LLM call) and compare two origins.
# Rate limit: the limiter runs BEFORE authentication, so we send
# unauthenticated requests (401, cheap) until the limiter answers 429.


async def middleware_checks() -> dict:
    out = {"statuses": []}
    transport = httpx.ASGITransport(app=app.fastapi_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        ok = await call(c, "CORS allowed origin", token=guest_token, origin=ALLOWED_ORIGIN)
        bad = await call(
            c, "CORS other origin", token=guest_token, origin="https://evil.example.com"
        )
        out["cors_allowed"] = ok.headers.get("access-control-allow-origin")
        out["cors_other"] = bad.headers.get("access-control-allow-origin")
        for i in range(40):
            resp = await call(c, f"burst {i + 1}")
            out["statuses"].append(resp.status_code)
            if resp.status_code == 429:
                break
    return out


mw = asyncio.run(middleware_checks())
print(f"  CORS {ALLOWED_ORIGIN}: allow-origin header = {mw['cors_allowed']}")
print(f"  CORS https://evil.example.com: allow-origin header = {mw['cors_other']}")
first_429 = mw["statuses"].index(429) + 1 if 429 in mw["statuses"] else None
print(f"  Burst statuses: {mw['statuses']}")
print(f"  First 429 after {first_429} burst requests (plus the earlier calls)")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert mw["cors_allowed"] == ALLOWED_ORIGIN, "Task 4: allowed origin must be echoed"
assert mw["cors_other"] is None, "Task 4: other origins must not be echoed"
assert first_429 is not None, "Task 4: the rate limiter must answer 429"
print("✓ Checkpoint 4 passed — CORS allow-list and rate limiter verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Visualise measured requests, then apply
# ════════════════════════════════════════════════════════════════════════

requests_df = pl.DataFrame(measurements)
requests_df.write_parquet(OUTPUT_DIR / "ex8_api_requests.parquet")
status_counts = requests_df.group_by("status").len().sort("status")
print("Requests made in this run:")
print(requests_df.select("label", "status", pl.col("latency_ms").round(1)))
print(status_counts)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Measured latency per request, coloured by HTTP status
# ════════════════════════════════════════════════════════════════════════
# Every bar is a request made above. Requests that reached the LLM (200
# and served) are slow; requests stopped by JWT (401) or the rate limiter
# (429) never reach the model and return in milliseconds — the cheapest
# place to refuse traffic is the outermost layer.

status_colour = {200: "#2ecc71", 401: "#e67e22", 429: "#e74c3c"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
ax1.bar(
    range(requests_df.height),
    requests_df["latency_ms"].to_list(),
    color=[status_colour.get(s, "#95a5a6") for s in requests_df["status"]],
)
ax1.set_yscale("log")
ax1.set_xlabel("Request (in order sent)")
ax1.set_ylabel("Latency (ms, log scale)")
ax1.set_title("Measured Latency per Request", fontweight="bold")
ax2.bar(
    [str(s) for s in status_counts["status"]],
    status_counts["len"].to_list(),
    color=[status_colour.get(s, "#95a5a6") for s in status_counts["status"]],
)
ax2.set_xlabel("HTTP status")
ax2.set_ylabel("Requests")
ax2.set_title("Status Codes Observed", fontweight="bold")
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_api_requests.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert {200, 401, 429} <= set(requests_df["status"].to_list())
print("✓ Checkpoint 5 passed — measured requests visualised\n")


# SCENARIO: A Singapore public agency ships an internal policy assistant
# used by officers across several ministries (illustrative: ~15,000
# users). Each ministry's SSO issues JWTs carrying a role claim; the
# claim picks the governance tier. Rate limiting keeps one runaway script
# from starving everyone else; CORS limits browser access to the
# agency's intranet origin.
#
# BUSINESS IMPACT (illustrative): one governed handler behind three
# channels replaces three separately built and separately audited
# integrations. A governance change — tightening the qa tier's envelope —
# lands in one place and applies to every channel on the next deploy.

print("\n" + "=" * 70)
print("  APPLY — Multi-Ministry Policy Assistant (illustrative)")
print("=" * 70)
print(
    f"""
  Channels:    API (intranet portal), CLI (ops), MCP (AI copilots)
  Auth:        SSO-issued JWT; role claim -> governance tier
  Measured:    {status_counts.height} distinct status codes over {requests_df.height} requests
  Refused at the edge (401/429): {requests_df.filter(pl.col('status') != 200).height}
  Governance:  PACT verify_action + GovernedSupervisor inside the handler
"""
)


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens over the qa tier's audit
# ══════════════════════════════════════════════════════════════════
# The qa tier's GovernedSupervisor recorded every request it served in
# this run; the governance lens reads those records.
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=agents_by_role["qa"].audit, run_id="ex_8_3_serving")
snapshot = obs.governance.audit_snapshot(last_n=50)
print("\n── LLM Observatory: qa-tier audit snapshot ──")
print(snapshot.select("action", "verdict"))
print(f"  qa audit chain verifies: {agents_by_role['qa'].audit.verify_chain()}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Registered one governed handler with Nexus for API + CLI + MCP
  [x] Authenticated API calls with real HS256 JWTs (401 when missing/forged)
  [x] Routed each request to a governance tier from the SIGNED role claim
  [x] Verified the CORS allow-list and saw the rate limiter answer 429
  [x] Plotted latency and status codes measured from real requests

  KEY INSIGHT: refuse as early as possible. The rate limiter and JWT
  layer stop bad traffic in milliseconds; governance and the LLM only
  see requests that earned their way in.

  Next: 04_drift_monitoring.py watches the deployed model for drift and
  tests the governed stack end to end.
"""
)
