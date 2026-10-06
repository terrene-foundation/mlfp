# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 3: Serve a Governed Endpoint (Reference Solution)

Withheld from students. Verified to pass grader.py across seeds. No LLM calls.

The stack: Nexus + NexusAuthPlugin (HS256 JWT + 6rpm/burst-2 rate limit) +
a one-item CORS allow-list. The handler reads the VERIFIED role from
request.state.user (never the body), prices the question ($0.02 over 280
chars, else $0.01), and runs PACT verify_action for the tier's envelope
before serving. Refusals are HTTP 200 with blocked=True; auth failures are
401 from the middleware, before the handler runs.
"""
# NOTE: no `from __future__ import annotations` here — Nexus binds the
# handler's `request: Request` parameter from its runtime annotation, and
# deferred (string) annotations would defeat the extractor.

import os
import secrets

from kailash.trust.auth.jwt import JWTConfig, JWTValidator
from kailash.trust.rate_limit.config import RateLimitConfig
from nexus import Nexus, NexusAuthPlugin
from starlette.requests import Request

from shared.mlfp06.ex_7 import compile_governance

ALLOWED_ORIGIN = "https://intranet.example.sg"
ENDPOINT = "/workflows/serve_qa/execute"

TIERS = {
    "qa": ("D1-R1-T1-R1", 0.015),
    "admin": ("D1-R1-T2-R1", 0.05),
}
LONG_QUESTION_CHARS = 280
COST_SHORT = 0.01
COST_LONG = 0.02

_HEAD_BY_ADDRESS = {"D1-R1-T1-R1": "D1-R1", "D1-R1-T2-R1": "D1-R1"}


def _attach_tier_envelopes(engine) -> None:
    from pact import (
        CommunicationConstraintConfig,
        ConfidentialityLevel,
        ConstraintEnvelopeConfig,
        DataAccessConstraintConfig,
        FinancialConstraintConfig,
        OperationalConstraintConfig,
        RoleEnvelope,
        TemporalConstraintConfig,
    )

    for role, (address, cap) in TIERS.items():
        config = ConstraintEnvelopeConfig(
            id=f"{role}_tier_envelope",
            description=f"{role} tier",
            confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
            financial=FinancialConstraintConfig(max_spend_usd=cap),
            operational=OperationalConstraintConfig(
                allowed_actions=["generate_answer"], blocked_actions=[]
            ),
            temporal=TemporalConstraintConfig(blackout_periods=[]),
            data_access=DataAccessConstraintConfig(
                read_paths=["/internal/*"], write_paths=[], blocked_data_types=[]
            ),
            communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
            max_delegation_depth=1,
        )
        engine.set_role_envelope(
            RoleEnvelope(
                id=f"{role}_tier_role_envelope",
                defining_role_address=_HEAD_BY_ADDRESS[address],
                target_role_address=address,
                envelope=config,
            )
        )


def solve() -> dict:
    engine, _org = compile_governance(apply_specs=False)
    _attach_tier_envelopes(engine)

    secret = os.environ.get("MLFP_JWT_SECRET") or secrets.token_urlsafe(48)
    jwt_config = JWTConfig(secret=secret, algorithm="HS256")
    issuer = JWTValidator(jwt_config)

    app = Nexus(
        api_port=8000,
        rate_limit=None,  # rate limiting lives on the auth plugin
        cors_origins=[ALLOWED_ORIGIN],
        enable_durability=False,
    )
    app.add_plugin(
        NexusAuthPlugin(
            jwt=jwt_config,
            rate_limit=RateLimitConfig(requests_per_minute=6, burst_size=2),
        )
    )

    async def serve_qa(question: str, request: Request) -> dict:
        user = getattr(request.state, "user", None)
        subject = getattr(user, "user_id", None)
        roles = list(getattr(user, "roles", None) or [])
        role = next((r for r in roles if r in TIERS), "")
        cost = COST_LONG if len(str(question)) > LONG_QUESTION_CHARS else COST_SHORT
        if not role:
            return {
                "blocked": True,
                "verdict": "blocked",
                "role": "",
                "user": subject,
                "error": f"no governed tier for roles {roles}",
            }
        address, _cap = TIERS[role]
        verdict = engine.verify_action(
            role_address=address, action="generate_answer", context={"cost": cost}
        )
        if not verdict.allowed:
            return {
                "blocked": True,
                "verdict": "blocked",
                "role": role,
                "user": subject,
                "error": str(verdict.reason),
            }
        return {
            "answer": f"[{role}] {question}",
            "role": role,
            "verdict": "served",
            "blocked": False,
            "user": subject,
            "cost": cost,
        }

    app.handler_extract("serve_qa", serve_qa, description="Governed policy QA")
    return {"app": app, "issuer": issuer, "endpoint": ENDPOINT}


if __name__ == "__main__":
    import asyncio

    import httpx

    out = solve()
    app, issuer = out["app"], out["issuer"]
    qa_token = issuer.create_access_token("officer.amy", roles=["qa"])

    async def demo():
        transport = httpx.ASGITransport(app=app.fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            ok = await c.post(
                ENDPOINT,
                json={"inputs": {"question": "What is the leave policy?"}},
                headers={"Authorization": f"Bearer {qa_token}"},
            )
            long_qa = await c.post(
                ENDPOINT,
                json={"inputs": {"question": "x" * 300}},
                headers={"Authorization": f"Bearer {qa_token}"},
            )
            no_tok = await c.post(ENDPOINT, json={"inputs": {"question": "hi"}})
        return ok, long_qa, no_tok

    ok, long_qa, no_tok = asyncio.run(demo())
    print("qa short:", ok.status_code, ok.json()["outputs"]["handler"])
    print("qa long:", long_qa.status_code, long_qa.json()["outputs"]["handler"])
    print("no token:", no_tok.status_code)
