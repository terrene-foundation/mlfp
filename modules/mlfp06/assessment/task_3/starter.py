# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 3: Serve a Governed Endpoint

Implement `solve()`. problem.md holds the tier table, the cost model, the
response contract, and the acceptance criteria. The grader drives your app
in-process with tokens it mints itself — a canned handler fails.

    python starter.py               # (you) build + smoke-test in-process
    python grader.py starter.py     # (instructor) grade an attempt

No LLM is involved in this task.
"""
# NOTE: no `from __future__ import annotations` here — Nexus binds the
# handler's `request: Request` parameter from its runtime annotation, and
# deferred (string) annotations would defeat the extractor.

ALLOWED_ORIGIN = "https://intranet.example.sg"
ENDPOINT = "/workflows/serve_qa/execute"

# The governance tiers from problem.md: role -> (role address, cap usd).
TIERS = {
    "qa": ("D1-R1-T1-R1", 0.015),
    "admin": ("D1-R1-T2-R1", 0.05),
}
LONG_QUESTION_CHARS = 280
COST_SHORT = 0.01
COST_LONG = 0.02


def solve() -> dict:
    """Build the Nexus app + JWT issuer per problem.md.

    Returns:
        {"app": Nexus, "issuer": JWTValidator, "endpoint": ENDPOINT}
    """
    raise NotImplementedError("Implement solve() — see problem.md")


if __name__ == "__main__":
    import asyncio

    import httpx

    out = solve()
    app, issuer = out["app"], out["issuer"]
    qa_token = issuer.create_access_token("officer.amy", roles=["qa"])

    async def demo():
        transport = httpx.ASGITransport(app=app.fastapi_app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            ok = await client.post(
                ENDPOINT,
                json={"inputs": {"question": "What is the leave policy?"}},
                headers={"Authorization": f"Bearer {qa_token}"},
            )
            no_tok = await client.post(
                ENDPOINT, json={"inputs": {"question": "hi"}}
            )
        return ok, no_tok

    ok, no_tok = asyncio.run(demo())
    print("qa short:", ok.status_code, ok.json()["outputs"]["handler"])
    print("no token:", no_tok.status_code)
