#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP06 Assessment Task 3 — Serve a Governed Endpoint.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

Ground truth the student cannot influence: the grader mints its own tokens
(per-run subjects and roles) with the returned issuer, drives the returned
app's full middleware stack in-process via httpx.ASGITransport, and reads the
handler's outputs. A forged token is minted with the grader's own secret. A
canned handler (fixed role/subject strings, body-role trust, no real JWT
verification) fails the per-run subject/role checks.

No LLM is contacted; no network port is opened.
"""
from __future__ import annotations

import asyncio
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main, quiet  # noqa: E402

WEIGHT = 30
GATES = ("returns_contract",)
ALLOWED_ORIGIN = "https://intranet.example.sg"
LONG = "x" * 300
SHORT = "What is the leave policy?"


def _handler_output(resp) -> dict:
    try:
        return resp.json()["outputs"]["handler"]
    except Exception:
        return {}


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m6_task3")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}", GATES)
    if not callable(getattr(st, "solve", None)):
        return finalize(checks, WEIGHT, seed, "Module does not define solve()", GATES)
    try:
        with quiet():
            r = st.solve()
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"solve() raised {type(e).__name__}: {e}", GATES)

    app = r.get("app") if isinstance(r, dict) else None
    issuer = r.get("issuer") if isinstance(r, dict) else None
    endpoint = r.get("endpoint") if isinstance(r, dict) else None
    try:
        probe_token = issuer.create_access_token("probe", roles=["qa"])
        gate_ok = hasattr(app, "fastapi_app") and isinstance(endpoint, str) and isinstance(probe_token, str)
    except Exception as e:
        gate_ok = False
        checks.add("returns_contract", False, f"issuer.create_access_token raised {type(e).__name__}: {e}")
    if gate_ok:
        checks.add("returns_contract", True)
    if not checks.results.get("returns_contract"):
        return finalize(checks, WEIGHT, seed, None, GATES)

    import httpx

    # Per-run identities: subjects and the unknown role are fresh every run.
    qa_subject = f"grader-qa-{seed}"
    admin_subject = f"grader-admin-{seed}"
    unknown_role = f"auditor-{seed}"
    qa_tok = issuer.create_access_token(qa_subject, roles=["qa"])
    admin_tok = issuer.create_access_token(admin_subject, roles=["admin"])
    unknown_tok = issuer.create_access_token(f"grader-x-{seed}", roles=[unknown_role])
    from kailash.trust.auth.jwt import JWTConfig, JWTValidator

    forged_tok = JWTValidator(
        JWTConfig(secret=secrets.token_urlsafe(48), algorithm="HS256")
    ).create_access_token("mallory", roles=["admin"])

    async def run_probes():
        out: dict[str, object] = {}
        transport = httpx.ASGITransport(app=app.fastapi_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            def auth(tok):
                return {"Authorization": f"Bearer {tok}"}

            out["qa_short"] = await c.post(
                endpoint, json={"inputs": {"question": SHORT}}, headers=auth(qa_tok)
            )
            out["qa_long"] = await c.post(
                endpoint, json={"inputs": {"question": LONG}}, headers=auth(qa_tok)
            )
            out["admin_long"] = await c.post(
                endpoint, json={"inputs": {"question": LONG}}, headers=auth(admin_tok)
            )
            out["escalate"] = await c.post(
                endpoint,
                json={"inputs": {"question": LONG, "role": "admin"}},
                headers=auth(qa_tok),
            )
            out["unknown"] = await c.post(
                endpoint, json={"inputs": {"question": SHORT}}, headers=auth(unknown_tok)
            )
            out["no_token"] = await c.post(endpoint, json={"inputs": {"question": SHORT}})
            out["forged"] = await c.post(
                endpoint, json={"inputs": {"question": SHORT}}, headers=auth(forged_tok)
            )
            out["cors_ok"] = await c.post(
                endpoint,
                json={"inputs": {"question": SHORT}},
                headers={**auth(qa_tok), "Origin": ALLOWED_ORIGIN},
            )
            out["cors_evil"] = await c.post(
                endpoint,
                json={"inputs": {"question": SHORT}},
                headers={**auth(qa_tok), "Origin": "https://evil.example.com"},
            )
            burst = []
            for _ in range(15):
                resp = await c.post(endpoint, json={"inputs": {"question": SHORT}})
                burst.append(resp.status_code)
                if resp.status_code == 429:
                    break
            out["burst"] = burst
        return out

    try:
        with quiet():
            api = asyncio.run(run_probes())
    except Exception as e:
        return finalize(
            checks,
            WEIGHT,
            seed,
            f"in-process probes raised {type(e).__name__}: {e}",
            GATES,
        )

    qa_short = api["qa_short"]
    qa_short_h = _handler_output(qa_short)
    checks.add(
        "qa_short_served",
        qa_short.status_code == 200
        and qa_short_h.get("role") == "qa"
        and qa_short_h.get("verdict") == "served"
        and qa_short_h.get("blocked") is False
        and qa_short_h.get("answer") == f"[qa] {SHORT}",
        f"status {qa_short.status_code}, handler {qa_short_h}",
    )

    qa_long_h = _handler_output(api["qa_long"])
    checks.add(
        "qa_long_blocked",
        api["qa_long"].status_code == 200
        and qa_long_h.get("blocked") is True
        and qa_long_h.get("role") == "qa",
        f"a 300-char question costs $0.02 > qa cap $0.015; handler returned {qa_long_h}",
    )

    admin_long_h = _handler_output(api["admin_long"])
    checks.add(
        "admin_long_served",
        api["admin_long"].status_code == 200
        and admin_long_h.get("verdict") == "served"
        and admin_long_h.get("role") == "admin",
        f"status {api['admin_long'].status_code}, handler {admin_long_h}",
    )

    esc_h = _handler_output(api["escalate"])
    checks.add(
        "body_role_ignored",
        api["escalate"].status_code == 200
        and esc_h.get("role") == "qa"
        and esc_h.get("blocked") is True,
        f"a body role=admin must not change the qa token's decision; handler returned {esc_h}",
    )

    unknown_h = _handler_output(api["unknown"])
    checks.add(
        "unknown_role_blocked",
        api["unknown"].status_code == 200 and unknown_h.get("blocked") is True,
        f"token role {unknown_role!r} should be refused; handler returned {unknown_h}",
    )

    checks.add(
        "subject_passthrough",
        qa_short_h.get("user") == qa_subject,
        f"handler user={qa_short_h.get('user')!r}; expected the verified token subject {qa_subject!r}",
    )
    checks.add(
        "no_token_401",
        api["no_token"].status_code == 401,
        f"missing token returned {api['no_token'].status_code}; expected 401",
    )
    checks.add(
        "forged_token_401",
        api["forged"].status_code == 401,
        f"forged token returned {api['forged'].status_code}; expected 401",
    )
    burst = api["burst"]
    checks.add(
        "rate_limit_429",
        429 in burst,
        f"burst statuses {burst}; the 6rpm/burst-2 limiter never answered 429",
    )
    checks.add(
        "cors_allowed_echoed",
        api["cors_ok"].headers.get("access-control-allow-origin") == ALLOWED_ORIGIN,
        f"allow-origin for {ALLOWED_ORIGIN}: {api['cors_ok'].headers.get('access-control-allow-origin')!r}",
    )
    checks.add(
        "cors_evil_not_echoed",
        api["cors_evil"].headers.get("access-control-allow-origin") is None,
        f"allow-origin for the evil origin: {api['cors_evil'].headers.get('access-control-allow-origin')!r}",
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
