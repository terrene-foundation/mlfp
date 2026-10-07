#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP06 Assessment Task 2 — Governed Agent Tools and Config.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

Ground truth the student cannot influence: the grader calls the registered
executors itself with fresh seeded inputs and compares against references it
computes; it reads the agent's envelope attributes directly; and it runs one
governed objective with a grader-supplied executor, checking the audit chain
grew and verifies. No LLM is contacted.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main, quiet  # noqa: E402

WEIGHT = 20
GATES = ("returns_registry",)
TOOL_NAMES = ["compute_statistics", "normalise_text", "convert_currency"]


def _exec(registry, name: str, args: dict):
    """Call a registered executor; returns (ok, parsed_json_or_error)."""
    try:
        raw = asyncio.run(registry.execute(name, args))
    except Exception as e:
        return False, f"raised {type(e).__name__}: {e}"
    try:
        return True, json.loads(raw)
    except Exception:
        return False, f"executor did not return a JSON string: {raw!r}"


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m6_task2")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}", GATES)
    if not callable(getattr(st, "build_tools", None)) or not callable(getattr(st, "build_agent", None)):
        return finalize(checks, WEIGHT, seed, "Module must define build_tools() and build_agent()", GATES)

    rng = np.random.default_rng(seed)

    try:
        with quiet():
            registry = st.build_tools()
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"build_tools() raised {type(e).__name__}: {e}", GATES)

    names = list(getattr(registry, "tool_names", []) or [])
    checks.add(
        "returns_registry",
        hasattr(registry, "execute") and hasattr(registry, "tool_names"),
        f"build_tools() returned {type(registry).__name__}; expected a Kaizen ToolRegistry",
    )
    if not checks.results["returns_registry"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    checks.add(
        "tool_names_exact",
        sorted(names) == sorted(TOOL_NAMES),
        f"registered {sorted(names)}; expected {sorted(TOOL_NAMES)}",
    )

    def statistics():
        bad = []
        for _ in range(3):
            nums = [round(float(v), 3) for v in rng.uniform(-50, 50, size=int(rng.integers(4, 9)))]
            ok, got = _exec(registry, "compute_statistics", {"numbers": nums})
            want = {
                "count": len(nums),
                "mean": round(sum(nums) / len(nums), 4),
                "min": min(nums),
                "max": max(nums),
            }
            if not ok:
                bad.append(f"executor {got}")
                continue
            for k, v in want.items():
                g = got.get(k)
                if k == "count":
                    match = g == v
                else:
                    try:
                        match = abs(float(g) - v) <= 1e-3 + 1e-3 * abs(v)
                    except (TypeError, ValueError):
                        match = False
                if not match:
                    bad.append(f"{nums}: {k} got {g} want {v}")
        return {"statistics_correct": (not bad, "; ".join(bad))}

    checks.guarded(["statistics_correct"], statistics)

    def normalise():
        bad = []
        words = ["Report", "URGENT", "claim", "Ticket", "Zone", "delta"]
        texts = []
        for _ in range(3):
            n = int(rng.integers(3, 6))
            ws = [str(w) for w in rng.choice(words, size=n)]
            sep = str(rng.choice(["  ", " \t ", "   ", " \n "]))
            texts.append("  " + sep.join(ws) + "   ")
        for text in texts:
            ok, got = _exec(registry, "normalise_text", {"text": text})
            want = " ".join(text.split()).lower()
            if not ok:
                bad.append(f"executor {got}")
            elif got.get("normalised") != want:
                bad.append(f"{text!r}: got {got.get('normalised')!r} want {want!r}")
        return {"normalise_correct": (not bad, "; ".join(bad))}

    checks.guarded(["normalise_correct"], normalise)

    def currency():
        bad = []
        for _ in range(3):
            amount = round(float(rng.uniform(1, 10_000)), 2)
            rate = round(float(rng.uniform(0.5, 1.8)), 4)
            ok, got = _exec(registry, "convert_currency", {"amount": amount, "rate": rate})
            want = round(amount * rate, 2)
            if not ok:
                bad.append(f"executor {got}")
                continue
            try:
                match = abs(float(got.get("converted")) - want) <= 0.011
            except (TypeError, ValueError):
                match = False
            if not match:
                bad.append(f"({amount}, {rate}): got {got.get('converted')} want {want}")
        return {"currency_correct": (not bad, "; ".join(bad))}

    checks.guarded(["currency_correct"], currency)

    def schemas():
        try:
            cards = registry.get_openai_tools()
        except Exception as e:
            return {"schemas_declared": (False, f"get_openai_tools() raised {type(e).__name__}: {e}")}
        want_args = {
            "compute_statistics": {"numbers"},
            "normalise_text": {"text"},
            "convert_currency": {"amount", "rate"},
        }
        bad = []
        by_name = {}
        for card in cards:
            fn = card.get("function", {}) if isinstance(card, dict) else {}
            by_name[fn.get("name")] = fn.get("parameters", {})
        for name, args in want_args.items():
            props = set((by_name.get(name) or {}).get("properties", {}).keys())
            if props != args:
                bad.append(f"{name}: properties {sorted(props)}; expected {sorted(args)}")
        return {"schemas_declared": (not bad, "; ".join(bad))}

    checks.guarded(["schemas_declared"], schemas)

    def agent():
        with quiet():
            agent = st.build_agent(registry)
        env = getattr(agent, "envelope", None)
        out: dict[str, tuple[bool, str]] = {}
        from pact import ConfidentialityLevel

        clearance = getattr(env, "confidentiality_clearance", None)
        out["agent_clearance_internal_alias"] = (
            clearance == ConfidentialityLevel.RESTRICTED,
            f"data_clearance='internal' should land on RESTRICTED; envelope has {clearance}",
        )
        fin = getattr(env, "financial", None)
        budget = getattr(fin, "max_spend_usd", None)
        out["agent_budget"] = (
            budget is not None and abs(float(budget) - 0.25) < 1e-9,
            f"envelope.financial.max_spend_usd is {budget}; expected 0.25 (set at construction, as budget_usd)",
        )
        op = getattr(env, "operational", None)
        tools = sorted(getattr(op, "allowed_actions", []) or [])
        out["agent_tools"] = (
            tools == sorted(TOOL_NAMES),
            f"envelope.operational.allowed_actions is {tools}; expected {sorted(TOOL_NAMES)}",
        )

        async def grader_executor(_spec, _inputs):
            return {"result": {"answer": "grader-ok"}, "cost": 0.0}

        before = len(agent.audit.to_list())
        try:
            with quiet():
                result = asyncio.run(agent.run("triage this ticket", execute_node=grader_executor))
            success = bool(getattr(result, "success", False))
        except Exception as e:
            success = False
            result = f"raised {type(e).__name__}: {e}"
        after = agent.audit.to_list()
        out["audit_trail"] = (
            success and len(after) > before and bool(agent.audit.verify_chain()),
            f"run success={success}, audit {before} -> {len(after)}, chain ok={agent.audit.verify_chain()} ({result if not success else 'ok'})",
        )
        return out

    checks.guarded(
        ["agent_clearance_internal_alias", "agent_budget", "agent_tools", "audit_trail"],
        agent,
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
