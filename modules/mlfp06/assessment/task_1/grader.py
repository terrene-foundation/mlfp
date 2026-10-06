#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP06 Assessment Task 1 — Operating Envelopes and Deny-Paths.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

Ground truth the student cannot influence: the grader probes the RETURNED
engine itself, with grader-drawn costs (multipliers fresh per run) on actions
chosen from the spec — inside, outside, and over budget — plus the installed
fail-open defaults (unknown address, envelope-less role). The tightening
checks feed grader-built ConstraintEnvelopeConfig pairs to the student's
validate_child(). A submission returning canned verdict dicts fails: nothing
student-reported is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main, quiet  # noqa: E402

WEIGHT = 30
GATES = ("returns_engine",)

# The envelope table from problem.md (the grader's own copy).
SPEC = {
    "data_analyst": ("D1-R1-T1-R1", 20.0, ["read_data", "summarise_data", "generate_report"]),
    "model_trainer": ("D1-R1-T2-R1", 100.0, ["train_model", "evaluate_model", "read_data"]),
    "risk_assessor": ("D2-R1-T1-R1", 200.0, ["read_data", "audit_model", "generate_report", "access_audit_log"]),
    "customer_agent": ("D3-R1-T1-R1", 5.0, ["answer_question", "search_faq"]),
}
OUTSIDE_ACTION = {
    "data_analyst": "deploy_model",
    "model_trainer": "deploy_model",
    "risk_assessor": "delete_all_records",
    "customer_agent": "read_data",
}
UNENVELOPED = [("D1-R1-T3-R1", "deploy_model"), ("D2-R1-T2-R1", "run_fairness_check")]


def _config(eid, clearance, cap, actions):
    from pact import (
        CommunicationConstraintConfig,
        ConstraintEnvelopeConfig,
        DataAccessConstraintConfig,
        FinancialConstraintConfig,
        OperationalConstraintConfig,
        TemporalConstraintConfig,
    )

    return ConstraintEnvelopeConfig(
        id=eid,
        description=eid,
        confidentiality_clearance=clearance,
        financial=FinancialConstraintConfig(max_spend_usd=cap),
        operational=OperationalConstraintConfig(
            allowed_actions=list(actions), blocked_actions=[]
        ),
        temporal=TemporalConstraintConfig(blackout_periods=[]),
        data_access=DataAccessConstraintConfig(
            read_paths=["/*"], write_paths=[], blocked_data_types=[]
        ),
        communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
        max_delegation_depth=3,
    )


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m6_task1")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}", GATES)
    if not callable(getattr(st, "solve", None)) or not callable(getattr(st, "validate_child", None)):
        return finalize(checks, WEIGHT, seed, "Module must define solve() and validate_child()", GATES)
    try:
        with quiet():
            r = st.solve()
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"solve() raised {type(e).__name__}: {e}", GATES)

    rng = np.random.default_rng(seed)
    engine = r.get("engine") if isinstance(r, dict) else None
    try:
        probe = engine.verify_action(
            role_address="D1-R1-T1-R1", action="read_data", context={"cost": 0.01}
        )
        gate_ok = hasattr(probe, "allowed") and hasattr(probe, "level")
    except Exception as e:
        gate_ok = False
        checks.add("returns_engine", False, f"engine.verify_action raised {type(e).__name__}: {e}")
    if gate_ok:
        checks.add("returns_engine", True)
    if not checks.results.get("returns_engine"):
        return finalize(checks, WEIGHT, seed, None, GATES)

    def verdict(addr, action, cost):
        v = engine.verify_action(role_address=addr, action=action, context={"cost": cost})
        return bool(v.allowed), str(v.level)

    def runtime_probes():
        out: dict[str, tuple[bool, str]] = {}
        # allow paths: each role's first action, cost well under the cap
        bad = []
        for role, (addr, cap, actions) in SPEC.items():
            ok, lvl = verdict(addr, actions[0], round(cap * float(rng.uniform(0.01, 0.2)), 2))
            if not ok:
                bad.append(f"{role}:{actions[0]} -> {lvl}")
        out["allow_paths"] = (not bad, f"within-envelope probes blocked: {bad}")

        # deny by action
        bad = []
        for role, (addr, cap, _actions) in SPEC.items():
            ok, lvl = verdict(addr, OUTSIDE_ACTION[role], round(cap * 0.05, 2))
            if ok:
                bad.append(f"{role}:{OUTSIDE_ACTION[role]} allowed ({lvl})")
        out["deny_by_action"] = (not bad, f"outside-envelope actions allowed: {bad}")

        # deny by budget (grader-drawn multipliers), allow at boundary
        bad_d, bad_a = [], []
        for role, (addr, cap, actions) in SPEC.items():
            over = round(cap * float(rng.uniform(1.5, 3.0)), 2)
            under = round(cap * float(rng.uniform(0.3, 0.8)), 2)
            ok_over, lvl = verdict(addr, actions[0], over)
            if ok_over:
                bad_d.append(f"{role}:{actions[0]} ${over} (cap {cap}) allowed ({lvl})")
            ok_under, lvl = verdict(addr, actions[0], under)
            if not ok_under:
                bad_a.append(f"{role}:{actions[0]} ${under} (cap {cap}) -> {lvl}")
        out["deny_by_budget"] = (not bad_d, f"over-budget probes allowed: {bad_d}")
        out["boundary_budget_allowed"] = (not bad_a, f"under-budget probes blocked: {bad_a}")

        # installed fail-open defaults
        ok, lvl = verdict("D99-R99-T99-R99", "read_data", 0.0)
        out["failopen_unknown_role"] = (
            ok and lvl == "auto_approved",
            f"D99-R99-T99-R99 -> allowed={ok} level={lvl!r}; the installed default is auto_approved",
        )
        bad = []
        for addr, action in UNENVELOPED:
            ok, lvl = verdict(addr, action, 1.0)
            if not (ok and lvl == "auto_approved"):
                bad.append(f"{addr}:{action} -> allowed={ok} level={lvl!r}")
        out["failopen_unenveloped_role"] = (
            not bad,
            f"envelope-less roles should auto-approve (envelopes attached wrongly?): {bad}",
        )
        return out

    checks.guarded(
        [
            "allow_paths",
            "deny_by_action",
            "deny_by_budget",
            "boundary_budget_allowed",
            "failopen_unknown_role",
            "failopen_unenveloped_role",
        ],
        runtime_probes,
    )

    def tightening():
        from pact import ConfidentialityLevel as C

        parent = _config("parent", C.CONFIDENTIAL, 50.0, ["read_data", "write_data"])
        cases = {
            "tightening_legal": _config("legal", C.CONFIDENTIAL, 25.0, ["read_data"]),
            "tightening_clearance_escalation": _config("esc", C.SECRET, 50.0, ["read_data", "write_data"]),
            "tightening_budget_widening": _config("bud", C.CONFIDENTIAL, 100.0, ["read_data", "write_data"]),
            "tightening_action_widening": _config("act", C.CONFIDENTIAL, 50.0, ["read_data", "write_data", "deploy_model"]),
            "tightening_restricted_child_legal": _config("rst", C.RESTRICTED, 50.0, ["read_data", "write_data"]),
        }
        expected = {
            "tightening_legal": True,
            "tightening_clearance_escalation": False,
            "tightening_budget_widening": False,
            "tightening_action_widening": False,
            "tightening_restricted_child_legal": True,
        }
        out: dict[str, tuple[bool, str]] = {}
        for name, child in cases.items():
            try:
                got = bool(st.validate_child(parent, child))
            except Exception as e:
                got = None
                out[name] = (False, f"validate_child raised {type(e).__name__}: {e}")
                continue
            want = expected[name]
            out[name] = (
                got is want,
                f"validate_child(parent=confidential/$50/[read,write], {name}) returned {got}; expected {want}",
            )
        return out

    checks.guarded(
        [
            "tightening_legal",
            "tightening_clearance_escalation",
            "tightening_budget_widening",
            "tightening_action_widening",
            "tightening_restricted_child_legal",
        ],
        tightening,
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
