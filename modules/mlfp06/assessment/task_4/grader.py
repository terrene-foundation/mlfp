#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Grader for MLFP06 Assessment Task 4 — Design the Governance Org.

    python grader.py starter.py          # grade a submission
    python grader.py solution.py         # verify the reference passes
    python grader.py solution.py --seed 123   # replay a grading run

The YAML is the artefact: the grader re-loads the student's org_yaml itself,
rebuilds the engine (GovernanceEngine + apply_governance_specs), discovers
role addresses from the rebuilt org, and probes that engine. The student's
returned engine is only a gate check — a submission cannot pre-cook verdicts.

No LLM is contacted.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grading_harness import Checks, finalize, load_student_module, main, quiet  # noqa: E402

WEIGHT = 20
GATES = ("returns_contract", "yaml_loads")

# The organisational brief from problem.md (the grader's own copy).
DEPARTMENTS = {"model_development", "operations"}
TEAMS = {"research_team", "deployment_team", "support_team"}
ROLES = {
    "chief_data_officer",
    "head_of_operations",
    "research_scientist",
    "ml_engineer",
    "support_agent",
}
REPORTS_TO = {
    "research_scientist": "chief_data_officer",
    "ml_engineer": "chief_data_officer",
    "support_agent": "head_of_operations",
}
HEADS = {"chief_data_officer": "model_development", "head_of_operations": "operations"}
CLEARANCES = {
    "chief_data_officer": "secret",
    "head_of_operations": "secret",
    "research_scientist": "confidential",
    "ml_engineer": "confidential",
    "support_agent": "public",
}
ENVELOPES = {
    "research_scientist": ("chief_data_officer", 50.0, {"read_data", "run_experiment", "train_model"}),
    "ml_engineer": ("chief_data_officer", 80.0, {"deploy_model", "monitor_model", "rollback_model", "read_data"}),
    "support_agent": ("head_of_operations", 5.0, {"answer_ticket", "search_kb"}),
}
LADDER = {"public": 0, "restricted": 1, "confidential": 2, "secret": 3, "top_secret": 4}


def _reload(org_yaml: str):
    """Grader-side rebuild: load the student's YAML and apply its specs."""
    from kailash.trust.pact.yaml_resolvers import apply_governance_specs
    from pact import GovernanceEngine, load_org_yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(org_yaml)
        path = f.name
    loaded = load_org_yaml(path)
    engine = GovernanceEngine(loaded.org_definition)
    apply_governance_specs(engine, loaded)
    return loaded, engine


def _addresses(engine) -> dict[str, str]:
    """role_id -> positional D/T/R address, discovered from the org nodes."""
    out = {}
    for addr, node in engine.get_org().nodes.items():
        if node.role_definition is not None:
            out[node.role_definition.role_id] = addr
    return out


def grade(student_path: Path, seed: int) -> dict:
    checks = Checks()
    try:
        st = load_student_module(student_path, "student_m6_task4")
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"Failed to import: {type(e).__name__}: {e}", GATES)
    if not callable(getattr(st, "solve", None)):
        return finalize(checks, WEIGHT, seed, "Module does not define solve()", GATES)
    try:
        with quiet():
            r = st.solve()
    except Exception as e:
        return finalize(checks, WEIGHT, seed, f"solve() raised {type(e).__name__}: {e}", GATES)

    org_yaml = r.get("org_yaml") if isinstance(r, dict) else None
    engine = r.get("engine") if isinstance(r, dict) else None
    engine_ok = False
    if hasattr(engine, "verify_action"):
        try:
            engine.verify_action(role_address="D1-R1", action="read_data", context={})
            engine_ok = True
        except Exception:
            engine_ok = False
    checks.add(
        "returns_contract",
        isinstance(org_yaml, str) and org_yaml.strip() != "" and engine_ok,
        "solve() must return {'org_yaml': non-empty str, 'engine': a working GovernanceEngine}",
    )
    if not checks.results["returns_contract"]:
        return finalize(checks, WEIGHT, seed, None, GATES)

    try:
        loaded, rebuilt = _reload(org_yaml)
        checks.add("yaml_loads", True)
    except Exception as e:
        checks.add("yaml_loads", False, f"grader-side reload raised {type(e).__name__}: {e}")
        return finalize(checks, WEIGHT, seed, None, GATES)

    od = loaded.org_definition

    def structure():
        got_d = {d.department_id for d in od.departments}
        got_t = {t.id for t in od.teams}
        got_r = {r.role_id for r in od.roles}
        return {
            "structure_counts": (
                len(od.departments) == 2 and len(od.teams) == 3 and len(od.roles) == 5,
                f"departments={len(od.departments)}, teams={len(od.teams)}, roles={len(od.roles)} (want 2/3/5)",
            ),
            "heads_and_reports": (
                got_d == DEPARTMENTS
                and got_t == TEAMS
                and got_r == ROLES
                and all(
                    getattr(r, "reports_to_role_id", None) == REPORTS_TO[r.role_id]
                    for r in od.roles
                    if r.role_id in REPORTS_TO
                )
                and all(
                    getattr(r, "is_primary_for_unit", None) == HEADS[r.role_id]
                    for r in od.roles
                    if r.role_id in HEADS
                ),
                "role ids, heads= targets and reports_to lines must match the brief "
                f"(got roles {sorted(got_r)})",
            ),
        }

    checks.guarded(["structure_counts", "heads_and_reports"], structure)

    def clearances():
        got = {c.role_id: str(c.level).lower() for c in loaded.clearances}
        bad = [f"{role}: {lvl!r} not in pact's ladder" for role, lvl in got.items() if lvl not in LADDER]
        if bad:
            return {"clearance_chain": (False, "; ".join(bad)),
                    "clearance_levels_exact": (False, "invalid level strings present")}
        by_role = {r.role_id: r for r in od.roles}
        chain_bad = []
        for role, head in REPORTS_TO.items():
            if LADDER[got.get(role, "public")] > LADDER[got.get(head, "public")]:
                chain_bad.append(f"{role} ({got.get(role)}) outranks {head} ({got.get(head)})")
        exact_bad = [
            f"{role}: {got.get(role)!r} != {want!r}"
            for role, want in CLEARANCES.items()
            if got.get(role) != want
        ]
        return {
            "clearance_chain": (not chain_bad, "; ".join(chain_bad)),
            "clearance_levels_exact": (not exact_bad, "; ".join(exact_bad)),
        }

    checks.guarded(["clearance_chain", "clearance_levels_exact"], clearances)

    def envelope_specs():
        bad = []
        by_target = {e.target: e for e in loaded.envelopes}
        for role, (head, cap, actions) in ENVELOPES.items():
            e = by_target.get(role)
            if e is None:
                bad.append(f"{role}: no envelope")
                continue
            if e.defined_by != head:
                bad.append(f"{role}: defined_by {e.defined_by!r} != {head!r}")
            got_cap = (e.financial or {}).get("max_spend_usd")
            if got_cap is None or abs(float(got_cap) - cap) > 1e-9:
                bad.append(f"{role}: cap {got_cap} != {cap}")
            got_actions = set((e.operational or {}).get("allowed_actions") or [])
            if got_actions != actions:
                bad.append(f"{role}: actions {sorted(got_actions)} != {sorted(actions)}")
        return {"envelope_specs": (not bad, "; ".join(bad))}

    checks.guarded(["envelope_specs"], envelope_specs)

    addr = _addresses(rebuilt)

    def probes():
        rng = np.random.default_rng(seed)

        def verdict(role_id, action, cost):
            v = rebuilt.verify_action(
                role_address=addr[role_id], action=action, context={"cost": cost}
            )
            return bool(v.allowed), str(v.level)

        out: dict[str, tuple[bool, str]] = {}
        bad = []
        for role, action, cost in (
            ("research_scientist", "run_experiment", round(50.0 * float(rng.uniform(0.05, 0.5)), 2)),
            ("ml_engineer", "deploy_model", round(80.0 * float(rng.uniform(0.05, 0.5)), 2)),
            ("support_agent", "answer_ticket", round(5.0 * float(rng.uniform(0.05, 0.5)), 2)),
        ):
            ok, lvl = verdict(role, action, cost)
            if not ok:
                bad.append(f"{role}:{action} ${cost} -> {lvl}")
        out["allow_probes"] = (not bad, f"within-envelope probes blocked: {bad}")

        bad = []
        for role, action in (("support_agent", "read_data"), ("ml_engineer", "run_experiment")):
            ok, lvl = verdict(role, action, 0.10)
            if ok:
                bad.append(f"{role}:{action} allowed ({lvl})")
        out["deny_action_probes"] = (not bad, f"outside-envelope actions allowed: {bad}")

        over = round(80.0 * float(rng.uniform(1.5, 3.0)), 2)
        ok, lvl = verdict("ml_engineer", "deploy_model", over)
        out["deny_budget_probe"] = (
            not ok,
            f"ml_engineer deploy_model ${over} (cap $80) -> allowed={ok} level={lvl!r}",
        )

        v_unknown = rebuilt.verify_action(
            role_address="D9-R9-T9-R9", action="read_data", context={"cost": 0.0}
        )
        ok_head, lvl_head = verdict("chief_data_officer", "read_data", 0.0)
        out["failopen_defaults"] = (
            bool(v_unknown.allowed)
            and str(v_unknown.level) == "auto_approved"
            and ok_head
            and lvl_head == "auto_approved",
            f"unknown address -> {v_unknown.allowed}/{v_unknown.level}; envelope-less head -> {ok_head}/{lvl_head} (both should be auto_approved under the installed default)",
        )
        return out

    checks.guarded(
        ["allow_probes", "deny_action_probes", "deny_budget_probe", "failopen_defaults"],
        probes,
    )
    return finalize(checks, WEIGHT, seed, None, GATES)


if __name__ == "__main__":
    main(grade)
