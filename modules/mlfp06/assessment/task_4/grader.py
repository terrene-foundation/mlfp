#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP06 Assessment Task 4 — PACT Governance.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The governance engine is a pure decision function, so the reference verdicts
are deterministic. All ten checks must pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

# Reference ground truth — independently fixed by the task specification.
EXPECTED_STATS = {"n_agents": 6, "n_delegations": 6, "n_departments": 3}
EXPECTED_VERDICTS = [True, False, False, True, False, True, True, True, False, False]
# Index groups for partial-credit checks.
ALLOW_IDX = [0, 3, 5, 6, 7]  # actions within envelope
DENY_ACTION_IDX = [1, 4, 8]  # action not in allowed_actions
DENY_BUDGET_IDX = [2, 9]  # cost exceeds financial cap


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task4", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(student_path: Path) -> dict:
    score: dict = {"passed": False, "checks": {}, "total": 0, "max": 0}
    try:
        student = load_student_module(student_path)
    except Exception as e:
        score["error"] = f"Failed to import: {type(e).__name__}: {e}"
        return score
    if not hasattr(student, "solve"):
        score["error"] = "Module does not define a solve() function"
        return score
    try:
        r = student.solve()
    except Exception as e:
        score["error"] = f"Runtime error in solve(): {type(e).__name__}: {e}"
        return score

    c = score["checks"]
    c["returns_dict"] = isinstance(r, dict)
    if not c["returns_dict"]:
        return _finalize(score)

    stats = r.get("org_stats", {}) if isinstance(r.get("org_stats"), dict) else {}
    c["n_agents_correct"] = stats.get("n_agents") == EXPECTED_STATS["n_agents"]
    c["n_delegations_correct"] = (
        stats.get("n_delegations") == EXPECTED_STATS["n_delegations"]
    )
    c["n_departments_correct"] = (
        stats.get("n_departments") == EXPECTED_STATS["n_departments"]
    )

    verdicts = r.get("verdicts")
    c["ten_verdicts_returned"] = isinstance(verdicts, list) and len(verdicts) == len(
        EXPECTED_VERDICTS
    )
    if c["ten_verdicts_returned"]:
        vb = [bool(v) for v in verdicts]
        c["allow_path_correct"] = all(vb[i] == EXPECTED_VERDICTS[i] for i in ALLOW_IDX)
        c["deny_by_action_correct"] = all(
            vb[i] == EXPECTED_VERDICTS[i] for i in DENY_ACTION_IDX
        )
        c["deny_by_budget_correct"] = all(
            vb[i] == EXPECTED_VERDICTS[i] for i in DENY_BUDGET_IDX
        )
        c["all_verdicts_match"] = vb == EXPECTED_VERDICTS
    else:
        c["allow_path_correct"] = False
        c["deny_by_action_correct"] = False
        c["deny_by_budget_correct"] = False
        c["all_verdicts_match"] = False

    c["escalation_caught"] = r.get("escalation_caught") is True

    return _finalize(score)


def _finalize(score: dict) -> dict:
    score["total"] = sum(1 for v in score["checks"].values() if v)
    score["max"] = len(score["checks"])
    score["passed"] = score["max"] > 0 and score["total"] == score["max"]
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("student", type=Path)
    args = parser.parse_args()
    result = grade(args.student)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["passed"] else 1)
