#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP06 Assessment Task 1 — Schema-Constrained Extraction.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The LLM runs at temperature 0 (deterministic greedy decoding), so extraction
is stable across runs. Factual fields whose values are explicit in the text
(the incident id) are graded exactly; the remaining LLM-extracted fields use a
high pass FLOOR (5 of 6) to tolerate at most one occasional drift — see the
inline notes. Schema and type checks are fully deterministic.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REQUIRED_KEYS = {
    "incident_id",
    "severity",
    "location",
    "parcels_affected",
    "claim_required",
}
SEVERITY_ENUM = {"low", "medium", "high"}

# Independent ground truth, keyed by incident id.
GROUND_TRUTH = {
    "INC-3001": {
        "severity": "high",
        "location": "tuas checkpoint",
        "parcels": 42,
        "claim": True,
    },
    "INC-3002": {
        "severity": "low",
        "location": "changi airfreight centre",
        "parcels": 3,
        "claim": False,
    },
    "INC-3003": {
        "severity": "medium",
        "location": "jurong port",
        "parcels": 17,
        "claim": True,
    },
    "INC-3004": {
        "severity": "high",
        "location": "woodlands checkpoint",
        "parcels": 58,
        "claim": True,
    },
    "INC-3005": {
        "severity": "low",
        "location": "pasir panjang terminal",
        "parcels": 1,
        "claim": False,
    },
    "INC-3006": {
        "severity": "medium",
        "location": "tampines logistics hub",
        "parcels": 9,
        "claim": False,
    },
}
N = len(GROUND_TRUTH)
FLOOR = 5  # of 6 — tolerate at most one drift on LLM-extracted fields


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task1", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _norm_id(v) -> str:
    return re.sub(r"\s+", "", str(v)).upper()


def _to_int(v):
    try:
        if isinstance(v, bool):
            return None
        return int(v)
    except (TypeError, ValueError):
        m = re.search(r"-?\d+", str(v))
        return int(m.group(0)) if m else None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "yes", "required", "1"}:
        return True
    if s in {"false", "no", "not required", "0"}:
        return False
    return None


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
    c["returns_list"] = isinstance(r, list)
    if not c["returns_list"]:
        return _finalize(score)

    c["correct_length"] = len(r) == N
    c["all_items_dict"] = len(r) > 0 and all(isinstance(x, dict) for x in r)
    if not (c["correct_length"] and c["all_items_dict"]):
        return _finalize(score)

    # Schema compliance: every required key present in every record.
    c["schema_keys_present"] = all(REQUIRED_KEYS.issubset(x.keys()) for x in r)

    # Type compliance: id is str-ish, parcels coerces to int, claim coerces to bool.
    c["types_correct"] = all(
        x.get("incident_id") is not None
        and _to_int(x.get("parcels_affected")) is not None
        and _to_bool(x.get("claim_required")) is not None
        for x in r
    )

    # Severity in the allowed enum (schema constraint), case-insensitive.
    c["severity_in_enum"] = all(
        str(x.get("severity", "")).strip().lower() in SEVERITY_ENUM for x in r
    )

    # Index records by extracted id for value comparison.
    by_id = {}
    for x in r:
        by_id[_norm_id(x.get("incident_id"))] = x

    # incident_id: exact match for ALL six (values are explicit in the text).
    c["incident_id_all_correct"] = all(gid in by_id for gid in GROUND_TRUTH)

    # The four LLM-extracted semantic fields — graded against a 5/6 floor.
    sev_ok = loc_ok = parcels_ok = claim_ok = 0
    for gid, gt in GROUND_TRUTH.items():
        rec = by_id.get(gid)
        if rec is None:
            continue
        if str(rec.get("severity", "")).strip().lower() == gt["severity"]:
            sev_ok += 1
        if gt["location"] in str(rec.get("location", "")).strip().lower():
            loc_ok += 1
        if _to_int(rec.get("parcels_affected")) == gt["parcels"]:
            parcels_ok += 1
        if _to_bool(rec.get("claim_required")) is gt["claim"]:
            claim_ok += 1

    c["severity_values_correct"] = sev_ok >= FLOOR
    c["location_values_correct"] = loc_ok >= FLOOR
    c["parcels_values_correct"] = parcels_ok >= FLOOR
    c["claim_required_values_correct"] = claim_ok >= FLOOR

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
