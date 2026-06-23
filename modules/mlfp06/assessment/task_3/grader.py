#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated grader for MLFP06 Assessment Task 3 — Tool-Using Agent.

Usage:
    python grader.py starter.py     # grade your attempt
    python grader.py solution.py    # verify the reference passes

The graded signal is TOOL SELECTION + ARGUMENTS, recorded by the tool wrappers
themselves — fully deterministic and independent of the model's prose. Because
each tool computes its result deterministically from real SST-2 data, "correct
tool + correct args" guarantees the correct computed value was produced as an
observation. The model's final wording is graded only as a soft floor (small
local models often fail to restate the value), so the pass criteria do not
depend on bit-stable text.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REQUIRED_TOOLS = {
    "dataset_size",
    "count_by_label",
    "average_review_length",
    "get_review_by_index",
}

# (expected_tool, arg_predicate) per question, in order.
EXPECTED = [
    ("dataset_size", None),
    ("count_by_label", lambda a: "positive" in str(a.get("label", "")).lower()),
    ("count_by_label", lambda a: "negative" in str(a.get("label", "")).lower()),
    ("average_review_length", None),
    ("get_review_by_index", lambda a: _as_int(a.get("index")) == 0),
]


def _as_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_student_module(path: Path):
    spec = importlib.util.spec_from_file_location("student_task3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _first_call(tools_called):
    """Return (name, args_dict) of the first tool call, or (None, {})."""
    if not tools_called:
        return None, {}
    entry = tools_called[0]
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return entry[0], (entry[1] if isinstance(entry[1], dict) else {})
    if isinstance(entry, str):
        return entry, {}
    return None, {}


def _names(tools_called):
    out = []
    for entry in tools_called or []:
        if isinstance(entry, (list, tuple)) and entry:
            out.append(entry[0])
        elif isinstance(entry, str):
            out.append(entry)
    return out


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

    tool_names = r.get("tool_names")
    transcripts = r.get("transcripts")

    c["tool_names_complete"] = isinstance(tool_names, list) and REQUIRED_TOOLS.issubset(
        set(tool_names)
    )
    c["transcripts_length"] = isinstance(transcripts, list) and len(transcripts) == len(
        EXPECTED
    )
    if not c["transcripts_length"]:
        return _finalize(score)

    c["transcript_keys"] = all(
        isinstance(t, dict) and {"question", "tools_called", "answer"} <= set(t.keys())
        for t in transcripts
    )
    if not c["transcript_keys"]:
        return _finalize(score)

    # Every question must have invoked at least one tool.
    c["every_question_used_a_tool"] = all(
        len(t.get("tools_called") or []) >= 1 for t in transcripts
    )

    # No hallucinated tool names — every called tool is registered.
    c["no_hallucinated_tool"] = all(
        all(n in REQUIRED_TOOLS for n in _names(t.get("tools_called")))
        for t in transcripts
    )

    # Correct tool selected for each question (first tool call).
    tool_ok = 0
    arg_ok_label = 0
    arg_ok_index = True
    for t, (exp_tool, pred) in zip(transcripts, EXPECTED):
        name, args = _first_call(t.get("tools_called"))
        if name == exp_tool:
            tool_ok += 1
        if exp_tool == "count_by_label" and pred is not None:
            if name == exp_tool and pred(args):
                arg_ok_label += 1
        if exp_tool == "get_review_by_index" and pred is not None:
            if not (name == exp_tool and pred(args)):
                arg_ok_index = False
    c["correct_tool_selected"] = tool_ok == len(EXPECTED)
    c["count_by_label_args_correct"] = arg_ok_label == 2  # positive + negative
    c["get_review_index_arg_correct"] = arg_ok_index

    # Deterministic coverage: across the five questions, every one of the four
    # tools is exercised at least once (Q0 size, Q1/Q2 count, Q3 avg, Q4 index).
    exercised = set()
    for t in transcripts:
        exercised.update(_names(t.get("tools_called")))
    c["all_tools_exercised"] = REQUIRED_TOOLS <= exercised

    # The agent produced a non-empty final answer for every question.
    c["answers_nonempty"] = all(
        isinstance(t.get("answer"), str) and t["answer"].strip() for t in transcripts
    )

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
