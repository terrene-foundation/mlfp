# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the MLFP05 assessment graders (instructor-side only).

Every grader follows the same pattern:

1. import the student's submission as a module;
2. build grader-held inputs the student never sees (fresh stratified splits,
   jittered variants, freshly-seeded synthetic series, models with planted
   pathologies) — all drawn with a fresh secret seed;
3. call the student's functions or probe the student's returned objects on
   those inputs, and compare against a reference the grader computes itself.

Because the grader re-derives its own ground truth and never reads a number
the submission reports about itself, a submission that hard-codes metrics,
echoes its input, or hands back an untrained model cannot pass.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import secrets
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


def quiet() -> contextlib.AbstractContextManager:
    """Redirect stdout to stderr while student code runs.

    Submissions print (training logs, torch.onnx progress, ...). The grader's
    own stdout is the JSON report, so student output is diverted to stderr —
    visible for debugging, never corrupting the report.
    """
    return contextlib.redirect_stdout(sys.stderr)


def load_student_module(path: Path, name: str):
    """Import the submission file at ``path`` as a fresh module."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    with quiet():
        spec.loader.exec_module(mod)
    return mod


def close(a: Any, b: Any, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    """Numeric closeness that returns False (never raises) on bad input."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if a != a or b != b:  # NaN never matches
        return False
    return abs(a - b) <= atol + rtol * abs(b)


class Checks:
    """Ordered collection of named pass/fail checks with diagnostic notes."""

    def __init__(self) -> None:
        self.results: dict[str, bool] = {}
        self.notes: dict[str, str] = {}

    def add(self, name: str, ok: bool, note: str = "") -> bool:
        self.results[name] = bool(ok)
        if note and not ok:
            self.notes[name] = note
        return bool(ok)

    def guarded(self, names: list[str], fn: Callable[[], dict[str, tuple[bool, str]]]) -> None:
        """Run ``fn`` (which returns {name: (ok, note)}); on any exception mark
        every name in ``names`` as failed with the error message."""
        try:
            out = fn()
        except Exception as exc:  # the student's code raised — record, never hide
            msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc(limit=3)
            for n in names:
                self.add(n, False, msg)
            self.notes.setdefault("_traceback", tb)
            return
        for n in names:
            ok, note = out.get(n, (False, "check not evaluated"))
            self.add(n, ok, note)


def finalize(checks: Checks, weight: int, seed: int, error: str | None = None,
             gates: tuple[str, ...] = ()) -> dict:
    """Marks = weight x non-gate checks passed / non-gate checks. A failed gate
    zeroes the task. Gates earn no marks of their own."""
    gates_ok = all(checks.results.get(g, False) for g in gates)
    earned = sum(1 for k, v in checks.results.items() if v and k not in gates)
    n = sum(1 for k in checks.results if k not in gates)
    total = sum(1 for v in checks.results.values() if v)
    marks = round(weight * earned / n, 2) if (n and gates_ok and error is None) else 0.0
    out: dict[str, Any] = {
        "passed": error is None and gates_ok and n > 0 and earned == n,
        "checks_passed": total,
        "checks_total": len(checks.results),
        "marks": marks,
        "weight": weight,
        "seed": seed,
        "checks": checks.results,
    }
    if gates:
        out["gates"] = {g: bool(checks.results.get(g, False)) for g in gates}
    if checks.notes:
        out["notes"] = checks.notes
    if error:
        out["error"] = error
    return out


def main(grade: Callable[[Path, int], dict]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="replay a previous grading run (default: fresh secret seed)",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(32)
    result = grade(args.submission.resolve(), seed)
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0 if result["passed"] else 1)
