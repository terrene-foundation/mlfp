# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Shared plumbing for the MLFP03 assessment graders (instructor-side only).

Every grader follows the same pattern:

1. import the student's submission as a module;
2. build grader-held inputs the student never sees (secret subsamples drawn
   with a fresh random seed, synthetic data with planted ground truth,
   held-out rows);
3. call the student's functions on those inputs and compare against a
   reference computed by the grader itself.

Because the inputs change on every run, a submission that hard-codes numbers,
echoes its input, or ignores the data it is given cannot pass.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import secrets
import sys
import traceback
from pathlib import Path
from typing import Any, Callable


def load_student_module(path: Path, name: str):
    """Import the submission file at ``path`` as a fresh module."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    mod = importlib.util.module_from_spec(spec)
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


def finalize(checks: Checks, weight: int, seed: int, error: str | None = None) -> dict:
    total = sum(1 for v in checks.results.values() if v)
    n = len(checks.results)
    marks = round(weight * total / n, 2) if n else 0.0
    out: dict[str, Any] = {
        "passed": n > 0 and total == n and error is None,
        "checks_passed": total,
        "checks_total": n,
        "marks": marks if error is None else 0.0,
        "weight": weight,
        "seed": seed,
        "checks": checks.results,
    }
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
