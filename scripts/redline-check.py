#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Automated redline validation for MLFP course materials.

Checks all 8 non-negotiable quality standards from specs/redlines.md.
Run: .venv/bin/python scripts/redline-check.py [--module mlfpNN]
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULES = ["mlfp01", "mlfp02", "mlfp03", "mlfp04", "mlfp05", "mlfp06"]
OVERFLOW_SCRIPT = REPO_ROOT / "scripts" / "check-deck-overflow.js"

BLOCKED_IMPORTS = [
    "import mlflow",
    "import pycaret",
    "import pandas",
    "from mlflow",
    "from pycaret",
    "from pandas",
]

FINDINGS: list[tuple[str, str, str, str]] = []  # (module, redline, severity, detail)


def finding(module: str, redline: str, severity: str, detail: str) -> None:
    FINDINGS.append((module, redline, severity, detail))
    marker = "FAIL" if severity == "BLOCKING" else "WARN"
    print(f"  [{marker}] R{redline}: {detail}")


_DEVICE_HELPERS = {"device", "DEVICE", "get_device", "init_environment"}


def _selects_device(source: str) -> bool:
    """True if the file picks its device automatically (Redline 8).

    Accepted: calling get_device() / init_environment() (both wrap
    kailash_helpers.get_device), importing ``device`` / ``DEVICE`` from a
    shared helper module (shared/mlfpNN/ex_N.py sets it via get_device()), or
    following a Hugging Face model's own placement (``model.device``).
    """
    if "get_device(" in source or "init_environment(" in source or "model.device" in source:
        return True
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").startswith("shared")
        and any(a.name in _DEVICE_HELPERS for a in node.names)
        for node in ast.walk(tree)
    )


def _places_on_device(source: str) -> bool:
    """True if the file moves tensors/models to a device (not just prose)."""
    return bool(re.search(r"\.to\(\s*\w*device|device\s*=\s*\w*device", source, re.I))


def check_module(module: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {module.upper()}")
    print(f"{'='*60}")

    sol_dir = REPO_ROOT / "modules" / module / "solutions"

    # Exercises are either single files (solutions/ex_N.py, M1) or R10
    # directories of technique files (solutions/ex_N/*.py, M2-M6). Globbing
    # only ex_*.py silently skipped every R10 module — R1/R7/R8 checked nothing.
    exercises: dict[str, list[Path]] = {}
    for p in sorted(sol_dir.glob("ex_*.py")):
        exercises[p.stem] = [p]
    for d in sorted(q for q in sol_dir.glob("ex_*") if q.is_dir()):
        files = sorted(f for f in d.glob("*.py") if not f.name.startswith("_"))
        if files:
            exercises[d.name] = files
    sol_files = [f for files in exercises.values() for f in files]

    # ── Redline 7: Blocked imports ──
    for sol in sol_files:
        content = sol.read_text()
        for blocked in BLOCKED_IMPORTS:
            if blocked in content:
                finding(module, "7", "BLOCKING", f"{sol.name}: uses '{blocked}'")

    # ── Redline 8: GPU/MPS ──
    for sol in sol_files:
        content = sol.read_text()
        if 'torch.device("cuda" if torch.cuda.is_available()' in content:
            finding(module, "8", "BLOCKING", f"{sol.name}: cuda-only device (no MPS)")
        if "import torch" in content and not _selects_device(content):
            if _places_on_device(content):
                finding(
                    module,
                    "8",
                    "BLOCKING",
                    f"{sol.name}: uses torch but no get_device()",
                )

    # ── Redline 1: Exercise depth (LOC as proxy) ──
    total_loc = 0
    ex_count = 0
    for files in exercises.values():
        total_loc += sum(len(f.read_text().splitlines()) for f in files)
        ex_count += 1
    if ex_count > 0:
        avg_loc = total_loc // ex_count
        if avg_loc < 400:
            finding(
                module,
                "1",
                "BLOCKING",
                f"avg {avg_loc} LOC/exercise (need >400 for 25h depth)",
            )
        elif avg_loc < 300:
            finding(
                module, "1", "BLOCKING", f"avg {avg_loc} LOC/exercise (critically thin)"
            )
    print(
        f"  [INFO] {ex_count} exercises, {total_loc} total LOC, avg {total_loc//max(ex_count,1)} LOC/ex"
    )

    # Redline 3 (deck overflow) is checked only by the rendered audit in
    # run_visual_overflow_check() — scripts/check-deck-overflow.js is the single
    # canonical detector. The old static "<pre> >18 lines without font-size"
    # heuristic contradicted it (false FAILs on .code-fit slides) and was removed.

    # ── Redline 6: MCQ check ──
    mcq_file = REPO_ROOT / "modules" / module / "quiz_mcq.py"
    if mcq_file.exists():
        finding(module, "6", "BLOCKING", f"quiz_mcq.py exists (MCQ is BANNED)")

    # ── Redline 7: Kailash engine usage ──
    engine_keywords = [
        "ExperimentTracker",
        "ModelRegistry",
        "TrainingPipeline",
        "ModelVisualizer",
        "DataExplorer",
        "PreprocessingPipeline",
        "AutoMLEngine",
        "EnsembleEngine",
        "HyperparameterSearch",
        "InferenceServer",
        "DriftMonitor",
        "OnnxBridge",
        "FeatureStore",
    ]
    engines_found = set()
    for sol in sol_files:
        content = sol.read_text()
        for eng in engine_keywords:
            if eng in content:
                engines_found.add(eng)
    if len(engines_found) < 2:
        finding(
            module,
            "7",
            "BLOCKING",
            f"Only {len(engines_found)} kailash engines used: {engines_found}",
        )
    else:
        print(
            f"  [PASS] R7: {len(engines_found)} engines used: {', '.join(sorted(engines_found))}"
        )

    # ── Summary ──
    module_findings = [f for f in FINDINGS if f[0] == module]
    blocking = [f for f in module_findings if f[3] == "BLOCKING"]
    if not module_findings:
        print(f"  [PASS] All checked redlines pass")


def run_visual_overflow_check(modules: list[str]) -> None:
    """Redline 3 visual check — render every slide and check scrollHeight.

    Delegates to scripts/check-deck-overflow.js (puppeteer + headless Chrome).
    Catches the failure mode where a slide renders but content is clipped
    at the bottom edge — invisible to a static HTML parse.
    """
    print(f"\n{'='*60}")
    print("  R3 (visual): Rendered slide overflow check")
    print(f"{'='*60}")

    if not OVERFLOW_SCRIPT.exists():
        print(
            f"  [SKIP] {OVERFLOW_SCRIPT.name} not found — install puppeteer to enable"
        )
        return

    deck_args = [f"modules/{m}" for m in modules]
    cmd = ["node", str(OVERFLOW_SCRIPT), "--json", *deck_args]
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError:
        print("  [SKIP] node not found on PATH — install Node.js to enable")
        return
    except subprocess.TimeoutExpired:
        finding("all", "3", "BLOCKING", "visual overflow check timed out after 300s")
        return

    if result.returncode == 2:
        print(f"  [SKIP] visual check setup error: {result.stderr.strip()[:200]}")
        return

    try:
        decks = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  [SKIP] visual check produced no JSON (stderr: {result.stderr[:200]})")
        return

    for d in decks:
        module = Path(d["deckPath"]).parent.name
        if d.get("error"):
            finding(module, "3", "BLOCKING", f"deck failed to load: {d['error']}")
            continue
        if not d["overflowing"]:
            print(f"  [PASS] {module}: {d['totalSlides']} slides fit 1280×720")
            continue
        for s in d["overflowing"]:
            finding(
                module,
                "3",
                "BLOCKING",
                f"slide {s['displayedIdx']} overflows by {s['overflowPx']}px: {s['title']}",
            )


def main() -> None:
    modules = MODULES
    skip_visual = "--no-visual" in sys.argv
    if "--module" in sys.argv:
        idx = sys.argv.index("--module")
        if idx + 1 < len(sys.argv):
            modules = [sys.argv[idx + 1]]

    print("MLFP Redline Validation")
    print("=" * 60)

    for module in modules:
        check_module(module)

    if not skip_visual:
        run_visual_overflow_check(modules)

    # ── Final summary ──
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    blocking = [f for f in FINDINGS if f[2] == "BLOCKING"]
    warnings = [f for f in FINDINGS if f[2] != "BLOCKING"]
    print(f"  BLOCKING: {len(blocking)}")
    print(f"  WARNINGS: {len(warnings)}")
    if blocking:
        print("\n  BLOCKING findings (must fix before shipping):")
        for mod, rl, _, detail in blocking:
            print(f"    {mod} R{rl}: {detail}")
        sys.exit(1)
    else:
        print("\n  All checked redlines PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
