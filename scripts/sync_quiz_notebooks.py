#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Re-inline ``quiz_harness.py`` into Cell 1 of a module's quiz notebooks.

The quiz notebooks (``modules/mlfpNN/quiz/mlfpNN_quiz{,_solutions}.ipynb``)
are self-contained for Colab: Cell 1 carries the module's ``quiz_harness.py``
inline. ``generate_selfcontained_notebook.py`` only covers exercises, so
without this script the inlined copy drifts from the source harness — the
M5 quiz shipped a pre-kailash-ml course-local DLDiagnostics for that reason.

Only Cell 1 is rewritten (and its stale outputs cleared); question cells are
hand-authored and left untouched.

Usage:
    .venv/bin/python scripts/sync_quiz_notebooks.py modules/mlfp05/quiz
    .venv/bin/python scripts/sync_quiz_notebooks.py modules/mlfp05/quiz --check
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_selfcontained_notebook import (  # noqa: E402
    make_code_cell,
    strip_copyright_header,
)

_FUTURE = re.compile(r"^\s*from\s+__future__\s+import\s+annotations\s*$")
_CELL1_MARKER = "# ── quiz_harness.py ──"


def build_cell1_source(harness: Path) -> str:
    body = strip_copyright_header(harness.read_text())
    body = "\n".join(line for line in body.split("\n") if not _FUTURE.match(line))
    source = f"from __future__ import annotations\n\n{_CELL1_MARKER}\n{body.strip()}\n"
    ast.parse(source)  # fail loudly before touching any notebook
    return source


def sync(quiz_dir: Path, *, check: bool) -> int:
    harness = quiz_dir / "quiz_harness.py"
    if not harness.exists():
        print(f"✗ {harness} not found", file=sys.stderr)
        return 1
    expected = make_code_cell(build_cell1_source(harness))
    notebooks = sorted(quiz_dir.glob("*_quiz*.ipynb"))
    if not notebooks:
        print(f"✗ no *_quiz*.ipynb under {quiz_dir}", file=sys.stderr)
        return 1

    drifted = 0
    for path in notebooks:
        nb = json.loads(path.read_text())
        cell1 = nb["cells"][1]
        if _CELL1_MARKER not in "".join(cell1["source"]):
            print(f"✗ {path}: Cell 1 is not the inlined harness cell", file=sys.stderr)
            return 1
        if cell1["source"] == expected["source"]:
            print(f"✓ {path} — in sync")
            continue
        drifted += 1
        if check:
            print(f"✗ {path} — Cell 1 drifted from {harness.name}")
            continue
        nb["cells"][1] = expected
        # Match the existing on-disk format so the diff is Cell 1 only.
        path.write_text(json.dumps(nb, indent=1))
        print(f"↻ {path} — Cell 1 re-inlined from {harness.name}")
    return 1 if (check and drifted) else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("quiz_dir", type=Path, help="e.g. modules/mlfp05/quiz")
    ap.add_argument("--check", action="store_true", help="report drift, write nothing")
    args = ap.parse_args()
    sys.exit(sync(args.quiz_dir, check=args.check))


if __name__ == "__main__":
    main()
