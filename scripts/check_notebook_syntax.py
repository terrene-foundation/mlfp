#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""AST-parse every code cell in every tracked notebook.

CI guard for notebook syntax. Run with no args to scan the whole repo;
pass paths or globs for a targeted run.

Rules:
- IPython magic lines (starting with `%`, `!`, `?`) are stripped before parse.
- Student cells containing `# TODO` or `____` placeholder markers are
  allowed to fail parse (those are intentional fill-in-blank scaffolds).
- Any other SyntaxError in a committed notebook fails the check.

Exit 0 on success, 1 on any unexpected SyntaxError. Intended for CI.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


def _strip_magics(src: str) -> str:
    """Replace IPython magic lines (`%`, `!`, `?`) with `pass`.

    Removing them outright breaks AST parse when a magic is the sole body
    of a control-flow block (e.g. `if not exists: !git clone ...`).
    Replacement with `pass` preserves indentation and block structure.

    Also folds shell line-continuations (`!pip install foo \\\n    bar`)
    into the `pass` so continuation lines don't dangle.
    """
    out: list[str] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        l = lines[i]
        stripped = l.lstrip()
        if stripped.startswith(("%", "!", "?")):
            indent = l[: len(l) - len(stripped)]
            out.append(f"{indent}pass")
            # Swallow continuation lines (ending with `\` or leading whitespace-
            # continued commands) so they don't parse as orphan code.
            while l.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                l = lines[i]
            i += 1
            continue
        out.append(l)
        i += 1
    return "\n".join(out)


def _has_intentional_blanks(src: str) -> bool:
    return "# TODO" in src or "____" in src


def _is_student_notebook(path: Path) -> bool:
    """Student-facing notebooks may ship with intentional `# TODO` / `____`
    scaffold blanks that don't AST-parse. Those are legitimate; any other
    SyntaxError in them is a real bug.

    Non-student (solution) notebooks must parse cleanly.
    """
    parts = set(path.parts)
    # Student-facing directories across delivery formats:
    #   colab-selfcontained/  — new self-contained Colab
    #   colab/                — classic Colab (requires git clone)
    #   notebooks/            — Jupyter format
    #   local/ex_N/           — VS Code .py (not .ipynb but included for safety)
    return (
        bool(parts & {"colab-selfcontained", "colab", "notebooks"})
        and "colab-selfcontained-solutions" not in parts
    )


def check_notebook(path: Path) -> list[tuple[int, str]]:
    """Return a list of (cell_index, error_msg) for unexpected SyntaxErrors."""
    try:
        nb = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return [(-1, f"invalid JSON: {e}")]
    errors: list[tuple[int, str]] = []
    allow_blanks = _is_student_notebook(path)
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        clean = _strip_magics(src).strip()
        if not clean:
            continue
        try:
            ast.parse(clean)
        except SyntaxError as e:
            if allow_blanks and _has_intentional_blanks(src):
                continue
            errors.append((i, f"line {e.lineno}: {e.msg}"))
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "paths", nargs="*", help="Notebook paths or glob roots (default: modules/)"
    )
    args = ap.parse_args()

    roots = [Path(p) for p in args.paths] if args.paths else [REPO_ROOT / "modules"]

    notebooks: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".ipynb":
            notebooks.append(root)
        elif root.is_dir():
            notebooks.extend(sorted(root.rglob("*.ipynb")))

    if not notebooks:
        print("no notebooks found", file=sys.stderr)
        return 1

    fail = 0
    for nb_path in notebooks:
        errs = check_notebook(nb_path)
        if errs:
            fail += 1
            for i, msg in errs:
                rel = (
                    nb_path.relative_to(REPO_ROOT) if nb_path.is_absolute() else nb_path
                )
                print(f"{rel}:cell={i}: {msg}")

    total = len(notebooks)
    if fail:
        print(
            f"\n{fail}/{total} notebooks have unexpected SyntaxErrors", file=sys.stderr
        )
        return 1
    print(f"✓ {total} notebooks parse cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
