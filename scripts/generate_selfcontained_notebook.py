#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Generate self-contained Colab notebooks from exercise .py files.

A self-contained notebook carries all `shared/` helpers inline in Cell 1,
so students can open it in Colab and run without git-cloning the course
repo or editing a FORK_URL.

Usage:
    # Single file
    python scripts/generate_selfcontained_notebook.py \
        modules/mlfp05/solutions/ex_1/01_standard_ae.py \
        --out modules/mlfp05/colab-selfcontained-solutions/ex_1/01_standard_ae.ipynb

    # Whole exercise directory (batch)
    python scripts/generate_selfcontained_notebook.py \
        --solutions modules/mlfp05/solutions \
        --local modules/mlfp05/local \
        --out-solutions modules/mlfp05/colab-selfcontained-solutions \
        --out-students modules/mlfp05/colab-selfcontained

Design contract
---------------
- Cell 0 = pip install + nest_asyncio + GPU check (no git clone).
- Cell 1 = inlined helpers (`shared/kailash_helpers.py` if imported,
  `shared/data_loader.py` if imported, per-module `shared/mlfpNN/ex_N.py`,
  and per-module `shared/mlfpNN/diagnostics` if imported).
- Cells 2+ = exercise content with every `from shared.*` removed
  (including full multi-line parenthesised forms) and duplicate
  `from __future__ import annotations` lines deduplicated.

Every emitted code cell is `ast.parse`d (after stripping IPython magics
and known `# TODO` / `____` scaffold lines) before the notebook is
written to disk. A SyntaxError aborts the generation with a pointer to
the offending cell.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────
# Import stripping
# ─────────────────────────────────────────────────────────────────────


def strip_shared_imports(source: str) -> str:
    """Remove every `from shared.*` import statement from `source`.

    Handles three forms:
        from shared.X import a, b, c
        from shared.X import (a, b, c)
        from shared.X import (
            a,
            b,
            c,
        )

    The multi-line form is the one that the original (buggy) M5
    generator mishandled, leaving an orphan indented tuple body.
    """
    lines = source.split("\n")
    out: list[str] = []
    i = 0
    single_line = re.compile(r"^\s*from\s+shared(\.|\s+import)")
    paren_open = re.compile(r"^\s*from\s+shared[\w.]*\s+import\s*\(\s*$")
    paren_inline = re.compile(r"^\s*from\s+shared[\w.]*\s+import\s*\([^)]*\)\s*$")
    while i < len(lines):
        line = lines[i]
        # Inline parenthesised form on a single line
        if paren_inline.match(line):
            i += 1
            continue
        # Multi-line parenthesised form: skip until the matching `)`
        if paren_open.match(line):
            i += 1
            while i < len(lines) and not re.match(r"^\s*\)\s*$", lines[i]):
                i += 1
            # Skip the closing `)` itself
            if i < len(lines):
                i += 1
            continue
        # Simple single-line form
        if single_line.match(line):
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def dedupe_future_imports(source: str) -> str:
    """Collapse repeated `from __future__ import annotations` to one.

    Inlined helpers + exercise source both declare it; duplicates are
    harmless but noisy.
    """
    seen = False
    out: list[str] = []
    for line in source.split("\n"):
        if re.match(r"^\s*from\s+__future__\s+import\s+annotations\s*$", line):
            if seen:
                continue
            seen = True
        out.append(line)
    return "\n".join(out)


def strip_copyright_header(source: str) -> str:
    """Remove the `# Copyright ... / # SPDX-License-Identifier: ...` prefix."""
    lines = source.split("\n")
    i = 0
    while i < len(lines) and (
        lines[i].startswith("# Copyright") or lines[i].startswith("# SPDX")
    ):
        i += 1
    return "\n".join(lines[i:])


# ─────────────────────────────────────────────────────────────────────
# Helper content inlining
# ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent


def load_helper(rel: str) -> str:
    """Read a shared helper file and strip its copyright header."""
    path = REPO_ROOT / rel
    if not path.exists():
        return ""
    return strip_copyright_header(path.read_text())


def collect_helper_imports(source: str) -> set[str]:
    """Return the set of `shared.*` top-level modules imported by `source`."""
    mods: set[str] = set()
    for line in source.split("\n"):
        m = re.match(r"^\s*from\s+(shared(?:\.[\w.]+)?)\s+import\b", line)
        if m:
            mods.add(m.group(1))
    return mods


def build_cell1_content(module: str, exercise_source: str) -> str:
    """Concatenate every `shared/` helper the exercise needs into one cell body."""
    imports = collect_helper_imports(exercise_source)
    parts: list[str] = [
        "# ══════════════════════════════════════════════════════════════════",
        "# MLFP inlined helpers — DO NOT EDIT (collapse this cell!)",
        f"# Auto-generated by scripts/generate_selfcontained_notebook.py for {module}",
        "# ══════════════════════════════════════════════════════════════════",
        "from __future__ import annotations",
        "",
    ]

    # Order matters: kailash_helpers is imported by diagnostics which is
    # imported by ex_N — inline top-down so forward references resolve.
    if "shared" in imports or any(i.startswith("shared.") for i in imports):
        parts.append("# ── shared/kailash_helpers.py ──")
        parts.append(load_helper("shared/kailash_helpers.py"))
        parts.append("")

    # data_loader is referenced by some ex_N helpers via `from shared.data_loader`
    for imp in sorted(imports):
        if imp == "shared.data_loader":
            parts.append("# ── shared/data_loader.py ──")
            parts.append(load_helper("shared/data_loader.py"))
            parts.append("")

    # Per-module diagnostics subpackage (M5 flat file; M6 subpackage)
    for imp in sorted(imports):
        if imp == f"shared.{module}.diagnostics":
            # M5 has shared/mlfp05/diagnostics.py (flat)
            flat = REPO_ROOT / "shared" / module / "diagnostics.py"
            if flat.exists():
                parts.append(f"# ── shared/{module}/diagnostics.py ──")
                parts.append(load_helper(f"shared/{module}/diagnostics.py"))
                parts.append("")
            else:
                # M6 has shared/mlfp06/diagnostics/*.py (subpackage)
                pkg = REPO_ROOT / "shared" / module / "diagnostics"
                if pkg.is_dir():
                    # Inline in a dependency-friendly order: leaves first, then
                    # high-level modules that reference them.
                    order = [
                        "_judges.py",
                        "_plots.py",
                        "_traces.py",
                        "retrieval.py",
                        "output.py",
                        "alignment.py",
                        "interpretability.py",
                        "governance.py",
                        "agent.py",
                        "observatory.py",
                        "__init__.py",
                    ]
                    for f in order:
                        fp = pkg / f
                        if not fp.exists():
                            continue
                        parts.append(f"# ── shared/{module}/diagnostics/{f} ──")
                        parts.append(load_helper(f"shared/{module}/diagnostics/{f}"))
                        parts.append("")

    # Per-exercise ex_N.py
    for imp in sorted(imports):
        m = re.match(rf"^shared\.{module}\.(ex_\d+)$", imp)
        if m:
            ex_file = f"shared/{module}/{m.group(1)}.py"
            if (REPO_ROOT / ex_file).exists():
                parts.append(f"# ── {ex_file} ──")
                parts.append(load_helper(ex_file))
                parts.append("")

    # Rewrite inlined helpers that themselves import `from shared.*`:
    # those references now resolve to names already defined in this cell.
    body = "\n".join(parts)
    body = strip_shared_imports(body)
    body = dedupe_future_imports(body)
    return body


# ─────────────────────────────────────────────────────────────────────
# Cell 0: pip + GPU boilerplate
# ─────────────────────────────────────────────────────────────────────


def detect_packages(source: str, helper_source: str) -> list[str]:
    """Which pip packages does the combined source need?"""
    combined = source + "\n" + helper_source
    packages = ["kailash", "polars", "plotly", "gdown", "python-dotenv", "nest-asyncio"]
    pkg_map = {
        "kailash_ml": "kailash-ml",
        "kailash_align": "kailash-align",
        "kailash_pact": "kailash-pact",
        "kailash_nexus": "kailash-nexus",
        "kailash_mcp": "kailash-mcp",
        "kaizen": "kailash-kaizen",
    }
    for prefix, pkg in pkg_map.items():
        if f"from {prefix}" in combined or f"import {prefix}" in combined:
            if pkg not in packages:
                packages.append(pkg)
    return packages


def cell0_source(packages: list[str]) -> str:
    pkgs = " ".join(packages)
    return (
        "# ══════════════════════════════════════════════════════════════════\n"
        "# Colab setup — self-contained. No git clone, no FORK_URL, no sys.path.\n"
        "# Before running: Runtime → Change runtime type → T4 GPU (free tier)\n"
        "# ══════════════════════════════════════════════════════════════════\n"
        f"!pip install -q {pkgs}\n"
        "\n"
        "import nest_asyncio; nest_asyncio.apply()\n"
        "\n"
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    print(f'✓ GPU available: {torch.cuda.get_device_name(0)}')\n"
        "else:\n"
        "    print('⚠ No GPU — training will be slow. Runtime → Change runtime type → T4 GPU')\n"
        "\n"
        "print('✓ Setup complete. All helpers defined in the next cell.')"
    )


# ─────────────────────────────────────────────────────────────────────
# Exercise body → cells (re-uses py_to_notebook.py logic)
# ─────────────────────────────────────────────────────────────────────


def py_to_cells(source: str) -> list[dict]:
    """Split a Python exercise file into notebook cells.

    Mirrors the segmentation used in `scripts/py_to_notebook.py`:
    - Module docstring → markdown cell
    - `# ══════` section headers → markdown h2
    - `# ── section ──` markers → markdown h3
    - Everything else between markers → code cells
    """
    lines = source.split("\n")
    cells: list[dict] = []
    current: list[str] = []
    in_doc = False
    doc_lines: list[str] = []

    def flush():
        nonlocal current
        code = "\n".join(current).strip()
        if code:
            cells.append(make_code_cell(code))
        current = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if '"""' in line and not in_doc:
            in_doc = True
            doc_lines = []
            if line.count('"""') >= 2:
                in_doc = False
                content = line.strip().strip('"').strip()
                if content:
                    cells.append(make_markdown_cell(content))
                i += 1
                continue
            i += 1
            continue
        if '"""' in line and in_doc:
            in_doc = False
            md = "\n".join(doc_lines).strip()
            if md:
                cells.append(make_markdown_cell(md))
            i += 1
            continue
        if in_doc:
            doc_lines.append(line)
            i += 1
            continue

        if re.match(r"^# [═]{10,}", line):
            flush()
            j = i + 1
            title_lines: list[str] = []
            while j < len(lines) and not re.match(r"^# [═]{10,}", lines[j]):
                if lines[j].startswith("# ") and lines[j] != "#":
                    title_lines.append(lines[j][2:])
                elif lines[j] == "#":
                    title_lines.append("")
                else:
                    break
                j += 1
            if j < len(lines) and re.match(r"^# [═]{10,}", lines[j]):
                j += 1
            if title_lines:
                cells.append(make_markdown_cell("## " + "\n".join(title_lines).strip()))
            i = j
            continue

        if re.match(r"^# ── .+──", line):
            flush()
            title = line.strip("# ─").strip()
            cells.append(make_markdown_cell(f"### {title}"))
            i += 1
            continue

        current.append(line)
        i += 1
    flush()
    return cells


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")],
    }


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
    }


def convert_asyncio(cells: list[dict]) -> list[dict]:
    """`asyncio.run(func())` → `await func()` for top-level await in Colab."""
    out: list[dict] = []
    for cell in cells:
        if cell["cell_type"] != "code":
            out.append(cell)
            continue
        src = "".join(cell["source"])
        src = re.sub(
            r"(\w[\w, ]*)\s*=\s*asyncio\.run\((\w+)\((.*?)\)\)",
            r"\1 = await \2(\3)",
            src,
        )
        src = re.sub(r"asyncio\.run\((\w+)\((.*?)\)\)", r"await \1(\2)", src)
        if "asyncio" not in src.replace("import asyncio", ""):
            src = re.sub(r"import asyncio\n?", "", src)
        out.append({**cell, "source": [l + "\n" for l in src.split("\n")]})
    return out


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ─────────────────────────────────────────────────────────────────────
# AST validation
# ─────────────────────────────────────────────────────────────────────


def _strip_magics(src: str) -> str:
    return "\n".join(
        l for l in src.splitlines() if not l.lstrip().startswith(("%", "!", "?"))
    )


def _has_scaffold_blanks(src: str) -> bool:
    """Student notebooks may have `# TODO` / `____` blanks that don't parse."""
    return "# TODO" in src or "____" in src


def validate_cells(cells: list[dict], *, path: Path, allow_blanks: bool) -> None:
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        clean = _strip_magics(src).strip()
        if not clean:
            continue
        try:
            ast.parse(clean)
        except SyntaxError as e:
            if allow_blanks and _has_scaffold_blanks(src):
                # Expected for student cells with `# TODO` placeholders
                continue
            raise SyntaxError(
                f"{path}: cell {i} failed AST parse at line {e.lineno}: {e.msg}"
            ) from e


# ─────────────────────────────────────────────────────────────────────
# Top-level build
# ─────────────────────────────────────────────────────────────────────


def build_notebook(source_py: Path, *, module: str, allow_blanks: bool) -> dict:
    raw = source_py.read_text()
    body = strip_copyright_header(raw)
    body = strip_shared_imports(body)
    body = dedupe_future_imports(body)

    cell1_body = build_cell1_content(module, raw)
    packages = detect_packages(raw, cell1_body)

    cells = [make_code_cell(cell0_source(packages))]
    cells.append(make_code_cell(cell1_body))
    cells.extend(convert_asyncio(py_to_cells(body)))

    validate_cells(cells, path=source_py, allow_blanks=allow_blanks)
    return make_notebook(cells)


def module_for(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"mlfp0\d", part):
            return part
    raise ValueError(f"cannot infer module from path: {path}")


def convert_one(source_py: Path, out_ipynb: Path, *, allow_blanks: bool) -> None:
    module = module_for(source_py)
    nb = build_notebook(source_py, module=module, allow_blanks=allow_blanks)
    out_ipynb.parent.mkdir(parents=True, exist_ok=True)
    with open(out_ipynb, "w") as f:
        json.dump(nb, f, indent=1)


def convert_tree(src_root: Path, out_root: Path, *, allow_blanks: bool) -> int:
    n = 0
    for py in sorted(src_root.rglob("*.py")):
        if py.name == "__init__.py":
            continue
        rel = py.relative_to(src_root).with_suffix(".ipynb")
        convert_one(py, out_root / rel, allow_blanks=allow_blanks)
        n += 1
    return n


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", help="Single .py file to convert")
    ap.add_argument("--out", help="Output .ipynb path (with --file)")
    ap.add_argument("--solutions", help="Solutions directory root (batch mode)")
    ap.add_argument("--local", help="Local/student directory root (batch mode)")
    ap.add_argument("--out-solutions", help="Output dir for solution notebooks")
    ap.add_argument("--out-students", help="Output dir for student notebooks")
    args = ap.parse_args()

    if args.file:
        convert_one(
            Path(args.file),
            Path(args.out) if args.out else Path(args.file).with_suffix(".ipynb"),
            allow_blanks=False,
        )
        print(f"✓ {args.file} → {args.out}")
        return

    if args.solutions and args.out_solutions:
        n = convert_tree(
            Path(args.solutions), Path(args.out_solutions), allow_blanks=False
        )
        print(f"✓ solutions: {n} notebooks → {args.out_solutions}")

    if args.local and args.out_students:
        n = convert_tree(Path(args.local), Path(args.out_students), allow_blanks=True)
        print(f"✓ students:  {n} notebooks → {args.out_students}")

    if not any([args.file, args.solutions, args.local]):
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
