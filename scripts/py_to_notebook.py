#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Convert local .py exercise files to Jupyter and Colab .ipynb notebooks.

Usage:
    python scripts/py_to_notebook.py modules/ascent01/local/ex_1.py
    python scripts/py_to_notebook.py --all          # Convert all modules
    python scripts/py_to_notebook.py --module ascent01  # Convert one module
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def py_to_cells(source: str) -> list[dict]:
    """Split a Python exercise file into notebook cells.

    Strategy:
    - The docstring becomes a markdown cell
    - Each `# ══════` TASK section becomes a markdown heading + code cell
    - Top-level code blocks become code cells
    - Comments starting with `# ──` become markdown cells
    """
    lines = source.split("\n")
    cells = []
    current_code: list[str] = []
    in_docstring = False
    docstring_lines: list[str] = []

    def flush_code():
        nonlocal current_code
        code = "\n".join(current_code).strip()
        if code:
            cells.append(make_code_cell(code))
        current_code = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect docstring start/end
        if '"""' in line and not in_docstring:
            in_docstring = True
            docstring_lines = []
            # Check if single-line docstring
            if line.count('"""') >= 2:
                in_docstring = False
                content = line.strip().strip('"').strip()
                if content:
                    cells.append(make_markdown_cell(content))
                i += 1
                continue
            i += 1
            continue
        elif '"""' in line and in_docstring:
            in_docstring = False
            # Convert docstring to markdown
            md = "\n".join(docstring_lines).strip()
            if md:
                cells.append(make_markdown_cell(md))
            i += 1
            continue
        elif in_docstring:
            docstring_lines.append(line)
            i += 1
            continue

        # Detect TASK section headers (═══)
        if re.match(r"^# [═]{10,}", line):
            flush_code()
            # Read the task title (next non-empty, non-border line)
            j = i + 1
            title_lines = []
            while j < len(lines) and not re.match(r"^# [═]{10,}", lines[j]):
                if lines[j].startswith("# ") and lines[j] != "#":
                    title_lines.append(lines[j][2:])
                elif lines[j] == "#":
                    title_lines.append("")
                else:
                    break
                j += 1
            # Skip closing border
            if j < len(lines) and re.match(r"^# [═]{10,}", lines[j]):
                j += 1

            if title_lines:
                md = "\n".join(title_lines).strip()
                cells.append(make_markdown_cell(f"## {md}"))

            i = j
            continue

        # Detect section markers (── )
        if re.match(r"^# ── .+──", line):
            flush_code()
            title = line.strip("# ─").strip()
            cells.append(make_markdown_cell(f"### {title}"))
            i += 1
            continue

        # Regular code line
        current_code.append(line)
        i += 1

    flush_code()
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


def make_notebook(cells: list[dict], kernel: str = "python3") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": kernel,
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _detect_packages(source: str) -> str:
    """Detect which Kailash packages are needed from import statements."""
    packages = ["polars", "plotly", "gdown", "python-dotenv"]
    # Map import prefixes to pip packages
    pkg_map = {
        "kailash_ml": "kailash-ml",
        "kailash_align": "kailash-align",
        "kailash_pact": "kailash-pact",
        "kailash_nexus": "kailash-nexus",
        "kailash_mcp": "kailash-mcp",
        "kailash.": "kailash",
        "kaizen": "kailash-kaizen",
    }
    for prefix, pkg in pkg_map.items():
        if f"from {prefix}" in source or f"import {prefix}" in source:
            if pkg not in packages:
                packages.append(pkg)
    return " ".join(packages)


def jupyter_setup_cell(source: str = "") -> dict:
    packages = _detect_packages(source)
    return make_code_cell(f"%pip install -q {packages}")


def colab_setup_cell(source: str = "") -> dict:
    packages = _detect_packages(source)
    return make_code_cell(
        "# Google Colab setup\n"
        f"!pip install -q {packages}\n"
        "\n"
        "# Mount Google Drive for datasets\n"
        "from google.colab import drive\n"
        'drive.mount("/content/drive")'
    )


def convert_asyncio_for_notebook(cells: list[dict]) -> list[dict]:
    """Replace asyncio.run(func()) with top-level await for notebooks."""
    converted = []
    for cell in cells:
        if cell["cell_type"] != "code":
            converted.append(cell)
            continue

        source = "".join(cell["source"])

        # Replace asyncio.run(func()) with await func()
        source = re.sub(
            r"(\w[\w, ]*)\s*=\s*asyncio\.run\((\w+)\((.*?)\)\)",
            r"\1 = await \2(\3)",
            source,
        )
        source = re.sub(
            r"asyncio\.run\((\w+)\((.*?)\)\)",
            r"await \1(\2)",
            source,
        )

        # Remove `import asyncio` if no longer needed
        if "asyncio" not in source.replace("import asyncio", ""):
            source = re.sub(r"import asyncio\n?", "", source)

        cell = {**cell, "source": [line + "\n" for line in source.split("\n")]}
        converted.append(cell)

    return converted


def convert_file(py_path: Path):
    """Convert a single .py exercise to Jupyter and Colab notebooks."""
    source = py_path.read_text()
    module = py_path.parent.parent.name  # e.g., ascent01

    # Parse into cells
    cells = py_to_cells(source)

    # Strip copyright/license from first cell (already in docstring)
    if cells and cells[0]["cell_type"] == "code":
        first_source = "".join(cells[0]["source"])
        if first_source.strip().startswith("# Copyright"):
            # Remove copyright lines from first code cell
            lines = first_source.split("\n")
            clean_lines = [
                l
                for l in lines
                if not l.startswith("# Copyright") and not l.startswith("# SPDX")
            ]
            cells[0]["source"] = [
                line + "\n" for line in "\n".join(clean_lines).strip().split("\n")
            ]

    # Jupyter notebook
    jupyter_cells = [jupyter_setup_cell(source)] + convert_asyncio_for_notebook(cells)
    jupyter_nb = make_notebook(jupyter_cells)

    notebook_dir = py_path.parent.parent / "notebooks"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = notebook_dir / py_path.with_suffix(".ipynb").name

    with open(notebook_path, "w") as f:
        json.dump(jupyter_nb, f, indent=1)
    print(f"  Jupyter: {notebook_path}")

    # Colab notebook
    colab_cells = [colab_setup_cell(source)] + convert_asyncio_for_notebook(cells)
    colab_nb = make_notebook(colab_cells)

    colab_dir = py_path.parent.parent / "colab"
    colab_dir.mkdir(parents=True, exist_ok=True)
    colab_path = colab_dir / py_path.with_suffix(".ipynb").name

    with open(colab_path, "w") as f:
        json.dump(colab_nb, f, indent=1)
    print(f"  Colab:   {colab_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert .py exercises to notebooks")
    parser.add_argument("file", nargs="?", help="Single .py file to convert")
    parser.add_argument("--all", action="store_true", help="Convert all modules")
    parser.add_argument("--module", help="Convert a specific module (e.g., ascent01)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent / "modules"

    if args.file:
        convert_file(Path(args.file))
    elif args.module:
        local_dir = root / args.module / "local"
        if not local_dir.exists():
            print(f"No local/ directory found at {local_dir}")
            sys.exit(1)
        for py_file in sorted(local_dir.glob("ex_*.py")):
            print(f"Converting {py_file.name}...")
            convert_file(py_file)
    elif args.all:
        for module_dir in sorted(root.iterdir()):
            local_dir = module_dir / "local"
            if local_dir.exists() and list(local_dir.glob("ex_*.py")):
                print(f"\n=== {module_dir.name} ===")
                for py_file in sorted(local_dir.glob("ex_*.py")):
                    print(f"Converting {py_file.name}...")
                    convert_file(py_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
