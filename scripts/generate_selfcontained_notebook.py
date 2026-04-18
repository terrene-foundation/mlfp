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


def collect_helper_imports_transitive(exercise_source: str) -> set[str]:
    """Walk the `shared.*` import graph to fixpoint.

    An exercise imports `shared.mlfp02.ex_1`; that helper then imports
    `shared.data_loader` and `shared.kailash_helpers`. Those must also be
    inlined or the helper crashes at runtime with NameError. This does a
    BFS over the imports so every transitive dependency surfaces.
    """
    seen: set[str] = set()
    frontier = collect_helper_imports(exercise_source)
    while frontier:
        next_frontier: set[str] = set()
        for mod in frontier:
            if mod in seen:
                continue
            seen.add(mod)
            path = _shared_import_to_path(mod)
            if path and path.exists():
                content = path.read_text()
                next_frontier |= collect_helper_imports(content)
        frontier = next_frontier - seen
    return seen


def _shared_import_to_path(imp: str) -> Path | None:
    """Map `shared.X` import string to the file on disk that defines it.

    Special case: `from shared import MLFPDataLoader` (top-level `shared`
    package) maps to `shared/__init__.py`, whose own `from shared.X import`
    statements the transitive walk will then pick up.
    """
    parts = imp.split(".")
    # e.g. ["shared"] → shared/__init__.py
    init_py = REPO_ROOT / Path(*parts) / "__init__.py"
    if init_py.exists():
        return init_py
    # e.g. ["shared", "mlfp05", "ex_1"] → shared/mlfp05/ex_1.py
    candidate_py = REPO_ROOT / Path(*parts).with_suffix(".py")
    if candidate_py.exists():
        return candidate_py
    return None


def strip_relative_imports(source: str) -> str:
    """Remove `from .x import y` and `from .x.y import z` lines.

    When a subpackage like `shared/mlfp06/diagnostics/` is flattened into
    one cell, its internal relative imports become invalid — the symbols
    are already defined in the same namespace from earlier inlined files.
    """
    lines = source.split("\n")
    out: list[str] = []
    i = 0
    single_line = re.compile(r"^\s*from\s+\.[\w.]*\s+import\b")
    paren_open = re.compile(r"^\s*from\s+\.[\w.]*\s+import\s*\(\s*$")
    paren_inline = re.compile(r"^\s*from\s+\.[\w.]*\s+import\s*\([^)]*\)\s*$")
    while i < len(lines):
        line = lines[i]
        if paren_inline.match(line):
            i += 1
            continue
        if paren_open.match(line):
            i += 1
            while i < len(lines) and not re.match(r"^\s*\)\s*$", lines[i]):
                i += 1
            if i < len(lines):
                i += 1
            continue
        if single_line.match(line):
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def build_cell1_content(module: str, exercise_source: str) -> str:
    """Concatenate every `shared/` helper the exercise needs into one cell body.

    Uses the transitive closure of `shared.*` imports — an exercise that
    only imports `shared.mlfpNN.ex_1` still pulls in `shared.data_loader`
    and `shared.kailash_helpers` if `ex_1.py` itself imports them.
    """
    imports = collect_helper_imports_transitive(exercise_source)
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

    # M6 Ollama bootstrap — every M6 ex_N.py imports from this module, so
    # it must be inlined BEFORE the per-module diagnostics and ex_N files
    # (which call make_delegate / DEFAULT_CHAT_MODEL from the bootstrap).
    if f"shared.{module}._ollama_bootstrap" in imports:
        bootstrap_path = f"shared/{module}/_ollama_bootstrap.py"
        if (REPO_ROOT / bootstrap_path).exists():
            parts.append(f"# ── {bootstrap_path} ──")
            parts.append(load_helper(bootstrap_path))
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
    # Also strip relative imports (`from .x import y`) that appear in the
    # M6 diagnostics subpackage — those symbols are co-located now.
    body = "\n".join(parts)
    body = strip_shared_imports(body)
    body = strip_relative_imports(body)
    # Flatten module-style references to leaf names. After inlining,
    # `_plots.py` contributes PRIMARY, TEMPLATE, etc. at top level; the
    # consuming files still write `_plots.PRIMARY` which no longer
    # resolves. Rewrite those to the bare name.
    body = flatten_module_references(body, ("_plots", "_judges", "_traces"))
    body = rewrite_repo_root_resolution(body)
    body = dedupe_future_imports(body)
    return body


def flatten_module_references(source: str, module_names: tuple[str, ...]) -> str:
    """Replace `module_name.X` with `X` for each named subpackage module.

    The inlining concatenates subpackage files into one namespace, so
    `_plots.PRIMARY` no longer resolves through a module object — but
    `PRIMARY` is defined at the same top level. The textual rewrite is
    safe because these names are chosen to be subpackage-private (leading
    underscore) and do not appear as prefixes of unrelated identifiers.
    """
    for name in module_names:
        pattern = re.compile(rf"\b{re.escape(name)}\.")
        source = pattern.sub("", source)
    return source


def rewrite_repo_root_resolution(source: str) -> str:
    """Make `REPO_ROOT = Path(__file__).resolve().parents[N]` Colab-safe.

    Helper files resolve the repo root from `__file__` so that data dirs
    live alongside the source tree. In a Colab cell, `__file__` is not
    defined — and when it IS defined (e.g. our tempfile test harness),
    `.parents[N]` escapes the filesystem. Rewrite to `Path.cwd()`, which
    is `/content` in Colab (writable, consistent) and the repo root when
    running tests from that CWD.
    """
    return re.sub(
        r"Path\(__file__\)\.resolve\(\)\.parents\[\d+\]",
        "Path.cwd()",
        source,
    )


# ─────────────────────────────────────────────────────────────────────
# Cell 0: pip + GPU boilerplate
# ─────────────────────────────────────────────────────────────────────


def detect_packages(source: str, helper_source: str) -> list[str]:
    """Which pip packages does the combined source need?"""
    combined = source + "\n" + helper_source
    packages = [
        "kailash",
        "polars",
        "plotly",
        "gdown",
        "python-dotenv",
        "nest-asyncio",
        "httpx",
    ]
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


# ─────────────────────────────────────────────────────────────────────
# Cell 0a: Ollama bootstrap (M6 only)
# ─────────────────────────────────────────────────────────────────────


def _lesson_from_path(path: Path) -> str | None:
    """Parse the ``ex_N`` lesson id from a solution path, or return None."""
    for part in path.parts:
        if re.fullmatch(r"ex_\d+", part):
            return part
    return None


# Mirrors LESSON_MODELS in shared/mlfp06/_ollama_bootstrap.py. Duplicated
# here (not imported) because this generator runs without the repo's
# .venv on its path in some CI paths, and we want the Cell 0 body to be
# deterministic text that does not require a runtime import from shared/.
_M6_LESSON_MODELS: dict[str, list[str]] = {
    "ex_1": ["llama3.2:3b"],
    "ex_2": ["qwen2.5:0.5b", "llama3.2:3b"],
    "ex_3": ["qwen2.5:0.5b", "llama3.2:3b"],
    "ex_4": ["llama3.2:3b", "nomic-embed-text"],
    "ex_5": ["llama3.2:3b"],
    "ex_6": ["llama3.2:3b"],
    "ex_7": ["llama3.2:3b"],
    "ex_8": ["llama3.2:3b"],
}


def ollama_bootstrap_cell_source(lesson_id: str) -> str:
    """Generate the Colab-only Ollama bootstrap cell for an M6 lesson.

    The cell is idempotent: re-running is free if Ollama is already
    installed and the models are already pulled. It is a no-op on local
    (where students run ``ollama serve`` in a terminal themselves).
    """
    models = _M6_LESSON_MODELS.get(lesson_id, ["llama3.2:3b"])
    models_literal = repr(models)
    return (
        "# ══════════════════════════════════════════════════════════════════\n"
        "# M6 Ollama bootstrap — local LLMs, no API keys\n"
        "# On Colab: installs + starts Ollama and pulls this lesson's models.\n"
        "# Locally: verifies the daemon is running and models are pulled.\n"
        "# ══════════════════════════════════════════════════════════════════\n"
        f"_LESSON_MODELS = {models_literal}\n"
        "import sys, os, time, shutil, subprocess\n"
        "import httpx\n"
        "\n"
        "_OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', "
        "'http://localhost:11434')\n"
        "_IN_COLAB = 'google.colab' in sys.modules\n"
        "\n"
        "def _daemon_up(timeout_s=1.0):\n"
        "    try:\n"
        "        r = httpx.get(f'{_OLLAMA_BASE_URL}/api/tags', timeout=timeout_s)\n"
        "        r.raise_for_status()\n"
        "        return r.json()\n"
        "    except Exception:\n"
        "        return None\n"
        "\n"
        "if _IN_COLAB and shutil.which('ollama') is None:\n"
        "    print('[ollama] installing binary …')\n"
        "    subprocess.run(\n"
        "        'curl -fsSL https://ollama.com/install.sh | sh',\n"
        "        shell=True, check=True,\n"
        "    )\n"
        "\n"
        "if _IN_COLAB and _daemon_up() is None:\n"
        "    print('[ollama] starting daemon …')\n"
        "    subprocess.Popen(\n"
        "        'nohup ollama serve > /tmp/ollama.log 2>&1 &',\n"
        "        shell=True,\n"
        "    )\n"
        "    deadline = time.monotonic() + 30\n"
        "    while time.monotonic() < deadline and _daemon_up() is None:\n"
        "        time.sleep(0.5)\n"
        "\n"
        "_tags = _daemon_up(timeout_s=2.0)\n"
        "if _tags is None:\n"
        "    raise RuntimeError(\n"
        "        f'Ollama daemon not reachable at {_OLLAMA_BASE_URL}. '\n"
        "        'On Colab: re-run this cell. Locally: run `ollama serve`.'\n"
        "    )\n"
        "\n"
        "_have = {m.get('name','').split(':',1)[0] for m in _tags.get('models', [])}\n"
        "for _model in _LESSON_MODELS:\n"
        "    _fam = _model.split(':', 1)[0]\n"
        "    if _fam in _have:\n"
        "        continue\n"
        "    if _IN_COLAB:\n"
        "        print(f'[ollama] pulling {_model} (one-time) …')\n"
        "        subprocess.run(['ollama', 'pull', _model], check=True)\n"
        "    else:\n"
        "        raise RuntimeError(\n"
        "            f'Model {_model!r} not pulled. Run: ollama pull {_model}'\n"
        "        )\n"
        "\n"
        "print(\n"
        "    '✓ Ollama ready at', _OLLAMA_BASE_URL,\n"
        "    '— models:', _LESSON_MODELS,\n"
        ")\n"
    )


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
# Exercise body → cells
# ─────────────────────────────────────────────────────────────────────


def py_to_cells(source: str) -> list[dict]:
    """Split a Python exercise file into notebook cells.

    Segmentation rules:
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

    # M6 lessons need a real LLM. Insert the Ollama bootstrap cell between
    # the pip-install cell and the inlined-helpers cell so the daemon is
    # already running by the time any helper imports kaizen_agents.
    if module == "mlfp06":
        lesson = _lesson_from_path(source_py)
        if lesson is not None:
            cells.append(make_code_cell(ollama_bootstrap_cell_source(lesson)))

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
