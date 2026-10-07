#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Check Python snippets in teaching material against the INSTALLED Kailash stack.

Decks, lesson pages, textbooks and speaker notes carry code students copy, but
nothing executes it — so it drifts from the real API silently (M1–M6 audit,
2026-10: most Kailash snippets in the decks/textbooks raised on kailash-ml 2.2.2).
This checker extracts every Python block and verifies, statically:

  * syntax (blocks with ``____`` scaffold blanks or ``...`` elisions are tolerated)
  * ``from X import Y`` / ``import X`` resolve for Kailash/course packages
  * ``Cls(...)`` keyword arguments match the constructor signature
  * ``obj.method(...)`` exists on the class ``obj`` was constructed from, and its
    keyword arguments match the method signature
  * ``loader.load("mlfpNN", "file")`` / ``"data/mlfpNN/file"`` paths exist in data/

A line carrying the comment ``# expect-error`` is exempt — for snippets that
demonstrate a failure on purpose (e.g. loading a file that does not exist).

It does not execute snippets, so it cannot judge semantics or printed outputs.

Usage:
    .venv/bin/python scripts/check_doc_snippets.py                 # all modules
    .venv/bin/python scripts/check_doc_snippets.py modules/mlfp03  # one module / file
    .venv/bin/python scripts/check_doc_snippets.py --json out.json
"""
from __future__ import annotations

import argparse
import ast
import html
import importlib
import inspect
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
CHECKED_ROOTS = ("kailash", "kaizen", "kaizen_agents", "dataflow", "nexus", "pact",
                 "kailash_ml", "kailash_align", "kailash_mcp", "shared")
DOC_GLOBS = ("deck.html", "lessons/*/slides.html", "lessons/*/textbook.html",
             "lessons/*/notes.html", "textbook.md", "speaker-notes.md", "README.md")
HTML_BLOCK = re.compile(r'<code[^>]*class="[^"]*language-python[^"]*"[^>]*>(.*?)</code>', re.S)
MD_BLOCK = re.compile(r"^```(?:python|py)\s*\n(.*?)^```", re.S | re.M)
DATA_REF = re.compile(r"""["'](?:data/)?(mlfp0\d|mlfp_assessment)/([\w./-]+\.(?:csv|parquet|json|jsonl|txt))["']""")
LOAD_CALL = re.compile(r"""\.load\(\s*["'](mlfp0\d|mlfp_assessment)["']\s*,\s*["']([\w./-]+)["']""")


@dataclass
class Finding:
    file: str
    line: int
    kind: str
    detail: str


@dataclass
class Ctx:
    file: str
    base_line: int
    findings: list[Finding] = field(default_factory=list)
    names: dict[str, object] = field(default_factory=dict)      # imported name -> object
    instances: dict[str, type] = field(default_factory=dict)    # var -> class it was built from

    def add(self, node_line: int, kind: str, detail: str) -> None:
        self.findings.append(Finding(self.file, self.base_line + node_line - 1, kind, detail))


_mod_cache: dict[str, object] = {}


def _import(mod: str):
    if mod not in _mod_cache:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _mod_cache[mod] = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 — report any import failure as a finding
            _mod_cache[mod] = e
    return _mod_cache[mod]


def _resolve(mod: str, name: str):
    m = _import(mod)
    if isinstance(m, Exception):
        return m
    if hasattr(m, name):
        return getattr(m, name)
    sub = _import(f"{mod}.{name}")
    return sub if not isinstance(sub, Exception) else AttributeError(f"{mod} has no {name}")


def _kwargs_problem(fn, kw: list[str]) -> str | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = sig.parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return None
    allowed = {n for n, p in params.items() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    bad = [k for k in kw if k not in allowed]
    return f"unexpected kwargs {bad}; accepts {sorted(allowed - {'self', 'cls'})}" if bad else None


def _strip_scaffold(src: str) -> str:
    # Scaffold blanks and elisions are pedagogy, not syntax errors.
    src = re.sub(r"\b_{4,}\b", "_BLANK_", src)
    return src


def check_block(ctx: Ctx, src: str) -> None:
    exempt = {i for i, l in enumerate(src.splitlines(), 1) if "expect-error" in l}
    before = len(ctx.findings)
    _check_block(ctx, src)
    ctx.findings[before:] = [f for f in ctx.findings[before:]
                             if f.line - ctx.base_line + 1 not in exempt]


def _check_block(ctx: Ctx, src: str) -> None:
    src = _strip_scaffold(src)
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        # Top-level await is legal in notebooks/Colab, so retry inside an async fn.
        try:
            tree = ast.parse("async def __snippet__():\n" + "\n".join("    " + l for l in src.splitlines()))
            ctx.base_line -= 1
        except SyntaxError:
            ctx.add(e.lineno or 1, "syntax", f"{e.msg}")
            return
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 \
                and node.module.split(".")[0] in CHECKED_ROOTS:
            for a in node.names:
                if a.name == "*":
                    continue
                obj = _resolve(node.module, a.name)
                if isinstance(obj, Exception):
                    ctx.add(node.lineno, "import", f"from {node.module} import {a.name}: {obj}")
                else:
                    ctx.names[a.asname or a.name] = obj
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in CHECKED_ROOTS:
                    m = _import(a.name)
                    if isinstance(m, Exception):
                        ctx.add(node.lineno, "import", f"import {a.name}: {m}")
                    elif a.asname:
                        ctx.names[a.asname] = m  # import kailash_ml as km -> km is kailash_ml
                    else:
                        ctx.names[a.name.split(".")[0]] = _import(a.name.split(".")[0])
    # Constructor + instance tracking in source order.
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) and node.value.func.id in ctx.names:
            cls = ctx.names[node.value.func.id]
            if inspect.isclass(cls):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        ctx.instances[t.id] = cls
        if isinstance(node, ast.Call):
            kw = [k.arg for k in node.keywords if k.arg]
            f = node.func
            if isinstance(f, ast.Name) and f.id in ctx.names:
                obj = ctx.names[f.id]
                target = obj.__init__ if inspect.isclass(obj) else obj
                if callable(target) and kw:
                    p = _kwargs_problem(target, kw)
                    if p:
                        ctx.add(node.lineno, "kwargs", f"{f.id}(...): {p}")
            elif isinstance(f, ast.Attribute):
                owner = None
                if isinstance(f.value, ast.Name):
                    owner = ctx.instances.get(f.value.id) or (
                        ctx.names.get(f.value.id) if inspect.isclass(ctx.names.get(f.value.id))
                        or inspect.ismodule(ctx.names.get(f.value.id)) else None)
                if owner is not None:
                    label = f"{getattr(owner, '__name__', owner)}.{f.attr}"
                    if not hasattr(owner, f.attr):
                        ctx.add(node.lineno, "attribute", f"{label} does not exist")
                    elif kw:
                        p = _kwargs_problem(getattr(owner, f.attr), kw)
                        if p:
                            ctx.add(node.lineno, "kwargs", f"{label}(...): {p}")
    for rx in (DATA_REF, LOAD_CALL):
        for m in rx.finditer(src):
            if not (REPO / "data" / m.group(1) / m.group(2)).exists():
                line = src[: m.start()].count("\n") + 1
                ctx.add(line, "data", f"data/{m.group(1)}/{m.group(2)} not found")


def blocks(path: Path):
    text = path.read_text(errors="ignore")
    rx = HTML_BLOCK if path.suffix == ".html" else MD_BLOCK
    for m in rx.finditer(text):
        body = m.group(1)
        if path.suffix == ".html":
            body = html.unescape(re.sub(r"<[^>]+>", "", body))
        yield text[: m.start(1)].count("\n") + 1, body


def doc_files(targets: list[str]) -> list[Path]:
    out: list[Path] = []
    for t in targets or [str(REPO / "modules")]:
        p = Path(t) if Path(t).is_absolute() else REPO / t
        if p.is_file():
            out.append(p)
        else:
            mods = [p] if p.name.startswith("mlfp0") else sorted(p.glob("mlfp0*"))
            for m in mods:
                for g in DOC_GLOBS:
                    out.extend(sorted(m.glob(g)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--json")
    args = ap.parse_args()
    findings: list[Finding] = []
    nblocks = 0
    for path in doc_files(args.targets):
        for line, src in blocks(path):
            nblocks += 1
            ctx = Ctx(str(path.relative_to(REPO)), line)
            check_block(ctx, src)
            findings.extend(ctx.findings)
    by_kind: dict[str, int] = {}
    for f in findings:
        by_kind[f.kind] = by_kind.get(f.kind, 0) + 1
        print(f"{f.file}:{f.line}: [{f.kind}] {f.detail}")
    print(f"\n{nblocks} python blocks checked; {len(findings)} findings {by_kind}")
    if args.json:
        Path(args.json).write_text(json.dumps([f.__dict__ for f in findings], indent=1))
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
