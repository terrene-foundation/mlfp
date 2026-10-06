#!/usr/bin/env python3
# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""Speaker-notes title-order gate: every lessons/NN/notes.html slide-block
<h3> title must match its lessons/NN/slides.html slide titles (h1/h2/h3) in
order, exactly. Notes that drift from the deck fail here. Run after any
notes or slides edit.

Usage: .venv/bin/python scripts/check_notes_titles.py [modules/mlfpNN ...]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
H_BLOCK = re.compile(r"<h3[^>]*>(.*?)</h3>", re.S)
H_SLIDE = re.compile(r"<h[123][^>]*>(.*?)</h[123]>", re.S)
STRIP = re.compile(r"<[^>]+>")


def titles(path: Path, rx) -> list[str]:
    if not path.exists():
        return []
    import html

    text = html.unescape(path.read_text(errors="ignore"))
    return [re.sub(r"\s+", " ", STRIP.sub("", t)).strip() for t in rx.findall(text)]


def main() -> int:
    modules = [Path(m) for m in sys.argv[1:]] or sorted(REPO.glob("modules/mlfp0[1-6]"))
    bad = 0
    for mod in modules:
        for lesson in sorted((mod / "lessons").glob("[0-9]*")):
            slides = titles(lesson / "slides.html", H_SLIDE)
            notes = titles(lesson / "notes.html", H_BLOCK)
            # Notes carry one header block before the slide blocks; drop the
            # leading block when the note has one more title than the slides.
            if len(notes) == len(slides) + 1:
                notes = notes[1:]
            if notes != slides:
                bad += 1
                print(f"✗ {mod.name}/lessons/{lesson.name}: "
                      f"{len(slides)} slides vs {len(notes)} note blocks")
                for i, (a, b) in enumerate(zip(slides, notes)):
                    if a != b:
                        print(f"    first mismatch at block {i+1}: slide {a!r} vs note {b!r}")
                        break
    if bad:
        print(f"\n{bad} lesson(s) drifted from their slides.")
        return 1
    print("✓ all speaker notes match their slides (title order)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
