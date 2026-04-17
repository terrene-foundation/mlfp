---
type: GAP
status: pending
session_id: redteam-2026-04-17
created: 2026-04-17
---

# M5 self-contained Colab generator never committed; 84 notebooks shipped broken in 6b28127

**What**: Commit `6b28127` (feat(m5): self-contained Colab notebooks, 2026-04-16) shipped 84 notebooks under `modules/mlfp05/colab-selfcontained*/ex_*/*.ipynb`. All 84 had `SyntaxError: unexpected indent` in Cell 3 — the agent that built them stripped only the first line of `from shared.mlfp05.ex_N import (\n  A,\n  B,\n)` leaving an orphan indented tuple body + closing `)`.

**How it slipped**: No generator script was committed in `scripts/`. The notebook build was a one-off agent workflow. No CI step runs `ast.parse` over notebook code cells, so the syntax error went undetected until a red-team scan in session 2026-04-17.

**Immediate fix (2026-04-17)**: Regex band-aid applied across all 84 notebooks (`\n\s*\n(    \w+,\n)+\)\n` → `\n`). Post-fix: 42/42 solutions parse, 4/42 student notebooks have intentional `# TODO` scaffolds (expected).

**Open gap**: The generator itself still doesn't exist as a reviewable, re-runnable script. If `shared/mlfp05/*` changes (new helper, renamed import), regenerating notebooks requires re-running an un-committed agent workflow with the same bug class.

**Impact**: Rolling out the same self-contained pattern to M1–M4 + M6 (~210 more notebooks) without first writing a proper generator would propagate the bug class at 5× scale.

**Recommended work**:

1. Write `scripts/generate-selfcontained-notebook.py` — reads `solutions/` + inlines `shared/mlfpNN/*` helpers + strips multi-line `from shared.*` blocks completely + dedupes `from __future__ import annotations`.
2. Add `ast.parse` smoke-test to CI for every code cell of every committed `.ipynb` (skipping IPython `%`/`!`/`?` magics and known `# TODO`/`____` scaffold patterns).
3. THEN roll the pattern out to M1–M4 + M6.

**Student repo status**: Fix is local to source repo only. Student repo (`terrene-foundation/pcml-run26-professional-certificate-in-machine-learning-pcml-run26-2601`) still carries the broken notebooks. Sync pending.

---

## Resolution (2026-04-17, same day)

1. **Generator shipped**: `scripts/generate_selfcontained_notebook.py` — strips all three `from shared.*` forms (single-line, inline-paren, multi-line paren), dedupes `from __future__`, inlines per-module helpers (including the M6 `diagnostics` subpackage in dependency order), AST-validates every generated cell before writing.
2. **CI guards shipped**:
   - `scripts/check_notebook_syntax.py` — AST-parses every `.ipynb` code cell repo-wide; student notebooks allowed `# TODO`/`____` scaffolds; shell-continuation lines folded into `pass`. 699 notebooks currently pass.
   - `scripts/check-deck-parity.sh` — rebuilds M5+M6 decks, `pdftotext`, diffs vs. committed baselines in `pdf/decks/.parity-baselines/`. Catches `katex-init.js` regressions.
3. **M5 regenerated**: 84 M5 notebooks re-generated from source (supersedes the 2026-04-17 regex band-aid). Also fixed two pre-existing M5 quiz notebooks shipped broken in commit `8c0dbe4` (same bug class).
4. **Rollout complete**: M1 (8+8), M2 (33+33), M3 (40+40), M4 (37+37), M6 (39+39) — 314 new self-contained notebooks plus M5's 84 = **398 total** under `modules/mlfp0?/colab-selfcontained{,-solutions}/`.

**Status**: Gap closed. Work is staged locally; pending user approval to commit. Student repo sync still pending (needs confirmed clone path).

## For Discussion

1. Counterfactual: would the `ast.parse` CI guard have caught commit `6b28127` before shipping? Yes — every M5 notebook's Cell 3 failed parse. The guard is cheap; only reason it didn't exist was no one had authored it. Is there a wider class of "obvious CI check missing because no one wrote it" that a single audit could surface?
2. Specific data: the 398 self-contained notebooks add ~25–200 KB of inlined helper code per notebook (M6 carries the largest payload at ~200 KB due to the diagnostics subpackage). Is the student-UX win (no git clone required) worth the ~50 MB of duplicated inline content across the student repo, or would a lean `pip install pcml-helpers` shared wheel be cleaner long-term?
3. The generator's dependency-order inline for M6 diagnostics (`_judges → _plots → retrieval → output → ... → __init__`) is currently hard-coded. If someone adds a new file to `shared/mlfp06/diagnostics/`, the generator won't know where to slot it. Worth inferring order via `ast.parse` of `import` statements, or accept that the list needs a manual update each time?
