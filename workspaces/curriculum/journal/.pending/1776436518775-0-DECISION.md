---
type: DECISION
status: pending
session_id: codify-2026-04-17
created: 2026-04-17
---

# Codify session: 13 redlines, self-contained Colab, transitive inlining contract

**Decision**: Elevate the session's self-contained Colab work into three new redlines (R11-R13) and update the two-format rule to full rule-authoring compliance. Codify the generator + CI guard chain as the mandated path for all future notebook authoring in MLFP.

## What was codified

1. **Redline 11** — Self-contained Colab is the canonical notebook format. Two shipping formats only (local .py + colab-selfcontained .ipynb); legacy `colab/`, `colab-instructor/`, `notebooks/` directories are BLOCKED from reintroduction.
2. **Redline 12** — Class-based equation markup (`<div class="equation-box"><span class="katex-display">…</span></div>`) with a shared idempotent renderer (`modules/assets/js/katex-init.js`). Dollar-delimiter syntax BLOCKED in decks that adopt the shared renderer.
3. **Redline 13** — Shared package structure + transitive inlining contract. Generator walks the `shared.*` import graph to fixpoint; subpackages (M6 diagnostics) are flattened with relative imports stripped and module-style references (`_plots.X`) rewritten.
4. **Redline 3** extended with the three-guard CI suite: `check-deck-overflow.js` (visual) + `check_notebook_syntax.py` (AST) + `check-deck-parity.sh` (pdftotext diff vs baseline).
5. **Rule `two-format.md`** rewritten to pass rule-authoring.md's MUST/MUST NOT + Why + DO/DO NOT + BLOCKED phrases + Origin line contract. Added path-scoping for shared/ and the generator.
6. **Agent `notebook-validator`** description + body fixed — was stale with "three delivery formats (local/Jupyter/Colab)" after the consolidation dropped Jupyter.
7. **Design-principles.md** §10 (two-format contract) and §11 (shared helper package layout) added.

## For Discussion

1. Counterfactual: if R11 had existed before commit `6b28127`, would the 84-notebook breakage have shipped? The audit command in R11 (AST-parse + Cell 1 exec) would have caught it at pre-commit. Is there any failure class the guard *still* wouldn't catch?
2. Specific data: the generator is now 555 LOC, mostly import-graph walking and subpackage flattening. Is that complexity indicating a deeper architectural issue (helpers should be re-packaged for Colab-first consumption), or is it inherent to supporting both `uv sync`-native and Colab-inlined paths from the same source?
3. M6 diagnostics subpackage flattening uses a hardcoded inline order (`_judges → _plots → _traces → retrieval → … → observatory`). When the next contributor adds `evaluation.py` to the diagnostics subpackage, the generator won't know where to slot it. Should the generator infer order from `import` statements (ast.parse + topological sort), or is the manual list good enough given M6 is the only subpackage for now?
