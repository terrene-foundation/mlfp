# M1–M6 Remediation Plan

Scope: all 513 findings in `01-analysis/audit-mlfp0N.md` (decisions in `../decisions.md`).

## Shards (per module N = 1..6)

Each shard owns a disjoint file set, so shards of one module can run in parallel
and merge without conflicts. The exercises are the source of truth for "what the
course's code looks like", so S1 lands before S2–S4 of the same module are merged.

| Shard          | Owns                                                                                  | Covers                                                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| S1 exercises   | `modules/mlfp0N/solutions/**`, `local/**`, `shared/mlfp0N/**` (M1 also `shared/*.py`) | exercise BLOCKING/MAJOR/MINOR: leakage, wrong theory, broken checkpoints, scaffold drift/leaks, fabricated outputs, anonymisation, env-model names |
| S2 slides      | `deck.html`, `lessons/*/slides.html`                                                  | snippet API/data errors (checker → 0), wrong facts, anonymisation, deck exercise descriptions                                                      |
| S3 textbook    | `textbook.md`, `lessons/*/textbook.html`                                              | same as S2 for textbook prose, worked examples, drills, solutions                                                                                  |
| S4 notes       | `speaker-notes.md`, `lessons/*/notes.html`                                            | regenerate (M1/M2/M5) or correct (M3/M4/M6)                                                                                                        |
| S5 assessment  | `assessment/**`                                                                       | grader exploits, wrong keys; Phase B: redesign to non-dictated tasks + coverage                                                                    |
| S6 spec gaps   | exercises + deck + textbook additions                                                 | Phase B: missing spec techniques (new technique files / sections)                                                                                  |
| S7 integration | generated artefacts                                                                   | regenerate Colab notebooks, PDFs, parity baselines; index/README; full solution run via trestle                                                    |

Cross-module: `specs/module-N.md` corrections ride with the module's S2.

## Gates (every shard)

- `scripts/check_doc_snippets.py <files>` → 0 findings for S2/S3/S4 files
- `scripts/check_notebook_syntax.py`, `ast.parse` on every touched `.py`
- local scaffold vs solution: blanks map 1:1, checkpoints/REFLECTION verbatim, no solution text in `local/`
- graders: reference solution passes; known-wrong submissions fail
- `scripts/check-deck-overflow.js` 0 clipped for touched decks
- S7: every solution executes (trestle), notebooks regenerated, `check-deck-parity.sh` refreshed

## Execution

Waves of ≤3 worktree agents, each on an explicit branch `audit/mNN-<shard>` cut
from the integration branch tip; merged into `fix/m1-m6-audit`, then one PR per
module. Phase A (S1–S5 correctness) for all modules, then Phase B (S5 redesign, S6).
