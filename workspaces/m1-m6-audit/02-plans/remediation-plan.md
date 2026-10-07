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

## Verification status (2026-10-06) — what each PR carries

- **#15 (base):** snippet checker 0 across all teaching files; redline-check 0 BLOCKING; all 594 slides pass
  the overflow check; all 400 notebooks parse clean (402).
- **#16 (M1):** solutions 8/8 on the fleet; speaker notes regenerated (78 slides 1:1); notebooks regenerated.
- **#17 (M2):** solutions 33/33 on the fleet; 5-task assessment graders pass their reference (adversarial fail);
  storey-typo + FeatureStore NOT-NULL fixes verified.
- **#18 (M3):** solutions 40/40 after the ex_7 search-budget cap (ex_7/01 and ex_7/05 confirmed locally; the
  hanging LightGBM workflow exercises bounded); 4-task assessment graders pass.
- **#19 (M4):** solutions 37/38 on the fleet (only `04_bertopic`, which needs `TOPIC_EMBED_MODEL`); new TF-IDF
  technique passes on the fleet; 5-task assessment graders pass (task_4 NMF best-of-24).
- **#20 (M5):** exercises confirmed passing locally (ex_1/01, ex_7/05 ONNX export → serving); fleet run queued.
- **#21 (M6):** 13/39 pass on the fleet = exactly the NON-LLM exercises (PACT/governance, drift, LoRA math).
  The other 26 call preflight_ollama and correctly fail loudly without an Ollama host (matches the exercise
  agent's "Need Ollama" list). Full M6 verification needs an Ollama host.

## Verification status (2026-10-07) — spec-gap lanes + hooks posture

- **Hooks advisory conversion (d8b3f006):** the three PostToolUse blockers (validate-workflow,
  validate-deployment, enforce-framework-first) no longer halt the session; findings reach the main
  agent via `additionalContext` with a resolve-now directive. Owner directive; destructive-command
  PreToolUse denies (never session-halting) unchanged.
- **lane-specgaps MERGED (dbc8e685), worktree reaped.** Nine M3 technique files closing the spec gaps:
  ex_1/06 temporal (HDB panel, shift-before-roll, group_by_dynamic forward-window gotcha taught;
  R² 0.681→0.701), ex_1/07 forward+backward SFS vs RFE (on credit — ICU frame is flat per P1;
  fwd∩bwd=6/12), ex_1/08 correlation-threshold + FeatureEngineer generate/select (P2 experimental
  warning acknowledged; importance-vs-CV disagreement taught); ex_4/05 from-scratch AdaBoost
  (reproduces sklearn within noise), ex_4/06 CatBoost native categoricals (+0.0053 AUC-PR);
  ex_5/06 log-loss taxonomy (agent), ex_5/07 regression metrics (R²/MAE/RMSE/MAPE, HDB;
  naive-mean R²≈0), ex_5/08 EnsembleEngine stack/blend (honest cosmetic-gain read + engine
  contribution-contamination gotcha taught); ex_6/06 KernelSHAP (probability-space additivity
  1e-16) + ModelExplainer unit-mismatch investigation (P6 upstream finding). All fleet-verified
  under `-W error::UserWarning`; 18 notebooks regenerated; deck exercise slides 3.1/3.4/3.5/3.6
  aligned (81b4aea2); lesson-08 model-card slide now Mitchell's nine sections (d9a9d4a7).
- **Shared-infra fixes riding the lane:** LGBM/sklearn-1.9 spurious feature-name warning filtered
  centrally (shared/mlfp03/**init**.py — whole ex_5 family was latent); shap LightGBM-binary
  output-shape notice filtered at the handled call site (shared/mlfp03/ex_6.py).
- **Notebook parity restored course-wide:** M2 ex_5/05-07 + ex_6/05 (89e8d85a), M4 ex_6/06 (44d77c48),
  M6 ex_2/07 (13cb2d65). Generator runs that write the tree must run LOCAL — fleet mirrors do not
  sync generated files back.
- **Deck parity baselines refreshed** (49a73888) — a7269663 + M5 diagnostics edits had red-drifted them.
- **lane-m5-specgaps MERGED (1179e00c), reaped** — 9 technique pairs (5.2 Mixup ablation, 5.3
  indicators + char-LSTM perplexity, 5.5 DCGAN w/ FID vs noise reference, 5.6 GIN/GCN on MUTAG,
  5.8 PPO/A2C/DDPG/SAC pure-torch), all fleet-passed; README/index aligned (73f15f5f).
- **lane-assessments MERGED (3dcaa2c4), reaped** — all 8 M5+M6 tasks rebuilt: graders score the
  returned model on grader-held ground truth with fresh secret seeds; adversarial stubs verified
  FAIL. Handoff noted in merge body (6bcf3386 auto-commit misdescribes; real task_1 in d0a49ef2).
- **Strict-gate hardening:** course-wide create_visualizer() factory sweep (94fa3398); RL exercises
  seeded + PPO re-budgeted (556928fc); Lightning nag filter + num_workers=0 (84a8f9a8); ex_0/00
  exempted with documented spawn-inheritance cause (360433f9); auto-format hook now reports
  on-disk rewrites (976c0550).
- **M3 suite 47/49 on fleet (bbsl6g6ja) + ex_7 pair verified locally under -W error::UserWarning
  with 4-thread caps (b1gcq1vm6)** — the two fleet failures were host memory pressure (MemoryError
  at execution start / mid-run) plus one remote 0%-CPU hang (fleet-monitor f9d9d48e; likely
  mirror sqlite lock-wait). Same files pass locally in minutes: M3 effectively 49/49.

## LANDING (2026-10-07)

Single integration PR opened as **#22** (owner decision over refreshing the 7-PR
stack); #13–#21 closed with traceable references. Verification grid at open:
M1 8/8, M2 41/41, M3 49/49 (47 fleet + 2 local), M4 38/38 (35 suite + 3 re-runs),
M5 52/52, M6 13/13 non-LLM (26 LLM exercises need an Ollama host). 720 commits,
1,313 files vs main.

## M6 Ollama verification + owner decisions executed (2026-10-08)

- **Owner approved all pending decisions.** Executed: P4 + P5 via PR #23; P1 + P2 via PR #24
  (ICU tables admission-anchored — 996/1,000 admissions with first-24h vitals, was 4; ripple prose
  honest about the fix; one shared storey cleaner); P3 stands sanctioned; P6 documented (D1
  precedent: teach real, no upstream filing); P7 holds for loom "TEMPLATE DONE: py".
- **Ollama activated by the owner.** Course models (llama3.2:3b, qwen2.5:0.5b, nomic-embed-text)
  pulled on the GPU host (100.71.125.70) — the fleet reaches it; the local Mac Ollama stays as
  rollback until one full green fleet suite, then idles (steward informed).
- **M6 fixed during verification:** ex_1/06 JSON value-validation (null confidence crashed
  float()); ex_6/03 tight_layout nag dropped (bbox_inches covers it); align 0.7.3 unconditional
  resume trap — stale checkpoints made re-runs train zero steps (PR #26: clear experiment dir
  first, recorded under P6); DPO output dir moved under OUTPUT_DIR convention.
- **M6 40/40 verified** (LLM + non-LLM), PRs #25/#26/#27 landed; definitive full-suite re-run
  via fleet + GPU host in flight.
