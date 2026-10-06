# M1–M6 Audit — Decisions

Audit date: 2026-10-02/03. Findings: 513 (104 BLOCKING, 199 MAJOR, 210 MINOR) — see `01-analysis/`.

| #   | Decision                                                                                                                                                  | Chosen                                                     | Consequence for fixes                                                                                                                                                       |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | kailash-pact 0.14.1 `verify_action` auto-approves unknown and envelope-less roles (fail-open), contradicting the fail-closed principle the course teaches | **Teach the real behaviour only**; nothing raised upstream | M6 states the actual default; every deny-path demo/test attaches an envelope first (`set_role_envelope`); spec 6.7 corrected to match                                       |
| D2  | Invented statistics, incidents and deployments attributed to real banks, hospitals, regulators, universities (M2–M6)                                      | **Anonymise**                                              | Replace with generic actors ("a Singapore bank", "a local university"); figures marked illustrative; keep a real public reference only where the fact is real and checkable |
| D3  | Remediation scope                                                                                                                                         | **Everything** — all 513 findings                          | Phase A (correctness) then Phase B (spec-gap content, assessment redesign to non-dictated tasks, MINORs) in the same programme                                              |
| D4  | `speaker-notes.md` for M1/M2/M5 written for older decks                                                                                                   | **Regenerate from the current decks**                      | Rebuild each from the deck's per-slide `<aside class="notes">`, then correct facts/API                                                                                      |
| D5  | Fleet hosts lack git-lfs, so trestle cannot snapshot this repo                                                                                            | Operator installs git-lfs                                  | Heavy verification (exercise execution, PDF builds) goes through `trestle run` once installed; fixers verify statically until then                                          |

Source of truth for every correction: the INSTALLED stack (`.venv`, kailash-ml 2.2.2, kailash-pact 0.14.1, …) and the actual data files in `data/`. Where `specs/module-N.md` contradicts the installed stack, the spec is corrected too.

## Pending owner decisions surfaced during remediation (2026-10-04)
- P1 ICU data (M3 ex_1): only 198/416,526 vital readings fall in the first 24h → first-24h features carry no signal (AUC ≈ 0.50). Regenerate dataset, move the cutoff, or reframe ex_1.
- P2 HDB data quality (M1/M2/M4): S$10 / S$9M prices, negative lease ages, letter-O storey typos — keep as deliberate M1 teaching defects with one shared cleaner downstream, or ship a clean variant.
- P3 mlxtend `.to_pandas()` at the call boundary (M4 4.5 + ex_5) vs the polars-only mandate.
- P4 Default model fallbacks: `SFT_BASE_MODEL` → "Qwen/Qwen2.5-0.5B-Instruct" (shared/mlfp06 ex_2/ex_3, slides), Ollama bootstrap defaults — env var first with a documented default (zero-config Colab) vs env-models.md "never hardcode".
- P5 Spec assessment lines ("Quiz + project") vs shipped auto-graded tasks — reconcile after the S5 redesign.
- P6 Upstream SDK issues (kailash-ml setup() leak, kailash-dataflow relative sqlite path, ONNX external data, Nexus CORS preflight 401, A2A un-awaited coroutine, HumanApprovalAgent placeholder, …) — D1 "no upstream" covered PACT only; decide whether to file the rest.

## P7 — Template re-point: kailash-coc-claude-py → kailash-coc-py (OWNER DECISION NEEDED)

Via trestle fleet steward (2026-10-07): loom has RETIRED the `kailash-coc-claude-*`
template variants; mlfp's upstream (`kailash-coc-claude-py` 3.21.10, synced
2026-06-22) will receive no more deliveries — including the lighter-hooks
coc-base delivery (4.5× fewer hook processes). Migration path per loom:
re-point upstream template to `kailash-coc-py`, run /sync-from-template,
restart sessions. **Timing gate:** hold until loom announces "TEMPLATE DONE: py"
(Python delivery queued behind Rust). Side note: routing the advisory-hooks
diff (d8b3f006) upstream requires a /codify proposal from this repo after the
re-point; loom will not adopt it from a relay.

### P1 — corroborating measurement (2026-10-07)

The ICU vitals timestamps are not admission-aligned at all: median first vital
is ~3,032 h BEFORE admission (range −26,680 h … +17,245 h); only 4 of 8,000
admissions have ANY vital inside the 24 h prediction window (198 readings
total). `build_vital_features` therefore produces nulls for ~99.95% of
admissions. Any M3 ex_1 redesign should regenerate vitals with timestamps
drawn inside each admission's stay. Measured via shared/mlfp03/ex_1.py
loaders on the current parquet.

### P6 — corroborating repro (2026-10-07): ModelExplainer additivity unit mismatch

kailash-ml 2.2.2 `ModelExplainer(model, X).explain_global()` raises
`shap.utils._exceptions.ExplainerError` on a binary sklearn-API LightGBM
model: its internal TreeExplainer emits LOG-ODDS SHAP values while
`assert_additivity` compares against `model.predict` PROBABILITY output
(measured: SHAP sum 0.9983 vs predict 0.6174). `check_additivity` is not
exposed, so the engine is unusable for this model class in 2.2.2. Repro and
unit-level analysis live in modules/mlfp03/solutions/ex_6/06 (Checkpoint 0).
Upstream fix: explain in predict's output space (model_output="probability")
or compare against raw margins consistently.

### P6 — further corroboration (2026-10-07): backend override + spawn warning hygiene

1. `km.use_device("cpu")` does NOT reach family fits: families still resolve
   backend='mps' on Apple Silicon and xgboost/lightgbm raise UnsupportedFamily
   inside `MLEngine.compare` (measured: families failed with 'mps' errors inside
   a use_device('cpu') block). The engine tolerates partial family failure —
   on Macs the "model comparison" silently compares fewer families.
2. `km.train` family workers are spawned processes: they inherit `-W` flags
   but NOT in-process `warnings.filterwarnings` — under a warnings-as-errors
   gate, Lightning's PossibleUserWarning ("GPU available but not used") kills
   the sklearn family too (error text becomes the family failure). Course
   handling: ex_0/00 exempted from the strict suite gate with the reason in
   the file header; suite runner has STRICT_SKIP.

### P6 — one more (2026-10-07): LocalRuntime._record_execution_metrics arity

kailash 2.44.1 logs `Error tracking conditional execution performance:
LocalRuntime._record_execution_metrics() missing 4 required positional
arguments: 'execution_time', 'node_count', 'skipped_nodes', 'execution_mode'`
when a SwitchNode-pruned plan completes (seen in mlfp03 ex_7/05 output).
Upstream logging-path bug; non-fatal but noise in every conditional
workflow run.
