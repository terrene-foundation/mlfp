# Cross-cutting issues (span modules; owned by integration/S6, not a single module shard)

## X-PP: kailash-ml PreprocessingPipeline.setup() fits before it splits (data leakage) — FIXED in course code (merged 4a821de7)
Course fix landed: `shared.kailash_helpers.split_then_preprocess` / `split_raw_train_test` / `preprocess_train_test`
(+ tests/test_split_then_preprocess.py, 7 tests). All mlfp03 shared loaders + mlfp01 ex_8 route through it; mlfp03
ex_2 CV re-fits preprocessing per fold. Uses the private `_apply_imbalance_correction` (guarded, typed error).
Remaining: regenerate the 36 affected notebooks; execute mlfp03 ex_2–ex_8 + mlfp01 ex_8 on the fleet (metrics may shift
slightly — credit splits now stratified; ex_2 scales on its 300 train rows).
Verified 2026-10-03 in kailash-ml 2.2.2 (engines/preprocessing.py): setup() runs _impute (l.354),
_encode_categoricals (l.359, incl. TARGET encoding l.747) and _scale_numerics (l.376) on ALL rows, and only
then _split (l.393). Test-set statistics leak into training; target encoding before the split is severe.
- Course fix: split first (hold out test rows), fit the pipeline on train only, apply `transform()` to test —
  via one shared helper used by every exercise that calls setup() (11 files under modules/*/solutions + shared/).
  M1 slides already teach "hold out test first" (audit/m01-s2-slides).
- Teaching material must never call setup(train_size=...) leak-free.
- Upstream: SDK bug in kailash-ml (kailash-py). Owner decision pending (D1 "no upstream" was PACT-specific).

## Upstream SDK notes collected so far (not course-fixable)
- kailash-ml: setup() fit-before-split (above); top-level `kailash_ml.FeatureSchema` rejected by FeatureStore;
  schema re-registration as v2 fails; registry `_kml_drift_reports` conflicts with DriftMonitor; TrainingPipeline
  fits LightGBM on unnamed numpy (Column_N); ModelVisualizer ExperimentalWarning.
- kailash-dataflow: close_async leaves pool tasks pending at exit; aiosqlite worker-thread error after loop close.
- kailash-nexus: "Unknown parameter(s) for HandlerNode" warning per request; CORS preflight 401 (JWT before CORS).
- kailash-mcp: @structured_tool(input_schema=…) + @server.tool() registers a tool with no parameters.
- kailash-pact 0.14.1: fail-open default (owner decision D1: teach real behaviour, do not raise).

## X-NUM: numeric claims must be reconciled across artefacts (integration consistency pass)
Different shards computed the same quantity with different parsing/cleaning, e.g. M1 hourly taxi demand:
slides 1,917–2,054; textbook 1,743–1,893; raw lenient parse 1,998–2,570. Integration must recompute each
cross-artefact number ONCE from the exercise's canonical (cleaned) output and make deck, lesson pages,
textbook, notes and exercises agree — or state the claim qualitatively. Start with M1 (hourly demand, r values
raw 0.47 vs cleaned 0.91), then grep each module's handoff file for numbers quoted in more than one artefact.
- M1 lesson 1.8 SLIDES use a shorter pipeline (47,547 rows, 7→44 cols, 12/12 alerts, hourly 1,917–2,054) while ex_8 + textbook.md + lesson 1.8 textbook page use 43,934 rows, 12→53 cols, 15→11 alerts. Bring the 1.8 slides (and deck capstone slides) in line with ex_8.
- M1 1.5 seasonality: page uses town-month medians (0.7% spread); textbook.md raw calendar-month counts (<1%) — both "no seasonality"; harmonise the method.

## X-DF: kailash-dataflow turns a RELATIVE sqlite URL into a path at filesystem ROOT
Verified in dataflow/core/engine.py (~l.9272): `file_path = db_url.replace("sqlite:///", "/")`, so
`sqlite:///x.db` (relative) becomes `/x.db` → "unable to open database file". Absolute URLs work.
- Course fix (integration): every URL passed to `DataFlow(...)` (FeatureStore, DriftCheck CRUD, governance, …)
  must be absolute — one helper (e.g. `shared.kailash_helpers.sqlite_url(path)` → f"sqlite:///{Path(path).resolve()}")
  used at every DataFlow call site; deck/textbook snippets showing `DataFlow("sqlite:///hdb.db")` must use an
  absolute path too (M2 2.8 slide). Known sites: shared/mlfp02/ex_8.py FEATURE_STORE_URL; grep `DataFlow(` repo-wide.
- Upstream: kailash-dataflow bug (relative sqlite paths).
- Also: the data loader finds data/ only when run from the repo root — integration runs start at the root.

## X-DATA: data-quality issues in shipped datasets (owner / S6 decision)
- data/mlfp01/hdb_resale.parquet: impossible prices (107 at S$10, 144 at S$9M), negative remaining-lease ages,
  letter-O typos in storey_range ("O4 TO 06"). M1 teaches them as planted defects; M2/M4 clean them ad hoc.
  Decide: keep as deliberate teaching defects (and make every downstream module clean them the same way via one
  shared cleaner) or ship a clean variant for M2+.
- M4 4.4 table: equal-blend "clustered" AUC is 0.97 (code 0.9748); textbook.md says 0.98 → reconcile.
- M4 4.8 worked example writes hdb_price_net.onnx into cwd; data loader only works from repo root.
- M5 numbers to reconcile (pages vs textbook.md vs slides): MUTAG test acc GCN 68.4% / GAT 71.1% / GIN 73.7%; 5.4 position task
  100% with PE vs 60.8% without (slides say "~59%"); GAN mode-coverage table; continuous PPO; churn DQN 11.0 / 8.0
  (textbook says 6.7 vs 8.7 "always call") — recompute once and align.
- torch 2.12 on Apple MPS: 2-layer nn.LSTM with dropout=0.1 trains badly (val MSE 0.15 vs 0.011 CPU / 0.012 MPS no-dropout)
  → check ex_4/03 and ex_4/05 LSTM configs on MPS (S7); upstream torch note.

## X-NUM reconciliation status (2026-10-05) — done without full suite
- M1 hourly taxi demand: 1,917–2,054 (ex_8-cleaned, 47,547 rows) — fixed in textbook.md (was 1,743–1,893).
- M4 equal-blend clustered AUC: 0.97 (code 0.9748) — fixed in textbook.md (was 0.98); matches lesson page.
- M5 5.4 no-positional-encoding: measured 60.8% (majority rate) — fixed in lesson slides (was ~59%).
- M5 churn RL fixed policies: never ≈ -2.5, always-call ≈ 10.4, random ≈ -0.2, discount ≈ -4.9; DQN ≈ 11.0 —
  fixed in textbook.md (was 8.7/6.7); DQN now beats the best heuristic (lesson page already teaches).
- M2 odds definition: p/(1-p) is the ODDS, not the "odds ratio" — fixed in textbook.md.
- Verified consistent (no change): M2 11,554/arm, CUPED 49%, SRM 40/35/15/10 (variant_c), conversion 19.1% vs 24.8%,
  R² 0.828, +9,096/sqm, 3,536 rows; M3 t* = 0.130, DI ≥0.97 race/gender & 0.26 age (unweighted) vs 0.94/0.96
  (class-weighted); M5 MUTAG GCN 68.4/GAT 71.1/GIN 73.7.
Still pending the full-suite run (trestle): recompute every cross-artefact number once from the canonical cleaned
output and confirm deck/textbook/lesson pages/exercises agree (grep each handoff file for numbers quoted in 2+ artefacts).

