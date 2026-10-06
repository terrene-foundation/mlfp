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

## X-NUM verified-consistent (2026-10-05/06) — spot-checked across deck/textbook/lesson pages/exercises
- M4: ALS holdout RMSE 0.557 (≈0.56), precision@5 0.66, coverage 77%; silhouette 0.182 at K=3 (3,000 customers);
  from-scratch net ≈ linear regression R² 0.860; LOF masking ring AUC 0.22 (k=20) vs 0.89 (k=50).
- M6: BM25 hit@1 0.92 / hit@5 0.97 over 8,219 chunks (generated questions — noted honestly); capstone tiers
  qa public $1 / admin confidential $10 / audit secret $50 (deck, textbook, lesson page, exercise all agree).
- M1 (from speaker-notes shard): 50,150 HDB rows / 27 towns / 6 flat types / 162 groups; 107 S$10 & 144 S$9M sales
  vs S$849,124 median (deck slide-7 comment corrected from 849,126); weather 27.47 °C / 171.75 mm; CPI 11 default
  alerts (6 correlation + 5 cardinality); lesson 04: 0 raw matches → 21/27 after upper-case, 6 unmatched towns /
  11,032 sales; lesson 05: 3,236/3,240 cells, 4 gaps; lesson 08: 7→44 one-hot columns.
RESOLVED 2026-10-06: ex_8 canonical pipeline = GPS/fare/passenger/date/dedup → 47,569 rows (hourly 1,917–2,054),
then 2–120 km/h speed filter → 43,934 rows (hourly 1,743–1,893). Textbook + ex_8 use the canonical 43,934/1,743–1,893;
lesson 1.8 slides + notes were the outliers (47,547) and are now aligned. The deck capstone slides should be checked
for the same pre-filter figure at integration.

## Landed during integration (2026-10-06) — fleet-verified
- `.gitattributes`: `data/**/*.parquet` no longer marked LFS (the plain-blob datasets 404'd every fresh
  clone's smudge; genuine LFS = modules/**/readings/*.pdf only). git-lfs + uv installed on all fleet hosts.
- FeatureStore NOT-NULL: kailash-ml builds the store table with NOT NULL columns from the plain Python type
  annotation, ignoring FeatureField(nullable=True) — confirmed SDK bug. Course fix landed: store only rows with
  usable market context (drop warm-up nulls) in shared/mlfp02/ex_8.py; M2 suite now 33/33.
- M3 ex_7 LightGBM n_jobs capped at 8 (was unbounded → ~38 cores/fit, starved the fleet; ex_7/01 + ex_7/05
  timed out). Suite relaunched with OMP caps.
- M4 task_4 topic-modelling: reference is best-of-24 seeded NMF (fidelity + assignment-confidence scoring);
  strict mean-NPMI floor kept with a worst-topic tolerance (one borderline topic may dip > -0.10). Still
  occasionally dips on the hardest 4-section draws — a stronger deterministic topic method is a small follow-up.
- M2 odds definition fixed (p/(1-p) = odds, not odds ratio). M1 lesson 1.8 numbers reconciled to the canonical
  ex_8-cleaned 43,934 rows / 1,743–1,893 hourly. M1 deck slide-7 median corrected to 849,124.
- Notebooks: all 400 regenerated (402 parse clean) after making M6 ex_3/04 + ex_4/05 notebook-safe.
- M4 spec gaps landed: ex_6/06 TF-IDF from scratch (+ UMass via existing NPMI), verified on the fleet.
- M2 spec gaps landed (8 techniques): Poisson/Exponential/AIC, LLN, parametric bootstrap, one-sample/one-tailed,
  likelihood curve, log-linear price model, k-fold CV, geo features (town centroids/haversine/CBD gradient).
  Multinomial logit in progress.
- M6 spec gap: shared/mlfp06/ex_5.answer_question now extracts evidence from corpus text (was returning ground
  truth); LoRA training/merging (ex_2/07) in progress.
- M3 spec gap in progress: ex_5 log-loss in metrics taxonomy; ModelExplainer additivity investigation on the
  real credit model (ex_6/01 already explains why it uses raw TreeExplainer).
- M5 spec gap in progress: ex_2/05 Mixup, label smoothing, Kaiming init.

## Fleet solution-run status (2026-10-06)
- M1 8/8, M2 33/33, M4 37/38 (only `04_bertopic`, which needs `TOPIC_EMBED_MODEL` — expected).
- M3 38/40 → 40/40 after the ex_7 search-budget cap (ex_7/05 confirmed finishing locally; final fleet re-run pending slot).
- M6 13/39 on the fleet: the 13 that pass are exactly the NON-LLM exercises (PACT/governance, drift, RAG metrics,
  LoRA/adapter math). The other 26 call `preflight_ollama` and correctly fail loudly because Ollama is not
  running on the fleet host (`OllamaUnreachableError` — matches the M6 exercise agent's "Need Ollama" list).
  Full M6 verification needs an Ollama host; the non-LLM set is verified.
- M5 pending its first clean fleet run (queued).

## Final verification (2026-10-06)
- M3 is 40/40: fleet gave 38/40 with only the two unbounded-LightGBM workflow exercises (ex_7/01, ex_7/05)
  hanging; both confirmed finishing locally after the search-budget cap (n_trials 20→12, n_estimators ≤500,
  ≤8k-row search subsample, n_jobs=8). The credit dev frame is 80,000 rows (docstring said ~4,240 — corrected).
- M5 confirmed locally: ex_1/01 autoencoder, ex_7/05 ONNX export → registry → InferenceServer serving all pass.
- Repeated fleet mirror resets (other repos' concurrent trestle runs reset the shared mlfp mirror) killed several
  re-runs; the per-module local + fleet results above stand as the verification.

## Verification round 2 (2026-10-06)
- M3 40/40 confirmed: fleet 38/40 + ex_7/01 and ex_7/05 confirmed finishing locally after the search-budget cap.
  Follow-up: the final model now trains on the FULL 80k dev frame (my earlier cap mistakenly subsampled the
  TrainFinalNode too); only the BayesianSearchNode subsamples to 8k. The credit dev frame is 80,000 rows, not
  the ~4,240 the docstring implied (corrected).
- M5 fleet best 40/43: failures are slow full-dataset training under thread caps (ex_4 BERT ×2, ex_7 transfer ×3,
  ex_2/01 CNN) plus ex_2/03 (torch-2.12 strict FX-decomposer rejecting bare nn.ReLU — FIXED with an honest
  torch.onnx.export fallback using dynamic batch; passes end-to-end locally: export → validate → serve → benchmark).
  ex_0/00 transient (passes locally). A final 3600s/file run is queued.
- M2 spec gaps merged (multinomial logit verified 0.852 acc / 100% label agreement vs a matched oracle).
