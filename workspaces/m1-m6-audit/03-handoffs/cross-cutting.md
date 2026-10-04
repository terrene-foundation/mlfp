# Cross-cutting issues (span modules; owned by integration/S6, not a single module shard)

## X-PP: kailash-ml PreprocessingPipeline.setup() fits before it splits (data leakage)
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
