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
