# Red-team synthesis — observability-first kailash proposal

Audit inputs: MLFP diagnostics inventory (7,267 LOC / 12 modules), kailash-py observability floor (17 ML engines + kaizen ObservabilityManager suite), kailash-rs observability floor (trait-based data-out primitives, no autodiff).

## 1. Scope reality check

### Initial framing (DISCOVERY)

> "5 helpers should upstream, 4,800 LOC."

### Expanded framing (user directive)

> "All 7 diagnostics included, cross-SDK, TensorBoard-class portal."

### Actual delivery surface

| Item                | Package                             | LOC     | Rust parity?                                     |
| ------------------- | ----------------------------------- | ------- | ------------------------------------------------ |
| TrainingDiagnostics | kailash-ml                          | 1,679   | **NO** (no autodiff)                             |
| LLMJudgeEngine      | kailash-ml + kaizen judge           | 615+435 | Partial (llama.cpp side)                         |
| AttentionExplorer   | kailash-ml (extend model_explainer) | 529     | **NO** (no HF transformers in rs)                |
| RAGDiagnostics      | kailash-ml                          | 705     | Partial (data-side works, judge needs py bridge) |
| AgentTracer         | kailash-kaizen                      | 668+360 | **YES** (kaizen observability exists)            |
| AlignmentMonitor    | kailash-align.diagnostics           | 649     | **NO** (rs align is serving-only, no training)   |
| GovernanceAuditor   | kailash-pact.diagnostics            | 716     | **YES** (pact observation exists)                |
| Observatory facade  | new kailash-observatory             | 538     | **YES** (Rust JSONL composition)                 |
| Portal UI           | new package                         | ~3-5k   | SDK-parallel                                     |
| Shared support      | \_plots / \_judges / \_traces       | 905     | \_traces yes, \_plots no, \_judges py-only       |

**Totals**: ~7,800 LOC existing code to upstream + portal that doesn't exist yet.

**Red-team verdict**: This is a **kailash-ml 1.0 / kailash-rs 2.0 milestone**, not a 0.13. Epic must honestly scope it that way; maintainers will reject "small feature add" framing.

## 2. Naming — medical metaphors don't ship

MLFP's naming is pedagogical: Stethoscope / Blood Test / X-Ray / ECG / Flight Recorder. That lands in a course; it does not land in an SDK public API. Proposal:

| MLFP name                   | SDK name (proposed)                                | Rationale                                                              |
| --------------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| DLDiagnostics               | `TrainingDiagnostics`                              | Follows `TrainingPipeline` sibling pattern                             |
| LLMDiagnostics (output)     | `LLMJudgeEngine` + `TextQualityMetrics`            | Separates judge-as-a-service from classical metrics                    |
| InterpretabilityDiagnostics | Extend `ModelExplainer` with `explain_attention()` | SHAP/LIME + attention are the same engine                              |
| RAGDiagnostics              | `RetrievalEvaluator`                               | Parallel to `AlignmentEvaluator` which already exists in kailash-align |
| AgentDiagnostics            | `AgentTracer`                                      | Kaizen owns traces; `tracer` matches the `tracing` crate vocabulary    |
| AlignmentDiagnostics        | `AlignmentMonitor`                                 | Parallel to `DriftMonitor` (continuous-signal observer)                |
| GovernanceDiagnostics       | `GovernanceAuditor`                                | Pact is audit-shaped, "monitor" is wrong register                      |
| LLMObservatory (facade)     | `ObservabilityFacade` OR keep as `Observatory`     | Facade is pure composition; name survives                              |

MLFP can keep its pedagogical metaphor wrapper around the SDK names — that's legitimate pedagogy. But the public API ships with production names.

## 3. Cross-SDK parity is NOT 1:1

**Red-team finding that must land in the Rust epic:** kailash-rs has no autodiff framework (ndarray-based classical ML + llama.cpp serving). `TrainingDiagnostics`, `AttentionExplorer`, and `AlignmentMonitor` **cannot ship in Rust today**. Attempting parity forces either (a) adding burn/candle as a workspace dep (enormous scope creep), or (b) wrapping tch-rs (reintroduces PyTorch as a Rust dep — violates the pure-Rust ethos of the workspace).

**Honest Rust scope for v1**:

| Engine               | Rust disposition                                                                                   |
| -------------------- | -------------------------------------------------------------------------------------------------- |
| TrainingDiagnostics  | **DEFER** — wait for burn/candle autodiff maturity. Document as "py-only" in v1.                   |
| LLMJudgeEngine       | **PARTIAL** — kailash-align-serving can run a local judge via llama.cpp. Cross-SDK JSONL contract. |
| AttentionExplorer    | **DEFER** — no HF transformers Rust port with hook surface.                                        |
| RetrievalEvaluator   | **YES** — pure-data (recall@k, precision, nDCG, MRR) has a clean Rust API.                         |
| AgentTracer          | **YES** — extends existing kailash-kaizen observability.                                           |
| AlignmentMonitor     | **DEFER** — no training loop in kailash-align-serving. Revisit when training arrives.              |
| GovernanceAuditor    | **YES** — extends existing kailash-pact observation sink.                                          |
| Observatory + Portal | **YES** — data-contract parity via JSONL + portal consumes both SDKs' output.                      |

Parity rule for v1: **data-contract parity, not class-hierarchy parity.** JSONL schemas (TraceEvent, JudgeVerdict, DriftReport, etc.) MUST be identical across SDKs so the portal consumes either producer. Class names and shapes are allowed to diverge to fit each language's idioms.

## 4. Dependency weight — hard deps vs extras

Upstreaming the 7 lenses pulls in: `deepeval`, `ragas`, `transformer_lens`, `sae_lens`, `captum`, `trl`, `rouge_score`, `sacrebleu`, `bert_score`, `scikit-learn`. **None of these can be base deps of kailash-ml** — they would bloat the install for every user who wants a `DataExplorer`.

**Extras plan**:

```toml
[project.optional-dependencies]
diagnostics = ["plotly>=5.0"]  # base: upgrade ModelVisualizer users
training-diagnostics = ["kailash-ml[diagnostics]", "torch>=2.0"]
llm-judge = ["kailash-ml[diagnostics]", "kailash-kaizen", "deepeval", "rouge-score", "sacrebleu", "bert-score"]
interpretability = ["kailash-ml[diagnostics]", "transformer_lens", "sae_lens", "captum", "scikit-learn"]
rag-eval = ["kailash-ml[diagnostics]", "ragas", "kailash-kaizen"]
all-diagnostics = ["kailash-ml[training-diagnostics,llm-judge,interpretability,rag-eval]"]
```

Per-lens optional imports MUST follow the MLFP pattern: loud `ImportError` with the exact `pip install` command, never silent `None`.

## 5. Portal architecture — reuse, don't rebuild

**The floor that already exists** (kaizen.monitoring):

- `PerformanceDashboard` — FastAPI + WebSocket + Plotly.js real-time refresh <1s
- `MetricsCollector` — Prometheus-format, <1ms overhead, singleton
- `AnalyticsAggregator` — rolling windows, percentile distributions, anomaly detection
- `AlertManager` — Email / Slack / webhook thresholds

**Red-team verdict**: Do NOT build a new portal from scratch. Extend `kaizen.monitoring.PerformanceDashboard` with diagnostics-specific panels. The TensorBoard-class experience is:

1. Live training following → already works (WebSocket)
2. Run-to-run comparison → add multi-run selector
3. Per-lens panels → add Plotly.js renderers for gradient flow / attention heatmaps / judge scores / agent traces
4. Persistent storage → extend `ExperimentTracker`'s SQLite schema with a `lens` dimension

**Rust side**: extend `kailash-ml-explorer::HtmlReport` pattern for static artifacts; for live, a new `kailash-observatory` crate (Leptos/Yew web app) consumes the JSONL event contract. Ships later than py.

**Portal package boundary**:

- **Option A** (recommended): Extend `kailash-kaizen.monitoring` — the dashboard already lives there.
- Option B: New `kailash-observatory` package — cleaner but fragments the install.
- Option C: JSONL-only, external viewers — cheap but doesn't meet "TensorBoard-class" bar.

Going with A for py. For rs, new `kailash-observatory` crate because extending kaizen's dashboard would bloat the agent framework.

## 6. Observatory facade — where does it live?

The facade composes 7 lenses from 4 packages (ml, kaizen, align, pact). It can't cleanly live in any one.

**Options**:

1. **New `kailash-observatory` meta-package** — peer deps on all 4 SDKs. `pip install kailash-observatory` pulls what's installed, works with what's there. **Recommended.**
2. Live in kailash-ml with optional peer deps — cleaner import path (`from kailash_ml import Observatory`) but creates bidirectional dep ambiguity.
3. Skip upstream, keep in MLFP — concedes pedagogy/production naming split.

Going with 1: kailash-observatory is the portal + facade home. Separate crate/package in both SDKs.

## 7. Foundation independence — framing discipline

Under `rules/independence.md` + `rules/terrene-naming.md`, the epic MUST NOT reference W&B, MLflow, Neptune, Aim, ClearML, or any commercial MLOps platform. "TensorBoard" is acceptable — it's an OSS project (Apache 2.0, Google-origin but community-governed). "PyTorch Lightning" acceptable.

Positive framing: "observability-first ML engineering" + "every Foundation ML project gets diagnostics for free" + "extends kaizen.monitoring with ML-lifecycle-specific panels." Not "MLflow for kailash" or "TensorBoard replacement."

## 8. Package-boundary decisions (resolved)

| Engine              | Package                                        | Module                                                                  |
| ------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- |
| TrainingDiagnostics | kailash-ml                                     | `kailash_ml.engines.training_diagnostics`                               |
| LLMJudgeEngine      | kailash-ml (engine) + kailash-kaizen (wrapper) | `kailash_ml.engines.llm_judge` + `kaizen.evaluation.llm_judge_callable` |
| TextQualityMetrics  | kailash-ml                                     | `kailash_ml.engines.text_quality`                                       |
| ModelExplainer      | kailash-ml (EXTEND existing)                   | Add `.explain_attention()` method                                       |
| RetrievalEvaluator  | kailash-ml                                     | `kailash_ml.engines.retrieval_evaluator`                                |
| AgentTracer         | kailash-kaizen                                 | `kaizen.observability.agent_tracer`                                     |
| AlignmentMonitor    | kailash-align                                  | `kailash_align.diagnostics`                                             |
| GovernanceAuditor   | kailash-pact                                   | `pact.diagnostics`                                                      |
| Observatory facade  | **new: kailash-observatory**                   | `kailash_observatory.Observatory`                                       |
| Portal (py)         | kailash-kaizen (EXTEND PerformanceDashboard)   | `kaizen.monitoring.observability_panels`                                |
| Portal (rs)         | **new crate: kailash-observatory**             | Leptos web app                                                          |

Submodule convention: `diagnostics/` under each domain package (`kailash_align.diagnostics`, `pact.diagnostics`, `kaizen.observability.*`). ADR enshrines this as the rule.

## 9. Support-module disposition

| Support module | LOC | Upstream destination                                                                    |
| -------------- | --- | --------------------------------------------------------------------------------------- |
| `_judges.py`   | 435 | `kaizen.evaluation.llm_judge_callable` — this IS the Kaizen judge primitive             |
| `_plots.py`    | 110 | `kailash_ml.viz._theme` — Plotly theme/palette constants                                |
| `_traces.py`   | 360 | `kaizen.observability.trace_schema` — `TraceEvent` is THE kaizen trace schema candidate |

All three are genuinely reusable platform primitives, not course-specific.

## 10. Delivery plan (phased, honest)

**Phase 1 — Convention + low-risk engines (kailash-ml 0.13 / kailash-rs v2.0)**

- Master ADR lands: observability-first convention, submodule pattern, JSONL data contract
- `kailash_ml.engines.training_diagnostics` (py only, DLDiagnostics → new engine)
- `kaizen.observability.agent_tracer` + trace_schema (py + rs)
- `kailash_ml.engines.retrieval_evaluator` (py + rs, pure-data)
- `pact.diagnostics.GovernanceAuditor` (py + rs)

**Phase 2 — LLM lenses (kailash-ml 0.14)**

- `kailash_ml.engines.llm_judge` + `kaizen.evaluation.llm_judge_callable` (py, rs partial)
- `kailash_ml.engines.text_quality` (classical metrics, py)
- `ModelExplainer.explain_attention()` (py only, HF transformers dep)
- `kailash_align.diagnostics.AlignmentMonitor` (py only)

**Phase 3 — Observatory + Portal (kailash-ml 0.15)**

- New `kailash-observatory` package (py facade + Leptos rs portal)
- Extend `kaizen.monitoring.PerformanceDashboard` with lens panels
- MLFP removes local helpers, re-imports from SDK

**Phase 4 — Portal polish & parity (kailash-ml 1.0 / kailash-rs 2.0)**

- Multi-run comparison
- Persistent experiment store schema upgrade
- OpenTelemetry wiring across all lenses
- Full Rust-side portal

## 11. Risks flagged for the epic

1. **Maintainer rejection of submodule convention** — if kailash-py maintainers don't accept "every domain package ships diagnostics", the convention ADR dies and each engine negotiates placement separately. Mitigation: open the ADR FIRST and get agreement before any code PR.
2. **Rust scope creep via burn/candle pressure** — users will ask "why no Rust DLDiagnostics?" Mitigation: the epic explicitly punts and links to upstream burn/candle readiness tickets.
3. **Portal scope explosion** — "TensorBoard-class" can mean 10k LOC of JS. Mitigation: extend kaizen PerformanceDashboard (already has WS/Plotly wiring), don't build from zero.
4. **Dep-extras complexity** — users hitting "module not found" with unhelpful errors. Mitigation: every lens's `__init__.py` raises a loud ImportError with the exact `pip install kailash-ml[X]` command.
5. **Observatory facade fragmentation** — if `kailash-observatory` is optional, most users won't install it and the facade goes unused. Mitigation: MLFP imports it by default; doc it as "the recommended entry point."

## 12. What goes in each artifact

| Artifact                 | Purpose                                                                                      | Destination                                                                |
| ------------------------ | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **ADR**                  | Convention: observability-first, submodule pattern, JSONL contract                           | `docs/adr/NNNN-observability-first.md` in kailash-py; mirror in kailash-rs |
| **kailash-py epic**      | 7 engines + portal extension + observatory package, phased milestones                        | GitHub issue on terrene-foundation/kailash-py                              |
| **kailash-rs epic**      | 4 engines (data-side) + observatory crate + JSONL contract, defer 3 engines pending autodiff | GitHub issue on esperie/kailash-rs                                         |
| **Portal design sketch** | Extend PerformanceDashboard; lens-panel contract; run model                                  | `docs/design/` or appendix to epic                                         |

## Open questions for user before filing

1. **Phase 1 scope confirmation** — is the 4-engines-in-v1 Rust disposition acceptable, or is the expectation full parity now (which requires burn/candle adoption)?
2. **Observatory package** — is a new `kailash-observatory` package OK, or preference to house in kailash-ml?
3. **Portal** — extend kaizen.monitoring or new package? I'm recommending extend; confirm.
4. **Filing order** — ADR PR first (gets convention agreement) then epics, OR epics first (parallel review) then ADR? Recommended: ADR first.
5. **ADR channel** — file as a PR in `docs/adr/` (formal), or as a GitHub Discussion (lower friction for maintainer feedback)?
