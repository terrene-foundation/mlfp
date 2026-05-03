# [Epic] Observability-first kailash-rs — data-contract parity with kailash-py + observatory crate

## TL;DR

Cross-SDK companion to **kailash-py#TBD**. Extend kailash-rs's existing observability floor (`kailash-ml::engine`, `kailash-kaizen::observability`, `kailash-pact::observation`, `kailash-core::telemetry`) with diagnostics engines that share **JSONL event-schema parity** with their Python counterparts. New `kailash-observatory` crate hosts the portal (Leptos) and the Observatory facade.

**Honest scoping**: three of the seven py engines have no coherent Rust port today — `kailash-rs` is ndarray-based classical ML plus llama.cpp serving, with no autodiff framework. Those engines defer to a future `burn`/`candle` adoption. This epic ships the four engines that Rust can support today at platform quality.

Paired with **ADR-0058** (cross-SDK convention: every package ships a `diagnostics` submodule).

## Motivation

The Rust floor is strong for data collection: `ExperimentTracker`, `DriftMonitor`, `ModelVisualizer`, `kailash-ml-explorer`, `kaizen::observability`, `kailash-pact::observation`, `kailash-core::telemetry` (OTel + OTLP/Jaeger/Zipkin). What's missing:

1. **No unified diagnostics API** — each crate has its own observation shape.
2. **No cross-SDK data contract** — py and rs engines can't share a portal because their event schemas diverge.
3. **No portal UI** — all modules are pure-data, rendering is downstream's problem.
4. **ML engines not wired to OpenTelemetry** — `TracingConfig` exists but engines use `tracing::debug!` ad-hoc, not instrumentation spans.

The kailash-py epic proposes 7 diagnostics engines filling gaps identified in Terrene Foundation's MLFP course (7,267 LOC of starter code). For Rust, the structural fix is the same (ADR-0058 convention) but the engine set must honestly reflect what the Rust ML stack can do.

## Rust reality — what ships v1, what defers

| Engine                | Rust disposition v1                                                                                  | Reason                                                               |
| --------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `TrainingDiagnostics` | **DEFER** to Rust autodiff maturity                                                                  | No PyTorch-equivalent in workspace; ndarray has no gradient hooks    |
| `LLMJudgeEngine`      | **PARTIAL** — judge-backend via kailash-align-serving (llama.cpp), py-side bridge for deepeval/ragas | Rust has the llama.cpp wiring; HF eval ecosystem is py-only          |
| `TextQualityMetrics`  | **YES** — pure-data metrics (ROUGE, BLEU, BERTScore can wrap `rust-bert` or remain Python-called)    | Some metrics work in pure Rust; bert-score needs model inference     |
| `AttentionSaliency`   | **DEFER**                                                                                            | No HF-transformers Rust port with hook surface                       |
| `RetrievalEvaluator`  | **YES** — pure-data (recall@k, precision@k, MRR, nDCG, coverage)                                     | Clean Rust API; compose with llama.cpp-based judge via JSONL         |
| `AgentTracer`         | **YES** — extends `kailash-kaizen::observability`                                                    | Trace schema fits existing `MetricsCollector`/`LogAggregator`        |
| `AlignmentMonitor`    | **DEFER**                                                                                            | kailash-align-serving is inference-only, no training loop to observe |
| `GovernanceAuditor`   | **YES** — extends `kailash-pact::observation`                                                        | `ObservationSink` trait already matches the audit pattern            |
| `Observatory` facade  | **YES** — consumes JSONL event streams from both Rust and Python producers                           | Data-contract parity makes this trivial                              |
| `Portal` (Leptos web) | **YES** — new `kailash-observatory` crate                                                            | No existing portal; clean-slate Leptos/Yew app                       |

**Parity rule**: data-contract (JSONL schemas) parity, NOT class-hierarchy parity. Rust types follow Rust idioms; schemas are identical.

## Deliverables

### ADR

- [ ] **ADR mirror of kailash-py ADR-0058** under `docs/adr/` (kailash-rs numbering convention). Same decision, Rust-specific implementation notes.

### Engines (Phase 1 — kailash-rs v2.0)

- [ ] `kailash_kaizen::observability::agent_tracer::AgentTracer` — wraps existing `MetricsCollector` + `LogAggregator` with trace-event semantics. Emits `TraceEvent` JSONL (schema-parity with py). Tool-use capture via `AgentExecutor` hooks.
- [ ] `kailash_ml::engines::retrieval_evaluator::RetrievalEvaluator` — trait + in-memory impl. `recall_at_k`, `precision_at_k`, `ndcg_at_k`, `mrr`, `context_utilisation`. Pure-data, serde-serializable, produces `RetrievalMetric` JSONL events.
- [ ] `kailash_pact::diagnostics::governance_auditor::GovernanceAuditor` — extends existing `ObservationSink` with audit-chain verification (SHA-256 prev_hash, matches Python `pact.diagnostics` behavior), budget consumption queries, negative-drill execution.

### Engines (Phase 2 — kailash-rs v2.1)

- [ ] `kailash_ml::engines::llm_judge::LLMJudgeEngine` — trait abstracting judge backends. Concrete impls: `LlamaCppJudge` (wraps `kailash-align-serving`), `PyBridgeJudge` (calls out to py via JSONL IPC for deepeval-backed judges). Emits `JudgeVerdict` JSONL (schema-parity with py).
- [ ] `kailash_ml::engines::text_quality::TextQualityMetrics` — Rust-native ROUGE, BLEU, exact-match, F1. BERTScore delegates to py bridge or `rust-bert` as an optional feature flag.

### Observatory + portal (Phase 3 — kailash-rs v2.2)

- [ ] `kailash-observatory` — new workspace crate. Three sub-modules:
  - `kailash_observatory::facade::Observatory` — composes available lenses (trait-object registry).
  - `kailash_observatory::ingest` — JSONL event-stream consumer; reads from files, Kafka, or `DomainEventBus`.
  - `kailash_observatory::portal` — Leptos SPA; per-lens panels; multi-run comparison; live-following via server-sent events.
- [ ] Extend `kailash-ml-explorer::HtmlReport` pattern: each lens optionally exports a self-contained HTML snapshot (Plotly.js inline, matches existing `DataExplorer.render()` precedent).

### Polish (Phase 4 — kailash-rs v2.0 → 2.3)

- [ ] Wire OpenTelemetry spans into all ML engines: `TrainingRun::record_step`, `DriftMonitor::check`, `ModelVisualizer::evaluate` all emit spans via existing `TracingConfig`. `lens`, `run_id`, `schema_version` attributes.
- [ ] Add `metrics` crate (Prometheus exporter) to `kailash-core` — currently missing; observability-first convention requires it.
- [ ] CI parity check: every public engine in `kailash-ml`, `kailash-kaizen`, `kailash-pact` has a diagnostics counterpart OR a documented exemption.

## Package-boundary map (Rust)

| Engine             | Crate                               | Module                                   | Feature flag           |
| ------------------ | ----------------------------------- | ---------------------------------------- | ---------------------- |
| AgentTracer        | kailash-kaizen                      | `observability::agent_tracer`            | base                   |
| RetrievalEvaluator | kailash-ml                          | `engines::retrieval_evaluator`           | `rag-eval`             |
| GovernanceAuditor  | kailash-pact                        | `diagnostics::governance_auditor`        | base                   |
| LLMJudgeEngine     | kailash-ml + kailash-align-serving  | `engines::llm_judge` + `llama_cpp_judge` | `llm-judge`            |
| TextQualityMetrics | kailash-ml                          | `engines::text_quality`                  | `text-quality`         |
| Observatory facade | kailash-observatory (**new crate**) | `facade::Observatory`                    | per-lens feature flags |
| Portal             | kailash-observatory                 | `portal` (Leptos SPA)                    | `portal`               |

## Rust API conventions

Derived from existing kailash-rs patterns:

1. **Trait-based engines**, not concrete structs. Example:
   ```rust
   pub trait DiagnosticsEngine: Send + Sync {
       type Event: Serialize + DeserializeOwned;
       fn record(&self, event: Self::Event) -> MlResult<()>;
       fn export_jsonl(&self) -> MlResult<String>;
       fn run_ids(&self) -> Vec<String>;
   }
   ```
2. **Pluggable backends** — every engine has at least `InMemory*` and `File*` impls, matching the `TrackerBackend` / `RegistryBackend` / `ObservationSink` precedent.
3. **Serde-driven serialization** — every event type derives `Serialize` + `Deserialize`. JSONL is the wire format.
4. **Lock-free atomics for hot-path metrics** — matches existing `MetricsCollector` in `kailash-kaizen`.
5. **OpenTelemetry via `tracing` crate** — `#[instrument]` macro on public engine methods; attributes include `lens`, `run_id`, `schema_version`.
6. **Optional feature flags** — heavy deps (`ragas` via py bridge, `rust-bert`, Leptos for portal) gated on Cargo features, never pulled into base.

## Cross-SDK data-contract parity

JSONL schemas are the bridge between Python producers and Rust producers. Schema files in `crates/kailash-observatory/schemas/` (mirror `kailash-py/docs/schemas/`). Every schema has a `schema_version` field; version bumps are minor revisions of the owning engine.

Schemas to define in this epic (shared with kailash-py epic):

- `TraceEvent` — agent trace record (ts, run_id, kind, tool, args, result, cost, latency, tokens)
- `JudgeVerdict` — LLM-judge output (score, rationale, criteria, judge_model, mode, latency)
- `DriftReport` — drift check (feature, metric_type, value, threshold, violated)
- `GradientSnapshot` — per-layer gradient state (batch, layer, grad_norm, grad_rms, update_ratio) — **py-emits-only v1**
- `ActivationSnapshot` — per-layer activation state — **py-emits-only v1**
- `AuditEntry` — governance audit row (ts, subject, action, verdict, reason, hash, prev_hash)
- `RetrievalMetric` — RAG metric (query_id, recall_at_k, precision_at_k, mrr, ndcg, k)
- `AlignmentSample` — alignment KL/reward sample — **py-emits-only v1**

Schema test suite: golden JSONL files in both repos; round-trip validation gates the CI.

## Dependencies — Cargo features, not base deps

```toml
[features]
default = []
rag-eval = []
text-quality = []
llm-judge = ["kailash-align-serving"]
llm-judge-pybridge = ["llm-judge", "pyo3"]
llm-judge-rust-bert = ["llm-judge", "rust-bert"]
portal = ["leptos", "axum", "tower"]
prometheus = ["metrics", "metrics-exporter-prometheus"]
```

Missing features → loud compile-time error via `#[cfg(not(feature = "X"))]` stubs that `compile_error!("enable feature 'X' with cargo install --features X")`.

## OpenTelemetry wiring

Existing `kailash_core::telemetry::TracingConfig` is configured but not consumed by ML engines. This epic wires it:

```rust
use tracing::instrument;

impl ExperimentTracker {
    #[instrument(skip(self, metrics), fields(lens = "experiment_tracker", run_id = %run_id, schema_version = 1))]
    pub fn record_step(&self, run_id: &str, step: u64, metrics: &BTreeMap<String, f64>) -> MlResult<()> {
        // ...
    }
}
```

Every public method on every diagnostics engine MUST be instrumented. The observatory portal consumes OTel spans (via Jaeger/Tempo backend) as its secondary data source, alongside JSONL files.

## Foundation independence

Same as py epic: no references to commercial MLOps platforms. Architectural references permitted for Apache 2.0 OSS tools. Rust ecosystem references (burn, candle, tch-rs, rust-bert, Leptos, Yew, egui, Tauri) are all OSS.

## Risks

1. **Rust ML stack evolution** — `burn` and `candle` mature on different timelines; TrainingDiagnostics deferral may need revisiting mid-epic. **Mitigation**: document the deferral clearly; re-evaluate at v2.1.
2. **py-bridge complexity** — pyo3-backed judge delegation is non-trivial. **Mitigation**: `LlamaCppJudge` ships first as pure-Rust; `PyBridgeJudge` is a v2.1 addition.
3. **Portal stack churn** — Leptos is young; Rust SPA frameworks shift. **Mitigation**: keep portal as a separate crate; swap frameworks without touching engines.
4. **Cross-SDK schema drift** — py schemas evolve; rs lags. **Mitigation**: golden-file tests that FAIL the build when py or rs emit unrecognized JSONL.
5. **Workspace bloat** — new `kailash-observatory` crate adds compile time. **Mitigation**: keep it as a peer crate, not a workspace dep for non-observability users.

## Acceptance criteria (epic-level)

- [ ] ADR mirror merged
- [ ] All Phase 1 engines shipped with Tier 1 (unit) + Tier 2 (real-infra) tests
- [ ] All Phase 2 engines shipped with tests
- [ ] `kailash-observatory` crate published to crates.io
- [ ] Portal renders JSONL from both py and rs producers
- [ ] OTel wiring across all ML engines verified (spans visible in Jaeger)
- [ ] Schema parity golden tests green against kailash-py
- [ ] Cross-SDK inspection rule (`rules/cross-sdk-inspection.md`) updated with JSONL-schema parity clause

## Open questions for maintainers

1. **Observatory crate location** — standalone in `kailash-rs` workspace (recommended), or a separate repo? Recommended: workspace for now; promote to own repo if it grows.
2. **Portal framework** — Leptos (recommended: SSR-friendly, ergonomic), Yew, or Dioxus? All are viable; Leptos has the best SSR story for live-following.
3. **py-bridge mechanism** — pyo3 embedded, subprocess JSONL, or gRPC? pyo3 is tightest but heaviest compile; subprocess is simplest. Recommended: subprocess v1, pyo3 for performance-critical paths in v2.
4. **Rust autodiff timing** — accept deferring TrainingDiagnostics / AttentionSaliency / AlignmentMonitor, or block epic pending burn/candle maturity? Recommended: accept the defer; document clearly.
5. **OTel wiring scope** — every engine method, or only hot-path methods? Recommended: every public method; `#[instrument]` is near-zero cost when no collector is attached.

## Source material

- **MLFP starter code** (Python, Apache 2.0): `courses/mlfp/shared/mlfp05/diagnostics.py` + `courses/mlfp/shared/mlfp06/diagnostics/` — reference implementations, 7,267 LOC. JSONL schemas derived from these.
- **Rust floor** — existing code to extend: `crates/kailash-ml/src/engine/` (17 engines), `crates/kailash-kaizen/src/observability/`, `crates/kailash-pact/src/observation/`, `crates/kailash-core/src/telemetry.rs`, `crates/kailash-ml-explorer/src/report.rs`.
- **Red-team synthesis**: `courses/mlfp/workspaces/curriculum/drafts/observability-epic-redteam.md`
- **ADR-0058 draft**: `courses/mlfp/workspaces/curriculum/drafts/adr-observability-first.md`
- **Py companion epic**: `terrene-foundation/kailash-py#TBD`

## Cross-SDK coordination

Changes to JSONL schemas require coordinated commits across both repos per the existing `rules/cross-sdk-inspection.md` protocol. Schema test-harness is the primary guardrail; schema-version bumps require paired PRs.
