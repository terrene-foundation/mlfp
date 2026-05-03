# ADR-0058: Observability-First Diagnostics Convention (Cross-SDK)

## Status

Proposed — 2026-04-20

## Context

The Kailash platform ships seventeen ML engines in `kailash-ml`, a mature observability suite in `kailash-kaizen` (`ObservabilityManager`, `TracingManager`, `MetricsCollector`, `PerformanceDashboard`), an audit trail in `kailash-pact` (`McpAuditTrail`), and an evaluator in `kailash-align` (`AlignmentEvaluator`). The floor is real.

What is missing is a **convention** that says diagnostics are a first-class engine axis: every ML / agent / alignment / governance surface ships with an observability layer the day it lands. Today, each engine lands without one, and every downstream project rebuilds its own — Terrene Foundation's ML Foundations for Professionals (MLFP) course has accumulated **7,267 LOC across 12 diagnostics modules** filling the gap (training-time gradient/activation instruments, LLM-judge evaluation, attention saliency, RAG recall metrics, agent trace capture, alignment health, governance audit, a composing Observatory facade, and Plotly theme + judge-wrapper + trace-schema support modules).

Every new Foundation ML project will rebuild these same helpers because the SDK has no convention that forces them upstream. The cost compounds: maintenance burden on project teams, framework dilution for students and users, and missed contribution opportunities.

This ADR establishes the structural fix.

## Decision

**Observability is a first-class engine axis in Kailash, parallel to execution.** Every Kailash package MUST expose a `diagnostics` namespace alongside its primary responsibility. New engines MUST ship with at least one diagnostics hook-point in the same PR.

### Scope

This convention applies to both SDKs:

- **kailash-py** — packages: `kailash-ml`, `kailash-kaizen`, `kailash-align`, `kailash-pact`
- **kailash-rs** — crates: `kailash-ml`, `kailash-kaizen`, `kailash-align-serving`, `kailash-pact`

A new peer package `kailash-observatory` (py) + crate `kailash-observatory` (rs) hosts the cross-package Observatory facade and the portal UI.

### Submodule layout

Each package MUST expose diagnostics at a predictable path:

| Package             | Diagnostics location                                        | Module convention                                                                   |
| ------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| kailash-ml          | `kailash_ml.engines.*` (existing) + new diagnostics engines | `TrainingDiagnostics`, `LLMJudgeEngine`, `TextQualityMetrics`, `RetrievalEvaluator` |
| kailash-kaizen      | `kaizen.observability.*` (existing) + `kaizen.evaluation.*` | `AgentTracer`, `LLMJudgeCallable`, `TraceEvent` schema                              |
| kailash-align       | `kailash_align.diagnostics.*` (new submodule)               | `AlignmentMonitor`                                                                  |
| kailash-pact        | `pact.diagnostics.*` (new submodule)                        | `GovernanceAuditor`                                                                 |
| kailash-observatory | `kailash_observatory.*` (new peer package)                  | `Observatory` facade, `Portal` runtime                                              |

### Data-contract parity (cross-SDK)

Class shapes and names MAY diverge between Python and Rust to fit each language's idioms. **JSONL event schemas MUST NOT.** Every diagnostics engine emits a documented JSONL event stream (`TraceEvent`, `JudgeVerdict`, `DriftReport`, `AuditEntry`, etc.). Schemas are versioned with a `schema_version` field and specified in `docs/schemas/` (py) and `crates/kailash-observatory/schemas/` (rs).

The Observatory portal consumes these event streams from either producer; a Rust trainer emitting JSONL events to the same schema shows up in the same portal as a Python trainer.

### Visualization standards

- **Python**: Plotly is canonical (matches existing `ModelVisualizer`, `DataExplorer`, `PerformanceDashboard`). All `plot_*` methods return `go.Figure`.
- **Rust**: pure-data primitives (serde-serializable structs). HTML reports via the `kailash-ml-explorer::HtmlReport` pattern for static artifacts; the live portal is a Leptos web app in `kailash-observatory`.
- **Portal**: extends `kaizen.monitoring.PerformanceDashboard` (FastAPI + WebSocket + Plotly.js). One entry point per lens; multi-run comparison; live-following.

### Dependency policy

Heavy third-party dependencies MUST be declared as optional extras, never as base dependencies of `kailash-ml`:

```toml
[project.optional-dependencies]
diagnostics = ["plotly>=5.0"]
training-diagnostics = ["kailash-ml[diagnostics]", "torch>=2.0"]
llm-judge = ["kailash-ml[diagnostics]", "kailash-kaizen", "deepeval", "rouge-score", "sacrebleu", "bert-score"]
interpretability = ["kailash-ml[diagnostics]", "transformer_lens", "sae_lens", "captum", "scikit-learn"]
rag-eval = ["kailash-ml[diagnostics]", "ragas", "kailash-kaizen"]
all-diagnostics = ["kailash-ml[training-diagnostics,llm-judge,interpretability,rag-eval]"]
```

Each lens MUST raise a loud, actionable `ImportError` when its extras are missing — never fall back to silent `None`, never fabricate readings. The error message MUST include the exact `pip install` command.

### OpenTelemetry wiring

Every diagnostics engine MUST emit OpenTelemetry spans via the existing `TracingConfig` (py: ADR-017, rs: `kailash_core::telemetry`). Span attributes MUST include `run_id`, `lens` (e.g. `lens=training_diagnostics`), and `schema_version`. The kaizen `ObservabilityManager` already wires OTel; diagnostics engines reuse its infrastructure.

### Foundation independence

The convention does not reference, compare with, or design against any commercial MLOps platform (W&B, MLflow, Neptune, Aim, ClearML, CometML, etc.). Reference implementations cited where architecturally informative: TensorBoard (Apache 2.0, community-governed), PyTorch Lightning callbacks (Apache 2.0).

## Alternatives considered

1. **Diagnostics as a single monolithic package (`kailash-diagnostics`)** — rejected. Each domain's diagnostics are tightly coupled to that domain's data shapes (alignment KL divergence needs Align's reward tensors; governance audit needs Pact's envelope schema). A monolith forces every diagnostics user to pull every domain SDK.

2. **Diagnostics only in `kailash-ml`, every other package feeds it** — rejected. Forces unidirectional dependency `kailash-align → kailash-ml` which inverts the conceptual hierarchy (ML is a consumer of align outputs, not a parent).

3. **No convention, per-issue negotiation** — rejected. This is the status quo that produced 7,267 LOC of external course code filling the gap. Predicts that the next Foundation ML project builds the same thing again.

4. **Extend `kailash-kaizen.observability` to cover everything** — rejected. Kaizen's observability is agent-lifecycle-shaped (spans, tool calls, budgets). ML training diagnostics (gradient flow, dead neurons) need hook-based introspection that doesn't fit the span model; governance audit needs immutable append-only records with chain verification that doesn't fit the metrics model.

## Consequences

### Positive

1. **Net deletion of ~7,267 LOC from MLFP** over three phases as each engine lands upstream.
2. **Consistent diagnostics API across the platform** — students learn one pattern (`engine.track_*() / .record_*() / .report() / .plot_*()`) that applies everywhere.
3. **Portal as a platform service** — every Kailash project gets TensorBoard-class live training + evaluation viewing for free.
4. **Future-proof** — new engines are blocked from merging without diagnostics coverage, so the gap cannot reopen.
5. **Cross-SDK JSONL contract** lets Rust trainers and Python analyzers share tooling.

### Negative / trade-offs

1. **Rust has structural gaps v1** — no autodiff framework means no `TrainingDiagnostics`, no attention-saliency, no training-loop alignment monitor. The Rust epic honestly punts these to post-burn/candle.
2. **Observatory package adds an install layer** — `kailash-observatory` must be discoverable (top-level import, starred from kailash-ml README).
3. **Dependency-extras surface** — users hitting `pip install kailash-ml` and expecting `DLDiagnostics` to work will see ImportErrors. Mitigated by loud, actionable error messages.
4. **ADR enforcement burden** — convention requires CI gates on new engine PRs to check for diagnostics. Needs a lightweight linter.

## Rollout plan

Referenced in full detail in the paired epics (`kailash-py#TBD`, `kailash-rs#TBD`). Phased:

- **Phase 1 (kailash-ml 0.13, kailash-rs v2.0)** — convention ADR + 4 low-risk engines (RetrievalEvaluator py+rs, AgentTracer py+rs, GovernanceAuditor py+rs, TrainingDiagnostics py-only)
- **Phase 2 (kailash-ml 0.14)** — LLM lenses: LLMJudge, TextQualityMetrics, AttentionExplorer (extend ModelExplainer), AlignmentMonitor
- **Phase 3 (kailash-ml 0.15)** — `kailash-observatory` package + portal extension of PerformanceDashboard
- **Phase 4 (kailash-ml 1.0 / kailash-rs 2.0)** — polish, multi-run comparison, full Rust portal via Leptos

## Governance

- Convention enforcement: CI check in kailash-py (new engine PR must touch `diagnostics/` or pass a documented exemption)
- Schema versioning: every JSONL schema change requires a minor version bump of its engine
- Cross-SDK drift: existing `rules/cross-sdk-inspection.md` extended to include JSONL schema parity

## References

- MLFP DISCOVERY: `courses/mlfp/workspaces/curriculum/journal/.pending/1776442000000-0-DISCOVERY.md` (the 12-module inventory and migration rationale)
- MLFP source implementations: `courses/mlfp/shared/mlfp05/diagnostics.py` + `courses/mlfp/shared/mlfp06/diagnostics/` (Apache 2.0 starter code for each engine)
- Red-team synthesis: `courses/mlfp/workspaces/curriculum/drafts/observability-epic-redteam.md`
- Existing floor: `kailash-ml` engines (17), `kailash-kaizen.observability` (mature), `kailash-pact.observation` (append-only audit), `kailash-ml-explorer::HtmlReport` (Rust HTML+Plotly.js precedent)
- ADR-0017 (py) — Multi-Workflow API Architecture (precedent for cross-package convention)
