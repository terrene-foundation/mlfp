# [Epic] Observability-first kailash-ml — 7 diagnostics engines + observatory portal

## TL;DR

Make observability a first-class engine axis across `kailash-ml`, `kailash-kaizen`, `kailash-align`, and `kailash-pact`. Land seven diagnostics engines, a cross-package `Observatory` facade, and a TensorBoard-class portal (extending `kaizen.monitoring.PerformanceDashboard`). Paired with **ADR-0058** (convention: every Kailash package ships a `diagnostics` submodule) and cross-SDK epic **kailash-rs#TBD**.

Apache-2.0 starter code already exists — 7,267 LOC across 12 modules in the Terrene Foundation's ML Foundations for Professionals course (MLFP). This epic moves it upstream.

## Motivation

`kailash-ml` has 17 production engines and `kailash-kaizen` has a mature observability suite, but no convention that says diagnostics are a first-class engine axis. Every downstream Foundation ML project rebuilds its own during-training, LLM, RAG, agent, alignment, and governance diagnostics.

Evidence: MLFP carries **7,267 LOC across 12 diagnostics modules** — platform code hiding in a course repo. Every new Foundation project will rebuild the same helpers because the SDK has no convention forcing them upstream.

Root cause: the SDK ships execution primitives without observability siblings. The structural fix is a convention, not seven one-off feature requests. **ADR-0058** is the convention; this epic is the code.

## Deliverables

### ADR

- [ ] **ADR-0058** — Observability-first diagnostics convention (cross-SDK). Lands before any child engine PR; gets convention agreement before code review overhead starts.

### Engines (Phase 1 — kailash-ml 0.13)

- [ ] `kailash_ml.engines.training_diagnostics.TrainingDiagnostics` — PyTorch hook-based. Gradient flow per layer, dead-neuron tracking, activation saturation, loss-curve dynamics, LR range test, Grad-CAM. 5 polars DataFrames, 11 Plotly figure types, `.report()` auto-diagnosis. **Source**: `shared/mlfp05/diagnostics.py` (1,679 LOC). Python-only v1 (kailash-rs has no autodiff framework).
- [ ] `kaizen.observability.agent_tracer.AgentTracer` + `kaizen.observability.trace_schema.TraceEvent` — Agent run capture: tool use, loop detection, cost breakdown, timeline plot. Extends existing `kaizen.observability`. **Source**: `shared/mlfp06/diagnostics/agent.py` (668 LOC) + `shared/mlfp06/diagnostics/_traces.py` (360 LOC). Rust parity in kailash-rs#TBD.
- [ ] `kailash_ml.engines.retrieval_evaluator.RetrievalEvaluator` — RAG pipeline diagnostics: recall@k, precision@k, nDCG, MRR, context utilization, retriever leaderboard. 2 DataFrames, 2x2 dashboard figure. **Source**: `shared/mlfp06/diagnostics/retrieval.py` (705 LOC). Rust parity in kailash-rs#TBD (pure-data, clean port).
- [ ] `pact.diagnostics.governance_auditor.GovernanceAuditor` — PACT envelope audit: decision chain verification (SHA-256 prev_hash), budget consumption, negative drill execution, audit snapshot. **Source**: `shared/mlfp06/diagnostics/governance.py` (716 LOC). Rust parity in kailash-rs#TBD.

### Engines (Phase 2 — kailash-ml 0.14)

- [ ] `kailash_ml.engines.llm_judge.LLMJudgeEngine` + `kaizen.evaluation.llm_judge_callable.LLMJudgeCallable` — LLM-as-judge with Kaizen Delegate, position-swap bias mitigation, budget cap, faithfulness scoring, self-consistency detection. **Source**: `shared/mlfp06/diagnostics/output.py` (615 LOC) + `shared/mlfp06/diagnostics/_judges.py` (435 LOC).
- [ ] `kailash_ml.engines.text_quality.TextQualityMetrics` — Classical NLP metrics extracted from `LLMDiagnostics`: ROUGE, BLEU, BERTScore, perplexity. Pure-data, no LLM calls. Rust parity in kailash-rs#TBD.
- [ ] `kailash_ml.engines.model_explainer.ModelExplainer.explain_attention()` — Extend existing `ModelExplainer` with attention saliency, logit lens, linear probing, SAE feature extraction. `transformer_lens` + `sae_lens` + `captum` as optional extras. **Source**: `shared/mlfp06/diagnostics/interpretability.py` (529 LOC).
- [ ] `kailash_align.diagnostics.alignment_monitor.AlignmentMonitor` — Fine-tuning health: KL divergence (`trl` or closed-form fallback), reward margin, win rate, reward-hacking detection. **Source**: `shared/mlfp06/diagnostics/alignment.py` (649 LOC). Py-only v1 (kailash-align-serving is inference-only).

### Observatory + portal (Phase 3 — kailash-ml 0.15)

- [ ] `kailash-observatory` — new peer package; houses `Observatory` facade composing the 7 lenses with optional peer-dep resolution (only lenses whose extras are installed show up). **Source**: `shared/mlfp06/diagnostics/observatory.py` (538 LOC). Cross-SDK via JSONL.
- [ ] `kaizen.monitoring.observability_panels` — extend existing `PerformanceDashboard` (FastAPI + WebSocket + Plotly.js) with lens-specific panel renderers. Multi-run selector, live-following, per-lens drilldown.
- [ ] `kailash_ml.viz._theme` — promote `_plots.py` Plotly theme/palette constants (110 LOC) to a shared theme.

### Polish (Phase 4 — kailash-ml 1.0)

- [ ] Persistent experiment store schema upgrade: `ExperimentTracker` SQLite gains a `lens` dimension so runs can be filtered by diagnostics lens.
- [ ] OpenTelemetry wiring across all lenses (every `record_*` call emits a span with `run_id`, `lens`, `schema_version` attributes).
- [ ] Multi-run comparison UI in portal.
- [ ] CI gate: new engine PRs must touch `diagnostics/` or pass a documented exemption.

## Package-boundary map

| Engine              | Package                                        | Module                                                                  | Extras                    |
| ------------------- | ---------------------------------------------- | ----------------------------------------------------------------------- | ------------------------- |
| TrainingDiagnostics | kailash-ml                                     | `kailash_ml.engines.training_diagnostics`                               | `[training-diagnostics]`  |
| LLMJudgeEngine      | kailash-ml + kailash-kaizen                    | `kailash_ml.engines.llm_judge` / `kaizen.evaluation.llm_judge_callable` | `[llm-judge]`             |
| TextQualityMetrics  | kailash-ml                                     | `kailash_ml.engines.text_quality`                                       | `[llm-judge]`             |
| AttentionSaliency   | kailash-ml (extend `ModelExplainer`)           | `kailash_ml.engines.model_explainer` adds `.explain_attention()`        | `[interpretability]`      |
| RetrievalEvaluator  | kailash-ml                                     | `kailash_ml.engines.retrieval_evaluator`                                | `[rag-eval]`              |
| AgentTracer         | kailash-kaizen                                 | `kaizen.observability.agent_tracer`                                     | base                      |
| AlignmentMonitor    | kailash-align                                  | `kailash_align.diagnostics.alignment_monitor`                           | `[alignment-diagnostics]` |
| GovernanceAuditor   | kailash-pact                                   | `pact.diagnostics.governance_auditor`                                   | base                      |
| Observatory facade  | kailash-observatory (**new**)                  | `kailash_observatory.Observatory`                                       | per-lens optional deps    |
| Portal              | kailash-kaizen (extend `PerformanceDashboard`) | `kaizen.monitoring.observability_panels`                                | base                      |

## Optional-dependency extras

```toml
[project.optional-dependencies]
diagnostics = ["plotly>=5.0"]
training-diagnostics = ["kailash-ml[diagnostics]", "torch>=2.0"]
llm-judge = ["kailash-ml[diagnostics]", "kailash-kaizen", "deepeval", "rouge-score", "sacrebleu", "bert-score"]
interpretability = ["kailash-ml[diagnostics]", "transformer_lens", "sae_lens", "captum", "scikit-learn"]
rag-eval = ["kailash-ml[diagnostics]", "ragas", "kailash-kaizen"]
alignment-diagnostics = ["kailash-align", "trl"]
all-diagnostics = ["kailash-ml[training-diagnostics,llm-judge,interpretability,rag-eval,alignment-diagnostics]"]
```

Each engine's `__init__` MUST raise a loud, actionable `ImportError` when its extras are missing — no silent `None`, no fake data. Per `rules/zero-tolerance.md` Rule 2.

## API conventions (all engines)

Derived from MLFP's existing implementations. Every engine ships with:

1. **Context manager** — `with EngineName(...) as diag:` for auto-cleanup of hooks and resources.
2. **Record methods** — `record_batch(...)`, `record_epoch(...)`, `record_step(...)` for streaming state.
3. **Accessor methods** — `gradients_df()`, `batches_df()`, etc. returning Polars DataFrames.
4. **Plot methods** — `plot_*()` returning `go.Figure`. Uniform theme via `kailash_ml.viz._theme`.
5. **Report method** — `report() -> dict` with severity + auto-diagnosis + actionable suggestions.
6. **Observatory integration** — engine emits JSONL events to a configurable sink; Observatory consumes.
7. **run_id propagation** — every public method accepts optional `run_id` for trace correlation.
8. **OpenTelemetry spans** — `record_*` and `plot_*` emit spans with `lens` + `schema_version` attributes.

## Cross-SDK data contract

JSONL event schemas are the parity contract between `kailash-py` and `kailash-rs`. Schema files live at `docs/schemas/` (py) and `crates/kailash-observatory/schemas/` (rs). The portal consumes either producer without modification.

Schemas to define in this epic: `TraceEvent`, `JudgeVerdict`, `DriftReport`, `GradientSnapshot`, `ActivationSnapshot`, `AuditEntry`, `RetrievalMetric`, `AlignmentSample`.

## Migration (MLFP-side, once each engine lands)

Per-engine deletion and re-import in MLFP. Net MLFP reduction: ~7,267 LOC across Phases 1-3.

## Foundation independence

Per `rules/independence.md` + `rules/terrene-naming.md`: no references to W&B, MLflow, Neptune, Aim, ClearML, CometML, SageMaker, Vertex, or any commercial MLOps platform. Architectural references permitted: TensorBoard (Apache 2.0), PyTorch Lightning callbacks (Apache 2.0).

## Risks

1. **Convention rejection** — if maintainers don't accept the submodule convention, each engine negotiates placement separately. **Mitigation**: ADR-0058 lands first.
2. **Dep-extras confusion** — users installing `kailash-ml` and expecting all lenses to work. **Mitigation**: loud ImportErrors with exact `pip install` commands; README lens table.
3. **Portal scope creep** — "TensorBoard-class" is elastic. **Mitigation**: extend existing `PerformanceDashboard`; do NOT build a new UI framework.
4. **Observatory package discoverability** — if `kailash-observatory` is optional, most users won't find it. **Mitigation**: top-level mention in kailash-ml README; `from kailash_observatory import Observatory` as the recommended entry point.
5. **CI enforcement burden** — linter for "new engines must ship diagnostics". **Mitigation**: lightweight check in `pyproject.toml` hook, documented exemption mechanism.

## Acceptance criteria (epic-level)

- [ ] ADR-0058 merged
- [ ] All Phase 1 engines shipped with tests (Tier 1 unit + Tier 2 integration against real services where applicable)
- [ ] All Phase 2 engines shipped with tests
- [ ] `kailash-observatory` package published to PyPI
- [ ] Portal extension merged to `kaizen.monitoring`
- [ ] MLFP migration PRs merged (local helpers deleted, SDK imports in place)
- [ ] Cross-SDK data-contract parity verified against `kailash-rs#TBD`
- [ ] CI convention gate active

## Open questions for maintainers

1. **ADR channel** — file as a PR in `docs/adr/` (formal) or a GitHub Discussion (lower friction)? Lean toward ADR PR for a convention this load-bearing.
2. **Rust scope** — accept the "py-only for TrainingDiagnostics / AttentionSaliency / AlignmentMonitor" honest disposition, or block the epic pending burn/candle Rust ML maturity?
3. **Observatory package boundary** — new `kailash-observatory` peer package (recommended), or fold into kailash-ml as a submodule with optional peer deps?
4. **Portal package** — extend `kaizen.monitoring` (recommended, has the WebSocket infra) or new `kailash-observatory.portal`?
5. **CI enforcement shape** — block engine PRs without diagnostics, or warn-only with escalation to maintainer?

## Source material

- **Starter code**: `courses/mlfp/shared/mlfp05/diagnostics.py` + `courses/mlfp/shared/mlfp06/diagnostics/` (Apache 2.0, 7,267 LOC, polars-native, Plotly-based)
- **DISCOVERY entry**: `courses/mlfp/workspaces/curriculum/journal/.pending/1776442000000-0-DISCOVERY.md`
- **Red-team synthesis**: `courses/mlfp/workspaces/curriculum/drafts/observability-epic-redteam.md`
- **ADR draft**: `courses/mlfp/workspaces/curriculum/drafts/adr-observability-first.md`
- **Cross-SDK companion**: `kailash-rs#TBD`

## Linked ADRs

- ADR-0017 (py): Multi-Workflow API Architecture — precedent for cross-package convention
- ADR-017 (kaizen): Observability & Performance Monitoring — floor that this epic extends
- ADR-0058 (py, **this epic**): Observability-first Diagnostics Convention
