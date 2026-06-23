# Observability Portal Design Sketch

Companion to **ADR-0058**, **kailash-py epic**, and **kailash-rs epic**.

## What the user sees

A TensorBoard-class web viewer that lives inside the Kailash platform, not beside it. Opens to a **Runs** page (left sidebar) + **Lens panel** (main) + **Compare strip** (bottom).

```
┌─────────────────────────────────────────────────────────────────┐
│ Observatory                                    [Runs] [Compare] │
├───────────┬─────────────────────────────────────────────────────┤
│ run_2601a │ Training Diagnostics — run_2601a                    │
│ run_2601b │ ┌──────────────┬──────────────┬──────────────────┐  │
│ run_2602a │ │ Loss curves  │ Gradient flow│ Activation stats │  │
│ run_2602b │ │              │              │                  │  │
│ ─────────  │ ├──────────────┼──────────────┼──────────────────┤  │
│ LIVE:     │ │ Dead neurons │ LR vs loss   │ Training dash    │  │
│ run_2603  │ │              │              │                  │  │
│ ●●●       │ └──────────────┴──────────────┴──────────────────┘  │
│           │                                                     │
│ [filter…] │ Report: ⚠ Vanishing gradients in layer conv2.0     │
│           │         ⚠ 45% dead neurons in relu3                │
├───────────┴─────────────────────────────────────────────────────┤
│ Compare: [run_2601a] vs [run_2601b] ─ loss delta +0.02          │
└─────────────────────────────────────────────────────────────────┘
```

Left sidebar: runs (searchable, filterable by tag / lens / date). Green pulse = live. Main pane: active lens's dashboard (`plot_training_dashboard()` etc). Compare strip: pin 2-N runs, overlaid plots across every lens.

## Lens panel catalogue

Each lens contributes one dashboard panel + one report card. All panels are Plotly.js (py renders), with a per-lens fallback to the Rust `HtmlReport`-style inline serialization.

| Lens                | Panel                                                                                      | Report card                               |
| ------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------- |
| TrainingDiagnostics | 2x3 grid (loss, grad flow, act stats, dead neurons, LR sweep, dashboard)                   | Severity-tagged findings + Rx suggestions |
| LLMJudgeEngine      | 2x2 (judge scores histogram, faithfulness, refusal bars, score by criteria)                | Judge confidence + over-refusal flags     |
| TextQualityMetrics  | Bar chart family (ROUGE / BLEU / BERTScore)                                                | Outlier samples table                     |
| AttentionSaliency   | Attention heatmap + logit-lens bar                                                         | Probe accuracy per layer                  |
| RetrievalEvaluator  | 2x2 (recall curve, context-util histogram, faithfulness vs context, retriever leaderboard) | Top-K drift per retriever                 |
| AgentTracer         | Timeline (tool calls + cumulative cost) + run comparison                                   | Loop detection + budget burn              |
| AlignmentMonitor    | 2x2 (reward curve, KL curve, win-rate, hack-scan scatter)                                  | Reward-hacking alerts (z-score > 2.5)     |
| GovernanceAuditor   | 2x2 (verdicts over time, budget bars, drill heatmap, chain timeline)                       | Chain integrity + budget breaches         |

## Data model

### Run

A **run** is a single training session / evaluation pass / agent execution / governance drill. Identified by `run_id`. Carries metadata (model name, dataset, seed, git_sha, tags) + per-lens event streams.

```python
@dataclass
class Run:
    run_id: str
    started_at: datetime
    ended_at: datetime | None      # None = live
    status: Literal["live", "complete", "failed"]
    model: str | None
    dataset: str | None
    tags: list[str]
    git_sha: str | None
    lenses_present: list[str]      # e.g. ["training_diagnostics", "llm_judge"]
    schema_versions: dict[str, int]
```

### Event stream (per lens, per run)

JSONL file. One event per line. Schema versioned.

```jsonl
{"schema": "TraceEvent/1", "ts": 1776442100.0, "run_id": "run_2603", "kind": "tool_call", "tool": "search", "args": {"q": "..."}, "latency_ms": 120, "cost_usd": 0.0003}
{"schema": "TraceEvent/1", "ts": 1776442101.0, "run_id": "run_2603", "kind": "response", "tokens_in": 412, "tokens_out": 89, "cost_usd": 0.0018}
```

### Experiment (cluster of runs)

Runs can be grouped into **experiments** — named clusters that share a comparison baseline (e.g. "fashion-mnist-ablation" contains 5 runs varying LR).

```python
@dataclass
class Experiment:
    experiment_id: str
    name: str
    runs: list[str]              # run_ids
    baseline_run: str | None     # for diff views
```

## Storage

**Python side**:

- `ExperimentTracker` SQLite (existing engine) — extended with a `lens` dimension on the metrics table. Run metadata already fits.
- JSONL event files in `observatory/{experiment_id}/{run_id}/{lens}.jsonl`. Configurable root via `KAILASH_OBSERVATORY_ROOT`.
- Plotly figures never persisted — always reconstituted from events.

**Rust side**:

- SQLite via existing `LocalTrackerBackend` — same dimension extension.
- JSONL files via existing `FileObservationSink` pattern (0o600 perms, append-only).
- HTML snapshots via `HtmlReport` pattern for offline viewing.

Both producers write to a shared `observatory/` root; portal consumes either.

## Transport — live vs static

**Live following** — WebSocket (py, extends existing `PerformanceDashboard`) / Server-Sent Events (rs, Axum + Leptos). New event → appends to in-memory buffer → pushes to open portal sessions. <1s refresh target (matches existing `kaizen.monitoring.PerformanceDashboard` spec).

**Static viewing** — REST `GET /runs/{run_id}/{lens}` returns the full JSONL + reconstituted figures. No backend compute needed for historic runs; the portal UI renders Plotly from cached JSONL.

**Multi-run comparison** — portal pins 2-N runs; for each lens, overlays equivalent metrics (losses on one axis, multiple series). Uses `plot_*()` signatures that accept a list of runs when supported; falls back to side-by-side for incomparable panels.

## Portal package layout

**Python**: extend `kaizen.monitoring.PerformanceDashboard`.

```
kaizen/monitoring/
  dashboard.py                 # existing FastAPI+WS+Plotly.js
  observability_panels.py      # NEW — lens renderers (accept Run + lens name, return HTML/JSON)
  observability_routes.py      # NEW — /runs, /experiments, /compare REST endpoints
  websocket_live.py            # NEW — live-follow WS channel multiplexing
```

**Rust**: new `kailash-observatory` crate.

```
crates/kailash-observatory/
  Cargo.toml                   # features: portal, rag-eval, llm-judge
  src/
    lib.rs
    facade/                    # Observatory composition
      mod.rs
      observatory.rs
    ingest/                    # JSONL readers, OTel span consumer
      mod.rs
      jsonl_reader.rs
      otel_consumer.rs
    portal/                    # Leptos SPA
      mod.rs
      app.rs
      panels/
        training_diagnostics.rs (stub — py-only v1)
        agent_tracer.rs
        retrieval_evaluator.rs
        governance_auditor.rs
        ...
      ssr.rs                   # Axum + SSE for live-following
  schemas/                     # JSONL schema fixtures (golden-file tests)
```

## Security / multi-tenancy

Observatory is a platform service; tenant isolation mandatory (`rules/tenant-isolation.md`):

- Cache keys: `observatory:v1:{tenant_id}:{run_id}:{lens}`
- Audit rows: `tenant_id` column indexed
- Portal auth: pluggable session middleware; defaults to deny-all without config

## Operational defaults

- `KAILASH_OBSERVATORY_ROOT=./observatory` — where JSONL events land
- `KAILASH_OBSERVATORY_RETENTION_DAYS=90` — auto-prune via a background task
- `KAILASH_OBSERVATORY_MAX_RUNS_LIVE=10` — concurrent live-follow channels
- Tiered storage: hot (last 7 days, SQLite-indexed), warm (compressed JSONL), cold (S3-compatible object storage via extras)

## Migration / interop

- **From MLFP `LLMObservatory`** — the facade maps directly; MLFP imports `from kailash_observatory import Observatory` and deletes its local copy.
- **From kailash-kaizen PerformanceDashboard** — existing dashboard remains live; observability panels are additive.
- **From TensorBoard logs** — out-of-scope v1; a community contribution could add a `TensorboardAdapter` that converts `events.out.tfevents.*` to JSONL.

## What's NOT in v1

- Hyperparameter-sweep visualization (deferred to Phase 4)
- Model-graph rendering (needs torch.fx / ONNX parsing — separate effort)
- Collaborative annotations (future, multi-user mode)
- Alerting integrations beyond what kaizen.monitoring already has (Email/Slack/webhook)
- Embedded notebook widget (future; JSONL contract makes this straightforward later)

## Success criteria (portal)

- [ ] A student running `DLDiagnostics` in MLFP sees their training live in the portal with zero extra configuration
- [ ] A developer running `LLMJudgeEngine` can load yesterday's evaluation run and compare against today's
- [ ] A governance engineer can inspect the audit chain verification for a specific subject across 30 runs in under 10 seconds
- [ ] Rust producers and Python producers appear in the same portal without translation
- [ ] Adding a new lens to the portal requires only: a Plotly renderer function, a panel layout spec, and a JSONL schema — no portal-core changes
