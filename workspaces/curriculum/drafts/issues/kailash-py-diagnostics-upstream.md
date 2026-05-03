# Offer: upstream MLFP's diagnostics helpers into kailash-ml / kaizen / align / pact

## Context

The Terrene Foundation's ML Foundations for Professionals (MLFP) course carries ~7,300 LOC of diagnostics helpers that are structurally platform code, not course pedagogy — polars-native, Plotly-based, Apache-2.0, framework-first (no raw OpenAI calls, routes through Kaizen Delegate).

Happy to upstream whichever ones you want. Opening this issue as a conversation-starter; will break out into per-helper PRs however you prefer.

## The helpers

| Helper                                   | Purpose                                                                                                                                               | LOC       | Source                                                                                                                                                                                                              |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DLDiagnostics`                          | PyTorch training instruments: gradient flow per layer, dead neurons, activation saturation, LR range test, Grad-CAM, auto-diagnosis                   | 1,679     | [`shared/mlfp05/diagnostics.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp05/diagnostics.py)                                                                                                 |
| `LLMDiagnostics` + `JudgeCallable`       | LLM-as-judge (Delegate-wrapped, position-swap bias mitigation, budget cap), faithfulness, self-consistency, ROUGE/BLEU/BERTScore, refusal calibration | 615 + 435 | [`output.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/output.py) + [`_judges.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/_judges.py) |
| `InterpretabilityDiagnostics`            | Attention heatmap, logit lens, linear probe, SAE features — open-weight models only (Llama/Gemma/Phi/Mistral)                                         | 529       | [`interpretability.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/interpretability.py)                                                                                         |
| `RAGDiagnostics`                         | Retrieval metrics: recall@k, precision@k, MRR, nDCG, context utilisation, retriever leaderboard                                                       | 705       | [`retrieval.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/retrieval.py)                                                                                                       |
| `AgentDiagnostics` + `TraceEvent` schema | Agent run capture: tool usage, loop detection, cost breakdown, timeline                                                                               | 668 + 360 | [`agent.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/agent.py) + [`_traces.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/_traces.py)   |
| `AlignmentDiagnostics`                   | Fine-tuning health: KL divergence, reward margin, win rate, reward-hacking detection (z-score threshold)                                              | 649       | [`alignment.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/alignment.py)                                                                                                       |
| `GovernanceDiagnostics`                  | Read-only audit inspector: chain verification (SHA-256 prev_hash), budget consumption, negative drills, envelope snapshots                            | 716       | [`governance.py`](https://github.com/terrene-foundation/mlfp/blob/main/shared/mlfp06/diagnostics/governance.py)                                                                                                     |

All share a common shape: context manager, polars DataFrames, `plot_*()` → `go.Figure`, `report()` → dict, `run_id` for correlation.

## Suggested placement (open to anything)

- Training / retrieval / text metrics → `kailash_ml.engines.*`
- Agent tracer + judge callable → `kaizen.observability.*` + `kaizen.judges.*`
- Alignment health → `kailash_align.diagnostics.*`
- Governance auditor → `pact.diagnostics.*`
- Attention interpretability → extend `ModelExplainer` or new bare class

Maintainer call.

## Known things to clean up before merge

- `AgentDiagnostics` has a Langfuse exporter hardcoded — will strip to a `TraceExporter` protocol before PR (no default commercial wiring, per Foundation independence).
- `AlignmentDiagnostics` has a `trl` fallback that can probably be dropped (closed-form KL estimator is already in the code).
- `AgentTrace` sums cost as floats; `kaizen.cost.CostTracker` uses microdollars — will route through CostTracker on port.
- Docstrings use MLFP's "medical instrument" metaphors (Stethoscope / X-Ray / ECG / Flight Recorder). Will replace with production-neutral docstrings for the SDK surface.

## Cross-SDK / kailash-rs

Happy to file parity issues on kailash-rs for the helpers where Rust parity makes sense (agent tracer, retrieval evaluator, governance auditor have clean Rust paths via existing kailash-kaizen / kailash-pact primitives). Other three (DL, interpretability, alignment) are py-only until there's a Rust autodiff / HF-transformers story.

Will follow your lead on repo coordination — happy to open cross-linked tickets or leave parity decisions to you.

## Ask

1. Which helpers (if any) do you want upstream?
2. Preferred placement for the ones you want?
3. One PR per helper, or grouped?

No pressure on scope — happy to land one, some, or all. Each helper stands alone.
