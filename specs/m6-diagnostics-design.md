# M6 Diagnostics Toolkit — Design Doc

**Status**: Phase 1 (design) — Phase 2 (deck) and Phase 3 (library + exercise integration) follow.
**Audience**: MLFP instructors and the exercise-designer agent.
**Source program**: Derived from ASCENT — M6 covers LLMs, agents, RAG, fine-tuning, governance, deployment.
**Companion**: `specs/module-6.md` (lesson spec), `shared/mlfp05/diagnostics.py` (reference implementation).

## 1. Motivation — Why M6 Needs Its Own Toolkit

M5 gave students a doctor's bag for deep networks: five instruments that answer "is the training loop healthy?" The failure modes are local — one loss curve, one gradient, one layer. A ten-line hook catches them.

M6 is a different system. The unit of analysis is not a loss value; it is a generation. The failure modes are distributed across six surfaces that did not exist in M5:

1. The **output** is fluent but wrong (hallucination, drift, refusal miscalibration).
2. The **internal computation** is opaque (attention heads, MLP circuits, SAE features).
3. The **retrieved context** is irrelevant or stale (RAG ingestion mistakes).
4. The **agent's plan** loops, stalls, or burns tool budget on dead ends.
5. The **alignment signal** rewards the wrong thing (reward hacking, over-refusal).
6. The **governance envelope** is bypassed, the audit trail is silently broken.

A professional working in this layer must be able to name, read, and act on all six. Re-running with a different prompt is the M6 equivalent of re-running with different hyperparameters — it feels faster than instrumenting, until the clock hits hour six.

This doc specifies an **LLM Observatory** with **six lenses** — the direct successor to M5's Doctor's Bag — and the library, per-lesson mapping, and Kailash integration that make it teachable.

## 2. The M5 Pattern We Inherit

Three properties of M5's toolkit MUST carry over:

- **A single object owns the plumbing.** Students never write a raw PyTorch hook in M5; they call `DLDiagnostics`. The M6 equivalent: students never hand-wire an LLM judge, a retrieval recall calculator, or an audit-chain verifier. They call the library.
- **Every lesson's first exercise begins with the protocol.** M5 Slide 5O: "Every lesson's first exercise begins with 'Run the diagnostic protocol. Report the five instrument readings. Then train.'" M6 mirror: "Run the observatory. Report the six lens readings. Then build/fine-tune/deploy."
- **The readings are written to disk.** M5 diagnostics are Polars DataFrames + Plotly figures. M6 keeps the Polars/Plotly rule and adds a structured trace format (JSONL) because agent and retrieval traces are sequences of events, not scalars per batch.

## 3. The Six Lenses

Each lens answers exactly one question. The lens is both a concept (taught in the deck) and a class in `shared/mlfp06/diagnostics.py` (used in the exercises). The first four lenses DIAGNOSE; the fifth evaluates the TRAINING SIGNAL; the sixth is the ORGANISATIONAL CONTRACT — the equivalent of M5's Prescription except the prescription is written against policy, not hyperparameters.

### Lens 1 — Output Lens (Stethoscope)

**Question**: Is the generation coherent, faithful, and on-task?
**Class**: `LLMDiagnostics`
**Reads**: model outputs, reference answers (when available), retrieved context (when RAG is in the loop).
**Methods**:

- `perplexity(texts)` — sliding-window perplexity for bounded tasks.
- `rouge(predictions, references)`, `bleu(...)`, `bertscore(...)` — classical metrics for summarisation/translation.
- `llm_as_judge(prompt, response, criteria, judge_model)` — wraps Kaizen Delegate as the judge. Returns structured verdict with score + rationale.
- `self_consistency(prompt, n=5)` — samples n completions, measures agreement, flags hallucination candidates.
- `faithfulness(response, context)` — checks every claim in response is supported by context (RAG grounding).
- `refusal_rate(prompts)`, `over_refusal(benign_prompts)` — refusal calibration.
- `plot_output_dashboard()` — Plotly grid of perplexity, judge scores, faithfulness, refusal metrics.

**Industry libraries leveraged**: `deepeval` (G-Eval, faithfulness, hallucination metrics), `ragas` (faithfulness, answer relevance), `rouge-score`, `sacrebleu`, `bert-score`. The Kailash wrapper is the Delegate-based judge.

### Lens 2 — Attention Lens (X-Ray)

**Question**: What does the model attend to, and what circuit produces the answer?
**Class**: `InterpretabilityDiagnostics`
**Reads**: model activations and attention weights from open-weight models (Llama, Gemma, Phi, Mistral). API-only models cannot be X-rayed — the lens explicitly reports "not applicable".
**Methods**:

- `attention_heatmap(prompt, layer, head)` — token-to-token attention weights as a Plotly heatmap.
- `logit_lens(prompt)` — predictions per layer via the early-exit trick. Surfaces where in the stack the answer crystallises.
- `probe(prompt, feature)` — linear probe for learned features (gender, sentiment, truthfulness).
- `sae_features(prompt, layer)` — loads a pre-trained sparse autoencoder (Gemma Scope for Gemma-2, Sparsify for Llama-3) and returns the top-k active features with their labels.
- `circuit_trace(prompt, target_token)` — uses activation patching to identify the minimal subgraph that produces the token.

**Industry libraries leveraged**: `transformer_lens` (HookedTransformer), `nnterp` (2025 unified interface), `sae_lens` + Gemma Scope, Anthropic's circuit-tracer patterns. Students do not train their own SAE — they load pre-trained ones. The lens is scaffolded: M6 teaches the READING of SAE features, not the training.

### Lens 3 — Retrieval Lens (RAG Diagnostic)

**Question**: Did we fetch the right context, and did the generator use it?
**Class**: `RAGDiagnostics`
**Reads**: the retriever (BM25, dense, hybrid), the retrieved chunks, and the generator's output.
**Methods**:

- `recall_at_k(queries, relevant_ids, k)`, `precision_at_k(...)`, `mrr(...)` — standard IR metrics.
- `chunk_relevance(query, chunks, judge_model)` — per-chunk LLM-as-judge relevance score.
- `context_utilisation(response, context)` — did the answer actually draw from the context or ignore it? Uses faithfulness + attribution.
- `compare_retrievers(retrievers, queries, k)` — runs BM25 vs dense vs hybrid vs HyDE on the same queries, returns a leaderboard.
- `chunk_size_sweep(documents, sizes, queries)` — plots recall@5 vs chunk size. Finds the sweet spot empirically, not by rule of thumb.
- `plot_retrieval_dashboard()` — Plotly grid of recall/precision curves, per-retriever leaderboard, chunk-size sweep.

**Industry libraries leveraged**: `ragas` (context precision, context recall, context relevance, faithfulness, answer relevance), `trulens-eval` (RAG triad: context relevance / groundedness / answer relevance), `llama-index` evaluators, `langchain` evaluators. The Kailash wrapper is the Kaizen-based judge + MLFPDataLoader-compatible chunk store.

### Lens 4 — Agent Trace Lens

**Question**: What did the agent actually do, and where did it spend its budget?
**Class**: `AgentDiagnostics`
**Reads**: Kaizen agent event streams (Thought / Action / Observation / Tool / Cost events).
**Methods**:

- `capture(agent, prompt)` — runs the agent with a structured event capture, writes a JSONL trace to `runs/<agent>/<timestamp>.jsonl`.
- `tool_usage_summary(trace)` — per-tool call count, success rate, p50/p99 latency, cost share.
- `thought_quality(trace, judge_model)` — rubric-scored thought coherence: does each thought advance the plan, or does the agent loop?
- `loop_detection(trace)` — finds repeated (thought, action) pairs, flags cycles.
- `budget_timeline(trace)` — Plotly step chart of cumulative cost vs wall clock, with tool boundaries marked.
- `context_utilisation(trace)` — tokens consumed vs model context window over time.
- `replay(trace, modified_tool_output)` — counterfactual replay: swap one observation and re-run from that step.

**Industry libraries leveraged**: `langfuse` (self-hosted agent trace UI, OTEL-native), `langsmith` (hosted alternative), `phoenix` (Arize, open-source trace UI). The Kailash wrapper converts Kaizen event streams into OpenTelemetry spans so any of these back-ends can consume them. The course default is Langfuse self-hosted because it's open-source and respects the Foundation Independence rule.

### Lens 5 — Alignment Lens

**Question**: Is the fine-tuning signal rewarding the right thing, and has the policy drifted from reference?
**Class**: `AlignmentDiagnostics`
**Reads**: the training pipeline (SFT, LoRA, DPO, GRPO), the reference model, and the current policy.
**Methods**:

- `kl_from_reference(policy, reference, prompts)` — per-prompt KL divergence from the frozen reference model. Spikes indicate the policy is drifting.
- `reward_margin(chosen_logps, rejected_logps)` — DPO implicit reward margin per pair; histogram across the batch.
- `win_rate(policy, reference, prompts, judge_model)` — blind pairwise win rate of the fine-tuned model against the base. Uses bias-mitigated LLM-as-judge (swap positions, normalise lengths).
- `benchmark_delta(policy, reference, suites=["mmlu", "hellaswag", "truthfulqa"])` — runs `lm-evaluation-harness` suites before/after fine-tuning and reports the delta. Flags regressions.
- `reward_hacking_scan(trace, rubric)` — samples N completions, applies the rubric, flags completions that score well on the proxy but fail a held-out check.
- `grpo_group_stats(trace)` — for GRPO runs: group-relative reward variance, advantage distribution, group collapse detection.
- `plot_alignment_dashboard()` — Plotly grid of KL trajectory, reward margin histogram, win-rate confidence interval, benchmark deltas.

**Industry libraries leveraged**: `trl` (GRPOTrainer, DPOTrainer surface their own internal stats — we expose them uniformly), `lm-evaluation-harness`, `deepeval` (bias/toxicity suites), `ragas` (for RAG-augmented alignment). The Kailash wrapper sits on `kailash-align.AlignmentPipeline`.

### Lens 6 — Governance Lens (PACT Audit)

**Question**: Is the system operating within its envelope, and is the audit chain intact?
**Class**: `GovernanceDiagnostics`
**Reads**: the PACT `GovernanceEngine`, `GovernedSupervisor.audit`, and the `RoleEnvelope` stack.
**Methods**:

- `envelope_snapshot(engine, role_address)` — current constraint values across all 5 dimensions (Financial, Operational, Temporal, Data Access, Communication) for a role.
- `verify_chain(audit_log)` — returns `True` iff the hash chain is intact. On failure, returns the index of the first tampered entry.
- `envelope_breaches(audit_log)` — filters the audit log for entries where `verdict.allowed == False`. Groups by role, action, reason.
- `budget_cascade_check(engine, root_role)` — walks the role tree and asserts every child's `FinancialConstraintConfig.max_spend_usd` is ≤ its parent's. Reports violations structurally, not at runtime.
- `clearance_coverage(model_registry)` — every field in every model has a declared clearance level. Unclassified fields are flagged as HIGH.
- `negative_verdict_drill(engine, role_addresses, actions)` — runs `engine.verify_action` over a grid of (role, action) pairs; asserts the expected deny set stays denied. This is the structural test that prevents governance from silently regressing.
- `plot_governance_dashboard()` — Plotly grid of envelope utilisation, breach timeline, audit chain status, clearance coverage.

**Industry libraries leveraged**: `kailash-pact` (GovernanceEngine, RoleEnvelope, ConstraintEnvelopeConfig), `kaizen-agents.GovernedSupervisor`. No external libraries — this lens is native to the Kailash stack because PACT is the Foundation's own governance framework.

## 4. The Unified Object — `LLMObservatory`

Mirroring `DLDiagnostics`, a single facade wraps the six lenses for lesson-first use. The facade is thin; students can also reach into each lens class directly when they need more surface area.

```python
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory()  # composes all six lenses, no args required for the simple path

# ── Output lens (any lesson) ─────────────────────────────────
obs.output.faithfulness(response, context)
obs.output.llm_as_judge(prompt, response, criteria="coherence,helpfulness,harmlessness")

# ── Attention lens (lesson 6.2, open-weight models only) ─────
obs.attention.logit_lens(prompt, model="meta-llama/Llama-3.2-1B")
obs.attention.sae_features(prompt, model="google/gemma-2-2b", layer=12)

# ── Retrieval lens (lesson 6.4) ──────────────────────────────
obs.retrieval.compare_retrievers([bm25, dense, hybrid, hyde], queries, k=5)
obs.retrieval.context_utilisation(response, retrieved_chunks)

# ── Agent trace lens (lessons 6.5, 6.6, 6.8) ─────────────────
with obs.agent.capture(my_react_agent) as trace:
    result = my_react_agent.run(task)
obs.agent.tool_usage_summary(trace)
obs.agent.budget_timeline(trace).show()

# ── Alignment lens (lessons 6.2, 6.3) ────────────────────────
obs.alignment.benchmark_delta(policy, reference, suites=["mmlu", "truthfulqa"])
obs.alignment.win_rate(policy, reference, prompts, judge_model=delegate)

# ── Governance lens (lessons 6.7, 6.8) ───────────────────────
obs.governance.verify_chain(supervisor.audit.to_list())
obs.governance.envelope_breaches(supervisor.audit.to_list())

# ── Combined dashboard (capstone) ────────────────────────────
obs.report()                       # text summary, all six lenses
obs.plot_observatory_dashboard()   # single 6-panel figure
```

**Design principle**: the facade is optional, not mandatory. Exercise 6.2 uses only `alignment` + `attention`; exercise 6.4 uses only `retrieval` + `output`. The observatory imports all six but initialises them lazily so the import is cheap.

## 5. Per-Lesson Mapping

Every M6 lesson's first exercise begins with: **"Run the observatory. Report the relevant lens readings. Then build."** The matrix below names the primary (P) and secondary (S) lenses per lesson, the diagnostic question the student must answer, and the industry-library back-end.

| Lesson | Topic                                   | Primary         | Secondary                 | Diagnostic question                                                                                                                                                           | Backend                                 |
| ------ | --------------------------------------- | --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| 6.1    | LLM fundamentals + prompting            | Output (P)      | —                         | Across zero-shot / few-shot / CoT / self-consistency, which technique gives the best accuracy-cost frontier for this task?                                                    | deepeval + Kaizen Delegate              |
| 6.2    | LoRA + adapters + fine-tuning landscape | Alignment (P)   | Attention (S), Output (S) | Does the LoRA-tuned model (a) match the reference on held-out benchmarks, (b) change the attention pattern on target-domain tokens, (c) improve task-specific output quality? | trl + transformer_lens + lm-eval        |
| 6.3    | DPO and GRPO                            | Alignment (P)   | Output (S)                | Has the policy drifted too far from reference (KL spike)? Has it reward-hacked the preference signal? Does win-rate vs base survive bias-mitigated judging?                   | trl DPOTrainer/GRPOTrainer + deepeval   |
| 6.4    | RAG systems                             | Retrieval (P)   | Output (P)                | Which retriever + chunk strategy + HyDE variant maximises both recall AND faithfulness? Where does the generator ignore retrieved context?                                    | ragas + trulens-eval                    |
| 6.5    | ReAct agents + tool use                 | Agent Trace (P) | Output (S)                | Is the agent reasoning or pattern-matching? Which tool burns the cost budget? Where does the agent loop?                                                                      | langfuse + Kaizen event stream          |
| 6.6    | Multi-agent + MCP                       | Agent Trace (P) | Governance (S)            | Do the specialists coordinate or duplicate work? Does any agent exceed its envelope when delegated to?                                                                        | langfuse + PACT                         |
| 6.7    | AI governance engineering               | Governance (P)  | Agent Trace (S)           | Does `verify_action` correctly deny the negative test set? Is the audit chain tamper-evident? Does the budget cascade hold structurally?                                      | PACT + kaizen-agents.GovernedSupervisor |
| 6.8    | Capstone — production platform          | **ALL SIX**     | —                         | Produce the full observatory dashboard for a deployed Nexus system and interpret every lens reading.                                                                          | full stack                              |

**Note on 6.1**: the attention lens is introduced in the deck during this lesson (as a preview) but is not used in the 6.1 exercise — API-only models (GPT, Claude, Gemini) cannot be X-rayed. The attention lens becomes the primary tool in 6.2 where students load Llama/Gemma/Phi weights.

## 6. Library Architecture

### 6.1 Module layout

```
shared/mlfp06/
  __init__.py
  diagnostics/
    __init__.py              # re-exports LLMObservatory + the six lens classes
    observatory.py           # LLMObservatory facade (50-80 LOC)
    output.py                # LLMDiagnostics
    interpretability.py      # InterpretabilityDiagnostics (requires transformer_lens)
    retrieval.py             # RAGDiagnostics
    agent.py                 # AgentDiagnostics
    alignment.py             # AlignmentDiagnostics
    governance.py            # GovernanceDiagnostics
    _judges.py               # shared LLM-as-judge primitives (bias-mitigated)
    _traces.py               # JSONL trace format + OpenTelemetry converter
    _plots.py                # shared Plotly theme matching M5 dashboards
  ex_1.py … ex_8.py          # per-exercise shared helpers (R10 structure)
```

Each lens file is capped at ~500 LOC load-bearing logic per `autonomous-execution.md` MUST Rule 1. The observatory facade is a thin composition layer, not a god-class.

### 6.2 Dependency declarations

Every dependency MUST be declared in the MLFP root `pyproject.toml` per `dependencies.md` MUST. The M6 diagnostics stack adds:

- `deepeval>=2.0` — LLM evaluation metrics, GEval, hallucination suite.
- `ragas>=0.2` — RAG faithfulness / context relevance / answer relevance.
- `trulens-eval>=1.0` — RAG triad evaluators.
- `transformer-lens>=2.0` — open-weight model X-ray.
- `nnterp>=0.5` — unified mechanistic interpretability interface (optional; transformer-lens is the primary).
- `sae-lens>=3.0` — sparse autoencoder loading (Gemma Scope, Sparsify).
- `lm-eval>=0.4` — lm-evaluation-harness for MMLU / HellaSwag / TruthfulQA.
- `langfuse>=2.0` — self-hosted agent trace UI (Foundation-Independence compliant).
- `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp` — for agent trace export.
- `sacrebleu`, `rouge-score`, `bert-score` — classical generation metrics.

Packages already in the stack (`kailash`, `kailash-kaizen`, `kailash-align`, `kailash-pact`, `kailash-nexus`, `polars`, `plotly`) are reused. No `pandas`, no `matplotlib`, per the framework-first + polars rules.

### 6.3 Core class contract

Every lens class MUST follow the same shape as `DLDiagnostics`:

```python
class LLMDiagnostics:
    """Output lens — faithfulness, coherence, refusal calibration."""

    def __init__(
        self,
        *,
        judge_model: str | None = None,   # defaults to OPENAI_PROD_MODEL from .env
        cache_dir: Path | None = None,    # per-exercise reproducibility cache
    ) -> None:
        # Loads .env (env-models.md), validates judge_model pairing with the right key,
        # creates cache_dir if absent. No network call at construction time.
        ...

    def __enter__(self) -> LLMDiagnostics: ...
    def __exit__(self, *exc) -> None: ...  # flushes cache, closes judge sessions

    # ── Instrument methods return Polars DataFrames or Plotly Figures ───────
    def faithfulness(self, response: str, context: list[str]) -> pl.DataFrame: ...
    def llm_as_judge(self, prompt: str, response: str, criteria: str) -> JudgeVerdict: ...
    def plot_output_dashboard(self) -> go.Figure: ...
    def report(self) -> str: ...   # text-mode auto-diagnosis
```

- All DataFrames are Polars.
- All plots are Plotly.
- All LLM calls route through Kaizen Delegate, never direct `openai.chat.completions.create` (framework-first MUST).
- Every judge call uses a bias-mitigated wrapper (position swap + length normalisation) by default.
- Every public method has a docstring with a one-line example, matching the M5 style.

### 6.4 Trace format

Agent and RAG traces are JSONL files, one event per line, with a stable schema:

```json
{"ts": "2026-04-15T12:01:03.142Z", "run_id": "r_abc123", "kind": "thought", "content": "...", "cost_usd": 0.0}
{"ts": "...", "run_id": "r_abc123", "kind": "action", "tool": "search", "args": {"q": "..."}, "cost_usd": 0.0}
{"ts": "...", "run_id": "r_abc123", "kind": "observation", "result": "...", "cost_usd": 0.002, "latency_ms": 842}
```

The schema matches the OpenTelemetry GenAI semantic conventions (2026-01 draft) so Langfuse, Langsmith, and Phoenix can all consume MLFP traces without translation. This is the agent-layer equivalent of M5's Polars DataFrame contract.

## 7. Integration with the Kailash Stack

The observatory is not a parallel observability system — it wraps and surfaces what Kailash already emits:

| Source                               | What Kailash emits                                       | What the observatory does                                          |
| ------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------------------ |
| `Delegate`                           | Streaming events (token, cost, tool)                     | `AgentDiagnostics.capture` materialises them into a trace          |
| `ReActAgent` / `ChainOfThoughtAgent` | TAOD events                                              | Same capture pipeline                                              |
| `GovernedSupervisor.audit`           | Hash-chained audit log                                   | `GovernanceDiagnostics.verify_chain` validates it                  |
| `GovernanceEngine.verify_action`     | `GovernanceVerdict` with `.allowed`, `.level`, `.reason` | `GovernanceDiagnostics.negative_verdict_drill` exercises the grid  |
| `AlignmentPipeline` (DPO/GRPO)       | Per-step reward margins, KL divergence                   | `AlignmentDiagnostics` aggregates and plots                        |
| `Nexus` request handlers             | Structured logs with `request_id`                        | `AgentDiagnostics.capture` correlates to the agent trace           |
| `kailash-mcp` tool calls             | MCP tool schemas + call results                          | `AgentDiagnostics.tool_usage_summary` attributes cost per MCP tool |

**Specialist delegation**: any exercise that wires a new diagnostic into Kaizen/PACT/Align MUST go through the relevant specialist per `agents.md` MUST Rule 1. The observatory is not a bypass; it is the structured consumer of the framework's existing surface.

## 8. What the Student DIAGNOSES — Per-Exercise Tasks

These are the first-contact diagnostic tasks, one per lesson, that `exercise-designer` expands into the R10 5-phase structure.

### 6.1 — Prompting technique frontier

Student receives a classification dataset and four prompting techniques (zero-shot, few-shot, CoT, self-consistency). They run all four under `LLMDiagnostics` and produce the accuracy-cost Pareto frontier. **Diagnostic outcome**: "at what cost per query does CoT stop being worth it?"

### 6.2 — LoRA behaviour attribution

After LoRA fine-tuning on IMDB sentiment (from scratch — `specs/module-6.md` § 6.2), the student runs:

- `AlignmentDiagnostics.benchmark_delta` (did general capability regress?)
- `InterpretabilityDiagnostics.logit_lens` on 5 in-domain and 5 out-of-domain prompts (where in the stack did the LoRA rewire the prediction?)
- `InterpretabilityDiagnostics.sae_features` on a sentiment prompt before and after (which features activated differently?)

**Diagnostic outcome**: a three-part report "what the LoRA changed, and what it broke".

### 6.3 — DPO reward hacking hunt

After DPO training on a preference dataset, the student:

- Plots `AlignmentDiagnostics.reward_margin` histogram. Bimodal? That's a win. Single-peak near zero? Reward model collapsed.
- Runs `kl_from_reference` per prompt. Spike region = policy drifted.
- Runs `win_rate` with position swap AND length normalisation, both toggled on/off, to show how much of the apparent win rate is bias.

**Diagnostic outcome**: "the naive win rate was 73%. After bias mitigation it's 54%. The 19-point gap is verbosity bias."

### 6.4 — RAG leaderboard

Student builds four retrievers (BM25, dense, hybrid, HyDE) over Singapore policy documents and runs `RAGDiagnostics.compare_retrievers` across 50 queries. Then `RAGDiagnostics.context_utilisation` per query-retriever pair to find queries where retrieval was good but generation ignored the context.

**Diagnostic outcome**: "hybrid wins on recall@5 by 12 points, but on faithfulness BM25 wins — because hybrid retrieves more chunks and the generator ignores the less-relevant ones, producing ungrounded answers."

### 6.5 — Agent trace autopsy

Student builds a ReAct data-analysis agent. Runs it on 3 tasks of increasing complexity under `AgentDiagnostics.capture`. Then:

- `tool_usage_summary` — which tool dominated cost?
- `loop_detection` — did the agent cycle?
- `budget_timeline` — at what wall-clock second did the agent burn 80% of budget?

**Diagnostic outcome**: "on task 3 the agent looped between `explore` and `search` for 18 steps. It solved task 2 in 4 steps. The prompt template collapses when the question contains more than one sub-task."

### 6.6 — Multi-agent handoff audit

Student builds the DataScientist → FeatureEngineer → ModelSelector → ReportWriter pipeline. Captures all four traces. Runs `AgentDiagnostics.tool_usage_summary` per agent AND across the pipeline. Runs `GovernanceDiagnostics.envelope_breaches` to confirm no agent exceeded its envelope when delegated to.

**Diagnostic outcome**: "the FeatureEngineer agent exceeds its temporal envelope on 2 of 10 runs — not a bug in the code, a mis-sized envelope. Tighten to 30s or widen to 120s; don't leave it at 60s."

### 6.7 — Governance red-team drill

Student defines an org with 5 roles, attaches envelopes, and builds a `GovernedSupervisor`. They run `GovernanceDiagnostics.negative_verdict_drill` with:

- Known-bad role addresses (`D99-R99-T99-R99`) — MUST be blocked.
- Privilege escalation attempts (`set_role_envelope` with a LOOSER child) — MUST be rejected at validation.
- Audit tampering (flip one entry's `content` byte) — `verify_chain` MUST return the tampered index.

**Diagnostic outcome**: "governance MUST fail closed across 12 negative tests AND MUST permit across 8 positive tests. The drill is green."

### 6.8 — Capstone observatory dashboard

Student deploys the full system via Nexus. Runs the observatory across all six lenses on a production-representative workload (50 queries). Produces the 6-panel dashboard AND the text report. The text report is the CAPSTONE DELIVERABLE, not the code. Assessment: can the student read their own dashboard?

## 9. Deck Section Preview (Phase 2 scope)

The M6 deck opens with a 15-slide section — "LLM Observatory — Six Lenses" — structurally mirroring M5's Slides 5A-5O:

- **6A** (title): "The LLM Observatory — Six Lenses"
- **6B** (overview): the six lens icons, diagnose/evaluate/govern split
- **6B2** (meet the object): `LLMObservatory` in one slide
- **6C-6H** (one slide per lens): the question, the method signature, a one-line example
- **6I-6L** (per-lesson integration): how each lens shows up in the 8 lessons
- **6M** (anti-pattern): "the three things that look like observability but aren't" — raw print, unstructured trace, silent governance
- **6N** (protocol): "every lesson's first exercise begins with 'run the observatory'"
- **6O** (transition): handoff to lesson 6.1

This section replaces M5's "Doctor's Bag" positioning with the LLM-native equivalent. The deck is Phase 2 work — specified here so the design is consistent when Phase 2 lands.

## 10. Phase 3 Scope (library + exercise integration)

Phase 3 produces:

1. `shared/mlfp06/diagnostics/` — the 6 lens classes + observatory facade, targeting ~2500 LOC total (~500 per lens, ~300 shared, ~100 facade).
2. Per-exercise integration: `shared/mlfp06/ex_N.py` wires the relevant lenses into each lesson's exercise directory.
3. Updates to `modules/mlfp06/solutions/ex_N/` to use the observatory (R10 5-phase structure: Theory / Build / Train / Visualise / Apply — the Visualise phase becomes the observatory dashboard).
4. Updates to `modules/mlfp06/local/ex_N/` (scaffolded student versions, ~20% scaffolding per M6 progressive-disclosure).
5. Colab conversion via `scripts/generate_selfcontained_notebook.py`.
6. Tier 2 integration tests under `tests/integration/test_mlfp06_diagnostics_*.py` that exercise each lens against real infrastructure (real Kaizen Delegate, real PACT engine, real transformer_lens model load). Unit-only tests are NOT sufficient per `orphan-detection.md` MUST Rule 2.

Phase 3 is ~3-5 autonomous cycles per `autonomous-execution.md` sharding budget. Proposed shards:

- **Shard A**: `output.py` + `retrieval.py` + judges module (most shared judge infrastructure).
- **Shard B**: `agent.py` + `_traces.py` + Langfuse integration.
- **Shard C**: `alignment.py` + `interpretability.py` (both depend on open-weight models; share load infrastructure).
- **Shard D**: `governance.py` + `observatory.py` + the 8 exercise wirings.

Each shard has a Tier 2 test that imports through the observatory facade, not the lens class directly — this is the `orphan-detection.md` contract.

## 11. Non-Goals

To keep the design tractable for one module, the observatory explicitly does NOT attempt:

- **Training sparse autoencoders.** Students load pre-trained SAEs (Gemma Scope, Sparsify). SAE training is a research topic, not an M6 skill.
- **Building a novel interpretability method.** The lens surfaces existing tools (logit lens, attention heatmap, activation patching). Students learn to READ mech interp output, not invent it.
- **Replacing lm-evaluation-harness.** The alignment lens wraps `lm-eval` for benchmark runs; it does not re-implement the benchmarks.
- **Replacing Langfuse/Langsmith.** The agent lens writes OTEL-compatible traces and offers a built-in minimal viewer for exercises that can't run a Langfuse container; for full UI, students launch Langfuse self-hosted.
- **Runtime governance.** The governance lens INSPECTS PACT; it does not replace the `GovernanceEngine`. PACT is the execution boundary; the lens is the audit boundary.

## 12. Open Questions (Resolve Before Phase 2)

1. **Model choice for the attention lens**: Llama-3.2-1B is small enough for Colab free tier (T4 GPU, 15GB VRAM) but Gemma-2-2B has better SAE coverage via Gemma Scope. Pick one as the course default?
2. **Langfuse self-hosted vs built-in viewer**: running Langfuse requires Docker — Colab does not have Docker. The built-in viewer would be a Plotly-based fallback. Is Plotly sufficient, or do we require Langfuse for lesson 6.5+?
3. **Grading the capstone's observatory report**: the student-facing deliverable is a text report, not a metric. We need a rubric that the `quiz-designer` agent can hold against it. Draft rubric in Phase 2.
4. **Cost budget for the full M6 module**: every lens uses LLM-as-judge, which costs money. Estimate expected `$ per student per exercise` and confirm against the MLFP cost cap before Phase 3 starts.

## 13. Compliance Summary

| Rule                                                  | Disposition                                                                                                      |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `framework-first.md` — Kaizen for all LLM calls       | MUST: all judges and agents route through Delegate / BaseAgent. No raw `openai.chat.completions`.                |
| `env-models.md` — no hardcoded model names            | MUST: `judge_model` defaults to `os.environ["OPENAI_PROD_MODEL"]`. All lenses load `.env` at construction.       |
| `dependencies.md` — latest versions                   | Each new dep gets `>=X.Y` floor only, no upper cap.                                                              |
| `independence.md` — no commercial coupling            | Langfuse self-hosted (open-source) over Langsmith (hosted); `deepeval` over proprietary eval suites.             |
| `two-format.md` — VS Code + Colab                     | Each exercise ships `.py` + `.ipynb` via `scripts/generate_selfcontained_notebook.py`.                           |
| `exercise-standards.md` — R10 5-phase                 | Each technique file follows Theory → Build → Train → Visualise → Apply.                                          |
| `orphan-detection.md` — Tier 2 wiring tests           | Each lens has a Tier 2 test that imports through `LLMObservatory`.                                               |
| `facade-manager-detection.md` — manager-shape classes | The 6 lens classes follow the `*Diagnostics` pattern; each has a wiring test.                                    |
| `observability.md` — structured logs                  | Every lens emits structured INFO/WARN via the framework logger, never `print`.                                   |
| `testing.md` — Tier 2 real infrastructure             | Lens tests use real Delegate, real PACT engine, real transformer_lens model — no mocks for the external surface. |

---

**End of design doc.** Phase 2 (deck section) and Phase 3 (library + exercise integration) proceed only after this design is approved.
