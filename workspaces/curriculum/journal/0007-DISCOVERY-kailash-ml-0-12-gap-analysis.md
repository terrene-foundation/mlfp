---
type: DISCOVERY
date: 2026-04-20
created_at: 2026-04-20T00:00:00Z
author: agent
session_id: m5-audit-and-gap-analysis-2026-04-20
session_turn: 1
project: mlfp
topic: kailash-ml 0.12 gap analysis — 5 helpers in MLFP that should be upstream
phase: implement
tags: [m5, m6, kailash-ml, diagnostics, upstream-contribution]
---

# DISCOVERY: ~4,800 LOC of MLFP "diagnostics" helpers belong in kailash-ml, not in the course

## Premise

The user observed that MLFP carries large `diagnostics` modules in
`shared/mlfp05/diagnostics.py` (1,679 LOC) and
`shared/mlfp06/diagnostics/` (5,588 LOC across 7 modules) and asked the
right question: why isn't this in kailash-ml? A teaching course should
not be re-implementing platform infrastructure.

This entry classifies every helper in those modules as DUPLICATE, EXTENDS,
GAP, or COURSE-SPECIFIC and pitches the GAP-class items as upstream
contributions.

## Findings

### Most M5 helpers are EXTENDS, not GAPS

The eight `shared/mlfp05/ex_*.py` modules (~2,000 LOC total) wire kailash-ml
engines (ExperimentTracker, ModelRegistry, ModelVisualizer, InferenceServer,
OnnxBridge) into lesson-specific orchestration: dataset loaders for the
chosen corpora, lesson-tied training loops, lesson-specific visualisations.
These are **EXTENDS** the SDK with course pedagogy — appropriately local.

Example: `shared/mlfp05/ex_1.py::setup_engines()` constructs an
`ExperimentTracker(conn)` + `ModelRegistry(conn)` + a SQLite ConnectionManager
all wired to `outputs/ex1_autoencoders/`. The engines themselves are SDK
primitives; the wiring is course pedagogy.

Verdict: keep local. Migration cost > benefit.

### Five GENUINE gaps — upstream candidates totalling ~4,800 LOC

| Helper                                                                           | LOC    | Why it's a GAP                                                                                                                                                                                                                                                                                                                                                                                   | Upstream slot                                                                                    |
| -------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **`shared/mlfp05/diagnostics.py::DLDiagnostics`**                                | ~1,200 | PyTorch training-time instrumentation: gradient flow per layer, dead-neuron tracking, activation saturation, loss-curve dynamics. Polars + Plotly. `kailash_ml.engines.model_explainer` exists but targets POST-training attribution (SHAP, LIME) — there is zero DURING-training observability. The four-failure-mode framing (Stethoscope/Blood Test/X-Ray/Prescription) is genuinely missing. | `kailash_ml.engines.training_diagnostics` (new module, sibling of `training_pipeline`)           |
| **`shared/mlfp06/diagnostics/output.py::LLMDiagnostics`**                        | 615    | LLM-as-judge evaluation (faithfulness, relevance, hallucination, refusal calibration). Wraps deepeval/ragas/sacrebleu/rouge_score + Kaizen Delegate (no raw `openai.*`). `model_explainer` covers classical ML; LLMs have no equivalent.                                                                                                                                                         | `kailash_ml.engines.model_explainer` (extend) OR `kailash_ml.engines.llm_judge` (new)            |
| **`shared/mlfp06/diagnostics/interpretability.py::InterpretabilityDiagnostics`** | 529    | Attention-saliency attribution for open-weight LLMs (Gemma, Llama). Uses transformer_lens / Captum. Gracefully short-circuits on closed-weight API models (`not_applicable`). Zero LLM attention support in kailash-ml today.                                                                                                                                                                    | `kailash_ml.engines.model_explainer` (extend)                                                    |
| **`shared/mlfp06/diagnostics/retrieval.py::RAGDiagnostics`**                     | 705    | RAG pipeline diagnostics: recall@k, precision, coverage, context relevance. Decouples retrieval quality from generation quality. Zero RAG observability in kailash-ml.                                                                                                                                                                                                                           | `kailash_ml.engines.rag_evaluator` (new) OR slot under `inference_server`                        |
| **`shared/mlfp06/diagnostics/agent.py::AgentDiagnostics`**                       | 668    | Agent execution trace capture: tool use, loop detection, budget accounting, multi-step reasoning trace. Built on the Kaizen Delegate event stream. SDK has no agent-run observability layer.                                                                                                                                                                                                     | `kailash_kaizen.observability.agent_tracer` (new in kaizen) OR `kailash_ml.engines.agent_tracer` |

### Two helpers reasonably stay local (course-tied)

- `shared/mlfp06/diagnostics/alignment.py::AlignmentDiagnostics` (649 LOC) — fine-tuning health (KL divergence, reward margins, hacking detection). Tightly coupled to Align SDK + the M6 fine-tuning narrative.
- `shared/mlfp06/diagnostics/governance.py::GovernanceDiagnostics` (716 LOC) — PACT envelope audit (D/T/R accountability, decision drilling). Tightly coupled to PACT SDK + the M6 governance narrative.

These COULD be upstreamed too but the boundary is fuzzier (they wrap
governance / fine-tuning APIs in pedagogically useful but opinionated ways).

### Observatory facade

`shared/mlfp06/diagnostics/observatory.py::LLMObservatory` (538 LOC) is a
composing facade across all six lenses. Its fate depends on whether the
lenses upstream — if 4 of 6 leave to kailash-ml, the Observatory
continues to make sense locally as a course-shaped composition.

## Why this matters

Every line of platform code we re-implement in the course is:

1. A maintenance burden on us (we own the bugs, the API drift, the SDK upgrades)
2. A teaching dilution (students learn course-specific abstractions instead of platform primitives they will use professionally)
3. A missed contribution opportunity (other Foundation projects also need these — they will reinvent the same thing)

The five GAP candidates above are ~4,800 LOC of well-designed,
polars-native, Plotly-based, Foundation-licensed (Apache-2.0) code that
could ship in `kailash-ml 0.13` or `kailash-kaizen 2.8`. The course would
then `from kailash_ml import DLDiagnostics` instead of carrying our own.

## Proposed disposition

1. **Short term** (this session): document the gaps (this DISCOVERY entry).
   Migration to upstream is multi-session work that needs SDK PR coordination.
2. **Medium term**: Open SDK issues for each of the 5 candidates citing
   this entry. Pitch them as 0.13 / 2.8 milestones.
3. **Long term**: When upstream lands, course removes its local copies
   and re-imports. Net LOC reduction in MLFP: ~4,800 lines.

## For Discussion

1. The DLDiagnostics module is the most obvious win — it has ZERO LLM
   coupling, runs purely on torch + polars + plotly, and addresses a
   universally-needed gap (every PyTorch trainer wants gradient-flow +
   dead-neuron observability). Should we open the kailash-ml issue this
   week, or wait to see if the user wants the four LLM lenses bundled
   into the same proposal?
2. The Agent Tracer pitch is interesting because it crosses the kailash-ml
   / kailash-kaizen package boundary. Does it belong in kaizen (it's
   coupled to Delegate event types) or in kailash-ml (it's an
   observability concern, parallel to model_explainer)? My read: kaizen,
   because the event stream is the schema and that schema lives in kaizen.
3. The Alignment + Governance lenses are borderline — they're course-
   specific in the sense of packaging, but the underlying diagnostics
   (reward-hacking detection, envelope audit) are genuinely useful
   primitives. Would the Foundation rather see two more SDK packages
   (`kailash-align-diagnostics`, `kailash-pact-diagnostics`) or keep these
   as opinionated course scaffolds?
