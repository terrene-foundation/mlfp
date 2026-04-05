# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 7: Multi-Agent Orchestration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Implement formal multi-agent coordination patterns using
#   A2A (Agent-to-Agent) protocol. Cover four production patterns:
#   supervisor-worker, sequential pipeline, parallel fan-out, and handoff.
#   Each pattern is demonstrated with a practical ML scenario.
#
# TASKS:
#   1. Supervisor-Worker: one coordinator dispatches to specialised workers
#   2. Sequential Pipeline: strict ordering, each agent builds on prior output
#   3. Parallel Fan-out: independent agents run concurrently, results merged
#   4. Handoff Pattern: agent transfers ownership based on conditions
#   5. Choosing the right pattern for ML production scenarios
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate, SupervisorWorkerPattern, SequentialPattern
from kaizen_agents import ParallelPattern, HandoffPattern

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

data_context = (
    f"Singapore Credit Scoring: {credit.height:,} rows, "
    f"{len(credit.columns)} columns, target=default "
    f"({credit['default'].mean():.1%} default rate)"
)

print(f"=== Multi-Agent Orchestration ===")
print(f"Dataset: {data_context}")
print(f"Patterns: supervisor-worker, sequential, parallel, handoff")


# ══════════════════════════════════════════════════════════════════════
# Define shared Signatures used across patterns
# ══════════════════════════════════════════════════════════════════════


class WorkerResult(Signature):
    """Generic worker agent output."""

    task: str = InputField(description="Assigned task")
    findings: list[str] = OutputField(description="Key findings")
    recommendation: str = OutputField(description="Primary recommendation")
    confidence: float = OutputField(description="Confidence in recommendation 0-1")


class SupervisorDecision(Signature):
    """Supervisor agent: plan tasks and synthesize worker results."""

    goal: str = InputField(description="High-level goal to achieve")
    worker_results: str = InputField(description="Aggregated results from workers")
    final_recommendation: str = OutputField(description="Synthesized recommendation")
    task_assignments: list[str] = OutputField(description="Tasks assigned to workers")
    gaps: list[str] = OutputField(description="Identified gaps needing follow-up")


class PipelineStageOutput(Signature):
    """Output from one stage of a sequential pipeline."""

    context: str = InputField(description="Context passed from previous stage")
    stage_output: str = OutputField(description="This stage's output")
    handoff_context: str = OutputField(description="Context to pass to next stage")
    stage_complete: bool = OutputField(description="Whether this stage is complete")


class ParallelAnalysisResult(Signature):
    """Output from one parallel analysis worker."""

    analysis_type: str = InputField(description="Type of analysis to perform")
    data_context: str = InputField(description="Dataset context")
    analysis: str = OutputField(description="Analysis results")
    key_metrics: list[str] = OutputField(description="Key metrics found")
    priority: str = OutputField(description="HIGH / MEDIUM / LOW priority")


class HandoffDecision(Signature):
    """Agent decides whether to handle task or hand off."""

    task: str = InputField(description="Task to evaluate")
    agent_capability: str = InputField(description="This agent's specialty")
    handle_or_handoff: str = OutputField(description="HANDLE or HANDOFF")
    handoff_reason: str = OutputField(description="Why handing off (if applicable)")
    result: str = OutputField(description="Result if handling, empty if handing off")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Supervisor-Worker Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: One supervisor decomposes a goal into subtasks, dispatches to
# specialised workers, then synthesizes results. The supervisor has
# global context; workers are domain experts.
#
# ML use case: Model audit — supervisor assigns data quality, bias,
# and performance audits to specialist workers.
#
#   Supervisor
#       ├── Worker A: data quality audit
#       ├── Worker B: bias and fairness audit
#       └── Worker C: model performance audit
#           ↓
#      Supervisor synthesizes


async def pattern_supervisor_worker():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 1: Supervisor-Worker")
    print(f"{'─' * 60}")
    print(f"Scenario: Autonomous model audit before production deployment")

    # --- Define worker agents ---

    data_quality_worker = Delegate(model=model, max_llm_cost_usd=0.5)
    bias_worker = Delegate(model=model, max_llm_cost_usd=0.5)
    performance_worker = Delegate(model=model, max_llm_cost_usd=0.5)

    # --- Build SupervisorWorkerPattern ---
    # The supervisor coordinates task dispatch; workers are registered by role.

    pattern = SupervisorWorkerPattern(
        supervisor_model=model,
        supervisor_signature=SupervisorDecision,
        workers={
            "data_quality": data_quality_worker,
            "bias_fairness": bias_worker,
            "model_performance": performance_worker,
        },
        supervisor_cost_budget=0.8,
        worker_cost_budget=0.5,
    )

    # Give each worker a specialisation prompt
    worker_prompts = {
        "data_quality": (
            f"You are a data quality auditor. Examine this credit dataset and report:\n"
            f"{data_context}\n"
            f"Assess: null rates, outliers, class imbalance, data leakage risk."
        ),
        "bias_fairness": (
            f"You are a fairness auditor. Examine this credit dataset:\n"
            f"{data_context}\n"
            f"Assess: potential protected attributes, disparate impact, demographic parity."
        ),
        "model_performance": (
            f"You are a model performance reviewer. Assess readiness of:\n"
            f"{data_context}\n"
            f"Assess: appropriate metrics (AUC-PR for imbalanced), validation strategy, "
            f"production readiness criteria."
        ),
    }

    result = await pattern.run(
        goal=(
            "Perform a comprehensive pre-deployment audit of the Singapore credit "
            "scoring model. Identify risks in data quality, fairness, and performance."
        ),
        worker_prompts=worker_prompts,
    )

    print(f"\nSupervisor task assignments:")
    for t in result.task_assignments:
        print(f"  → {t}")
    print(f"\nSynthesized recommendation:")
    print(f"  {result.final_recommendation[:300]}...")
    print(f"\nGaps identified: {result.gaps}")

    return result


supervisor_result = asyncio.run(pattern_supervisor_worker())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Sequential Pipeline Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: Strict ordering. Each agent receives the previous agent's
# handoff context and builds on it. No parallelism — order matters.
#
# ML use case: Feature engineering pipeline where each stage depends
# on the prior stage's decisions.
#
#   DataProfiler → FeatureSelector → TransformDesigner → ValidationPlan
#       handoff       →   handoff      →   handoff       → final output


async def pattern_sequential():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 2: Sequential Pipeline")
    print(f"{'─' * 60}")
    print(f"Scenario: Feature engineering — each stage depends on prior output")

    # Each stage is a separate Delegate — they share context via handoff
    stage_agents = [
        Delegate(model=model, max_llm_cost_usd=0.5),  # Stage 1: profile
        Delegate(model=model, max_llm_cost_usd=0.5),  # Stage 2: select
        Delegate(model=model, max_llm_cost_usd=0.5),  # Stage 3: transform
        Delegate(model=model, max_llm_cost_usd=0.5),  # Stage 4: validate
    ]

    stage_prompts = [
        (
            "Stage 1 — Data Profiler: Profile this credit dataset and identify the "
            "most important characteristics for feature engineering:\n"
            f"{data_context}\n"
            "Output a concise handoff context for the feature selector."
        ),
        (
            "Stage 2 — Feature Selector: Based on the profiler's findings, select "
            "the most informative features for predicting default. Explain your "
            "selection rationale. Output a handoff context for the transformer."
        ),
        (
            "Stage 3 — Transform Designer: Design the transformation pipeline for "
            "the selected features. Specify: encoding, scaling, imputation strategies. "
            "Output a handoff context for the validation planner."
        ),
        (
            "Stage 4 — Validation Planner: Design a validation strategy for this "
            "feature engineering pipeline. Specify: CV strategy, held-out test, "
            "metrics to track. Output the final validation plan."
        ),
    ]

    pattern = SequentialPattern(
        agents=stage_agents,
        stage_names=[
            "data_profiler",
            "feature_selector",
            "transform_designer",
            "validation_planner",
        ],
    )

    result = await pattern.run(
        initial_context=data_context,
        stage_prompts=stage_prompts,
    )

    print(f"\nPipeline stages completed: {len(result.stage_outputs)}")
    for i, (name, output) in enumerate(result.stage_outputs.items()):
        print(f"\n  Stage {i+1} ({name}):")
        print(f"    {output[:150]}...")

    print(f"\nFinal stage output:")
    print(f"  {result.final_output[:300]}...")

    return result


sequential_result = asyncio.run(pattern_sequential())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Parallel Fan-out Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: Independent tasks dispatched concurrently; results merged.
# Each agent is unaware of the others. Fastest pattern — wall-clock
# time is max(individual times), not sum(individual times).
#
# ML use case: Model comparison — evaluate 3 model architectures in
# parallel and merge results for final selection.
#
#   ┌─ LightGBM analyst ──┐
#   ├─ XGBoost analyst ───┤ → Merge → Final recommendation
#   └─ Neural Net analyst ┘


async def pattern_parallel():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 3: Parallel Fan-out")
    print(f"{'─' * 60}")
    print(f"Scenario: Model architecture comparison — 3 agents run concurrently")

    # Independent agents — same model, different tasks
    lgbm_agent = Delegate(model=model, max_llm_cost_usd=0.5)
    xgb_agent = Delegate(model=model, max_llm_cost_usd=0.5)
    nn_agent = Delegate(model=model, max_llm_cost_usd=0.5)

    parallel_tasks = {
        "lightgbm": (
            lgbm_agent,
            (
                f"You are a LightGBM specialist. Assess whether LightGBM is the right "
                f"model for this task:\n{data_context}\n"
                f"Evaluate: strengths, weaknesses, expected AUC-PR, training time, "
                f"interpretability. Give a priority rating (HIGH/MEDIUM/LOW)."
            ),
        ),
        "xgboost": (
            xgb_agent,
            (
                f"You are an XGBoost specialist. Assess whether XGBoost is the right "
                f"model for this task:\n{data_context}\n"
                f"Evaluate: strengths, weaknesses, expected AUC-PR, training time, "
                f"interpretability. Give a priority rating (HIGH/MEDIUM/LOW)."
            ),
        ),
        "neural_net": (
            nn_agent,
            (
                f"You are a neural network specialist. Assess whether a TabNet/MLP is "
                f"right for this task:\n{data_context}\n"
                f"Evaluate: strengths, weaknesses, expected AUC-PR, training time, "
                f"interpretability. Give a priority rating (HIGH/MEDIUM/LOW)."
            ),
        ),
    }

    pattern = ParallelPattern(max_concurrency=3)

    # Launch all agents concurrently
    result = await pattern.run(tasks=parallel_tasks)

    print(f"\nParallel execution complete:")
    print(f"  Wall-clock time: {result.wall_clock_seconds:.2f}s")
    print(f"  Sum of individual times: {result.total_agent_seconds:.2f}s")
    print(
        f"  Speedup: {result.total_agent_seconds / max(result.wall_clock_seconds, 0.01):.1f}x"
    )

    print(f"\nAgent results:")
    for agent_name, agent_output in result.agent_results.items():
        print(f"\n  [{agent_name}]: {agent_output[:200]}...")

    # Merge phase: a synthesis agent picks the winner
    merger = Delegate(model=model, max_llm_cost_usd=0.5)
    merge_prompt = (
        f"Three ML specialists have assessed model architectures for credit default prediction.\n\n"
        + "\n\n".join(
            f"--- {name} ---\n{output}" for name, output in result.agent_results.items()
        )
        + "\n\nSynthesize these assessments. Which architecture do you recommend and why?"
    )
    merged_text = ""
    async for event in merger.run(merge_prompt):
        if hasattr(event, "text"):
            merged_text += event.text

    print(f"\nMerged recommendation:")
    print(f"  {merged_text[:300]}...")

    return result, merged_text


parallel_result, merged_recommendation = asyncio.run(pattern_parallel())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Handoff Pattern
# ══════════════════════════════════════════════════════════════════════
#
# Pattern: An agent evaluates a task and either handles it or transfers
# ownership to a more appropriate specialist. Creates a routing layer
# without a central coordinator.
#
# ML use case: Intelligent request routing — an incoming model query
# is routed to the right specialist based on query type.
#
#   Query → Router Agent
#                ├── HANDLE (within capability)
#                └── HANDOFF → Specialist Agent
#                                  └── Result


async def pattern_handoff():
    print(f"\n{'─' * 60}")
    print(f"PATTERN 4: Handoff")
    print(f"{'─' * 60}")
    print(f"Scenario: Query routing — route ML questions to the right specialist")

    # Define specialist agents
    specialists = {
        "data_quality": Delegate(model=model, max_llm_cost_usd=0.5),
        "model_performance": Delegate(model=model, max_llm_cost_usd=0.5),
        "deployment": Delegate(model=model, max_llm_cost_usd=0.5),
        "governance": Delegate(model=model, max_llm_cost_usd=0.5),
    }

    # Handoff pattern with a router
    pattern = HandoffPattern(
        router_model=model,
        specialists=specialists,
        router_cost_budget=0.3,
        specialist_cost_budget=0.5,
        routing_criteria={
            "data_quality": "data issues, null values, distributions, profiling",
            "model_performance": "AUC, precision, recall, metrics, evaluation",
            "deployment": "serving, latency, InferenceServer, production, scaling",
            "governance": "fairness, bias, compliance, audit, PACT, regulations",
        },
    )

    queries = [
        "What is the null rate in the annual_income column?",
        "Why is AUC-PR better than AUC-ROC for this credit dataset?",
        "How do I reduce inference latency below 20ms?",
        "Does the credit model satisfy MAS FEAT requirements?",
    ]

    print(f"\nRouting {len(queries)} queries:")
    for q in queries:
        result = await pattern.run(query=q)
        print(f"\n  Query: {q}")
        print(f"  Routed to: {result.routed_to}")
        print(f"  Response: {result.response[:150]}...")

    return pattern


handoff_pattern = asyncio.run(pattern_handoff())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Pattern selection guide
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'═' * 60}")
print(f"   PATTERN SELECTION GUIDE")
print(f"{'═' * 60}")
print(
    """
┌──────────────────────┬──────────────────────────────────────────┐
│ Pattern              │ When to use                              │
├──────────────────────┼──────────────────────────────────────────┤
│ Supervisor-Worker    │ - Complex goal needing decomposition     │
│                      │ - Workers have different specialisations │
│                      │ - Need global synthesis across workers   │
│                      │ - Example: model audit, research report  │
├──────────────────────┼──────────────────────────────────────────┤
│ Sequential Pipeline  │ - Strict ordering required               │
│                      │ - Each stage needs prior output          │
│                      │ - State must accumulate progressively    │
│                      │ - Example: ETL, feature engineering      │
├──────────────────────┼──────────────────────────────────────────┤
│ Parallel Fan-out     │ - Independent subtasks                   │
│                      │ - Latency matters more than cost         │
│                      │ - Results need merging at end            │
│                      │ - Example: model comparison, search      │
├──────────────────────┼──────────────────────────────────────────┤
│ Handoff              │ - Heterogeneous query types              │
│                      │ - Single entry point, multiple handlers  │
│                      │ - Agents know their own limits           │
│                      │ - Example: support routing, triage       │
└──────────────────────┴──────────────────────────────────────────┘

Cost comparison (approx for this exercise):
  Supervisor-Worker: N_workers × worker_cost + supervisor_cost
  Sequential:        Sum of all stage costs (sequential billing)
  Parallel:          Max of individual costs (wall-clock savings)
  Handoff:           router_cost + 1 × specialist_cost

In Module 6, all patterns are wrapped with PACT governance:
  → Each agent gets a frozen GovernanceContext
  → Supervisor cannot grant workers more than its own permissions
  → Every handoff is logged in the AuditChain
"""
)

print("✓ Exercise 7 complete — four multi-agent orchestration patterns")
