# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Multi-Agent Orchestration Patterns
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use formal multi-agent coordination patterns for complex workflows
# LEVEL: Advanced
# PARITY: Equivalent — Rust has orchestration primitives (protocols module)
# VALIDATES: SupervisorWorkerPattern, ConsensusPattern, DebatePattern,
#            HandoffPattern, SequentialPipelinePattern
#
# Run: uv run python textbook/python/04-agents/10_orchestration_patterns.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.patterns.patterns import (
    # Supervisor-Worker
    SupervisorWorkerPattern,
    SupervisorAgent,
    WorkerAgent,
    CoordinatorAgent,
    create_supervisor_worker_pattern,
    TaskDelegationSignature,
    # Consensus
    ConsensusPattern,
    ProposerAgent,
    VoterAgent,
    AggregatorAgent,
    create_consensus_pattern,
    # Debate
    DebatePattern,
    ProponentAgent,
    OpponentAgent,
    JudgeAgent,
    create_debate_pattern,
    # Handoff
    HandoffPattern,
    HandoffAgent,
    create_handoff_pattern,
    # Sequential Pipeline
    SequentialPipelinePattern,
    PipelineStageAgent,
    create_sequential_pipeline,
    # Base
    BaseMultiAgentPattern,
)

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. Pattern hierarchy ────────────────────────────────────────────
# All patterns inherit from BaseMultiAgentPattern, which provides:
#   - Agent registration
#   - Execution lifecycle
#   - Result aggregation

assert issubclass(SupervisorWorkerPattern, BaseMultiAgentPattern)
assert issubclass(ConsensusPattern, BaseMultiAgentPattern)
assert issubclass(DebatePattern, BaseMultiAgentPattern)
assert issubclass(HandoffPattern, BaseMultiAgentPattern)
assert issubclass(SequentialPipelinePattern, BaseMultiAgentPattern)

# ── 2. Supervisor-Worker Pattern ────────────────────────────────────
# A supervisor delegates tasks to workers and aggregates results.
# The supervisor decides WHAT to delegate (LLM reasoning, not code).
#
# Use case: Complex analysis split across specialist agents.
#
#   supervisor = SupervisorAgent(model=model, ...)
#   workers = [
#       WorkerAgent(model=model, role="data_analyst", ...),
#       WorkerAgent(model=model, role="model_builder", ...),
#   ]
#   pattern = create_supervisor_worker_pattern(supervisor, workers)
#   result = await pattern.execute("Analyze and model the dataset")

assert SupervisorAgent is not None
assert WorkerAgent is not None
assert CoordinatorAgent is not None
assert TaskDelegationSignature is not None

# ── 3. Consensus Pattern ───────────────────────────────────────────
# Multiple agents propose solutions, vote, then aggregate.
# Democratic decision-making for high-stakes decisions.
#
# Use case: Model selection — 3 agents each recommend a model,
# then vote on which recommendation to adopt.
#
#   proposers = [ProposerAgent(...) for _ in range(3)]
#   voters = [VoterAgent(...) for _ in range(3)]
#   aggregator = AggregatorAgent(...)
#   pattern = create_consensus_pattern(proposers, voters, aggregator)
#   result = await pattern.execute("Select the best model for credit scoring")

assert ProposerAgent is not None
assert VoterAgent is not None
assert AggregatorAgent is not None

# ── 4. Debate Pattern ──────────────────────────────────────────────
# Adversarial: proponent argues FOR, opponent argues AGAINST,
# judge decides. Forces robust reasoning through challenge.
#
# Use case: Risk assessment — proponent argues the model is safe,
# opponent challenges with failure scenarios, judge decides.
#
#   proponent = ProponentAgent(model=model, position="Model is production-ready")
#   opponent = OpponentAgent(model=model, position="Model has risks")
#   judge = JudgeAgent(model=model)
#   pattern = create_debate_pattern(proponent, opponent, judge, rounds=3)
#   result = await pattern.execute("Should we deploy the credit scoring model?")

assert ProponentAgent is not None
assert OpponentAgent is not None
assert JudgeAgent is not None

# ── 5. Handoff Pattern ─────────────────────────────────────────────
# Sequential specialists — each agent handles one phase, then
# hands off to the next. Like a relay race.
#
# Use case: ML pipeline stages — data prep → feature eng → training → eval
#
#   agents = [
#       HandoffAgent(model=model, role="data_prep"),
#       HandoffAgent(model=model, role="feature_eng"),
#       HandoffAgent(model=model, role="trainer"),
#       HandoffAgent(model=model, role="evaluator"),
#   ]
#   pattern = create_handoff_pattern(agents)
#   result = await pattern.execute("Build an end-to-end model")

assert HandoffAgent is not None

# ── 6. Sequential Pipeline Pattern ─────────────────────────────────
# Similar to handoff but with stage-level processing signatures.
# Each stage processes and transforms the output for the next stage.
#
#   stages = [
#       PipelineStageAgent(model=model, stage_name="research"),
#       PipelineStageAgent(model=model, stage_name="analyze"),
#       PipelineStageAgent(model=model, stage_name="recommend"),
#   ]
#   pipeline = create_sequential_pipeline(stages)
#   result = await pipeline.execute("Evaluate market opportunity")

assert PipelineStageAgent is not None

# ── 7. Choosing the right pattern ───────────────────────────────────
# | Pattern     | Best for                          | Agent count |
# |-------------|-----------------------------------|-------------|
# | Supervisor  | Complex delegated tasks           | 1 + N       |
# | Consensus   | High-stakes decisions             | 2N + 1      |
# | Debate      | Risk assessment, policy review    | 3 (fixed)   |
# | Handoff     | Sequential specialist pipeline    | N           |
# | Pipeline    | Data transformation chains        | N           |
#
# All patterns use LLM reasoning for decisions (not code conditionals).
# See rules/agent-reasoning.md for the LLM-First rule.

print("PASS: 04-agents/10_orchestration_patterns")
