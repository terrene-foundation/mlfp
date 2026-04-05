# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Pipeline Composition
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compose agents into multi-agent pipelines
# LEVEL: Advanced
# PARITY: Equivalent — Rust has orchestration runtime with pipeline patterns
# VALIDATES: Pipeline, Pipeline.sequential(), Pipeline.router(),
#            Pipeline.parallel(), Pipeline.ensemble(), Pipeline.supervisor_worker(),
#            SequentialPipeline, Pipeline.to_agent()
#
# Run: uv run python textbook/python/04-agents/08_pipeline.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.patterns.pipeline import Pipeline, SequentialPipeline
from kaizen.core.base_agent import BaseAgent, BaseAgentConfig
from kaizen.signatures import InputField, OutputField, Signature

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. Pipeline — abstract base class ─────────────────────────────────
# Pipeline is the composable abstraction for multi-agent workflows.
# It provides:
#   .run(**inputs)  -> execute the pipeline
#   .to_agent()     -> convert pipeline to a BaseAgent for composition
#
# Static factory methods create specific pipeline patterns:
#   Pipeline.sequential()       -> linear agent chain
#   Pipeline.router()           -> intelligent request routing
#   Pipeline.parallel()         -> concurrent execution
#   Pipeline.ensemble()         -> multi-perspective collaboration
#   Pipeline.supervisor_worker() -> delegated task execution
#   Pipeline.blackboard()       -> iterative specialist collaboration
#   Pipeline.consensus()        -> democratic voting
#   Pipeline.debate()           -> adversarial reasoning
#   Pipeline.handoff()          -> tier escalation

# ── 2. Custom Pipeline subclass ───────────────────────────────────────
# Pipelines implement run() with multi-step logic.


class DataProcessingPipeline(Pipeline):
    """Example: a three-step data processing pipeline."""

    def run(self, **inputs):
        data = inputs.get("data", "")
        # Step 1: Clean
        cleaned = data.strip().lower()
        # Step 2: Transform
        transformed = cleaned.replace(" ", "_")
        # Step 3: Validate
        is_valid = len(transformed) > 0
        return {
            "original": data,
            "cleaned": cleaned,
            "transformed": transformed,
            "valid": is_valid,
        }


pipeline = DataProcessingPipeline()
result = pipeline.run(data="  Hello World  ")

assert result["original"] == "  Hello World  "
assert result["cleaned"] == "hello world"
assert result["transformed"] == "hello_world"
assert result["valid"] is True

# ── 3. Pipeline.to_agent() — pipeline as agent ────────────────────────
# Convert any pipeline into a BaseAgent for use in larger orchestrations.
# This enables composition: pipelines within pipelines.

agent = pipeline.to_agent(
    name="data_processor",
    description="Cleans and transforms data",
)

assert isinstance(agent, BaseAgent)
assert agent.agent_id == "data_processor"

# The agent delegates to the pipeline's run()
agent_result = agent.run(data="  Test Input  ")
assert agent_result["transformed"] == "test_input"

# ── 4. SequentialPipeline — linear agent chain ────────────────────────
# SequentialPipeline executes agents in order, passing each agent's
# output as input to the next agent.
#
# Input -> [Agent 1] -> [Agent 2] -> [Agent 3] -> Output


# Create minimal agents for testing
class StepAgent(BaseAgent):
    """Minimal agent that adds a step marker to input."""

    def __init__(self, step_name: str):
        config = BaseAgentConfig(llm_provider="mock", model="mock")

        class StepSig(Signature):
            """Process a pipeline step."""

            data: str = InputField(description="Input data")
            result: str = OutputField(description="Step output")

        super().__init__(config=config, signature=StepSig())
        self.step_name = step_name

    def run(self, **inputs):
        data = inputs.get("data", inputs.get("result", "start"))
        return {
            "result": f"{data} -> {self.step_name}",
            "data": f"{data} -> {self.step_name}",
        }


agent_a = StepAgent("extract")
agent_b = StepAgent("transform")
agent_c = StepAgent("load")

# Create sequential pipeline via factory method
seq_pipeline = Pipeline.sequential(agents=[agent_a, agent_b, agent_c])

assert isinstance(seq_pipeline, SequentialPipeline)
assert len(seq_pipeline.agents) == 3

# Execute the sequential pipeline
seq_result = seq_pipeline.run(data="raw_data")

assert "final_output" in seq_result
assert "intermediate_results" in seq_result
assert len(seq_result["intermediate_results"]) == 3

# Each step is recorded
steps = seq_result["intermediate_results"]
assert steps[0]["step"] == 1
assert steps[0]["agent"] == "StepAgent"
assert steps[1]["step"] == 2
assert steps[2]["step"] == 3

# ── 5. SequentialPipeline.to_agent() — composable ─────────────────────
# Sequential pipelines can also be converted to agents.

seq_agent = seq_pipeline.to_agent(name="etl_pipeline")
assert isinstance(seq_agent, BaseAgent)

# ── 6. Pipeline.router() — intelligent routing ────────────────────────
# Routes requests to the best agent based on A2A capability matching.
# Falls back to round-robin when A2A is unavailable.
#
# Pipeline.router(
#     agents=[code_agent, data_agent, writing_agent],
#     routing_strategy="semantic",  # A2A-based
# )
#
# Routing strategies:
#   "semantic"    -> A2A capability matching (recommended)
#   "round-robin" -> alternating between agents
#   "random"      -> random selection
#
# NOTE: We do not instantiate the router here because it requires
# MetaControllerPipeline internal imports. The factory method is the
# intended entry point.

# ── 7. Pipeline.parallel() — concurrent execution ─────────────────────
# Executes all agents concurrently, then aggregates results.
# Achieves significant speedup over sequential execution.
#
# pipeline = Pipeline.parallel(
#     agents=[agent1, agent2, agent3],
#     aggregator=combine_results,  # Optional custom aggregator
#     max_workers=5,
#     timeout=30.0,
# )
#
# result = pipeline.run(input="test_data")

# ── 8. Pipeline.ensemble() — multi-perspective ────────────────────────
# Selects top-k agents with best capability matches (via A2A),
# executes them, then synthesizes perspectives into a unified result.
#
# pipeline = Pipeline.ensemble(
#     agents=[code_agent, data_agent, writing_agent, research_agent],
#     synthesizer=synthesis_agent,
#     discovery_mode="a2a",  # or "all"
#     top_k=3,
# )

# ── 9. Pipeline.supervisor_worker() — delegated execution ─────────────
# Supervisor delegates tasks to workers with A2A semantic matching.
#
# pattern = Pipeline.supervisor_worker(
#     supervisor=supervisor_agent,
#     workers=[code_expert, data_expert, writing_expert],
#     selection_mode="semantic",  # or "round-robin"
# )
#
# tasks = pattern.delegate("Process 100 documents")
# results = pattern.aggregate_results(tasks[0]["request_id"])

# ── 10. Pipeline.blackboard() — iterative specialists ─────────────────
# Maintains shared state, iteratively selects specialists based on
# evolving needs, and uses a controller to determine completion.
#
# pipeline = Pipeline.blackboard(
#     specialists=[solver, analyst, optimizer, validator],
#     controller=controller_agent,
#     selection_mode="semantic",
#     max_iterations=5,
# )

# ── 11. Pipeline pattern summary ──────────────────────────────────────
# Each pattern serves a different coordination need:
#
# | Pattern           | Use Case                          | Agents  |
# |-------------------|-----------------------------------|---------|
# | sequential        | Linear processing chain           | N       |
# | router            | Best-agent selection              | N       |
# | parallel          | Concurrent independent tasks      | N       |
# | ensemble          | Multi-perspective synthesis       | N + 1   |
# | supervisor_worker | Delegated task management         | 1 + N   |
# | blackboard        | Iterative specialist refinement   | N + 1   |
# | consensus         | Democratic voting                 | N       |
# | debate            | Adversarial argument evaluation   | 2+      |
# | handoff           | Tier-based escalation             | N tiers |

print("PASS: 04-agents/08_pipeline")
