# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 6: Multi-Agent A2A Coordination
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Full multi-agent orchestration using A2A protocol: research
#   → analyze → engineer → review agents. End-to-end autonomous ML
#   pipeline driven by agent coordination.
#
# TASKS:
#   1. Define agent roles and A2A message protocol
#   2. Build research agent (data discovery)
#   3. Build analysis agent (EDA + profiling)
#   4. Build engineering agent (feature + model)
#   5. Build review agent (quality gate)
#   6. Orchestrate full pipeline with GovernedSupervisor preview
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import Signature, InputField, OutputField
from kaizen.core.base_agent import BaseAgent
from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define agent roles and A2A message protocol
# ══════════════════════════════════════════════════════════════════════

# A2A (Agent-to-Agent) protocol: agents communicate via structured messages
# Each Signature defines the input and output contract for one agent role.


# TODO: Define ResearchOutput with one InputField and three OutputFields.
# InputField: task (str) — description of the research task
# OutputField: dataset_summary (str), quality_issues (list[str]), feasibility (str)
class ResearchOutput(Signature):
    """Research agent output: dataset discovery and assessment."""

    # Hint: task: str = InputField(description="Research task")
    task: str = ____

    # Hint: dataset_summary: str = OutputField(description="Summary of available data")
    dataset_summary: str = ____
    # Hint: quality_issues: list[str] = OutputField(description="Data quality issues found")
    quality_issues: list[str] = ____
    # Hint: feasibility: str = OutputField(description="Assessment of ML feasibility")
    feasibility: str = ____


# TODO: Define AnalysisOutput with one InputField and three OutputFields.
# InputField: research_context (str) — research findings to build on
# OutputFields: eda_findings (list[str]), feature_recommendations (list[str]),
#               modeling_approach (str)
class AnalysisOutput(Signature):
    """Analysis agent output: EDA results and recommendations."""

    # Hint: research_context: str = InputField(description="Research findings to build on")
    research_context: str = ____

    eda_findings: list[str] = ____
    feature_recommendations: list[str] = ____
    modeling_approach: str = ____


# TODO: Define EngineeringOutput with one InputField and three OutputFields.
# InputField: analysis_context (str)
# OutputFields: model_architecture (str), preprocessing_steps (list[str]),
#               expected_performance (str)
class EngineeringOutput(Signature):
    """Engineering agent output: model design and implementation plan."""

    analysis_context: str = ____

    model_architecture: str = ____
    preprocessing_steps: list[str] = ____
    expected_performance: str = ____


# TODO: Define ReviewOutput with one InputField and three OutputFields.
# InputField: pipeline_context (str)
# OutputFields: approved (bool), issues (list[str]), improvements (list[str])
class ReviewOutput(Signature):
    """Review agent output: quality gate decision."""

    pipeline_context: str = ____

    approved: bool = ____
    issues: list[str] = ____
    improvements: list[str] = ____


# ══════════════════════════════════════════════════════════════════════
# TASK 2-5: Build specialized agents
# ══════════════════════════════════════════════════════════════════════


async def run_pipeline():
    """Run the full multi-agent pipeline."""

    data_summary = (
        f"Singapore Credit Scoring: {credit.height:,} rows, "
        f"{len(credit.columns)} cols, target=default ({credit['default'].mean():.1%}), "
        f"columns: {', '.join(credit.columns[:10])}..."
    )

    # Step 1: Research Agent
    print("=== Agent 1: Research ===")
    # TODO: create a Delegate for the research agent with max_llm_cost_usd=1.0
    # Hint: Delegate(model=model, max_llm_cost_usd=1.0)
    research_agent = ____

    # TODO: build a prompt instructing the agent to assess ML feasibility
    # Include data_summary and request: dataset summary, quality issues, ML feasibility
    research_prompt = ____

    # TODO: stream the agent response into research_text
    # Hint: async for event in research_agent.run(research_prompt):
    #           if hasattr(event, "text"): research_text += event.text
    research_text = ""
    ____  # your streaming loop here

    print(f"Research: {research_text[:200]}...")

    # Step 2: Analysis Agent (receives research context via A2A)
    print("\n=== Agent 2: Analysis ===")
    # TODO: create a Delegate for the analysis agent with max_llm_cost_usd=1.0
    analysis_agent = ____

    # TODO: build a prompt using research_text[:500] as context
    # Request: EDA findings, feature recommendations, modeling approach
    analysis_prompt = ____

    # TODO: stream the analysis_agent response into analysis_text
    analysis_text = ""
    ____  # your streaming loop here

    print(f"Analysis: {analysis_text[:200]}...")

    # Step 3: Engineering Agent
    print("\n=== Agent 3: Engineering ===")
    # TODO: create a Delegate for the engineering agent with max_llm_cost_usd=1.0
    engineering_agent = ____

    # TODO: build a prompt using analysis_text[:500] as context
    # Request: model architecture, preprocessing pipeline, expected performance
    engineering_prompt = ____

    # TODO: stream the engineering_agent response into engineering_text
    engineering_text = ""
    ____  # your streaming loop here

    print(f"Engineering: {engineering_text[:200]}...")

    # Step 4: Review Agent (quality gate)
    print("\n=== Agent 4: Review (Quality Gate) ===")
    # TODO: create a Delegate for the review agent with max_llm_cost_usd=1.0
    review_agent = ____

    # TODO: build a review_prompt combining research[:300], analysis[:300],
    #       engineering[:300] context; ask agent to approve/reject with issues list
    review_prompt = ____

    # TODO: stream the review_agent response into review_text
    review_text = ""
    ____  # your streaming loop here

    print(f"Review: {review_text[:200]}...")

    return {
        "research": research_text,
        "analysis": analysis_text,
        "engineering": engineering_text,
        "review": review_text,
    }


pipeline_results = asyncio.run(run_pipeline())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: GovernedSupervisor preview (seeds Module 6)
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== GovernedSupervisor Preview ===")
print(
    """
In Module 6, this multi-agent pipeline gets wrapped with PACT governance:

  from pact import GovernanceEngine, PactGovernedAgent

  # Each agent gets a frozen GovernanceContext
  governed_research = PactGovernedAgent(
      agent=research_agent,
      governance_context=GovernanceContext(
          cost_budget_usd=1.0,
          data_access=["ascent03/*"],  # Can only access credit data
          tools_allowed=["profile_data", "describe_column"],
      )
  )

  # GovernedSupervisor orchestrates with oversight
  supervisor = GovernedSupervisor(
      agents=[governed_research, governed_analysis, governed_engineering, governed_review],
      governance_engine=governance_engine,
  )

Key governance properties:
  1. Monotonic tightening: child agents cannot exceed parent permissions
  2. GovernanceContext is FROZEN: agents cannot modify their own governance
  3. AuditChain: every agent action logged with tamper-evident chain
  4. Fail-closed: if governance check fails, access is DENIED (not allowed)

This is NOT optional. In regulated industries (banking, healthcare),
ungoverned AI agents are a liability.
"""
)

# Compare autonomous vs manual
print(f"=== Autonomous vs Manual Pipeline ===")
print(f"Manual (Modules 2-4): You wrote every step, chose every parameter")
print(f"Autonomous (Module 5): Agents suggested, reasoned, decided")
print(f"Governed (Module 6): Same autonomy, but with formal accountability")
print(f"\nThe progression: manual → autonomous → governed autonomous")
print(f"This IS the future of production ML.")

print("\n✓ Exercise 6 complete — multi-agent A2A coordination")
print(
    "  Module 5 complete: 6 exercises covering LLMs, agents, RAG, and multi-agent systems"
)
