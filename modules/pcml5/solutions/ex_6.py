# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 6: Multi-Agent A2A Coordination
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
# TASK 1: Define agent roles and message protocol
# ══════════════════════════════════════════════════════════════════════

# A2A (Agent-to-Agent) protocol: agents communicate via structured messages


class ResearchOutput(Signature):
    """Research agent output: dataset discovery and assessment."""

    task: str = InputField(description="Research task")
    dataset_summary: str = OutputField(description="Summary of available data")
    quality_issues: list[str] = OutputField(description="Data quality issues found")
    feasibility: str = OutputField(description="Assessment of ML feasibility")


class AnalysisOutput(Signature):
    """Analysis agent output: EDA results and recommendations."""

    research_context: str = InputField(description="Research findings to build on")
    eda_findings: list[str] = OutputField(description="Key EDA findings")
    feature_recommendations: list[str] = OutputField(description="Recommended features")
    modeling_approach: str = OutputField(description="Suggested modeling approach")


class EngineeringOutput(Signature):
    """Engineering agent output: model design and implementation plan."""

    analysis_context: str = InputField(description="Analysis findings")
    model_architecture: str = OutputField(description="Chosen model architecture")
    preprocessing_steps: list[str] = OutputField(
        description="Data preprocessing pipeline"
    )
    expected_performance: str = OutputField(description="Expected model performance")


class ReviewOutput(Signature):
    """Review agent output: quality gate decision."""

    pipeline_context: str = InputField(description="Full pipeline description")
    approved: bool = OutputField(description="Whether the pipeline passes quality gate")
    issues: list[str] = OutputField(description="Issues found during review")
    improvements: list[str] = OutputField(description="Suggested improvements")


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
    research_agent = Delegate(model=model, max_llm_cost_usd=1.0)
    research_prompt = (
        f"You are a data research agent. Assess this dataset for ML feasibility:\n"
        f"{data_summary}\n"
        f"Provide: dataset summary, quality issues, ML feasibility assessment."
    )
    research_text = ""
    async for event in research_agent.run(research_prompt):
        if hasattr(event, "text"):
            research_text += event.text
    print(f"Research: {research_text[:200]}...")

    # Step 2: Analysis Agent (receives research context via A2A)
    print("\n=== Agent 2: Analysis ===")
    analysis_agent = Delegate(model=model, max_llm_cost_usd=1.0)
    analysis_prompt = (
        f"You are a data analysis agent. Based on research findings:\n"
        f"{research_text[:500]}\n\n"
        f"Provide: EDA findings, feature recommendations, modeling approach."
    )
    analysis_text = ""
    async for event in analysis_agent.run(analysis_prompt):
        if hasattr(event, "text"):
            analysis_text += event.text
    print(f"Analysis: {analysis_text[:200]}...")

    # Step 3: Engineering Agent
    print("\n=== Agent 3: Engineering ===")
    engineering_agent = Delegate(model=model, max_llm_cost_usd=1.0)
    engineering_prompt = (
        f"You are an ML engineering agent. Based on analysis:\n"
        f"{analysis_text[:500]}\n\n"
        f"Provide: model architecture choice, preprocessing pipeline, expected performance."
    )
    engineering_text = ""
    async for event in engineering_agent.run(engineering_prompt):
        if hasattr(event, "text"):
            engineering_text += event.text
    print(f"Engineering: {engineering_text[:200]}...")

    # Step 4: Review Agent (quality gate)
    print("\n=== Agent 4: Review (Quality Gate) ===")
    review_agent = Delegate(model=model, max_llm_cost_usd=1.0)
    review_prompt = (
        f"You are an ML review agent. Review this pipeline:\n"
        f"Research: {research_text[:300]}\n"
        f"Analysis: {analysis_text[:300]}\n"
        f"Engineering: {engineering_text[:300]}\n\n"
        f"Approve or reject. List issues and improvements."
    )
    review_text = ""
    async for event in review_agent.run(review_prompt):
        if hasattr(event, "text"):
            review_text += event.text
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
