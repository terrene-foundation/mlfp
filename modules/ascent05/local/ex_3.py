# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 3: ReActAgent with Tools
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ReActAgent with custom tools for autonomous data
#   exploration. Observe reasoning-action traces and tool selection.
#
# TASKS:
#   1. Define custom tools (DataExplorer, FeatureEngineer, ModelVisualizer)
#   2. Build ReActAgent with tool access
#   3. Run autonomous data exploration
#   4. Observe and analyse reasoning-action trace
#   5. Safety: what happens without cost budgets?
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.react import ReActAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define tools for the ReActAgent
# ══════════════════════════════════════════════════════════════════════


# Tools are functions the agent can call during reasoning
async def tool_profile_data(dataset_name: str) -> str:
    """Profile a dataset using DataExplorer and return summary."""
    # TODO: Import DataExplorer from kailash_ml and profile the credit dataset
    # Hint: from kailash_ml import DataExplorer
    #       explorer = DataExplorer()
    #       profile = await explorer.profile(credit.sample(...))
    # TODO: Return a formatted string with: dataset name, rows, columns,
    #       alert count, type summary, and top 5 alert types
    # Hint: profile has .n_rows, .n_columns, .alerts, .type_summary attributes
    ____


async def tool_check_correlations(threshold: float = 0.8) -> str:
    """Find highly correlated features."""
    # TODO: Import DataExplorer and AlertConfig from kailash_ml
    # Hint: from kailash_ml import DataExplorer, AlertConfig
    #       Use AlertConfig(high_correlation_threshold=threshold)
    # TODO: Profile a 5000-row sample and extract pairs where abs(corr) > threshold
    # Hint: profile.correlation_matrix is a dict[str, dict[str, float]]
    #       Skip self-correlations (col_a != col_b) and duplicates
    # TODO: Return formatted string listing high-correlation pairs
    ____


def tool_describe_column(column_name: str) -> str:
    """Get statistics for a specific column."""
    # TODO: Check if column_name exists in credit.columns; return error message if not
    # Hint: if column_name not in credit.columns: return f"Column '____' not found..."
    # TODO: For numeric columns (Float64, Float32, Int64, Int32): return mean, std, min, max, nulls
    # Hint: col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)
    # TODO: For categorical columns: return unique count, null count, top 5 value_counts
    ____


def tool_default_rate_by(column_name: str) -> str:
    """Compute default rate grouped by a column."""
    # TODO: Check if column_name exists; return error if not
    # TODO: Group credit by column_name, aggregate default mean (rate) and count
    # Hint: credit.group_by(column_name).agg(
    #           pl.col("default").mean().alias("default_rate"),
    #           pl.col("default").count().alias("n"),
    #       ).sort("default_rate", descending=True).head(10)
    # TODO: Format and return the grouped result as a readable string
    ____


# TODO: Build the tool registry dict mapping string names to functions
# Hint: tools = {"profile_data": tool_profile_data, "check_correlations": ____, ...}
tools = ____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build ReActAgent
# ══════════════════════════════════════════════════════════════════════


class DataExplorationResult(Signature):
    """Autonomous data exploration result."""

    # TODO: Define one InputField: task (data exploration task description)
    task: str = ____

    # TODO: Define three OutputFields: findings (list[str] of key findings),
    #       tools_used (list[str] of tools invoked), recommendation (str)
    findings: list[str] = ____
    tools_used: list[str] = ____
    recommendation: str = ____


async def run_react_agent():
    # TODO: Create a ReActAgent with DataExplorationResult signature,
    #       the tools dict, budget $3.00, and max_iterations=10
    # Hint: ReActAgent(signature=____, model=____, tools=____, max_llm_cost_usd=____, max_iterations=____)
    agent = ____

    task = (
        "Explore the Singapore credit scoring dataset. "
        "Profile the data, check for high correlations, "
        "find which features most strongly predict default, "
        "and recommend preprocessing steps for a classification model."
    )

    print(f"\n=== ReActAgent Exploration ===")
    print(f"Task: {task}")
    print(f"Available tools: {list(tools.keys())}")
    print(f"Max iterations: 10")
    print(f"Cost budget: $3.00\n")

    # TODO: Run the agent passing task as a keyword argument
    # Hint: await agent.run(task=____)
    result = ____

    print(f"\n=== Results ===")
    print(f"Tools used: {result.tools_used}")
    print(f"\nFindings:")
    for i, finding in enumerate(result.findings):
        print(f"  {i+1}. {finding}")
    print(f"\nRecommendation: {result.recommendation}")

    return result


react_result = asyncio.run(run_react_agent())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Analyse reasoning-action trace
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Reasoning-Action Trace ===")
print("ReAct loop: Thought → Action → Observation → Thought → ...")
print(f"The agent autonomously decided which tools to call and in what order.")
print(f"This is the key difference from CoT (Exercise 2):")
print(f"  CoT: reason step-by-step, produce answer")
print(f"  ReAct: reason, ACT on the world (call tools), observe results, reason again")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Safety — cost budgets and runaway agents
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Safety: Cost Budgets ===")
print(
    """
What happens if you REMOVE the cost budget?

1. The agent could loop indefinitely (calling tools → reasoning → more tools)
2. Each LLM call costs money — 100 iterations at $0.05/call = $5.00
3. What if the agent calls DataExplorer on a 100GB dataset?
   → Memory exhaustion, cluster costs, timeout failures

max_llm_cost_usd is NOT optional. It is a governance requirement.

In production:
  - Set budgets proportional to task complexity
  - Monitor actual spend vs budget
  - PACT (Module 6) enforces budgets as operating envelopes
  - Agents cannot modify their own budget (frozen GovernanceContext)
"""
)

print("\n✓ Exercise 3 complete — ReActAgent with tools and safety analysis")
