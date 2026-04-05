# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 3: ReActAgent with Tools
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
    from kailash_ml import DataExplorer

    explorer = DataExplorer()
    profile = await explorer.profile(
        credit.sample(n=min(10_000, credit.height), seed=42)
    )
    return (
        f"Dataset: {dataset_name}\n"
        f"Rows: {profile.n_rows}, Columns: {profile.n_columns}\n"
        f"Alerts: {len(profile.alerts)}\n"
        f"Types: {profile.type_summary}\n"
        f"Top alerts: {[a['type'] + ':' + str(a.get('column', '')) for a in profile.alerts[:5]]}"
    )


async def tool_check_correlations(threshold: float = 0.8) -> str:
    """Find highly correlated features."""
    from kailash_ml import DataExplorer, AlertConfig

    explorer = DataExplorer(
        alert_config=AlertConfig(high_correlation_threshold=threshold)
    )
    profile = await explorer.profile(credit.sample(n=5000, seed=42))

    if profile.correlation_matrix:
        high_corrs = []
        seen = set()
        for col_a, row in profile.correlation_matrix.items():
            for col_b, corr in row.items():
                if (
                    col_a != col_b
                    and abs(corr) > threshold
                    and (col_b, col_a) not in seen
                ):
                    seen.add((col_a, col_b))
                    high_corrs.append(f"{col_a} <-> {col_b}: {corr:.3f}")
        return f"High correlations (>{threshold}):\n" + "\n".join(high_corrs[:10])
    return "No correlation matrix available"


def tool_describe_column(column_name: str) -> str:
    """Get statistics for a specific column."""
    if column_name not in credit.columns:
        return f"Column '{column_name}' not found. Available: {credit.columns}"
    col = credit[column_name]
    if col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        return (
            f"{column_name}: mean={col.mean():.3f}, std={col.std():.3f}, "
            f"min={col.min()}, max={col.max()}, nulls={col.null_count()}"
        )
    return f"{column_name}: unique={col.n_unique()}, nulls={col.null_count()}, top={col.value_counts().head(5)}"


def tool_default_rate_by(column_name: str) -> str:
    """Compute default rate grouped by a column."""
    if column_name not in credit.columns:
        return f"Column '{column_name}' not found"
    result = (
        credit.group_by(column_name)
        .agg(
            pl.col("default").mean().alias("default_rate"),
            pl.col("default").count().alias("n"),
        )
        .sort("default_rate", descending=True)
        .head(10)
    )
    rows = [
        f"  {row[column_name]}: {row['default_rate']:.3f} (n={row['n']})"
        for row in result.iter_rows(named=True)
    ]
    return f"Default rate by {column_name}:\n" + "\n".join(rows)


# Tool registry
tools = {
    "profile_data": tool_profile_data,
    "check_correlations": tool_check_correlations,
    "describe_column": tool_describe_column,
    "default_rate_by": tool_default_rate_by,
}


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build ReActAgent
# ══════════════════════════════════════════════════════════════════════


class DataExplorationResult(Signature):
    """Autonomous data exploration result."""

    task: str = InputField(description="Data exploration task")

    findings: list[str] = OutputField(description="Key findings from exploration")
    tools_used: list[str] = OutputField(description="Tools invoked during exploration")
    recommendation: str = OutputField(description="Recommendation based on findings")


async def run_react_agent():
    agent = ReActAgent(
        signature=DataExplorationResult,
        model=model,
        tools=tools,
        max_llm_cost_usd=3.0,
        max_iterations=10,
    )

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

    result = await agent.run(task=task)

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
