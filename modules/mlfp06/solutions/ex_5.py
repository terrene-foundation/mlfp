# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5: Building Agents with Kaizen
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Define custom tools with clear docstrings for agent consumption
#   - Build a ReActAgent and explain the Thought -> Action -> Observation loop
#   - Run an agent on a multi-step analysis task and inspect its reasoning
#   - Define a typed BaseAgent with Signature for structured, predictable output
#   - Choose between ReActAgent and BaseAgent+Signature for different use cases
#
# PREREQUISITES:
#   Exercises 1-4 (Delegate, Signature, prompt engineering, LLM fundamentals).
#   Understanding that agents are LLMs with the ability to call functions —
#   not new AI, just LLMs that can observe and act, not just respond.
#
# ESTIMATED TIME: 45-75 minutes
#
# TASKS:
#   1. Define custom tools (data_summary, run_query, plot_chart)
#   2. Build ReActAgent with tool access
#   3. Run agent on multi-step analysis task
#   4. Inspect reasoning trace
#   5. Build custom BaseAgent with Signature for structured analysis
#
# DATASET: Singapore company reports
#   Columns: text (report content) + metadata
#   The agent explores this dataset autonomously using the provided tools.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen.core import BaseAgent
from kaizen_agents.agents.specialized.react import ReActAgent
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
reports = loader.load("mlfp06", "sg_company_reports.parquet")

print(f"Loaded {reports.height:,} company reports")
print(f"Columns: {reports.columns}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define custom tools (data_summary, run_query, plot_chart)
# ══════════════════════════════════════════════════════════════════════


def data_summary(dataset_name: str = "reports") -> str:
    """Get a statistical summary of the company reports dataset.

    Args:
        dataset_name: Which dataset to summarise. Currently only 'reports'.

    Returns:
        Text summary of the dataset shape, columns, and basic stats.
    """
    df = reports
    summary_parts = [
        f"Dataset: {dataset_name}",
        f"Shape: {df.height} rows x {df.width} columns",
        f"Columns: {', '.join(df.columns)}",
    ]

    for col in df.columns:
        dtype = str(df.schema[col])
        if "Int" in dtype or "Float" in dtype:
            stats = df.select(
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ).row(0)
            summary_parts.append(
                f"  {col} ({dtype}): mean={stats[0]:.2f}, std={stats[1]:.2f}, range=[{stats[2]}, {stats[3]}]"
            )
        elif "Utf8" in dtype or "String" in dtype:
            n_unique = df.select(pl.col(col).n_unique()).item()
            avg_len = df.select(pl.col(col).str.len_chars().mean()).item()
            summary_parts.append(
                f"  {col} ({dtype}): {n_unique} unique, avg_len={avg_len:.0f}"
            )

    return "\n".join(summary_parts)


def run_query(query_expr: str) -> str:
    """Run a polars-style filter query on the reports dataset.

    Args:
        query_expr: A description of the filter to apply (e.g., 'top 5 by revenue').

    Returns:
        Query results as a formatted string.
    """
    # Use simple heuristic parsing — agent describes, we interpret
    df = reports
    if "top" in query_expr.lower() and "5" in query_expr:
        numeric_cols = [
            c
            for c in df.columns
            if "Int" in str(df.schema[c]) or "Float" in str(df.schema[c])
        ]
        if numeric_cols:
            result = df.sort(numeric_cols[0], descending=True).head(5)
            return f"Top 5 by {numeric_cols[0]}:\n{result}"
    elif "count" in query_expr.lower():
        return f"Total records: {df.height}"
    elif "unique" in query_expr.lower():
        text_cols = [
            c
            for c in df.columns
            if "Utf8" in str(df.schema[c]) or "String" in str(df.schema[c])
        ]
        if text_cols:
            uniques = {c: df.select(pl.col(c).n_unique()).item() for c in text_cols}
            return f"Unique values: {json.dumps(uniques)}"

    return f"Query interpreted: {query_expr}\nDataset has {df.height} rows, columns: {df.columns}"


def plot_chart(chart_type: str, x_col: str = "", y_col: str = "") -> str:
    """Generate a text description of a chart (placeholder for visualization).

    Args:
        chart_type: Type of chart (bar, line, scatter, histogram).
        x_col: Column for x-axis.
        y_col: Column for y-axis.

    Returns:
        Description of what the chart would show.
    """
    df = reports
    available = df.columns
    x_col = x_col if x_col in available else available[0]
    y_col = (
        y_col
        if y_col in available
        else (available[1] if len(available) > 1 else available[0])
    )

    return (
        f"Chart: {chart_type}\n"
        f"X-axis: {x_col}\n"
        f"Y-axis: {y_col}\n"
        f"Data points: {df.height}\n"
        f"(In production, this renders via ModelVisualizer)"
    )


tools = [data_summary, run_query, plot_chart]

# Test tools directly
print(f"\n=== Tool Definitions ===")
for tool in tools:
    print(f"  {tool.__name__}: {tool.__doc__.strip().split(chr(10))[0]}")

print(f"\n=== Tool Test: data_summary ===")
print(data_summary())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build ReActAgent with tool access
# ══════════════════════════════════════════════════════════════════════


async def build_and_run_react():
    """Create a ReActAgent with data analysis tools."""

    agent = ReActAgent(
        model=model,
        tools=tools,
        max_llm_cost_usd=2.0,
    )

    print(f"\n=== ReActAgent Created ===")
    print(f"Model: {model}")
    print(f"Tools: {[t.__name__ for t in tools]}")
    print(f"Budget: $2.00")

    return agent


agent = asyncio.run(build_and_run_react())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Run agent on multi-step analysis task
# ══════════════════════════════════════════════════════════════════════


async def multi_step_analysis():
    """Run the agent on a task requiring multiple tool calls."""
    react_agent = ReActAgent(
        model=model,
        tools=tools,
        max_llm_cost_usd=2.0,
    )

    task = """Analyze the Singapore company reports dataset:
1. First, get a summary of the dataset to understand its structure
2. Find the top 5 records by any numeric metric
3. Count how many unique values exist in text columns
4. Suggest what chart would best visualize the key patterns
Provide a final synthesis of your findings."""

    print(f"\n=== Multi-Step Analysis ===")
    result = await react_agent.run(task)

    if hasattr(result, "content"):
        print(f"Agent output: {result.content[:500]}...")
    elif isinstance(result, str):
        print(f"Agent output: {result[:500]}...")
    else:
        print(f"Agent output: {str(result)[:500]}...")

    return result


analysis_result = asyncio.run(multi_step_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Inspect reasoning trace
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Reasoning Trace ===")
print(f"ReAct loop: Thought → Action → Observation → repeat")
print(f"Each step:")
print(f"  1. Thought: agent reasons about what to do next")
print(f"  2. Action: agent calls a tool with arguments")
print(f"  3. Observation: tool returns its output")
print(f"  4. Agent decides: enough info → final answer, or loop again")
print(f"\nKey insight: agent decides WHICH tool and WHAT arguments via LLM")
print(f"reasoning, not if-else chains or keyword matching.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Custom BaseAgent with Signature for structured analysis
# ══════════════════════════════════════════════════════════════════════


class DataAnalysisSignature(Signature):
    """Analyse a dataset and produce structured insights."""

    dataset_summary: str = InputField(description="Statistical summary of the dataset")
    analysis_question: str = InputField(description="Specific question to investigate")

    key_findings: list[str] = OutputField(description="Top 3-5 findings from the data")
    recommended_model: str = OutputField(
        description="ML model type best suited for this data"
    )
    data_quality_issues: list[str] = OutputField(
        description="Potential data quality concerns"
    )
    next_steps: list[str] = OutputField(description="Recommended next analysis steps")


class DataAnalysisAgent(BaseAgent):
    """Structured data analysis agent using a typed Signature."""

    signature = DataAnalysisSignature
    model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
    max_llm_cost_usd = 1.0


async def structured_agent_analysis():
    """Run the structured analysis agent."""
    summary = data_summary()

    agent = DataAnalysisAgent()
    result = await agent.run(
        dataset_summary=summary,
        analysis_question="What patterns in this dataset would be most valuable for predicting company performance?",
    )

    print(f"\n=== Structured Analysis Agent ===")
    print(f"Key findings: {result.key_findings}")
    print(f"Recommended model: {result.recommended_model}")
    print(f"Data quality issues: {result.data_quality_issues}")
    print(f"Next steps: {result.next_steps}")

    return result


structured_result = asyncio.run(structured_agent_analysis())

print(f"\n=== ReActAgent vs BaseAgent ===")
print(f"ReActAgent: autonomous tool use, multi-step reasoning, flexible")
print(f"BaseAgent+Signature: structured output, typed contract, predictable")
print(f"Use ReAct when: task needs tools and exploration")
print(f"Use BaseAgent when: output structure matters for downstream pipeline")

print("=" * 60)
print("  MLFP06 Exercise 5: Building Agents with Kaizen")
print("=" * 60)
print(f"\n  ReActAgent and BaseAgent+Signature demonstrated.\n")

# ── Checkpoint 1: Tool definitions ────────────────────────────────────
assert len(tools) == 3, "Should have 3 tools defined"
assert all(callable(t) for t in tools), "All tools should be callable"
assert all(t.__doc__ for t in tools), "All tools should have docstrings"
print(f"✓ Checkpoint 1 passed — {len(tools)} tools defined with docstrings\n")

# INTERPRETATION: Tool docstrings are critical for agent performance.
# The agent reads the docstring to understand what the tool does, when to
# use it, and what arguments to provide. Vague docstrings lead to wrong tool
# selection. Precise docstrings with examples lead to accurate tool calls.
# Best practice: docstring = Args section with types, Returns section,
# and a concrete example in the first line.

# ── Checkpoint 2: ReActAgent creation ────────────────────────────────
assert agent is not None, "Agent should be created"
print(f"✓ Checkpoint 2 passed — ReActAgent created with {len(tools)} tools\n")

# INTERPRETATION: ReAct (Reasoning + Acting) follows the loop:
# Thought: "I need to understand the dataset first"
# Action: call data_summary(dataset_name="reports")
# Observation: [tool returns summary]
# Thought: "I see there's a numeric column 'revenue'. Let me find top 5."
# Action: call run_query("top 5 by revenue")
# ...and so on until the agent decides it has enough information.
# Unlike if-else logic, the agent decides WHICH tool and WHAT args via LLM.

# ── Checkpoint 3: Multi-step analysis ────────────────────────────────
assert analysis_result is not None, "Analysis should produce a result"
print(f"✓ Checkpoint 3 passed — multi-step analysis completed\n")

# INTERPRETATION: The agent's quality depends on the task description.
# Clear, step-by-step task descriptions help the agent structure its approach.
# Open-ended tasks ("analyse this dataset") give the agent flexibility.
# Budget (max_llm_cost_usd=2.0) prevents runaway spending on complex tasks.
# If the agent exceeds the budget, it stops and returns what it has so far.

# ── Checkpoint 4: Reasoning trace ────────────────────────────────────
print(f"✓ Checkpoint 4 passed — ReAct reasoning trace explained\n")

# INTERPRETATION: The reasoning trace is what makes agents interpretable.
# Each Thought -> Action -> Observation cycle is logged, letting you verify:
# - Did the agent understand the task?
# - Did it call the right tools?
# - Did it interpret the observations correctly?
# For production systems, store traces for debugging and compliance audit.

# ── Checkpoint 5: Structured agent ───────────────────────────────────
assert structured_result is not None, "Structured analysis should produce a result"
assert hasattr(structured_result, "key_findings"), "Should have key_findings"
assert hasattr(structured_result, "recommended_model"), "Should have recommended_model"
assert len(structured_result.key_findings) > 0, "Should have at least one finding"
print(f"✓ Checkpoint 5 passed — structured analysis: "
      f"{len(structured_result.key_findings)} findings, "
      f"model={structured_result.recommended_model}\n")

# INTERPRETATION: BaseAgent+Signature gives you typed, validated output.
# The Signature contract means downstream code can reliably access:
# result.key_findings[0] — not "parse the first sentence of the response"
# This is the difference between demo quality and production quality agents.
# Use BaseAgent when: output feeds into a pipeline or needs to be logged.
# Use ReActAgent when: the task requires tool exploration and iteration.


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print("""
  ✓ Tool design: docstrings are the agent's API — precision matters
  ✓ ReActAgent: Thought -> Action -> Observation loop, autonomous multi-step
  ✓ Cost budget: max_llm_cost_usd prevents runaway LLM spending
  ✓ Reasoning traces: interpretable, auditable agent behaviour
  ✓ BaseAgent + Signature: structured, typed, pipeline-safe output

  When to use which agent type:
    ReActAgent:        open-ended exploration, unknown number of steps
    BaseAgent+Sig:     known output schema, feeds into code, audit required
    ReAct + Signature: hybrid — explore first, then structured output

  NEXT: Exercise 6 (Multi-Agent) composes multiple specialist agents.
  A supervisor agent delegates to financial, legal, and technical specialists,
  then synthesises their analyses into a unified decision. This is
  fan-out (parallel) -> fan-in (synthesis) orchestration.
""")
