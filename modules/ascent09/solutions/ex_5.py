# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 5: Building Agents with Kaizen
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ReActAgent with custom tools for autonomous data
#   analysis — tool definition, reasoning loops, structured output.
#
# TASKS:
#   1. Define custom tools (data_summary, run_query, plot_chart)
#   2. Build ReActAgent with tool access
#   3. Run agent on multi-step analysis task
#   4. Inspect reasoning trace
#   5. Build custom BaseAgent with Signature for structured analysis
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

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

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

print("\n✓ Exercise 5 complete — ReActAgent with tools + BaseAgent with Signature")
