# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 7: MCP Integration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build an MCP server exposing kailash-ml tools, then connect
#   an agent to use those tools for ML operations.
#
# TASKS:
#   1. Create MCPServer with tool definitions
#   2. Expose DataExplorer as MCP tool
#   3. Expose TrainingPipeline as MCP tool
#   4. Connect ReActAgent to MCP server
#   5. Run agent-driven ML workflow via MCP
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen_agents.agents.specialized.react import ReActAgent
from kailash_ml import DataExplorer, TrainingPipeline
from kailash_mcp import MCPServer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Create MCPServer with tool definitions
# ══════════════════════════════════════════════════════════════════════

# TODO: Create an MCPServer with name "kailash-ml-tools"
server = MCPServer(______)

print(f"=== MCP Server: kailash-ml-tools ===")
print(f"MCP (Model Context Protocol) exposes tools as a standard interface.")
print(f"Any MCP-compatible agent can discover and call these tools.")
print(f"This decouples the agent from the tool implementation.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Expose DataExplorer as MCP tool
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

# Global data context for tools
_data_cache: dict[str, pl.DataFrame] = {}


@server.tool()
async def explore_dataset(dataset_name: str) -> str:
    """Load a dataset and return a statistical summary using DataExplorer.

    Args:
        dataset_name: Name of the dataset to explore (e.g., 'sg_company_reports')
    """
    df = loader.load("ascent09", f"{dataset_name}.parquet")
    _data_cache[dataset_name] = df

    explorer = DataExplorer()
    profile = await explorer.profile(df)

    summary = f"Dataset: {dataset_name}\n"
    summary += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
    summary += f"Columns: {', '.join(df.columns)}\n"
    summary += f"Numeric columns: {profile.get('numeric_columns', [])}\n"
    summary += f"Missing values: {profile.get('missing_summary', 'none')}\n"
    summary += f"Sample:\n{df.head(3)}"
    return summary


@server.tool()
async def get_column_stats(dataset_name: str, column: str) -> str:
    """Get detailed statistics for a specific column.

    Args:
        dataset_name: Name of the loaded dataset
        column: Column name to analyze
    """
    if dataset_name not in _data_cache:
        return f"Dataset '{dataset_name}' not loaded. Call explore_dataset first."

    df = _data_cache[dataset_name]
    if column not in df.columns:
        return f"Column '{column}' not found. Available: {df.columns}"

    col = df[column]
    if col.dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]:
        stats = {
            "mean": col.mean(),
            "std": col.std(),
            "min": col.min(),
            "max": col.max(),
            "median": col.median(),
            "null_count": col.null_count(),
        }
    else:
        value_counts = col.value_counts().sort("count", descending=True).head(10)
        stats = {
            "type": str(col.dtype),
            "unique": col.n_unique(),
            "null_count": col.null_count(),
            "top_values": value_counts.to_dicts(),
        }

    return str(stats)


print(f"\n=== MCP Tools Registered ===")
print(f"  explore_dataset(dataset_name) → DataExplorer profile")
print(f"  get_column_stats(dataset_name, column) → Column statistics")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Expose TrainingPipeline as MCP tool
# ══════════════════════════════════════════════════════════════════════


@server.tool()
async def train_classifier(
    dataset_name: str,
    target: str,
    features: str,
    algorithm: str = "gradient_boosting",
) -> str:
    """Train a classification model using TrainingPipeline.

    Args:
        dataset_name: Name of the loaded dataset
        target: Target column name
        features: Comma-separated feature column names
        algorithm: Algorithm to use (gradient_boosting, random_forest, logistic_regression)
    """
    if dataset_name not in _data_cache:
        return f"Dataset '{dataset_name}' not loaded."

    df = _data_cache[dataset_name]
    feature_list = [f.strip() for f in features.split(",")]

    # TODO: Create TrainingPipeline with model_type, target, features, and config dict
    pipeline = TrainingPipeline(____)

    n_train = int(df.height * 0.8)
    result = pipeline.fit(df[:n_train])

    # Evaluate
    predictions = pipeline.predict(df[n_train:])
    y_true = df[n_train:][target].to_list()
    y_pred = predictions["prediction"].to_list()
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

    return (
        f"Model trained: {algorithm}\n"
        f"Training metrics: {result.metrics}\n"
        f"Test accuracy: {accuracy:.4f}\n"
        f"Features used: {feature_list}"
    )


@server.tool()
async def list_datasets() -> str:
    """List all currently loaded datasets and their shapes."""
    if not _data_cache:
        return "No datasets loaded. Use explore_dataset to load one."
    return "\n".join(
        f"  {name}: {df.shape[0]} rows × {df.shape[1]} columns"
        for name, df in _data_cache.items()
    )


print(f"  train_classifier(dataset, target, features, algo) → Training results")
print(f"  list_datasets() → Currently loaded datasets")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Connect ReActAgent to MCP server
# ══════════════════════════════════════════════════════════════════════

# Get tools from MCP server for agent use
mcp_tools = server.get_tools()


async def agent_with_mcp():
    # TODO: Create ReActAgent with model, mcp_tools, and max_llm_cost_usd=3.0
    agent = ReActAgent(____)

    print(f"\n=== ReActAgent + MCP ===")
    print(f"Agent has access to {len(mcp_tools)} MCP tools")
    print(f"The agent discovers tools at runtime — no hardcoded tool knowledge.")

    return agent


agent = asyncio.run(agent_with_mcp())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Run agent-driven ML workflow via MCP
# ══════════════════════════════════════════════════════════════════════


async def run_ml_workflow():
    agent = ReActAgent(
        model=model,
        tools=mcp_tools,
        max_llm_cost_usd=5.0,
    )

    print(f"\n=== Agent-Driven ML Workflow ===")
    print(f"The agent will autonomously:")
    print(f"  1. Explore the dataset")
    print(f"  2. Identify suitable features and target")
    print(f"  3. Train a classifier")
    print(f"  4. Report results")

    # TODO: Run the agent with an instruction to explore, train, and report
    result = await agent.run(____)

    print(f"\n--- Agent Result ---")
    print(f"Final answer: {str(result)[:500]}...")

    # Inspect reasoning trace
    if hasattr(result, "trace"):
        print(f"\n--- Reasoning Trace ---")
        for step in result.trace:
            print(f"  Thought: {step.get('thought', '')[:100]}...")
            print(f"  Action:  {step.get('action', '')}")
            print(f"  Result:  {str(step.get('observation', ''))[:100]}...")
            print()

    return result


workflow_result = asyncio.run(run_ml_workflow())

print(f"\n=== MCP Architecture Summary ===")
print(f"MCP separates tool providers from tool consumers:")
print(f"  Server: defines tools (explore, train, predict)")
print(f"  Client: agent discovers and uses tools at runtime")
print(f"  Protocol: standard JSON-RPC over stdio/SSE/HTTP")
print(f"Benefits:")
print(f"  - Tools are reusable across different agents")
print(f"  - Agents don't need to know implementation details")
print(f"  - New tools can be added without changing agent code")
print(f"  - Security: tool access can be gated per agent")

print("\n✓ Exercise 7 complete — MCP server with kailash-ml tools + ReActAgent")
