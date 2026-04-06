# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 5: MCP Servers and Tool Integration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build an MCP server that exposes kailash-ml tools (DataExplorer,
#   ModelVisualizer) to agents at scale. Understand the MCP protocol, tool
#   registration pattern, and SSE transport setup.
#
# TASKS:
#   1. Understand the MCP architecture (server / client / protocol)
#   2. Create an MCP server with MCPServer
#   3. Register ML tools (profile_data, visualize_feature, describe_column)
#   4. Configure SSE transport for production use
#   5. Test tool invocation through the MCP protocol
#   6. Connect an agent to the MCP server as a tool provider
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash.mcp_server import MCPServer, MCPTool, MCPToolResult
from kailash.mcp_server.transports import SSETransport, StdioTransport
from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")
customers = loader.load("ascent04", "ecommerce_customers.parquet")

print(f"=== MCP Server Exercise ===")
print(f"Credit dataset: {credit.shape}")
print(f"Customers dataset: {customers.shape}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: MCP Architecture
# ══════════════════════════════════════════════════════════════════════
#
# MCP (Model Context Protocol) is a standard for agents to discover and
# call tools hosted on a server, independent of agent implementation.
#
# Architecture:
#
#   ┌─────────────────────┐       MCP protocol        ┌──────────────────────┐
#   │   Agent / Client    │ ─────────────────────────► │    MCPServer         │
#   │  (Delegate/ReAct)   │ ◄───────────────────────── │  (tool registry)     │
#   └─────────────────────┘   tool_call / tool_result  └──────────────────────┘
#                                                                │
#                                                    ┌───────────┼───────────┐
#                                                    ▼           ▼           ▼
#                                             profile_data  visualize   describe
#                                             (DataExplorer)(ModelViz)  (column)
#
# Transport options:
#   StdioTransport  — subprocess pipe, great for local/testing
#   SSETransport    — HTTP Server-Sent Events, production/remote
#   WebSocketTransport — bidirectional, streaming use cases
#
# Why MCP?
#   - Tool definitions live on the server, not hardcoded in agent
#   - Agents discover tools dynamically at runtime
#   - Same server can serve many different agents
#   - Centralised tool versioning and access control

print(f"\n=== MCP Architecture ===")
print(f"Protocol: tool_list → tool_call → tool_result")
print(f"Transport: StdioTransport (local), SSETransport (production)")
print(f"Discovery: agent calls list_tools() at startup")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create MCPServer with metadata
# ══════════════════════════════════════════════════════════════════════


def create_ml_mcp_server() -> MCPServer:
    """Build an MCPServer that wraps kailash-ml tools."""
    server = MCPServer(
        name="kailash-ml-tools",
        version="1.0.0",
        description=(
            "ML analysis tools powered by kailash-ml. "
            "Provides data profiling, feature analysis, and visualisation."
        ),
    )
    return server


server = create_ml_mcp_server()
print(f"\n=== MCPServer Created ===")
print(f"Name: {server.name}")
print(f"Version: {server.version}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Register ML tools
# ══════════════════════════════════════════════════════════════════════
# Each tool has: name, description, input_schema, and an async handler.
# The input_schema uses JSON Schema so agents know what arguments to pass.


@server.tool(
    name="profile_dataset",
    description=(
        "Profile a dataset using DataExplorer. Returns row count, column types, "
        "null rates, distribution statistics, and data quality alerts."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "enum": ["credit", "customers"],
                "description": "Which dataset to profile",
            },
            "sample_size": {
                "type": "integer",
                "description": "Number of rows to sample (default 5000)",
                "default": 5000,
            },
        },
        "required": ["dataset"],
    },
)
async def tool_profile_dataset(dataset: str, sample_size: int = 5000) -> MCPToolResult:
    """Profile a dataset using DataExplorer."""
    from kailash_ml import DataExplorer

    df = credit if dataset == "credit" else customers
    sample = df.sample(n=min(sample_size, df.height), seed=42)

    explorer = DataExplorer()
    profile = await explorer.profile(sample)

    summary = (
        f"Dataset: {dataset}\n"
        f"Rows: {profile.n_rows:,}, Columns: {profile.n_columns}\n"
        f"Null rates: {profile.null_summary}\n"
        f"Alerts ({len(profile.alerts)}): "
        f"{[a['type'] + ':' + str(a.get('column', '')) for a in profile.alerts[:5]]}\n"
        f"Numeric columns: {profile.numeric_columns}\n"
        f"Categorical columns: {profile.categorical_columns}"
    )

    return MCPToolResult(content=summary, is_error=False)


@server.tool(
    name="describe_column",
    description=(
        "Get detailed statistics for a specific column in a dataset. "
        "Returns mean, std, min, max, null count, and value distribution."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "enum": ["credit", "customers"],
                "description": "Which dataset to query",
            },
            "column": {
                "type": "string",
                "description": "Column name to describe",
            },
        },
        "required": ["dataset", "column"],
    },
)
async def tool_describe_column(dataset: str, column: str) -> MCPToolResult:
    """Describe a specific column with statistics."""
    df = credit if dataset == "credit" else customers

    if column not in df.columns:
        return MCPToolResult(
            content=f"Column '{column}' not found. Available: {df.columns}",
            is_error=True,
        )

    col = df[column]
    if col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16):
        result = (
            f"Column: {column} ({col.dtype})\n"
            f"  Mean:   {col.mean():.4f}\n"
            f"  Std:    {col.std():.4f}\n"
            f"  Min:    {col.min()}\n"
            f"  Max:    {col.max()}\n"
            f"  Nulls:  {col.null_count()} ({col.null_count()/len(col):.1%})\n"
            f"  P25:    {col.quantile(0.25):.4f}\n"
            f"  Median: {col.quantile(0.5):.4f}\n"
            f"  P75:    {col.quantile(0.75):.4f}"
        )
    else:
        vc = col.value_counts().sort("count", descending=True).head(5)
        result = (
            f"Column: {column} ({col.dtype})\n"
            f"  Unique: {col.n_unique()}\n"
            f"  Nulls:  {col.null_count()}\n"
            f"  Top values:\n"
            + "\n".join(
                f"    {row[column]}: {row['count']}" for row in vc.iter_rows(named=True)
            )
        )

    return MCPToolResult(content=result, is_error=False)


@server.tool(
    name="target_analysis",
    description=(
        "Analyse the relationship between a feature column and a target column. "
        "Returns group-level statistics and default/conversion rates."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "enum": ["credit", "customers"],
            },
            "feature": {
                "type": "string",
                "description": "Feature column to group by",
            },
            "target": {
                "type": "string",
                "description": "Target column to aggregate (e.g., 'default')",
            },
        },
        "required": ["dataset", "feature", "target"],
    },
)
async def tool_target_analysis(
    dataset: str, feature: str, target: str
) -> MCPToolResult:
    """Analyse feature-target relationship."""
    df = credit if dataset == "credit" else customers

    if feature not in df.columns:
        return MCPToolResult(
            content=f"Feature '{feature}' not found. Available: {df.columns}",
            is_error=True,
        )
    if target not in df.columns:
        return MCPToolResult(
            content=f"Target '{target}' not found. Available: {df.columns}",
            is_error=True,
        )

    feat_col = df[feature]
    if feat_col.dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32):
        # Bin numeric features
        binned = df.with_columns(
            pl.col(feature)
            .cut(
                breaks=[
                    feat_col.quantile(0.25),
                    feat_col.quantile(0.5),
                    feat_col.quantile(0.75),
                ]
            )
            .alias(f"{feature}_bin")
        )
        result_df = (
            binned.group_by(f"{feature}_bin")
            .agg(
                pl.col(target).mean().alias(f"{target}_rate"),
                pl.col(target).count().alias("n"),
            )
            .sort(f"{target}_rate", descending=True)
        )
    else:
        result_df = (
            df.group_by(feature)
            .agg(
                pl.col(target).mean().alias(f"{target}_rate"),
                pl.col(target).count().alias("n"),
            )
            .sort(f"{target}_rate", descending=True)
            .head(10)
        )

    rows = [
        f"  {row[feature if feature in result_df.columns else f'{feature}_bin']}: "
        f"{row[f'{target}_rate']:.4f} (n={row['n']})"
        for row in result_df.iter_rows(named=True)
    ]
    return MCPToolResult(
        content=f"{target} rate by {feature} (top 10):\n" + "\n".join(rows),
        is_error=False,
    )


@server.tool(
    name="list_columns",
    description="List all columns in a dataset with their data types.",
    input_schema={
        "type": "object",
        "properties": {
            "dataset": {
                "type": "string",
                "enum": ["credit", "customers"],
            }
        },
        "required": ["dataset"],
    },
)
async def tool_list_columns(dataset: str) -> MCPToolResult:
    """List all columns in a dataset."""
    df = credit if dataset == "credit" else customers
    col_info = "\n".join(
        f"  {col}: {dtype}" for col, dtype in zip(df.columns, df.dtypes)
    )
    return MCPToolResult(
        content=f"Dataset '{dataset}' ({df.height:,} rows):\n{col_info}",
        is_error=False,
    )


print(f"\n=== Tools Registered ===")
for tool in server.list_tools():
    print(f"  {tool.name}: {tool.description[:60]}...")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Configure transports
# ══════════════════════════════════════════════════════════════════════
#
# StdioTransport: used for local testing and subprocess-based agent
#   connections. The server reads JSON-RPC from stdin, writes to stdout.
#
# SSETransport: used for production. Clients connect via HTTP GET to
#   an /events endpoint, receive events as a stream. Tool calls go via
#   HTTP POST to /call. This is what hosted MCP servers use.

stdio_transport = StdioTransport()

sse_transport = SSETransport(
    host="0.0.0.0",
    port=8765,
    path="/mcp",  # Base path — tools served at /mcp/call
    cors_origins=["*"],  # Restrict in production
    auth_token=os.environ.get("MCP_AUTH_TOKEN"),  # None = no auth (dev only)
)

print(f"\n=== Transport Configuration ===")
print(f"StdioTransport: stdin/stdout (local dev / subprocess)")
print(f"SSETransport: http://0.0.0.0:8765/mcp (production)")
print(f"  → tool list:  GET  /mcp/tools")
print(f"  → tool call:  POST /mcp/call")
print(f"  → events:     GET  /mcp/events (SSE stream)")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Test tool invocation directly
# ══════════════════════════════════════════════════════════════════════
# Before wiring the agent, verify each tool works in isolation.


async def test_tools_directly():
    """Invoke registered tools directly to verify they work."""
    print(f"\n=== Direct Tool Tests ===")

    # Tool 1: list columns
    result = await server.call_tool("list_columns", {"dataset": "credit"})
    print(f"\ntool: list_columns(credit)")
    print(result.content[:300])

    # Tool 2: describe a specific column
    result = await server.call_tool(
        "describe_column", {"dataset": "credit", "column": "annual_income"}
    )
    print(f"\ntool: describe_column(credit, annual_income)")
    print(result.content)

    # Tool 3: target analysis
    result = await server.call_tool(
        "target_analysis",
        {"dataset": "credit", "feature": "late_payments_12m", "target": "default"},
    )
    print(f"\ntool: target_analysis(credit, late_payments_12m, default)")
    print(result.content[:400])

    # Tool 4: profile
    result = await server.call_tool(
        "profile_dataset", {"dataset": "credit", "sample_size": 2000}
    )
    print(f"\ntool: profile_dataset(credit, sample_size=2000)")
    print(result.content[:400])


asyncio.run(test_tools_directly())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Connect an agent to the MCP server
# ══════════════════════════════════════════════════════════════════════
# Rather than hard-coding tool functions in the agent, the agent
# discovers tools from the MCP server at runtime. This decouples
# agent logic from tool implementation.


async def agent_with_mcp_tools():
    """Run a Delegate agent with tools sourced from the MCP server."""

    # In production this is a network call: agent.connect_mcp("http://host:8765/mcp")
    # For this exercise we use the in-process client for testing.
    mcp_client = server.create_in_process_client()

    # Discover available tools
    available_tools = await mcp_client.list_tools()
    print(f"\n=== Agent Tool Discovery via MCP ===")
    print(f"Tools available: {[t.name for t in available_tools]}")

    # Build tool callables from MCP definitions
    async def mcp_list_columns(dataset: str) -> str:
        r = await mcp_client.call_tool("list_columns", {"dataset": dataset})
        return r.content if not r.is_error else f"Error: {r.content}"

    async def mcp_describe_column(dataset: str, column: str) -> str:
        r = await mcp_client.call_tool(
            "describe_column", {"dataset": dataset, "column": column}
        )
        return r.content if not r.is_error else f"Error: {r.content}"

    async def mcp_target_analysis(dataset: str, feature: str, target: str) -> str:
        r = await mcp_client.call_tool(
            "target_analysis",
            {"dataset": dataset, "feature": feature, "target": target},
        )
        return r.content if not r.is_error else f"Error: {r.content}"

    async def mcp_profile_dataset(dataset: str) -> str:
        r = await mcp_client.call_tool("profile_dataset", {"dataset": dataset})
        return r.content if not r.is_error else f"Error: {r.content}"

    # Wire into agent as standard tools
    from kaizen_agents.agents.specialized.react import ReActAgent
    from kaizen import Signature, InputField, OutputField

    class CreditAnalysis(Signature):
        """Autonomous credit dataset analysis using MCP tools."""

        task: str = InputField(description="Analysis task")
        key_findings: list[str] = OutputField(description="Key findings from analysis")
        recommended_features: list[str] = OutputField(
            description="Features most predictive of default"
        )
        next_steps: list[str] = OutputField(description="Recommended next steps")

    agent = ReActAgent(
        signature=CreditAnalysis,
        model=model,
        tools={
            "list_columns": mcp_list_columns,
            "describe_column": mcp_describe_column,
            "target_analysis": mcp_target_analysis,
            "profile_dataset": mcp_profile_dataset,
        },
        max_llm_cost_usd=3.0,
        max_iterations=8,
    )

    task = (
        "Analyse the Singapore credit scoring dataset. "
        "List the columns, profile the data, and investigate which features "
        "are most strongly associated with default. "
        "Recommend the top 5 predictive features."
    )

    print(f"\n=== Agent via MCP Tools ===")
    print(f"Task: {task}")
    result = await agent.run(task=task)

    print(f"\nFindings:")
    for i, f in enumerate(result.key_findings):
        print(f"  {i+1}. {f}")
    print(f"\nTop features: {result.recommended_features}")
    print(f"\nNext steps: {result.next_steps}")

    return result


mcp_result = asyncio.run(agent_with_mcp_tools())


# ══════════════════════════════════════════════════════════════════════
# Summary: MCP in Production
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== MCP in Production ===")
print(
    """
Why MCP matters at scale:

  Traditional (M3-M5 pattern):
    - Tool functions defined inside each agent script
    - Changing a tool requires redeploying the agent
    - Different agents duplicate the same tool logic

  MCP server pattern:
    - Tools live on the server, agents discover them at runtime
    - Update the server → all agents get the updated tool
    - Add auth, rate limiting, versioning at the server layer
    - One server can serve many agents (internal + external)

Production deployment flow:
  1. python -m kailash.mcp_server serve ex_5_server.py --transport sse --port 8765
  2. Agent: await delegate.connect_mcp("http://localhost:8765/mcp")
  3. Agent discovers tools automatically → uses them in reasoning loop

In Module 6, MCP tool access is governed by PACT:
  → GovernanceEngine controls which agents can call which MCP tools
  → MCPServer validates the GovernanceContext token on each call
"""
)

print("✓ Exercise 5 complete — MCP server with tool registration and SSE transport")
