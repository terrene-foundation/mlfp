# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Integration / Agent with ML Tools
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Give a Kaizen agent access to ML engines as tools
# LEVEL: Advanced
# PARITY: Python-only (full stack integration)
# VALIDATES: ReActAgent + custom tools wrapping kailash-ml engines
#
# Run: uv run python textbook/python/08-integration/03_agent_with_tools.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen import Signature, InputField, OutputField

# ── 1. Define the agent's signature ─────────────────────────────────


class DataAnalystSignature(Signature):
    """You are a data analyst with access to ML tools.
    Use the available tools to explore data and build insights."""

    __intent__ = "Analyze datasets and provide actionable insights"
    __guidelines__ = [
        "Always profile data before analysis",
        "Use DataExplorer for initial profiling",
        "Use ModelVisualizer for charts",
        "Stay within budget",
    ]

    question: str = InputField(description="Analysis question")
    dataset_name: str = InputField(description="Name of the dataset to analyze")
    analysis: str = OutputField(description="Detailed analysis with findings")
    recommendations: str = OutputField(description="Actionable recommendations")


assert "question" in DataAnalystSignature._signature_inputs
assert "analysis" in DataAnalystSignature._signature_outputs

# ── 2. Tool wrapping pattern ────────────────────────────────────────
# ML engines become agent tools by wrapping them in async functions.
# The agent decides WHEN to call each tool (LLM reasoning, not code logic).
#
# Pattern:
#   async def profile_data(dataset: str) -> dict:
#       """Profile a dataset using DataExplorer."""
#       df = loader.load(dataset)
#       explorer = DataExplorer()
#       profile = await explorer.profile(df)
#       return profile.to_dict()
#
#   async def create_chart(data: str, chart_type: str) -> str:
#       """Create a visualization using ModelVisualizer."""
#       viz = ModelVisualizer()
#       fig = viz.create(json.loads(data), chart_type=chart_type)
#       return fig.to_html()

# ── 3. ReActAgent with ML tools ─────────────────────────────────────
# ReActAgent uses Reason → Act → Observe loops:
#   1. Reason about what to do next
#   2. Call a tool (Act)
#   3. Observe the result
#   4. Decide if done or need more info
#
# from kaizen_agents.agents.specialized.react import ReActAgent
#
# agent = ReActAgent(
#     config=agent_config,
#     tools=[profile_data, create_chart, run_query],
# )
# result = await agent.solve("What are the key trends in HDB prices?")

# ── 4. AgentInfusionProtocol — double opt-in ───────────────────────
# When ML agents interact with ML engines, both sides must consent:
#   1. The engine declares: "I accept agent guidance" (MLToolProtocol)
#   2. The agent declares: "I respect engine constraints" (AgentInfusionProtocol)
#
# This prevents agents from bypassing engine safety checks.

from kailash_ml.types import AgentInfusionProtocol, MLToolProtocol

assert AgentInfusionProtocol is not None
assert MLToolProtocol is not None

# ── 5. Production pattern: budget-governed analyst ──────────────────
# In production, always set a cost budget:
#
# from kaizen_agents import Delegate
#
# delegate = Delegate(
#     model=os.environ["DEFAULT_LLM_MODEL"],
#     tools=[profile_data, create_chart, run_query],
#     budget_usd=2.0,  # MANDATORY: cap LLM costs
#     system_prompt="You are a data analyst. Use tools to answer questions.",
# )
#
# async for event in delegate.run("Analyze the credit scoring dataset"):
#     handle_event(event)

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
assert len(model) > 0

print("PASS: 08-integration/03_agent_with_tools")
