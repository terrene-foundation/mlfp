# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / ReAct Agent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ReActAgent that reasons and acts with tools
# LEVEL: Intermediate
# PARITY: Python-only — no Rust equivalent
# VALIDATES: ReActAgent, ReActConfig, ReActSignature, MultiCycleStrategy,
#            ActionType, convergence detection
#
# Run: uv run python textbook/python/04-agents/03_react_agent.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.react import (
    ReActAgent,
    ReActConfig,
    ReActSignature,
    ActionType,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.strategies.multi_cycle import MultiCycleStrategy

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. ReAct pattern: Reason + Act + Observe ──────────────────────────
# ReAct agents operate in iterative cycles:
#   1. REASON: Think about the current state and what to do next
#   2. ACT: Execute an action (tool call, clarification, or finish)
#   3. OBSERVE: Examine the result of the action
#   4. REPEAT: Until converged (finish action or high confidence)
#
# This is fundamentally different from single-shot agents (SimpleQA).
# ReAct is autonomous: it decides its own execution path.

# ── 2. ActionType — structured action vocabulary ──────────────────────
# The ReAct agent uses a fixed vocabulary of action types:

assert ActionType.TOOL_USE.value == "tool_use"
assert ActionType.FINISH.value == "finish"
assert ActionType.CLARIFY.value == "clarify"

# ── 3. ReActSignature — multi-field structured I/O ────────────────────
# The signature captures the full ReAct cycle state:
#   - task, context, available_tools, previous_actions (inputs)
#   - thought, action, action_input, confidence, need_tool, tool_calls (outputs)

assert "task" in ReActSignature._signature_inputs
assert "context" in ReActSignature._signature_inputs
assert "available_tools" in ReActSignature._signature_inputs
assert "previous_actions" in ReActSignature._signature_inputs

assert "thought" in ReActSignature._signature_outputs
assert "action" in ReActSignature._signature_outputs
assert "action_input" in ReActSignature._signature_outputs
assert "confidence" in ReActSignature._signature_outputs
assert "need_tool" in ReActSignature._signature_outputs
assert (
    "tool_calls" in ReActSignature._signature_outputs
), "ADR-013: tool_calls enables objective convergence detection"

# ── 4. ReActConfig — configuration with ReAct-specific settings ───────

config = ReActConfig()

assert config.max_cycles == 10, "Default: up to 10 reasoning cycles"
assert config.confidence_threshold == 0.7, "Converge at 0.7 confidence"
assert config.mcp_discovery_enabled is True, "Autonomous agents discover tools"
assert config.enable_parallel_tools is False, "Sequential tool execution by default"

# Custom configuration
custom_config = ReActConfig(
    max_cycles=15,
    confidence_threshold=0.8,
    enable_parallel_tools=True,
)
assert custom_config.max_cycles == 15

# ── 5. ReActAgent instantiation ───────────────────────────────────────
# ReActAgent uses MultiCycleStrategy (not AsyncSingleShotStrategy).
# This is the CRITICAL difference from interactive agents.

agent = ReActAgent(
    llm_provider="mock",
    model=model,
    max_cycles=10,
    confidence_threshold=0.7,
)

assert isinstance(agent, ReActAgent)
assert isinstance(agent, BaseAgent)
assert agent.react_config.max_cycles == 10

# Verify MultiCycleStrategy is used
assert isinstance(
    agent.strategy, MultiCycleStrategy
), "ReAct MUST use MultiCycleStrategy for iterative execution"

# ── 6. Convergence detection — objective (tool_calls) ─────────────────
# ADR-013: Objective convergence uses the tool_calls field.
# This implements Claude Code's while(tool_call_exists) pattern.
#
# tool_calls present and non-empty -> NOT converged (continue)
# tool_calls present but empty     -> CONVERGED (stop)

# Simulate: tool calls present -> continue
result_with_tools = {
    "tool_calls": [{"name": "search", "params": {"query": "flights"}}],
    "action": "tool_use",
    "confidence": 0.5,
}
assert (
    agent._check_convergence(result_with_tools) is False
), "Non-empty tool_calls means NOT converged"

# Simulate: empty tool calls -> converged
result_empty_tools = {
    "tool_calls": [],
    "action": "finish",
    "confidence": 0.9,
}
assert (
    agent._check_convergence(result_empty_tools) is True
), "Empty tool_calls means CONVERGED"

# ── 7. Convergence detection — subjective fallback ────────────────────
# When tool_calls is absent, fall back to action/confidence checks.

# action == "finish" -> converged
result_finish = {"action": "finish", "confidence": 0.6}
assert agent._check_convergence(result_finish) is True

# confidence >= threshold -> converged
result_high_conf = {"action": "tool_use", "confidence": 0.9}
assert agent._check_convergence(result_high_conf) is True

# action == "tool_use" with low confidence -> NOT converged
result_continue = {"action": "tool_use", "confidence": 0.3}
assert agent._check_convergence(result_continue) is False

# No signals -> default safe fallback (converged)
assert agent._check_convergence({}) is True

# ── 8. Input validation ───────────────────────────────────────────────
# Empty tasks return an error immediately without LLM calls.

empty_result = agent.run(task="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["action"] == ActionType.FINISH.value
assert empty_result["confidence"] == 0.0
assert empty_result["cycles_used"] == 0

whitespace_result = agent.run(task="   ")
assert whitespace_result["error"] == "INVALID_INPUT"

# ── 9. Node metadata ──────────────────────────────────────────────────

assert ReActAgent.metadata.name == "ReActAgent"
assert "react" in ReActAgent.metadata.tags
assert "multi-cycle" in ReActAgent.metadata.tags
assert "tool-use" in ReActAgent.metadata.tags

# ── 10. Return structure ──────────────────────────────────────────────
# ReActAgent.run() returns:
# {
#     "thought": "I need to search for flights...",
#     "action": "tool_use",
#     "action_input": {"tool": "search", "query": "flights to Paris"},
#     "confidence": 0.85,
#     "need_tool": True,
#     "cycles_used": 3,
#     "total_cycles": 10,
# }
#
# NOTE: We do not call run() with real tasks here because
# it requires an LLM API key. The return format is documented above.

print("PASS: 04-agents/03_react_agent")
