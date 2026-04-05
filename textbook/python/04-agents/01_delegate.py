# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Delegate with Tools & Events
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use Delegate with tool registration and event handling patterns
# LEVEL: Basic
# PARITY: Equivalent — Rust has DelegateEngine with same tool/event model
# VALIDATES: ToolRegistry, ToolDef, Delegate tool integration, event stream
#
# Run: uv run python textbook/python/04-agents/01_delegate.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents import Delegate
from kaizen_agents.delegate.loop import ToolRegistry, ToolDef
from kaizen_agents.delegate.events import (
    DelegateEvent,
    TextDelta,
    ToolCallStart,
    ToolCallEnd,
    TurnComplete,
    BudgetExhausted,
    ErrorEvent,
)

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. ToolRegistry — create and register tools ───────────────────────
# ToolRegistry holds tools the Delegate can call. Each tool has:
#   - name: unique identifier used in function calling
#   - description: human-readable description for the model
#   - parameters: JSON Schema for the tool's arguments
#   - executor: async callable that performs the action

registry = ToolRegistry()

assert isinstance(registry, ToolRegistry)
assert len(registry.tool_names) == 0, "Fresh registry is empty"


# Define an async executor function (tools are always async)
async def read_file_executor(path: str) -> str:
    """Read a file and return its contents."""
    return f"Contents of {path}"


# Register the tool with name, description, schema, and executor
registry.register(
    name="read_file",
    description="Read a file from the filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file"},
        },
        "required": ["path"],
    },
    executor=read_file_executor,
)

assert registry.has_tool("read_file"), "Tool was registered"
assert not registry.has_tool("nonexistent"), "Unknown tool not found"
assert registry.tool_names == ["read_file"]

# ── 2. Multiple tool registration ─────────────────────────────────────
# Real agents typically have many tools. Each follows the same pattern.


async def search_executor(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return f"Results for '{query}' (max {max_results})"


registry.register(
    name="search",
    description="Search for information in the codebase",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "description": "Max results",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    executor=search_executor,
)

assert len(registry.tool_names) == 2
assert "search" in registry.tool_names

# ── 3. OpenAI function-calling format ──────────────────────────────────
# ToolRegistry converts tools to the OpenAI function-calling format
# expected by LLM APIs. This is the wire format the model sees.

openai_tools = registry.get_openai_tools()
assert len(openai_tools) == 2

first_tool = openai_tools[0]
assert first_tool["type"] == "function"
assert "function" in first_tool
assert first_tool["function"]["name"] == "read_file"
assert "parameters" in first_tool["function"]

# ── 4. ToolDef — individual tool definition ────────────────────────────
# ToolDef is the dataclass that holds a single tool's metadata.

tool_def = ToolDef(
    name="write_file",
    description="Write content to a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)

assert tool_def.name == "write_file"
openai_format = tool_def.to_openai_format()
assert openai_format["type"] == "function"
assert openai_format["function"]["name"] == "write_file"

# ── 5. Delegate with tool registry ────────────────────────────────────
# Pass a pre-built ToolRegistry to the Delegate. The Delegate's
# AgentLoop uses these tools during the TAOD (Think-Act-Observe-Decide)
# loop: the model decides which tool to call, the loop executes it.

delegate_with_tools = Delegate(
    model=model,
    tools=registry,
    system_prompt="You are a code reviewer. Use the available tools.",
    budget_usd=5.0,
)

assert delegate_with_tools.tool_registry is registry
assert delegate_with_tools.tool_registry.has_tool("read_file")
assert delegate_with_tools.tool_registry.has_tool("search")

# ── 6. Event type system (pattern matching) ────────────────────────────
# Delegate.run() yields typed events. Consumers match on type rather
# than parsing raw strings. This is the complete event hierarchy:
#
#   DelegateEvent (base)
#   +-- TextDelta          text from the model (streaming)
#   +-- ToolCallStart      a tool invocation has begun
#   +-- ToolCallEnd        a tool invocation has completed
#   +-- TurnComplete       the model finished responding
#   +-- BudgetExhausted    budget cap exceeded
#   +-- ErrorEvent         an error occurred

# All events carry event_type discriminator and monotonic timestamp
td = TextDelta(text="Hello")
assert td.event_type == "text_delta"
assert td.text == "Hello"
assert td.timestamp > 0, "Monotonic timestamp is always positive"

tcs = ToolCallStart(call_id="call_001", name="read_file")
assert tcs.event_type == "tool_call_start"
assert tcs.call_id == "call_001"
assert tcs.name == "read_file"

tce = ToolCallEnd(call_id="call_001", name="read_file", result="file contents")
assert tce.event_type == "tool_call_end"
assert tce.result == "file contents"
assert tce.error == "", "No error by default"

tc = TurnComplete(
    text="Analysis complete", usage={"prompt_tokens": 100, "completion_tokens": 50}
)
assert tc.event_type == "turn_complete"
assert tc.usage["prompt_tokens"] == 100

be = BudgetExhausted(budget_usd=5.0, consumed_usd=5.01)
assert be.event_type == "budget_exhausted"
assert be.consumed_usd > be.budget_usd

err = ErrorEvent(error="Connection timeout", details={"exception_type": "TimeoutError"})
assert err.event_type == "error"
assert err.details["exception_type"] == "TimeoutError"

# ── 7. Event stream pattern (how to consume) ──────────────────────────
# The standard pattern for consuming Delegate events:
#
#   async for event in delegate.run("prompt"):
#       match event:
#           case TextDelta(text=t):
#               print(t, end="")
#           case ToolCallStart(name=n):
#               show_spinner(n)
#           case ToolCallEnd(name=n, result=r):
#               hide_spinner(n)
#           case TurnComplete(text=t, usage=u):
#               print(f"\nTokens: {u}")
#           case BudgetExhausted():
#               print("Budget exceeded")
#           case ErrorEvent(error=e):
#               print(f"Error: {e}")
#
# NOTE: We do not call run() here as it requires an LLM API key.
# The above shows the intended consumption pattern.

# ── 8. Tool execution error handling ───────────────────────────────────
# When a tool call fails, ToolCallEnd carries the error message.

tce_err = ToolCallEnd(
    call_id="call_002",
    name="read_file",
    result="",
    error="FileNotFoundError: /nonexistent.py",
)
assert tce_err.error != "", "Error field populated on failure"
assert tce_err.result == "", "No result on failure"

# ── 9. Budget tracking with tool-using Delegate ───────────────────────
# Budget tracking applies to the full session including tool calls.
# Each LLM turn (not each tool call) incurs token cost.

assert delegate_with_tools.budget_usd == 5.0
assert delegate_with_tools.consumed_usd == 0.0
assert delegate_with_tools.budget_remaining == 5.0

print("PASS: 04-agents/01_delegate")
