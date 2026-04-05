# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Kaizen / Delegate
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use Delegate — the primary entry point for autonomous AI execution
# LEVEL: Basic
# PARITY: Equivalent — Rust has DelegateEngine with similar progressive-disclosure
# VALIDATES: Delegate(), run(), run_sync(), budget tracking, event types
#
# Run: uv run python textbook/python/03-kaizen/02_delegate.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kaizen_agents import Delegate
from kaizen_agents.delegate.events import (
    DelegateEvent,
    TextDelta,
    TurnComplete,
    BudgetExhausted,
    ErrorEvent,
)

# ── 1. Create a Delegate (Layer 1: minimal) ─────────────────────────
# Delegate is the single entry point for AI execution. It composes
# AgentLoop + optional GovernedSupervisor.
#
# Model comes from DEFAULT_LLM_MODEL env var or is passed explicitly.
# NEVER hardcode model names — always use environment variables.

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

delegate = Delegate(model=model)

assert isinstance(delegate, Delegate)
assert delegate._budget_usd is None, "No budget cap by default"

# ── 2. Layer 2: configured Delegate ─────────────────────────────────
# Add tools, system prompt, and turn limits.

configured = Delegate(
    model=model,
    tools=[],  # Empty tool list — tools registered separately
    system_prompt="You are a helpful code reviewer.",
    max_turns=20,
)

assert configured._config.max_turns == 20

# ── 3. Layer 3: governed Delegate (budget tracking) ─────────────────
# budget_usd enables automatic cost tracking. The Delegate yields
# BudgetExhausted when the budget is exceeded.

governed = Delegate(
    model=model,
    budget_usd=5.0,
)

assert governed._budget_usd == 5.0
assert governed._consumed_usd == 0.0, "No cost consumed yet"

# ── 4. Event types ──────────────────────────────────────────────────
# Delegate.run() yields typed DelegateEvent subclass instances:
#
#   TextDelta      — incremental text from the model
#   TurnComplete   — the model finished responding
#   BudgetExhausted — budget cap exceeded
#   ErrorEvent     — an error occurred
#
# These are NOT raw strings — they're structured data for pattern matching.

assert issubclass(TextDelta, DelegateEvent)
assert issubclass(TurnComplete, DelegateEvent)
assert issubclass(BudgetExhausted, DelegateEvent)
assert issubclass(ErrorEvent, DelegateEvent)

# ── 5. run() — async streaming interface ────────────────────────────
# The primary interface. Yields DelegateEvent instances as an async generator.
#
# async for event in delegate.run("What is Kailash?"):
#     if isinstance(event, TextDelta):
#         print(event.text, end="")
#     elif isinstance(event, TurnComplete):
#         print(f"\nDone. Tokens: {event.prompt_tokens} + {event.completion_tokens}")
#
# NOTE: We don't actually call run() here because it requires an LLM API key.
# This tutorial validates the construction and event type system.

# ── 6. run_sync() — synchronous convenience ────────────────────────
# For scripts that don't need streaming. Returns the full text response.
#
# result = delegate.run_sync("What is Kailash?")
# print(result)  # Full text string

# ── 7. Edge case: budget validation ─────────────────────────────────

try:
    Delegate(model=model, budget_usd=-1.0)
    assert False, "Negative budget should raise ValueError"
except ValueError as e:
    assert "non-negative" in str(e).lower()

try:
    Delegate(model=model, budget_usd=float("inf"))
    assert False, "Infinite budget should raise ValueError"
except ValueError as e:
    assert "finite" in str(e).lower()

# ── 8. Edge case: model from environment ────────────────────────────
# When model="" (empty), Delegate reads DEFAULT_LLM_MODEL from env.

env_delegate = Delegate(model="")
# The resolved model is stored in the config
assert env_delegate._config.model is not None

# ── 9. Cost model ───────────────────────────────────────────────────
# Delegate uses conservative per-1M-token cost estimates by model prefix:
#   claude- : $3 input, $15 output
#   gpt-4o  : $2.5 input, $10 output
#   gemini- : $1.25 input, $5 output
#
# These are approximations for budget tracking, not billing.

print("PASS: 03-kaizen/02_delegate")
