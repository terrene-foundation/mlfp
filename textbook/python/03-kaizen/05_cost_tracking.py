# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Kaizen / Cost Tracking and Budgets
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Track LLM costs and enforce budget limits on agents
# LEVEL: Advanced
# PARITY: Full — Rust has cost module with same budget semantics
# VALIDATES: Delegate budget_usd, cost estimation, BudgetExhausted event
#
# Run: uv run python textbook/python/03-kaizen/05_cost_tracking.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kaizen_agents import Delegate
from kaizen_agents.delegate.events import BudgetExhausted

# ── 1. Budget-tracked Delegate ──────────────────────────────────────
# budget_usd sets a hard cap on estimated cost. When the accumulated
# cost exceeds the budget, the Delegate yields BudgetExhausted and stops.

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

delegate = Delegate(model=model, budget_usd=1.0)

assert delegate._budget_usd == 1.0
assert delegate._consumed_usd == 0.0, "Starts at zero"

# ── 2. Cost estimation internals ────────────────────────────────────
# The Delegate estimates cost after each turn using model prefix heuristics.
# This is NOT billing data — it's an approximation for budget enforcement.
#
# Example: claude-sonnet with 1000 prompt tokens + 500 completion tokens
#   input cost  = 1000 / 1_000_000 * $3  = $0.003
#   output cost = 500 / 1_000_000 * $15  = $0.0075
#   total = $0.0105 per turn


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Replicate the Delegate's cost estimation logic."""
    rates = {
        "claude-": (3.0, 15.0),
        "gpt-4o": (2.5, 10.0),
        "gemini-": (1.25, 5.0),
    }
    input_rate, output_rate = 3.0, 15.0  # defaults
    for prefix, (ir, otr) in rates.items():
        if model.startswith(prefix):
            input_rate, output_rate = ir, otr
            break
    return (prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000


# Verify cost calculation
cost = estimate_cost("claude-sonnet-4-20250514", 1000, 500)
assert 0.01 < cost < 0.02, f"Expected ~$0.01, got ${cost:.4f}"

# ── 3. BudgetExhausted event ────────────────────────────────────────
# When budget is exceeded during run(), the Delegate yields:
#   BudgetExhausted(budget_usd=X, consumed_usd=Y)
# This is a DelegateEvent, not an exception — the caller handles it.

event = BudgetExhausted(budget_usd=1.0, consumed_usd=1.05)
assert event.budget_usd == 1.0
assert event.consumed_usd == 1.05

# ── 4. No budget = unlimited ────────────────────────────────────────

unlimited = Delegate(model=model)
assert unlimited._budget_usd is None, "No budget cap"

# ── 5. Budget validation ────────────────────────────────────────────
# Budget must be finite and non-negative

try:
    Delegate(model=model, budget_usd=-0.01)
    assert False, "Negative budget should fail"
except ValueError:
    pass

try:
    Delegate(model=model, budget_usd=float("nan"))
    assert False, "NaN budget should fail"
except ValueError:
    pass

# Zero budget is valid (immediately exhausted on first turn)
zero_budget = Delegate(model=model, budget_usd=0.0)
assert zero_budget._budget_usd == 0.0

# ── 6. Production pattern: budget per request ───────────────────────
# In production, create a new Delegate per request with a per-request budget:
#
#   async def handle_request(user_prompt: str):
#       delegate = Delegate(model=model, budget_usd=0.50)
#       async for event in delegate.run(user_prompt):
#           if isinstance(event, BudgetExhausted):
#               return "Request exceeded cost limit"
#           yield event

print("PASS: 03-kaizen/05_cost_tracking")
