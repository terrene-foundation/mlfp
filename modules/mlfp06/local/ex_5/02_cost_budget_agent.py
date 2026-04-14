# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.2: Cost-Bounded Agents (LLMCostTracker)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Attach a dollar budget to any ReActAgent via max_llm_cost_usd
#   - Watch budget enforcement stop a looping agent gracefully
#   - Understand budget hierarchy: session -> task -> step
#   - Connect agent budgets to PACT governance (Ex 7)
#
# PREREQUISITES: 01_react_agent.py (ReAct loop, tools)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load data + tools
#   2. Build two agents — one with tight budget, one with normal budget
#   3. Run both on an intentionally expensive task
#   4. Visualise: when the budget trips vs when the task completes
#   5. Apply: Singapore SME chatbot cost-safety scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from kaizen_agents.agents.specialized.react import ReActAgent

from shared.mlfp06.ex_5 import MODEL, load_hotpotqa, make_tools

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Data and tools
# ════════════════════════════════════════════════════════════════════════

# TODO: Load HotpotQA and bind tools (see 01_react_agent.py Task 1)
qa_data = ____
tools = ____
print(f"Loaded {qa_data.height} QA examples + {len(tools)} tools\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(tools) == 4
print("✓ Checkpoint 1 passed — infra ready\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why agents need a financial operating envelope
# ════════════════════════════════════════════════════════════════════════
# A looping agent can burn money fast.  LLMCostTracker is the first line
# of defence:
#
#   Session budget ($50) -> Task budget ($5) -> Step budget ($1)
#
# When the budget is exceeded, the agent is forced to produce a final
# answer and subsequent tool calls are refused.  This is the AGENT layer
# of the envelope — PACT (Ex 7) adds the ORGANISATION layer above it.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build two agents with very different budgets
# ════════════════════════════════════════════════════════════════════════

# TODO: Build a ReActAgent with max_llm_cost_usd=0.10 (intentionally tight)
low_budget_agent = ____
# TODO: Build a ReActAgent with max_llm_cost_usd=2.00 (enough to complete)
normal_agent = ____

print(f"Low-budget agent:    $0.10 (will likely trip)")
print(f"Normal-budget agent: $2.00 (should complete)")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert low_budget_agent is not None
assert normal_agent is not None
print("\n✓ Checkpoint 2 passed — two budgeted agents built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Run both on an intentionally expensive task
# ════════════════════════════════════════════════════════════════════════

EXPENSIVE_TASK = """Perform an exhaustive analysis of the dataset:
1. Get summary statistics
2. Count all question types
3. Count all difficulty levels
4. Search for 10 different topics (churn, fraud, growth, retention, AI,
   governance, policy, climate, risk, regulation)
5. Find the longest documents
6. Synthesise a multi-paragraph report."""


async def run_with_budget(agent: ReActAgent, label: str) -> tuple[str, bool]:
    """Run the agent and return (output_preview, tripped_budget)."""
    print(f"--- {label} ---")
    try:
        # TODO: Await agent.run(EXPENSIVE_TASK)
        result = ____
        output = str(result)[:300] if result else "No output"
        print(f"  Completed. Result preview: {output}...")
        return output, False
    except Exception as e:
        msg = str(e)
        print(f"  Budget tripped: {msg[:200]}")
        return msg, True


async def compare_budgets():
    print("Running tight-budget agent on expensive task...")
    low_output, low_tripped = await run_with_budget(low_budget_agent, "LOW $0.10")

    print("\nRunning normal-budget agent on the same task...")
    normal_output, normal_tripped = await run_with_budget(normal_agent, "NORMAL $2.00")

    return low_output, low_tripped, normal_output, normal_tripped


low_output, low_tripped, normal_output, normal_tripped = asyncio.run(compare_budgets())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert low_output is not None
assert normal_output is not None
print("\n✓ Checkpoint 3 passed — both agents ran to completion or trip\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the outcome table
# ════════════════════════════════════════════════════════════════════════

import polars as pl

outcomes = pl.DataFrame(
    {
        "Agent": ["low_budget", "normal_budget"],
        "Budget USD": [0.10, 2.00],
        "Budget Tripped": [low_tripped, normal_tripped],
        "Output Preview": [
            low_output[:80].replace("\n", " ") + "...",
            normal_output[:80].replace("\n", " ") + "...",
        ],
    }
)
print("=" * 70)
print("  Budget enforcement comparison")
print("=" * 70)
print(outcomes)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert outcomes.height == 2
print("\n✓ Checkpoint 4 passed — outcome table visualised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore SME chatbot cost safety
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore SME deploys a support chatbot.  An attacker
# crafts a prompt that loops the agent.  Without a budget: ~$200 per
# attacker session.  With max_llm_cost_usd=$0.25: $0.25 per attacker,
# 800x reduction in denial-of-wallet exposure.  MAS FSM-GL-04 requires
# fraud-resistant cost controls on AI customer interactions.

print("=" * 70)
print("  KEY TAKEAWAY: Every production agent needs a budget")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Attached max_llm_cost_usd to a ReActAgent
  [x] Observed budget enforcement stop a looping agent
  [x] Understood the session -> task -> step budget hierarchy
  [x] Quantified the denial-of-wallet risk for production chatbots
  [x] Chose a budget using the 95th-percentile-times-two rule

  Next: 03_structured_agent.py switches from tool-using ReAct to
  typed structured output via BaseAgent + Signature...
"""
)
