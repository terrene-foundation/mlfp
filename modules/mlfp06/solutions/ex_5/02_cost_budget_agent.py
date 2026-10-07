# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.2: Bounded Agents — Turn Ceilings and Dollar Budgets
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Bound a runaway ReAct agent with a hard turn ceiling (max_turns)
#   - Attach a dollar budget via budget_limit_usd AT CONSTRUCTION, and
#     see why setting it afterwards silently does nothing
#   - Measure what an agent actually consumed: turns, tool calls, tokens
#   - Know which control works where: dollar caps on priced providers,
#     turn and token ceilings on free local models
#   - Connect agent budgets to PACT governance (Ex 7)
#
# PREREQUISITES: 01_react_agent.py (ReAct loop, tools)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load data + tools
#   2. Build a tight and a normal bounded agent; configure a dollar cap
#   3. Run both on an intentionally expensive task
#   4. Visualise measured consumption against each ceiling
#   5. Apply: Singapore SME chatbot cost-safety scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import matplotlib.pyplot as plt
import polars as pl
from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent, BaseAgentConfig

from shared.mlfp06._ollama_bootstrap import (
    OLLAMA_BASE_URL,
    make_delegate,
    preflight_ollama,
)
from shared.mlfp06.diagnostics import LLMObservatory
from shared.mlfp06.ex_5 import (
    MODEL,
    OUTPUT_DIR,
    build_tool_registry,
    load_hotpotqa,
    make_tools,
    require_llm_trace,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Data and tools
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)
print(f"Loaded {qa_data.height} QA examples + {len(tools)} tools\n")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0
assert len(tools) == 4
print("✓ Checkpoint 1 passed — infra ready\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why agents need an operating envelope
# ════════════════════════════════════════════════════════════════════════
# A ReAct agent decides for itself when to stop.  On a badly phrased task
# it may never decide — it keeps calling tools, and every turn is another
# LLM call.  Three ceilings bound it:
#
#   1. TURN CEILING (max_turns) — the Delegate's loop stops after N
#      Thought->Action->Observation cycles, whatever the model wants.
#      Works on every provider, including free local models.
#   2. DOLLAR CAP — BaseAgentConfig.budget_limit_usd (and Delegate's
#      budget_usd) stop an agent once its PRICED spend reaches the cap.
#      It is only as good as the price table behind it: Kaizen prices a
#      local Ollama call at $0.00, so on this course's stack a dollar cap
#      can never trip.  make_delegate() deliberately sets budget_usd=None
#      for the same reason.  On a hosted, priced provider it is the
#      control you want.
#   3. TOKEN ACCOUNTING — every run reports prompt + completion tokens.
#      Tokens x your provider's price is your real spend.
#
# One trap with the dollar cap: BaseAgent copies budget_limit_usd into
# its enforcement context ONLY in __init__.  Assigning
# agent.config.budget_limit_usd after construction changes the config
# object but not the enforcement — the agent runs uncapped.
#
# This is the AGENT layer of the envelope.  The ORGANISATION layer sits
# above it in PACT (Ex 7), where a governance engine bounds spend per
# role and cascades budgets from parent to child.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build two bounded agents and configure a dollar cap
# ════════════════════════════════════════════════════════════════════════

TIGHT_TURNS = 2  # intentionally too few for the expensive task
NORMAL_TURNS = 12  # enough headroom to finish
SYSTEM_PROMPT = (
    "You are a research analyst. Use the tools to gather evidence, one "
    "step at a time, then write a plain-text report."
)


def build_bounded_agent(max_turns: int):
    """Ollama-backed tool-using Delegate with a hard turn ceiling."""
    return make_delegate(
        tools=build_tool_registry(tools),
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )


tight_agent = build_bounded_agent(TIGHT_TURNS)
normal_agent = build_bounded_agent(NORMAL_TURNS)
print(f"Tight agent:  max_turns={TIGHT_TURNS}  (should hit the ceiling)")
print(f"Normal agent: max_turns={NORMAL_TURNS} (should finish)")


class BriefingSignature(Signature):
    """Write a one-paragraph briefing from a dataset summary."""

    dataset_summary: str = InputField(description="Statistical summary")
    briefing: str = OutputField(description="One-paragraph briefing")


# Dollar cap — the right way: pass budget_limit_usd at construction.
capped_agent = BaseAgent(
    config=BaseAgentConfig(
        llm_provider="ollama",
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        use_async_llm=True,
        budget_limit_usd=0.10,
    ),
    signature=BriefingSignature(),
)
# Dollar cap — the trap: assign it after construction.
late_capped_agent = BaseAgent(
    config=BaseAgentConfig(
        llm_provider="ollama",
        model=MODEL,
        base_url=OLLAMA_BASE_URL,
        use_async_llm=True,
    ),
    signature=BriefingSignature(),
)
late_capped_agent.config.budget_limit_usd = 0.10

print("\nDollar cap enforcement context:")
print(f"  set at construction: {capped_agent.execution_context.budget_limit}")
print(f"  set afterwards:      {late_capped_agent.execution_context.budget_limit}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert tight_agent.tool_registry.tool_names == normal_agent.tool_registry.tool_names
assert (
    capped_agent.execution_context.budget_limit == 0.10
), "Task 2: budget_limit_usd must be passed when the agent is constructed"
assert (
    late_capped_agent.execution_context.budget_limit is None
), "Assigning config.budget_limit_usd after construction is not enforced"
print("\n✓ Checkpoint 2 passed — two bounded agents + a correctly capped agent\n")


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

preflight_ollama(required_models=[MODEL])  # fails loudly if Ollama is down
obs = LLMObservatory(run_id="ex_5_bounded")


async def run_bounded(agent, label: str, max_turns: int) -> dict:
    """Run one bounded agent and return what it measurably consumed."""
    print(f"--- {label} (max_turns={max_turns}) ---")
    trace = await obs.agent.capture_run(agent, EXPENSIVE_TASK, run_id=label)
    require_llm_trace(trace)
    steps = [ev for ev in trace.events if ev.kind in ("token", "tool_end", "error")]
    turns_used = agent.loop.usage.turns
    # The ceiling stopped the agent if it used every turn and its last
    # action was a tool call rather than a written answer.
    stopped_by_ceiling = turns_used >= max_turns and bool(steps) and (
        steps[-1].kind != "token"
    )
    answer = "".join(ev.content or "" for ev in trace.events if ev.kind == "token")
    outcome = {
        "label": label,
        "max_turns": max_turns,
        "turns_used": turns_used,
        "tool_calls": len(trace.filter_kind("tool_start")),
        "total_tokens": agent.loop.usage.total_tokens,
        "stopped_by_ceiling": stopped_by_ceiling,
        "answer": answer,
    }
    status = "STOPPED BY TURN CEILING" if stopped_by_ceiling else "finished"
    print(f"  {status}: {turns_used} turns, {outcome['tool_calls']} tool calls")
    return outcome


async def compare_bounds() -> tuple[dict, dict]:
    tight = await run_bounded(tight_agent, "tight", TIGHT_TURNS)
    normal = await run_bounded(normal_agent, "normal", NORMAL_TURNS)
    return tight, normal


tight_run, normal_run = asyncio.run(compare_bounds())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert tight_run["turns_used"] <= TIGHT_TURNS, "The turn ceiling must hold"
assert normal_run["turns_used"] <= NORMAL_TURNS, "The turn ceiling must hold"
assert normal_run["tool_calls"] >= 1, "The normal agent should call tools"
print("\n✓ Checkpoint 3 passed — both agents stayed inside their ceilings\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise measured consumption against each ceiling
# ════════════════════════════════════════════════════════════════════════

# Tokens are what a hosted provider bills.  The price below is an
# ILLUSTRATIVE reference rate, not a quote — substitute your provider's.
REFERENCE_USD_PER_MILLION_TOKENS = 1.00

outcomes = pl.DataFrame([tight_run, normal_run]).drop("answer")
outcomes = outcomes.with_columns(
    (pl.col("total_tokens") / 1e6 * REFERENCE_USD_PER_MILLION_TOKENS).alias(
        "est_hosted_cost_usd"
    )
)
print("=" * 70)
print("  Measured consumption (local Ollama: actual dollar cost is $0)")
print("=" * 70)
print(outcomes)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert outcomes.height == 2, "Task 4: should compare two runs"
print("\n✓ Checkpoint 4 passed — outcome table visualised\n")

# INTERPRETATION: the tight agent's turns_used equals its ceiling — the
# loop was cut off, whatever the model "wanted".  The normal agent's
# turns_used is what the task really needs.  The gap between the two is
# the "runaway zone": set production ceilings above the normal need but
# far below the point where a looping agent becomes expensive.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore SME chatbot cost safety
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative figures, all in USD): A Singapore SME deploys a
# customer-support agent on a HOSTED model.  It answers "where is my
# order?" and "what is your return policy?" with ~10 tools (order
# lookup, product catalog, shipping tracker, refund processor, FAQ
# search, ...).
#
# THE THREAT: a malicious user crafts a prompt that keeps the agent
# looping — "keep searching until you find my missing parcel."  The
# agent calls search_orders, search_shipments, search_refunds, repeat.
#
# THE ARITHMETIC:
#   Unbounded: ~$0.20 per iteration x 1,000 iterations = $200 per session
#              10 attacker sessions/day                = $2,000 per day
#              x 365 days                              = ~$730,000 per year
#   Bounded at $0.25 per session (dollar cap on the hosted provider):
#              10 sessions x $0.25 x 365               = ~$912 per year
#   Reduction in worst-case exposure: $200 / $0.25 = 800x per session.
#
# THE PATTERN: a dollar cap on the priced provider PLUS a turn ceiling
# (which also protects latency and works on self-hosted models).  Both
# are one constructor argument.  An agent with neither is a
# denial-of-wallet vulnerability.


print("=" * 70)
print("  KEY TAKEAWAY: every production agent needs a structural ceiling")
print("=" * 70)
print(
    """
  A ceiling is a structural defence, not a soft limit.  The agent cannot
  exceed it even if the LLM tries.  Think of it like a fuse in an
  electrical system — the agent isn't smart enough to know when to
  stop, so the fuse decides for it.

  Rule of thumb: set the ceiling to 2-5x the expected need.  If you
  don't know the expected need, run 10 representative tasks with a
  generous ceiling, take the 95th percentile of turns (or spend), and
  multiply by 2.
"""
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Ceiling vs measured consumption
# ════════════════════════════════════════════════════════════════════════

labels = [f"{r['label']}\n(max {r['max_turns']})" for r in (tight_run, normal_run)]
ceilings = [tight_run["max_turns"], normal_run["max_turns"]]
used = [tight_run["turns_used"], normal_run["turns_used"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
x = range(len(labels))
ax1.bar([i - 0.15 for i in x], ceilings, 0.3, label="Turn ceiling", color="#90CAF9")
ax1.bar([i + 0.15 for i in x], used, 0.3, label="Turns used", color="#FF7043")
ax1.set_xticks(list(x))
ax1.set_xticklabels(labels)
ax1.set_ylabel("LLM turns")
ax1.set_title("Turn Ceiling vs Turns Used (measured)")
ax1.legend()
for i, run in enumerate((tight_run, normal_run)):
    if run["stopped_by_ceiling"]:
        ax1.annotate(
            "STOPPED",
            (i + 0.15, used[i]),
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
            fontweight="bold",
        )
ax2.bar(labels, outcomes["total_tokens"].to_list(), color="#7E57C2")
ax2.set_ylabel("Tokens (prompt + completion)")
ax2.set_title("Tokens Consumed (measured)")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_budget_utilization.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '02_budget_utilization.png'}")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — the agent lens on this run
# ══════════════════════════════════════════════════════════════════
# Only the Agent Trace lens applies here; it recorded both runs above.
print("\n── LLM Observatory — agent lens ──")
print(obs.agent.report())
# Reading it: the tight run should show fewer tool calls than the
# normal run; a "stuck loop" line on the normal run means the ceiling
# is what ended it — tighten the prompt or the tool docstrings.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Bounded a ReAct agent with a hard turn ceiling (max_turns)
  [x] Passed budget_limit_usd at construction — and saw that assigning
      it afterwards leaves the agent uncapped
  [x] Measured real consumption: turns, tool calls, tokens
  [x] Quantified the denial-of-wallet risk for a hosted chatbot
  [x] Chose a ceiling using the 95th-percentile-times-two rule

  KEY INSIGHT: ceilings are the cheapest insurance policy in the AI
  stack.  One constructor argument each, and they turn an unbounded
  worst case into a known one.  Use the dollar cap where calls are
  priced, the turn ceiling everywhere.

  Next: 03_structured_agent.py switches from tool-using ReAct to
  typed structured output via BaseAgent + Signature...
"""
)
