# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 5.1: ReAct Agent — Tool-Using Autonomous Reasoning
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a ReAct agent: the Thought -> Action -> Observation loop
#   - Register Python tools with JSON schemas in a Kaizen ToolRegistry
#   - Run multi-step analysis where the LLM chooses the tool order
#   - Capture and interpret the REAL reasoning trace with the agent lens
#   - Understand function-calling protocol (auto / required / specific)
#
# PREREQUISITES: MLFP06 Ex 1-4 (Delegate, Signature, prompt engineering)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Load HotpotQA multi-hop dataset + bind tools
#   2. Register the tools and build a tool-using Delegate
#   3. Run multi-step analysis tasks under trace capture
#   4. Visualise the real reasoning trace
#   5. Apply: Singapore banking research analyst scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

import matplotlib.pyplot as plt
import polars as pl

from shared.mlfp06._ollama_bootstrap import make_delegate, preflight_ollama
from shared.mlfp06.diagnostics import LLMObservatory
from shared.mlfp06.ex_5 import (
    MODEL,
    OUTPUT_DIR,
    build_tool_registry,
    load_hotpotqa,
    make_tools,
    print_tool_registry,
    require_llm_trace,
    tool_schemas,
)

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and bind tools
# ════════════════════════════════════════════════════════════════════════

qa_data = load_hotpotqa()
tools = make_tools(qa_data)
print_tool_registry(tools)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert qa_data.height > 0, "Task 1: dataset should not be empty"
assert len(tools) == 4, "Task 1: should have 4 tools bound to qa_data"
assert all(callable(t) for t in tools), "All tools should be callable"
assert all(
    t.__doc__ for t in tools
), "All tools need docstrings — the agent reads them as API documentation"
print("\n✓ Checkpoint 1 passed — 4 tools registered with HotpotQA\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The ReAct Loop
# ════════════════════════════════════════════════════════════════════════
# ReAct = Reasoning + Acting.  The agent interleaves thinking and doing:
#
#   1. THOUGHT: "I need to understand the dataset first."
#   2. ACTION:  data_summary(dataset_name="qa_data")
#   3. OBSERVATION: <tool output: rows, columns, average text lengths>
#   4. THOUGHT: "Now I know the columns.  Let me count question types."
#   5. ACTION:  run_query("count question types")
#   6. OBSERVATION: <bridge / comparison counts>
#   7. THOUGHT: "I have enough — let me synthesise the answer."
#   8. FINAL ANSWER: <synthesised response>
#
# Unlike an if-else pipeline, the agent decides WHICH tool and WHAT
# arguments at each step.  The loop is autonomous — no human choreography.
#
# HOW KAIZEN RUNS IT: a Kaizen Delegate is a ReAct loop.  Each LLM turn
# either (a) emits one or more tool calls — the Thought + Action — which
# the Delegate executes against its ToolRegistry and feeds back as the
# Observation, or (b) answers in plain text, which ends the loop.  A
# tool only exists for the model if it is REGISTERED: name, description,
# JSON schema for the arguments, and an async executor.  Handing the
# Delegate bare Python functions registers nothing — the model then has
# no tools and simply guesses.  (Kaizen also ships a ReActAgent class,
# but in the installed release its tool executor only dispatches MCP
# tools, not Python functions — so for Python tools we use the Delegate.)
#
# ANALOGY: A research analyst with a filing cabinet.  You ask "which
# clients are at risk of churn?"  The analyst doesn't follow a script.
# They open the cabinet, pull a summary, realise they need the activity
# log, pull that, cross-reference — each step informed by the last.
# ReAct is that analyst, but the filing cabinet is your tool registry.
#
# WHY IT MATTERS: Business questions rarely decompose into fixed pipelines.
# ReAct lets one agent handle a whole class of questions without you
# writing a new pipeline for each.  One agent, N questions.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Register the tools and build the tool-using Delegate
# ════════════════════════════════════════════════════════════════════════

REACT_SYSTEM_PROMPT = (
    "You are a research analyst working on the HotpotQA dataset. "
    "Use the available tools to gather evidence before you answer. "
    "Call a tool whenever you need data; do not invent numbers. "
    "When you have enough evidence, reply with a concise plain-text report."
)
MAX_TURNS = 8  # hard ceiling on Thought->Action->Observation cycles


def build_react_agent():
    """Fresh Ollama-backed Delegate with the four tools registered.

    A Delegate keeps its conversation history between runs, so every
    independent task in this file gets its own instance.
    """
    registry = build_tool_registry(tools)
    return make_delegate(
        tools=registry,
        system_prompt=REACT_SYSTEM_PROMPT,
        max_turns=MAX_TURNS,
    )


react_agent = build_react_agent()
print("ReAct agent built:")
print(f"  Model:      {MODEL}  (from OLLAMA_CHAT_MODEL)")
print(f"  Tools:      {react_agent.tool_registry.tool_names}")
print(f"  Max turns:  {MAX_TURNS}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert react_agent is not None, "Task 2: agent should be created"
assert react_agent.tool_registry.tool_names == [
    t.__name__ for t in tools
], "Task 2: every tool must be registered — an unregistered tool is invisible"
print("\n✓ Checkpoint 2 passed — ReAct agent ready with 4 registered tools\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train (run) the agent on multi-step tasks, capturing the trace
# ════════════════════════════════════════════════════════════════════════
# "Train" here means exercise the agent end-to-end — LLM agents are not
# gradient-trained at runtime; the "training" is the reasoning trajectory.
# The Observatory's agent lens records every tool call the Delegate makes.

preflight_ollama(required_models=[MODEL])  # fails loudly if Ollama is down
obs = LLMObservatory(run_id="ex_5_react")

sample_q = qa_data["question"][0]
comparison_q = qa_data.filter(pl.col("type") == "comparison")["question"][0]
bridge_q = qa_data.filter(pl.col("type") == "bridge")["question"][1]

TASKS = {
    "multi_step": f"""Analyse the HotpotQA multi-hop reasoning dataset to understand
its structure and answer the question: "{sample_q}"

Steps:
1. Get a dataset summary to understand the columns and types
2. Count question types (comparison vs bridge) and difficulty levels
3. Search for documents relevant to the question above
4. Extract the answer evidence (the tool never sees answer labels)
5. Synthesise your findings into a clear report.""",
    "comparison": f'Find evidence in the corpus and answer: "{comparison_q}"',
    "bridge": f'Find evidence in the corpus and answer: "{bridge_q}"',
}


async def run_traced(label: str, task: str, *, _retried: bool = False) -> dict:
    """Run one task on a fresh agent and return measured trace statistics.

    Small local models occasionally answer without calling any tool. The
    production mitigation is ONE retry with a firmer tool-use instruction —
    not a silent pass, and not an infinite loop.
    """
    agent = build_react_agent()
    t0 = time.perf_counter()
    trace = await obs.agent.capture_run(agent, task, run_id=f"react_{label}")
    latency_s = time.perf_counter() - t0
    require_llm_trace(trace)
    tool_sequence = [ev.tool for ev in trace.events if ev.kind == "tool_start"]
    if not tool_sequence and not _retried:
        print(f"  [{label}] no tool call on the first pass — one retry with a firmer instruction")
        return await run_traced(
            label,
            task + "\n\nYou MUST call at least one tool before answering.",
            _retried=True,
        )
    answer = "".join(ev.content or "" for ev in trace.events if ev.kind == "token")
    return {
        "label": label,
        "run_id": trace.run_id,
        "tool_sequence": tool_sequence,
        "llm_turns": agent.loop.usage.turns,
        "total_tokens": agent.loop.usage.total_tokens,
        "latency_s": latency_s,
        "answer": answer,
    }


async def run_all() -> list[dict]:
    return [await run_traced(label, task) for label, task in TASKS.items()]


print(f"Main task: {TASKS['multi_step'][:200]}...\n")
runs = asyncio.run(run_all())
main_run = runs[0]
print(f"Agent answer (first 500 chars):\n{main_run['answer'][:500]}...")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert main_run["answer"].strip(), "Task 3: the agent should produce an answer"
assert len(main_run["tool_sequence"]) >= 1, (
    "Task 3: the agent answered without calling a single tool — check that "
    "the tools are registered and the model is tool-capable"
)
print("\n✓ Checkpoint 3 passed — multi-step analysis ran real tool calls\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the reasoning trace + function-calling protocol
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  Measured reasoning traces")
print("=" * 70)
for run in runs:
    print(
        f"  {run['label']:11s} turns={run['llm_turns']}  "
        f"tool calls={len(run['tool_sequence'])}  "
        f"latency={run['latency_s']:.1f}s  tokens={run['total_tokens']}"
    )
    print(f"              order: {' -> '.join(run['tool_sequence']) or '(none)'}")

print("\nTool usage on the main task (agent lens):")
print(obs.agent.tool_usage(main_run["run_id"]))
loops = obs.agent.detect_loops(main_run["run_id"])
print(f"Stuck loops detected (same tool + args 3x in a row): {loops.height}")

print(
    """
Good trace signals:
  ✓ Logical step order (general → specific)
  ✓ No redundant tool calls
  ✓ Arguments match the tool schema
  ✓ Final answer synthesises ALL observations, not just the last one

Bad trace signals:
  ✗ Random tool ordering
  ✗ Same tool called twice with identical args
  ✗ Arguments that don't match schema
  ✗ Final answer ignores some observations
"""
)

# The structured JSON schemas the model actually receives.
schemas = tool_schemas(tools)
print(f"Function-calling schemas ({len(schemas)} tools):")
for s in schemas:
    print(f"  {s['name']:20s} params={list(s['parameters']['properties'].keys())}")

print(
    """
Function-calling protocol (tool_choice, as exposed by most chat APIs):
  auto      — model decides whether to call a tool or respond directly
  required  — model MUST call at least one tool (force data grounding)
  specific  — pin to one named function (pipeline step)

Parallel calls — a model may emit several tool calls in ONE turn, e.g.
  [search_documents("churn"), run_query("count types")]
The Delegate executes all of them concurrently (asyncio.gather) before
the next LLM turn and returns every observation together.
"""
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(schemas) == 4, "Task 4: should generate schemas for all 4 tools"
assert all(
    "name" in s and "parameters" in s for s in schemas
), "Every schema needs name + parameters"
print("✓ Checkpoint 4 passed — trace interpretation and schemas visualised\n")

# INTERPRETATION: read the "order" lines above.  A good agent goes
# general -> specific (summary, then query, then search).  A poor agent
# repeats the same call or skips to an answer without looking at the
# data — a signal the prompt or the tool docstrings need work.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore banking research analyst
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative figures): A private bank in Singapore runs a
# research desk that answers wealth-advisor questions like "which
# regions had the most regulatory changes last quarter?"  Each question
# decomposes into search, filter, count, and synthesis — the same shape
# as the HotpotQA tasks above.
#
# BEFORE REACT: 4 research analysts at ~S$120K/year each (S$480K/year)
# answer ~2,000 such questions per year — ~S$240 per answer with a
# 1-2 day turnaround, and answers go stale before advisors use them.
#
# WITH REACT: one tool-using agent over the bank's internal search,
# summary, and lookup tools answers in the latency you measured above.
# On a self-hosted model the marginal cost is compute, not per-token
# fees; analysts move to reviewing and extending the agent's reports.
#
# BUSINESS IMPACT:
#   - Turnaround:         1-2 days -> the per-task latency printed above
#   - Advisor experience: stale reports -> live conversation support
#   - Analysts pivot to   higher-value qualitative work (fund manager
#                         interviews, thesis development)
#
# THE RISK: an unconstrained agent can loop.  Technique 2
# (02_cost_budget_agent.py) bounds the loop.


print("=" * 70)
print("  KEY TAKEAWAY: a ReAct agent turns a pipeline into a conversation")
print("=" * 70)
print(
    """
  Before: one pipeline per question shape, brittle and expensive.
  After:  one agent + N registered tools, answers any question that
          composes them.

  The tool docstring + schema is now your most important artifact.
  It's the contract the LLM reads to decide what to do next.  Precise
  tool docs = accurate agents.  Vague tool docs = wrong tool, wrong
  arguments, wasted turns.
"""
)


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Agent reasoning step profile (measured)
# ════════════════════════════════════════════════════════════════════════

labels = [r["label"] for r in runs]
tool_calls = [len(r["tool_sequence"]) for r in runs]
latencies_s = [r["latency_s"] for r in runs]

fig, ax1 = plt.subplots(figsize=(8, 4))
x = range(len(labels))
ax1.bar(x, tool_calls, color="#2196F3", alpha=0.8, label="Tool calls")
ax1.set_ylabel("Tool calls (measured)", color="#2196F3")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=15, ha="right")

ax2 = ax1.twinx()
ax2.plot(x, latencies_s, "o-", color="#FF5722", linewidth=2, label="Latency (s)")
ax2.set_ylabel("Latency (s, measured)", color="#FF5722")

ax1.set_title("ReAct Agent: Tool Calls & Latency per Task")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_react_steps.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '01_react_steps.png'}")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — the agent lens on this run
# ══════════════════════════════════════════════════════════════════
# The LLM Observatory has six lenses (Output, Attention, Retrieval,
# Agent Trace, Alignment, Governance).  Only the Agent Trace lens
# applies to a tool-using agent, and it already recorded every run
# above — this is its plain-text Prescription Pad over those runs.
print("\n── LLM Observatory — agent lens ──")
print(obs.agent.report())
# Reading it: tools_used should be > 1 for the multi-step task; an
# error event means a tool raised; a "stuck loop" line means the model
# called the same tool with the same arguments 3+ times — tighten the
# tool docstrings so the model knows what each tool returns.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a ReAct agent: a Delegate with registered, schema-typed tools
  [x] Ran multi-step tasks where the LLM chose the tool order
  [x] Inspected the real reasoning trace (tool order, loops, latency)
  [x] Understood function-calling protocol and parallel calls
  [x] Mapped the technique to a Singapore private-banking use case

  KEY INSIGHT: Agents are LLMs with the ability to call functions.
  No new AI — just LLMs that observe and act instead of just responding.
  The novelty is that YOU design the tool surface; the LLM orchestrates.

  Next: 02_cost_budget_agent.py bounds the loop so a confused agent
  cannot run forever...
"""
)
