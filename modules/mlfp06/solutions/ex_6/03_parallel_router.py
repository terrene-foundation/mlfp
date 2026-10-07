# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.3: Parallel Execution + LLM-Based Routing
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run independent specialists truly concurrently with asyncio.gather
#   - Prove the latency win: parallel ≈ max(stages), not sum
#   - Build an LLM router that picks the right specialist from each
#     specialist's capability card (and see why Kaizen's Pipeline.router()
#     needs care)
#   - Measure keyword routing (brittle) against LLM routing on the same
#     queries, including paraphrases that avoid the keywords
#
# PREREQUISITES: 02_sequential_pipeline.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load corpus + specialists
#   2. Build the parallel asyncio.gather orchestrator
#   3. Measure parallel vs sequential latency
#   4. Build the LLM router (capability cards + routing Signature)
#   5. Route six queries with both routers and score them
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import polars as pl
from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import OLLAMA_BASE_URL, preflight_ollama
from shared.mlfp06.ex_6 import (
    MODEL,
    OUTPUT_DIR,
    build_specialists,
    load_squad_corpus,
    run_checked,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Parallel Execution and LLM Routing
# ════════════════════════════════════════════════════════════════════════
# Parallel execution: when specialists are INDEPENDENT (no stage
# depends on another), you can launch all of them simultaneously
# with asyncio.gather. Total latency collapses from
#   sum(stages)  →  max(stages)
# For 3 specialists at ~2s each: sequential 6s, parallel 2s.
#
# LLM routing: when you have many specialists and one incoming query,
# you need a dispatcher. Keyword routing ("if query contains 'revenue',
# call financial_agent") is brittle — it misses paraphrases. LLM
# routing reads each specialist's description (capability card) and
# reasons about which specialist best matches the query intent.
#
# Non-technical analogy: a hospital triage desk. The keyword approach
# is a flowchart on the wall ("if patient says 'chest', call cardio").
# The LLM approach is a senior nurse who LISTENS to the patient,
# knows every department's scope, and routes by intent. The senior
# nurse handles paraphrases; the flowchart doesn't.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus + specialists
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load corpus + specialists")
print("=" * 70)

passages = load_squad_corpus()
factual_agent, semantic_agent, structural_agent = build_specialists()
print(f"Corpus: {passages.height}, three specialists instantiated")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0
assert factual_agent and semantic_agent and structural_agent
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the parallel orchestrator
# ════════════════════════════════════════════════════════════════════════


async def parallel_analysis(doc: str, question: str) -> dict:
    """Launch all specialists simultaneously with asyncio.gather."""
    t0 = time.perf_counter()

    factual_task = run_checked(factual_agent, document=doc, question=question)
    semantic_task = run_checked(semantic_agent, document=doc, question=question)
    structural_task = run_checked(structural_agent, document=doc, question=question)

    factual_r, semantic_r, structural_r = await asyncio.gather(
        factual_task, semantic_task, structural_task
    )

    elapsed = time.perf_counter() - t0
    return {
        "factual_claims": factual_r["factual_claims"],
        "themes": semantic_r["main_themes"],
        "entities": structural_r["key_entities"],
        "latency_s": elapsed,
    }


async def sequential_baseline(doc: str, question: str) -> float:
    """Same work, but one agent at a time — for latency comparison."""
    t0 = time.perf_counter()
    await run_checked(factual_agent, document=doc, question=question)
    await run_checked(semantic_agent, document=doc, question=question)
    await run_checked(structural_agent, document=doc, question=question)
    return time.perf_counter() - t0


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Measure parallel vs sequential latency
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Parallel vs sequential latency")
print("=" * 70)

doc = passages["text"][0]
question = passages["question"][0]


async def run_comparison():
    par = await parallel_analysis(doc, question)
    seq_latency = await sequential_baseline(doc, question)
    return par, seq_latency


preflight_ollama(required_models=[MODEL])  # fails loudly if Ollama is down
par_result, seq_latency = asyncio.run(run_comparison())

print(f"Parallel latency:   {par_result['latency_s']:5.1f}s  (~max of stages)")
print(f"Sequential latency: {seq_latency:5.1f}s  (~sum of stages)")
print(f"Speedup:            {seq_latency / max(par_result['latency_s'], 0.01):.2f}×")
print(f"\nFactual claims (top 3): {par_result['factual_claims'][:3]}")
print(f"Themes (top 3): {par_result['themes'][:3]}")
print(f"Entities (top 3): {par_result['entities'][:3]}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert par_result["factual_claims"], "Parallel run should produce claims"
# asyncio.gather overlaps the three specialist calls, so parallel wins WHEN the
# backend serves concurrent requests in parallel. A single local Ollama daemon
# serves one request at a time on one model instance, so the three "parallel"
# calls queue behind each other and the wall-clock lands at — or just above,
# due to scheduling jitter — the sequential baseline. We therefore assert the
# parallel path is not MATERIALLY slower (within a 25% jitter band) rather than
# strictly faster; the real speedup appears once the backend serves requests
# concurrently (hosted APIs, or Ollama with OLLAMA_NUM_PARALLEL>1).
assert (
    par_result["latency_s"] <= seq_latency * 1.25
), "Parallel run should not be materially slower than sequential"
print("\n✓ Checkpoint 2 passed — parallel execution verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Build the LLM router
# ════════════════════════════════════════════════════════════════════════
# The router is itself a small structured agent: it reads the query and
# one capability card per specialist (the agent's `description`) and
# returns the name of the specialist to call.  Every decision is a
# visible, typed field — you can log it, score it, and audit it.
#
# Kaizen also packages this idea as Pipeline.router(agents=[...]).  Two
# behaviours of the installed release make it a poor teaching tool here:
# if capability scoring fails (e.g. the LLM is unreachable) it silently
# falls back to the FIRST agent, and its result does not say which agent
# it picked.  Building the router explicitly keeps both visible.

print("=" * 70)
print("TASK 4: LLM-Based Query Routing")
print("=" * 70)

SPECIALISTS = {
    "factual": factual_agent,
    "semantic": semantic_agent,
    "structural": structural_agent,
}
CAPABILITY_CARDS = "\n".join(
    f"{name}: {agent.description}" for name, agent in SPECIALISTS.items()
)


class RoutingSignature(Signature):
    """Pick the single specialist whose capability best matches the query."""

    query: str = InputField(description="The user's question")
    capability_cards: str = InputField(
        description="One line per specialist, formatted 'name: capability'"
    )
    specialist: str = OutputField(
        description="Exactly one specialist name from the capability cards"
    )
    rationale: str = OutputField(description="One sentence explaining the choice")


@dataclass
class RouterConfig:
    llm_provider: str = "ollama"
    model: str = MODEL
    base_url: str = OLLAMA_BASE_URL
    temperature: float = 0.0  # routing should be deterministic
    use_async_llm: bool = True
    response_format: dict = field(default_factory=lambda: {"type": "json_object"})
    structured_output_mode: str = "explicit"


class RoutingAgent(BaseAgent):
    description = "Dispatcher: maps a query to the best-matching specialist"

    def __init__(self, config: RouterConfig | None = None):
        super().__init__(config=config or RouterConfig(), signature=RoutingSignature())


router = RoutingAgent()
print("Capability cards the router reads:")
print(CAPABILITY_CARDS)

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert isinstance(router.signature, RoutingSignature), "Task 4: router signature"
assert set(SPECIALISTS) == {"factual", "semantic", "structural"}
print("\n✓ Checkpoint 3 passed — router configured\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Route six queries with both routers and score them
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Routing intent → specialist (keyword vs LLM)")
print("=" * 70)

# (query, expected specialist, style).  The paraphrases ask for the same
# thing as the canonical query of the same intent, without its keywords.
test_queries = [
    ("What specific dates and numbers are mentioned in this passage?", "factual", "canonical"),
    ("What is the underlying theme of the author's argument?", "semantic", "canonical"),
    ("How is the passage organised and what entities are discussed?", "structural", "canonical"),
    ("In which year did this happen, and what figures are cited?", "factual", "paraphrase"),
    ("What is the writer really getting at beneath the surface?", "semantic", "paraphrase"),
    ("Who are the people and groups mentioned, and how are they linked?", "structural", "paraphrase"),
]

KEYWORDS = {
    "factual": ["date", "number", "how many", "when", "fact"],
    "semantic": ["theme", "meaning", "imply", "argument"],
    "structural": ["organis", "structure", "entit", "relationship"],
}


def keyword_route(query: str) -> str:
    """The brittle baseline: first specialist whose keyword appears."""
    q = query.lower()
    for name, words in KEYWORDS.items():
        if any(w in q for w in words):
            return name
    return "unrouted"


async def llm_route(query: str) -> str:
    """Ask the routing agent; unknown names are kept as-is (scored wrong)."""
    decision = await run_checked(
        router, query=query, capability_cards=CAPABILITY_CARDS
    )
    return str(decision["specialist"]).strip().lower()


async def route_all() -> list[str]:
    return [await llm_route(q) for q, _, _ in test_queries]


llm_choices = asyncio.run(route_all())
route_log = pl.DataFrame(
    {
        "query": [q for q, _, _ in test_queries],
        "expected": [e for _, e, _ in test_queries],
        "style": [s for _, _, s in test_queries],
        "keyword_choice": [keyword_route(q) for q, _, _ in test_queries],
        "llm_choice": llm_choices,
    }
).with_columns(
    (pl.col("keyword_choice") == pl.col("expected")).alias("keyword_correct"),
    (pl.col("llm_choice") == pl.col("expected")).alias("llm_correct"),
)
keyword_accuracy = route_log["keyword_correct"].mean()
llm_accuracy = route_log["llm_correct"].mean()

for r in route_log.iter_rows(named=True):
    print(f"\n  Query:    {r['query']}")
    print(
        f"  Expected: {r['expected']:10s} keyword -> {r['keyword_choice']:10s} "
        f"LLM -> {r['llm_choice']}"
    )
print(f"\nRouting accuracy: keyword {keyword_accuracy:.0%}, LLM {llm_accuracy:.0%}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert route_log.height == len(test_queries), "Task 5: route every query"
assert all(route_log["llm_choice"].str.len_chars() > 0), "Router must name a specialist"
print("\n✓ Checkpoint 4 passed — every query routed by both routers\n")

# INTERPRETATION: compare the two accuracy figures on the paraphrase
# rows.  The keyword router can only route words it was told about, so
# it fails as soon as users phrase things differently.  Where the LLM
# router is wrong, read its choice — the capability card is usually the
# thing to sharpen.

trace_path = OUTPUT_DIR / "ex6_parallel_router_trace.txt"
trace_path.write_text(
    f"Parallel latency: {par_result['latency_s']:.2f}s\n"
    f"Sequential latency: {seq_latency:.2f}s\n"
    f"Speedup: {seq_latency / max(par_result['latency_s'], 0.01):.2f}x\n"
    f"Routing accuracy: keyword {keyword_accuracy:.0%}, LLM {llm_accuracy:.0%}\n"
    + "\n".join(
        f"{r['expected']:10s} kw={r['keyword_choice']:10s} llm={r['llm_choice']:10s} "
        f"{r['query']}"
        for r in route_log.iter_rows(named=True)
    )
    + "\n"
)
print(f"\nTrace written to: {trace_path}")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Parallel vs sequential latency + routing distribution
# ════════════════════════════════════════════════════════════════════════
# Two panels: (1) measured parallel vs sequential latency; (2) measured
# routing accuracy of the keyword router vs the LLM router, split into
# the canonical queries and the paraphrases.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: latency comparison
bars = ax1.bar(
    ["Parallel\n(~max of stages)", "Sequential\n(~sum of stages)"],
    [par_result["latency_s"], seq_latency],
    color=["#2ecc71", "#e74c3c"],
    width=0.5,
)
speedup = seq_latency / max(par_result["latency_s"], 0.01)
ax1.set_ylabel("Latency (seconds)")
ax1.set_title("Parallel vs Sequential Latency", fontweight="bold")
for bar, val in zip(bars, [par_result["latency_s"], seq_latency]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.1,
        f"{val:.1f}s",
        ha="center",
        fontsize=11,
        fontweight="bold",
    )
ax1.text(
    0.5,
    max(par_result["latency_s"], seq_latency) * 0.5,
    f"{speedup:.1f}x\nspeedup",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="#2c3e50",
    transform=ax1.get_xaxis_transform(),
)

# Right: routing accuracy by router and query style (measured)
acc = route_log.group_by("style").agg(
    pl.col("keyword_correct").mean().alias("keyword"),
    pl.col("llm_correct").mean().alias("llm"),
).sort("style")
styles = acc["style"].to_list()
xs = range(len(styles))
ax2.bar([i - 0.2 for i in xs], acc["keyword"].to_list(), 0.4, label="Keyword router", color="#e67e22")
ax2.bar([i + 0.2 for i in xs], acc["llm"].to_list(), 0.4, label="LLM router", color="#3498db")
ax2.set_xticks(list(xs))
ax2.set_xticklabels(styles)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Routing accuracy (measured)")
ax2.set_title("Keyword vs LLM Routing", fontweight="bold")
ax2.legend(fontsize=8)

# bbox_inches='tight' on savefig handles the canvas fit; plt.tight_layout()
# is dropped — on headless hosts its decoration fit check nags (UserWarning)
# without changing the saved figure.
fname = OUTPUT_DIR / "ex6_parallel_router_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: helpdesk triage at a Smart Nation agency
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative figures): A Singapore government agency runs a citizen helpdesk that
# handles ~4,000 tickets/day across three teams: policy/regulation,
# technical/IT, and case/eligibility. Current triage uses keyword
# routing, which mis-routes ~18% of tickets because citizens phrase
# issues in plain language ("my MyInfo doesn't load" gets routed to
# policy because of the word "info").
#
# An LLM router reads three capability cards and dispatches by intent.
# Suppose it brings mis-routing down to ~3% (an assumption — measure it
# on your own tickets, exactly as Task 5 did on six queries).
#
# PARALLEL BONUS: when a ticket genuinely spans teams (policy
# question that also has a technical sub-question), the dispatcher
# can asyncio.gather the relevant specialists and return both
# answers in ~max-of-stages latency instead of waiting for a
# manual re-route.
#
# IMPACT:
#   Baseline mis-route rate:        18% → rework rate 18%
#   LLM-routed mis-route rate:      3%
#   Tickets saved from rework:     ~600 / day
#   Avg rework handling time:       ~8 min
#   Daily labour saved:             ~80 hours
#   Fully-loaded agent rate:        S$35/hour
#   Daily savings:                  ~S$2,800
#   Annual (250 working days):      ~S$700K

print("=" * 70)
print("  SINGAPORE APPLICATION: Smart Nation Helpdesk Triage")
print("=" * 70)
print(
    """
  Volume: 4,000 tickets/day
  Keyword router mis-route rate:   18%
  LLM router mis-route rate:       3%  (assumed — measure your own)
  Rework saved:                    ~600 tickets/day × 8 min = 80 hours/day
  Fully-loaded agent rate:         S$35/hour
  Daily savings:                   ~S$2,800
  Annual savings (250 days):       ~S$700K
  Plus: parallel specialist calls for cross-team tickets
"""
)


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT
# ══════════════════════════════════════════════════════════════════
# The diagnostics here are the two measurements you produced: the
# parallel/sequential speedup (≈1x means your backend serialises
# requests) and the routing table.  A router that is right on the
# canonical queries but wrong on paraphrases is pattern-matching words,
# not intent — rewrite the capability cards before blaming the model.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] asyncio.gather for concurrent specialist execution
  [x] Measured parallel latency ≈ max(stages) vs sequential = sum
  [x] An LLM router over capability cards, every decision visible
      (and why Pipeline.router()'s silent first-agent fallback matters)
  [x] Measured why keyword routing is brittle: paraphrases miss it
  [x] Smart Nation helpdesk triage — illustrative scale + dollar impact

  KEY INSIGHT: Parallelism is a latency optimisation; routing is a
  dispatch optimisation. Use both when you have many specialists AND
  many incoming intents — the combination is how you scale a multi-
  agent system from a demo to production.

  Next: 04_mcp_server.py — exposing your specialists as tools that
  other agents can discover via the Model Context Protocol.
"""
)
