# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.1: Supervisor-Worker Multi-Agent Pattern
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build specialist agents with domain-specific Kaizen Signatures
#   - Orchestrate the supervisor-worker (fan-out / fan-in) pattern
#   - Fan-out: dispatch the same question to three independent specialists
#     concurrently with asyncio.gather
#   - Fan-in: a supervisor synthesises specialist outputs into one answer
#   - Why decomposing analysis across specialists beats one mega-prompt
#   - Audit trail: every specialist's contribution is structured and traceable
#
# PREREQUISITES: Exercise 5 (BaseAgent, Signature, ReActAgent, single-agent)
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Load SQuAD 2.0 multi-domain corpus
#   2. Instantiate three specialists and a synthesis supervisor
#   3. Build the supervisor-worker async orchestrator
#   4. Run it against a real passage and inspect the audit trail
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

import matplotlib.pyplot as plt

from shared.mlfp06._ollama_bootstrap import preflight_ollama
from shared.mlfp06.ex_6 import (
    MODEL,
    OUTPUT_DIR,
    build_specialists,
    build_synthesis,
    load_squad_corpus,
    run_checked,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Supervisor-Worker (Fan-Out / Fan-In)
# ════════════════════════════════════════════════════════════════════════
# One supervisor sits above N specialist workers. The supervisor receives
# a complex task, fans it out to specialists (each with a focused prompt
# and a structured Signature), then fans the results back in and
# synthesises them into a single decision.
#
# Non-technical analogy: a medical case conference. The GP presents a
# patient; a cardiologist, a radiologist and a pharmacologist each give
# their independent read; then the attending physician (supervisor)
# synthesises all three opinions into a treatment plan. Nobody writes
# a mega-prompt that says "be a cardiologist AND radiologist AND
# pharmacologist" — you get better judgement by keeping the specialists
# focused and letting the attending decide.
#
# WHY IT BEATS ONE MEGA-PROMPT:
#   - Each specialist has a narrow, high-signal Signature → less drift
#   - The supervisor sees STRUCTURED specialist output, not free text
#   - Audit trail: you can trace exactly which specialist said what
#   - Costs are predictable per specialist: each has its own config
#     (budget_limit_usd caps priced spend on a hosted provider; locally
#     you meter calls and tokens)
#   - Independent specialists can run CONCURRENTLY (asyncio.gather), so
#     the fan-out costs about max(specialist latencies), not their sum —
#     provided the backend serves requests in parallel (hosted APIs, or
#     Ollama with OLLAMA_NUM_PARALLEL > 1; a single-slot daemon queues
#     them).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load the SQuAD 2.0 corpus
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load SQuAD 2.0 Multi-Domain Corpus")
print("=" * 70)

# TODO: Call load_squad_corpus() from shared.mlfp06.ex_6
passages = ____
print(f"Passages: {passages.height}, unique titles: {passages['title'].n_unique()}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0, "Task 1: corpus should not be empty"
assert passages["title"].n_unique() > 10, "Corpus should span many titles"
print("✓ Checkpoint 1 passed — multi-domain corpus loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the specialists and the synthesis supervisor
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Specialist + Supervisor Agents")
print("=" * 70)

# TODO: Use build_specialists() — returns (factual, semantic, structural)
factual_agent, semantic_agent, structural_agent = ____
# TODO: Use build_synthesis() to create the supervisor
synthesis_agent = ____

specialists = [factual_agent, semantic_agent, structural_agent]
print(f"Created {len(specialists)} specialists + 1 supervisor")
for agent in specialists:
    print(f"  {agent.__class__.__name__}: {agent.description}")
print(f"  {synthesis_agent.__class__.__name__}: {synthesis_agent.description}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(specialists) == 3, "Task 2: should have 3 specialists"
assert synthesis_agent is not None, "Task 2: supervisor should exist"
print("✓ Checkpoint 2 passed — four agents wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the supervisor-worker orchestrator
# ════════════════════════════════════════════════════════════════════════


async def timed(name: str, agent, **inputs) -> tuple[str, dict, float]:
    """Run one agent (failing loudly on LLM errors) and time it."""
    t0 = time.perf_counter()
    result = await run_checked(agent, **inputs)
    return name, result, time.perf_counter() - t0


async def supervisor_worker_analysis(doc: str, question: str) -> dict:
    """Run the full fan-out / fan-in pattern for one (doc, question)."""
    t0 = time.perf_counter()

    # Fan-out: the three specialists are independent, so launch them
    # concurrently and wait for all three.
    # TODO: Launch the three specialists concurrently and await all three.
    # Hint: asyncio.gather(timed("factual", factual_agent, document=doc,
    #       question=question), ...) — one timed(...) per specialist, in the
    #       order factual, semantic, structural
    fan_out = ____
    fan_out_wall_s = time.perf_counter() - t0
    results = {name: result for name, result, _ in fan_out}
    stage_latency = {name: dt for name, _, dt in fan_out}
    factual_result = results["factual"]
    semantic_result = results["semantic"]
    structural_result = results["structural"]

    # Fan-in: supervisor synthesises the three structured outputs
    # TODO: Fan-in — await timed("synthesis", synthesis_agent, ...) passing:
    #   document=doc, question=question,
    #   factual_analysis    = claims + evidence_quality from factual_result
    #   semantic_analysis   = main_themes + implicit_info from semantic_result
    #   structural_analysis = structure_type + key_entities from structural_result
    _, synthesis_result, synthesis_s = ____

    stage_latency["synthesis"] = synthesis_s

    elapsed = time.perf_counter() - t0
    return {
        "answer": synthesis_result["unified_answer"],
        "confidence": synthesis_result["confidence"],
        "reasoning": synthesis_result["reasoning_chain"],
        "factual_claims": factual_result["factual_claims"],
        "themes": semantic_result["main_themes"],
        "entities": structural_result["key_entities"],
        "stage_latency_s": stage_latency,
        "fan_out_wall_s": fan_out_wall_s,
        "latency_s": elapsed,
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Run it and inspect the audit trail
# ════════════════════════════════════════════════════════════════════════

doc = passages["text"][0]
question = passages["question"][0]
print(f"Question: {question}")
print(f"Passage title: {passages['title'][0]}")

preflight_ollama(required_models=[MODEL])  # fails loudly if Ollama is down
# TODO: Use asyncio.run() to execute supervisor_worker_analysis(doc, question)
sv_result = ____

print(f"\nUnified answer: {sv_result['answer'][:300]}...")
print(f"Confidence: {sv_result['confidence']:.2f}")
print(f"Reasoning steps: {len(sv_result['reasoning'])}")
print(f"Latency: {sv_result['latency_s']:.1f}s total")
for name, dt in sv_result["stage_latency_s"].items():
    print(f"  {name:10s} {dt:5.1f}s")
print(
    f"  fan-out wall clock {sv_result['fan_out_wall_s']:.1f}s vs "
    f"sum of specialists "
    f"{sum(v for k, v in sv_result['stage_latency_s'].items() if k != 'synthesis'):.1f}s"
)
print("\n--- Audit trail (who said what) ---")
print(f"  Factual claims (top 3): {sv_result['factual_claims'][:3]}")
print(f"  Semantic themes (top 3): {sv_result['themes'][:3]}")
print(f"  Structural entities (top 3): {sv_result['entities'][:3]}")

trace_path = OUTPUT_DIR / "ex6_supervisor_worker_trace.txt"
trace_path.write_text(
    f"Question: {question}\n\n"
    f"Unified Answer:\n{sv_result['answer']}\n\n"
    f"Confidence: {sv_result['confidence']:.2f}\n"
    f"Factual Claims: {sv_result['factual_claims']}\n"
    f"Semantic Themes: {sv_result['themes']}\n"
    f"Structural Entities: {sv_result['entities']}\n"
)
print(f"\nTrace written to: {trace_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert sv_result["answer"], "Task 3: should produce a unified answer"
assert 0 <= sv_result["confidence"] <= 1, "Confidence should be in [0, 1]"
assert sv_result["factual_claims"], "Factual specialist should contribute claims"
print("\n✓ Checkpoint 3 passed — supervisor-worker pattern complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Agent contribution and latency breakdown
# ════════════════════════════════════════════════════════════════════════
# Visual proof that the supervisor-worker pattern distributes work across
# specialists. The bar chart shows each specialist's contribution count
# and the overall latency, giving students a concrete sense of the
# fan-out / fan-in trade-off.

agents = ["Factual", "Semantic", "Structural", "Supervisor"]
contributions = [
    len(sv_result["factual_claims"]),
    len(sv_result["themes"]),
    len(sv_result["entities"]),
    len(sv_result["reasoning"]),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

# Left: agent contribution counts
colors = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
ax1.bar(agents, contributions, color=colors)
ax1.set_ylabel("Items contributed")
ax1.set_title("Specialist Contributions (Fan-Out)", fontweight="bold")
for i, c in enumerate(contributions):
    ax1.text(i, c + 0.1, str(c), ha="center", fontsize=10)

# Right: measured latency per stage vs the concurrent fan-out wall clock
stage_names = list(sv_result["stage_latency_s"].keys()) + ["fan-out\nwall clock"]
stage_secs = list(sv_result["stage_latency_s"].values()) + [sv_result["fan_out_wall_s"]]
ax2.barh(stage_names, stage_secs, color=colors + ["#34495e"], height=0.5)
ax2.set_xlabel("Seconds (measured)")
ax2.set_title("Per-Stage Latency vs Fan-Out Wall Clock", fontweight="bold")
for i, sec in enumerate(stage_secs):
    ax2.text(sec + 0.05, i, f"{sec:.1f}s", va="center", fontsize=9)

plt.tight_layout()
fname = OUTPUT_DIR / "ex6_supervisor_worker_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: insurance claims triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative figures): A Singapore general insurer handles ~8,000 personal-injury
# claims a month. Each claim has a narrative report (doctor notes,
# police statement, claimant description). A single-agent triage bot
# currently reads every claim and flags suspicious ones, but misses
# ~12% of fraud cases — the agent skims narratives and locks onto
# surface keywords like "accident" without auditing the evidence.
#
# The supervisor-worker pattern fixes this:
#   Factual specialist: "What events, dates, amounts are claimed?"
#   Semantic specialist: "What is implied but not stated?"
#   Structural specialist: "Which parties are involved and how?"
#   Supervisor: merges the three into a triage recommendation
#
# EXPECTED IMPACT: the catch rate applies to FRAUDULENT claims, not to
# all 8,000.  If ~5% of claims are fraudulent (400/month), raising the
# catch rate from 88% to 95% catches 7% x 400 = 28 more fraud cases a
# month.  At an average fraud claim of S$8,000 that is ~S$224,000/month
# in loss prevention — far more than the ~4x LLM calls vs the
# single-agent baseline.
#
# AUDIT BONUS: when a supervisor or regulator asks "why was this claim
# flagged?", the answer is a structured per-specialist trace, not a
# black-box decision.

print("=" * 70)
print("  SINGAPORE APPLICATION: Insurance Claims Triage")
print("=" * 70)
print(
    """
  Scale: 8,000 personal-injury claims/month (illustrative)
  Assumed fraudulent share:                ~5% -> 400 cases/month
  Baseline single-agent fraud catch rate: 88%
  Supervisor-worker target:                95%
  Net additional fraud caught:             7% x 400 = ~28 cases/month
  Average fraud claim size:                S$8,000
  Monthly loss-prevention delta:           ~S$224,000
  LLM calls vs single-agent:               4 per claim instead of 1
  Audit trail:                             structured per-specialist outputs
"""
)


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT
# ══════════════════════════════════════════════════════════════════
# The Agent Trace lens captures tool-using Delegates; these specialists
# are single structured calls, so the diagnostic is what you printed:
# per-specialist contributions (the audit trail) and the measured
# per-stage latency.  Watch for (a) a specialist contributing 0 items —
# it is not pulling its weight; (b) a fan-out wall clock close to the
# SUM of the specialists — your backend is serving them one at a time.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Specialist agents with domain-specific Signatures (Factual,
      Semantic, Structural)
  [x] Supervisor-worker pattern: fan-out to specialists, fan-in to
      the synthesis supervisor
  [x] Structured audit trail: which specialist contributed which claim
  [x] Why a focused Signature beats a mega-prompt for quality AND cost
  [x] The Singapore insurance triage scenario — quantified impact of
      multi-agent decomposition on a regulated workflow

  KEY INSIGHT: Decomposition is not about parallelism — it is about
  keeping each LLM call focused enough to produce high-signal structured
  output. The supervisor is the audit boundary; each specialist is a
  single-purpose expert whose output can be traced, tested, and priced.

  Next: 02_sequential_pipeline.py — when specialists must build on
  each other rather than run independently.
"""
)
