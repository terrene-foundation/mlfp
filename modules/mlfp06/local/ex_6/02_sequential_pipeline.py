# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6.2: Sequential Multi-Agent Pipeline (A → B → C)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build a 3-stage sequential agent pipeline
#   - When sequential beats fan-out: downstream needs upstream output
#   - Latency model: sequential = sum of stage latencies
#
# PREREQUISITES: 01_supervisor_worker.py
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Load the corpus and wire a fresh specialist set
#   2. Instantiate the stage-2 InterpretationAgent
#   3. Build the 3-stage pipeline orchestrator
#   4. Run it and reason about sequential latency
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import time

from shared.mlfp06.ex_6 import (
    InterpretationAgent,
    OUTPUT_DIR,
    build_specialists,
    build_synthesis,
    load_squad_corpus,
)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load corpus + specialists
# ════════════════════════════════════════════════════════════════════════

# TODO: Load the SQuAD corpus
passages = ____

# TODO: Build specialists — we only need factual_agent here; use _ for the others
factual_agent, _semantic_agent, _structural_agent = ____

# TODO: Build synthesis supervisor
synthesis_agent = ____

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert passages.height > 0, "Task 1: corpus should be non-empty"
assert factual_agent is not None, "Task 1: factual agent should exist"
print("✓ Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Instantiate the stage-2 interpreter
# ════════════════════════════════════════════════════════════════════════

# TODO: Create an InterpretationAgent instance
interpreter = ____

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert interpreter is not None, "Task 2: interpreter should exist"
print("✓ Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the 3-stage sequential pipeline
# ════════════════════════════════════════════════════════════════════════


async def sequential_pipeline(doc: str, question: str) -> dict:
    """Run 3-stage sequential pipeline: extract → interpret → synthesise."""
    t0 = time.perf_counter()
    per_stage: list[tuple[str, float]] = []

    # TODO: Stage 1 — await factual_agent.run_async(document=doc, question=question)
    # Note: BaseAgent.run_async returns a dict; read outputs via factual["factual_claims"].
    s1_t = time.perf_counter()
    factual = ____
    per_stage.append(("factual extraction", time.perf_counter() - s1_t))

    # TODO: Stage 2 — await interpreter.run_async(...) with
    #       factual_claims=str(factual["factual_claims"]),
    #       document=doc, question=question. Stage 2 consumes stage 1's dict output.
    s2_t = time.perf_counter()
    interpreted = ____
    per_stage.append(("contextual interpretation", time.perf_counter() - s2_t))

    # TODO: Stage 3 — await synthesis_agent.run_async(...) passing the interpreted
    #       output as factual_analysis/semantic_analysis/structural_analysis.
    #       Read interpreted fields via dict indexing — interpreted["interpreted_facts"].
    s3_t = time.perf_counter()
    final = ____
    per_stage.append(("synthesis", time.perf_counter() - s3_t))

    elapsed = time.perf_counter() - t0
    return {
        "answer": final["unified_answer"],
        "confidence": final["confidence"],
        "stages": per_stage,
        "latency_s": elapsed,
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Run and inspect the latency breakdown
# ════════════════════════════════════════════════════════════════════════

doc = passages["text"][0]
question = passages["question"][0]

# TODO: asyncio.run(sequential_pipeline(doc, question))
result = ____

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert result["answer"], "Task 3: should produce a final answer"
assert len(result["stages"]) == 3, "Should log 3 stages"
print("✓ Checkpoint 3 passed — sequential pipeline complete\n")


# ════════════════════════════════════════════════════════════════════════
# APPLY — Singapore scenario: REIT peer disclosure benchmarking
# ════════════════════════════════════════════════════════════════════════
# A Singapore-listed REIT's compliance team reviews 40 peer filings
# per quarter (~60 human-hours). A 3-stage sequential pipeline
# (extract numbers → rank relevance → draft benchmark) reduces
# this to ~2 hours of approval, saving ~S$7K/quarter per analyst
# with an SGX-defensible structured trace.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Sequential pipeline (A → B → C)
  [x] Latency = sum of stages (vs parallel = max)
  [x] Stage 2 consumes stage 1's structured output

  Next: 03_parallel_router.py — parallel execution and LLM routing.
"""
)
