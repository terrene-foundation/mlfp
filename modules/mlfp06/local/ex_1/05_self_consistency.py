# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.5: Self-Consistency (Sample N Paths, Majority Vote)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Sample multiple INDEPENDENT CoT paths for the same input
#   - Aggregate them with majority vote
#   - Understand when variance across paths beats single-path accuracy
#   - See the linear cost scaling (N samples = N x cost)
#
# PREREQUISITES: 03_chain_of_thought.py
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    get_eval_docs,
    normalise_label,
    print_summary,
    run_delegate,
)

load_dotenv()

N_SAMPLES = 3  # independent CoT paths per query


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Independent Samples Help
# ════════════════════════════════════════════════════════════════════════
# LLM generation is stochastic. Sample N times, vote. If most paths are
# right, the majority converges. If one path goes astray, the others
# overrule it. This is the LLM equivalent of an ensemble.
# Cost: N x everything. Returns diminish beyond N=5 for binary tasks.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def cot_once(text: str) -> tuple[str, float, float]:
    """One CoT sample."""
    # TODO: Build a CoT prompt (positive/negative, think step by step).
    prompt = ____
    # TODO: run_delegate, normalise, return (label, cost, elapsed)
    ____


async def self_consistency_classify(
    text: str,
) -> tuple[str, list[str], float, float]:
    """Sample N_SAMPLES CoT paths in parallel, return majority vote.

    Returns (majority_label, votes, total_cost, max_elapsed).
    """
    # TODO: Build a list of N_SAMPLES cot_once coroutines and await them
    # in parallel with asyncio.gather. Collect votes, sum costs, take max elapsed.
    ____

    # TODO: Use collections.Counter to find the majority vote
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate on a small subset for cost reasons)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs().head(10)
    results: list[dict] = []
    # TODO: Loop, call self_consistency_classify, record dict including "votes"
    ____
    return results


print("\n" + "=" * 70)
print(f"  Self-Consistency — {N_SAMPLES} parallel CoT samples + majority vote")
print("=" * 70)
sc_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(sc_results) > 0, "Task 3: self-consistency should produce results"
assert all(
    "votes" in r and len(r["votes"]) == N_SAMPLES for r in sc_results
), f"Each result must record exactly {N_SAMPLES} votes"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in sc_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — self-consistency evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(sc_results, f"Self-Consistency (N={N_SAMPLES})")

unanimous = sum(1 for r in sc_results if len(set(r["votes"])) == 1)
split = len(sc_results) - unanimous
print(f"\n  Vote agreement: {unanimous}/{len(sc_results)} unanimous, {split} split")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: PACT Ethics Review for Legal Research
# ════════════════════════════════════════════════════════════════════════
# Big-4 law firm screens discovery documents for privileged content.
# Single-CoT error rate (~3%) is unacceptable — privileged leaks cost
# S$500K+ per incident. N=7 self-consistency drops error to <0.5%.
# PACT governance policy REQUIRES multi-sample consensus for decisions
# with >S$100K downside.
#
# BUSINESS IMPACT: 36K classifications/mo, 2.5% error reduction avoids
# S$4.5M/mo in exposure, vs S$28K/mo in LLM cost. 160x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Sampled N independent CoT paths and aggregated with majority vote
  [x] Parallelised with asyncio.gather (N x cost, 1x latency)
  [x] Sized N x cost against catastrophic-downside scenarios

  Next: 06_structured_output.py — ditch string parsing, use Kaizen Signatures.
"""
)
