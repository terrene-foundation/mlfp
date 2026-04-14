# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.4: Zero-Shot CoT ("Let's Think Step by Step")
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Trigger step-by-step reasoning WITHOUT a hand-crafted template
#   - Understand why one magic phrase replaces 4 reasoning steps
#   - Compare zero-shot CoT against full CoT and zero-shot
#   - Grasp the cost/quality ratio sweet spot
#
# PREREQUISITES: 03_chain_of_thought.py
# ESTIMATED TIME: ~25 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    get_eval_docs,
    normalise_label,
    print_summary,
    run_delegate,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Magic Phrase
# ════════════════════════════════════════════════════════════════════════
# Kojima et al. (2022) showed that appending "Let's think step by step."
# unlocks most of the CoT benefit with no hand-crafted template. The
# phrase patterns the LLM into tutorial-style reasoning from pre-training.
# Cost: cheaper than full CoT, slightly lower accuracy on tricky inputs.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_cot_classify(text: str) -> tuple[str, str, float, float]:
    """Classify by appending the Kojima trigger phrase."""
    # TODO: Build a minimal prompt that asks for positive/negative
    # classification, includes the review (truncated 800 chars), and
    # ends with "Let's think step by step."
    prompt = ____

    # TODO: run_delegate, extract reasoning, normalise, return
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    # TODO: Loop, call zero_shot_cot_classify, append results dicts
    ____
    return results


print("\n" + "=" * 70)
print("  Zero-Shot CoT — 'Let's think step by step'")
print("=" * 70)
zs_cot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(zs_cot_results) > 0, "Task 3: zero-shot CoT should produce results"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in zs_cot_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — zero-shot CoT evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(zs_cot_results, "Zero-Shot CoT")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: SingPost Delivery-Complaint Triage
# ════════════════════════════════════════════════════════════════════════
# SingPost receives ~6K delivery complaints/day. Messages need urgent/
# informational tagging for the dispatch priority queue. Zero-shot CoT
# hits the Pareto sweet spot: 80% of full-CoT accuracy at 40% of cost.
# No regulatory audit needed — unlike SGH.
#
# BUSINESS IMPACT: Each urgent msg triaged in 30min (vs 4h baseline)
# saves ~S$22 in re-delivery + appeasement. 6% accuracy lift = S$230K/year,
# vs S$18K/year in LLM cost. 13x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Triggered reasoning with one sentence instead of a 4-step template
  [x] Compared the cost/accuracy Pareto vs zero-shot and full CoT

  Next: 05_self_consistency.py — sample N paths and vote.
"""
)
