# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.1: Zero-Shot Classification with Kaizen Delegate
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Call an LLM with zero examples using Kaizen Delegate
#   - Write a minimal classification prompt (task + categories + input)
#   - Normalise free-form LLM text into a discrete label
#   - Measure accuracy, cost, and latency across a sample
#
# PREREQUISITES: M5 (transformers, attention). Understanding that LLMs
# predict the next token — prompts shift which tokens become likely.
#
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — why zero-shot works
#   2. Build — write the zero-shot prompt
#   3. Train — there is no training; we EVALUATE on SST-2 eval docs
#   4. Visualise — per-doc predictions + headline metrics
#   5. Apply — Singapore DBS multilingual review triage
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
# THEORY — Why Zero-Shot Works
# ════════════════════════════════════════════════════════════════════════
# A large LLM has been pre-trained on trillions of tokens. Every sentiment
# word ("wonderful", "tedious", "masterpiece") already has a representation
# in the model. Zero-shot exploits that prior — no examples, no fine-tuning.
#
# Cost/quality trade-off: cheapest, fastest, least consistent. Use it as
# your baseline before climbing the prompting ladder.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the zero-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with zero examples. Returns (label, cost, elapsed)."""
    # TODO: Build a prompt that (a) names the task, (b) lists the categories
    # from CATEGORIES, (c) includes the review text, (d) asks for ONLY the
    # category name. Truncate text to 800 chars.
    prompt = ____

    # TODO: Call run_delegate(prompt) and unpack (response, cost, elapsed)
    response, cost, elapsed = ____

    # TODO: Normalise the free-form response into a discrete label via
    # normalise_label(). Return (label, cost, elapsed).
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate across SST-2 eval docs)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    # TODO: Iterate over zip(docs["text"].to_list(), docs["label"].to_list()),
    # call zero_shot_classify for each, and append a dict with keys:
    # text, pred, true, correct, cost, elapsed. Print the first 5.
    ____
    return results


print("\n" + "=" * 70)
print("  Zero-Shot Classification on SST-2")
print("=" * 70)
zero_shot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(zero_shot_results) > 0, "Task 3: zero-shot should produce results"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in zero_shot_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — zero-shot evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE — headline metrics
# ════════════════════════════════════════════════════════════════════════
print_summary(zero_shot_results, "Zero-Shot")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: DBS Bank Multilingual Review Triage
# ════════════════════════════════════════════════════════════════════════
# DBS Bank receives ~40K app-store reviews/month across English, Mandarin,
# Malay, Tamil. Zero-shot is the right tool: no labelled data exists for
# Malay/Tamil, the LLM already knows sentiment, and cost matters at scale.
#
# BUSINESS IMPACT: Each viral complaint prevented is worth ~S$8K. Catching
# 20 extra negatives/month = S$160K/mo, vs S$120/mo in LLM cost. 1,300x ROI.
#
# LIMITATIONS: Sarcasm and mixed reviews are hard — those need CoT (Ex 1.3).


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Invoked an LLM via Kaizen Delegate with a cost budget
  [x] Wrote a minimal zero-shot classification prompt
  [x] Normalised free-form LLM output into discrete labels
  [x] Measured accuracy, cost, and latency on a real SST-2 sample
  [x] Identified a production scenario where zero-shot is optimal

  Next: 02_few_shot.py — add a handful of examples and watch accuracy improve.
"""
)
