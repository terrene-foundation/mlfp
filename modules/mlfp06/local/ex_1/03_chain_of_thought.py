# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.3: Chain-of-Thought (CoT) Prompting
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Force step-by-step reasoning with an explicit CoT template
#   - Extract the final label from a multi-line reasoning trace
#   - See why CoT improves accuracy on ambiguous cases
#   - Understand the latency/cost penalty of reasoning tokens
#
# PREREQUISITES: 01_zero_shot.py, 02_few_shot.py
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why "thinking out loud" helps
#   2. Build — explicit 4-step reasoning template
#   3. Train — evaluate and preserve reasoning traces
#   4. Visualise — compare accuracy vs zero-shot/few-shot
#   5. Apply — SGH clinical triage notes
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
# THEORY — Why CoT Helps
# ════════════════════════════════════════════════════════════════════════
# Forcing the model to write reasoning tokens BEFORE committing to a
# label makes each reasoning step a commitment the answer must be
# consistent with. The output is AUDITABLE — critical for regulated
# industries. Cost: 5-10x more output tokens, 3-5x more latency.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the CoT classifier
# ════════════════════════════════════════════════════════════════════════


async def cot_classify(text: str) -> tuple[str, str, float, float]:
    """Classify with an explicit 4-step reasoning template.

    Returns (label, reasoning_trace, cost_usd, elapsed_s).
    """
    # TODO: Build a prompt that asks the model to (1) identify opinion words,
    # (2) assess tone, (3) consider sarcasm, (4) state final classification
    # as "positive" or "negative". End with "Step-by-step reasoning:".
    prompt = ____

    # TODO: run_delegate(prompt)
    response, cost, elapsed = ____

    # TODO: The reasoning is the stripped response. normalise_label extracts
    # the final label from the last line. Return (label, reasoning, cost, elapsed).
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    # TODO: Loop over docs, call cot_classify, record a dict with keys:
    # text, pred, true, correct, cost, elapsed, reasoning. Print reasoning
    # excerpt (first 180 chars) for the first 3 docs.
    ____
    return results


print("\n" + "=" * 70)
print("  Chain-of-Thought Classification on SST-2")
print("=" * 70)
cot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(cot_results) > 0, "Task 3: CoT should produce results"
assert all(
    "reasoning" in r and len(r["reasoning"]) > 0 for r in cot_results
), "Each CoT result should preserve a non-empty reasoning trace"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in cot_results
), "Predictions must normalise to a known category"
print("\n[ok] Checkpoint passed — CoT evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(cot_results, "Chain-of-Thought")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore General Hospital Clinical Triage
# ════════════════════════════════════════════════════════════════════════
# SGH pilot: LLM classifies triage nurse dictations as "ambulatory" or
# "resuscitation". CoT is mandatory because every decision must be
# AUDITABLE for quarterly clinical governance review. The reasoning
# trace is the artefact that survives the audit — and is also used to
# train junior nurses by comparing LLM reasoning against human decisions.
#
# BUSINESS IMPACT: Moving from 85% to 92% accuracy on 200 intakes/day
# avoids S$4.6M/year in incident costs, vs S$110K/year in LLM cost. 42x ROI.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a chain-of-thought prompt with an explicit 4-step template
  [x] Preserved full reasoning traces for downstream audit
  [x] Recognised CoT's role in regulated, auditability-critical settings

  Next: 04_zero_shot_cot.py — one magic phrase replaces the template.
"""
)
