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
# PREREQUISITES: 03_chain_of_thought.py (for the full-CoT comparison)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Theory — Kojima et al. 2022 and why the trigger phrase works
#   2. Build — the tiny prompt change
#   3. Train — evaluate on SST-2
#   4. Visualise — compare tokens/accuracy vs full CoT
#   5. Apply — delivery-complaint triage at a national postal operator
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    compute_metrics,
    get_eval_docs,
    load_technique_metrics,
    normalise_label,
    plot_comparison_bars,
    plot_tokens_vs_accuracy,
    print_summary,
    run_delegate,
    save_technique_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Magic Phrase
# ════════════════════════════════════════════════════════════════════════
# Kojima et al. (2022, "Large Language Models are Zero-Shot Reasoners")
# discovered that simply appending "Let's think step by step." to a
# prompt unlocks much of the CoT benefit without any hand-crafted
# reasoning template. With text-davinci-002 it lifted MultiArith from
# 17.7% to 78.7% and GSM8K from 10.4% to 40.7% — five words, no examples.
#
# Why it works: the training distribution contains millions of StackOverflow,
# textbook, and tutoring exchanges where "let's think step by step" precedes
# a careful, sequential explanation. The phrase is a PROMPT SHORTCUT into
# that style of generation. No examples, no template, just one sentence.
#
# TRADE-OFFS VS FULL COT:
#   + cheaper (no 4-step template to pay input tokens for)
#   + faster (slightly shorter output, same order of magnitude)
#   - less consistent reasoning format (harder to parse deterministically)
#   - slightly lower accuracy than hand-crafted CoT on tricky tasks
#
# Zero-shot CoT is the "best bang for the buck" reasoning technique for
# moderate-stakes tasks that don't need a strict audit format.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_cot_classify(text: str) -> tuple[str, str, float, float]:
    """Classify by appending the Kojima trigger phrase. Returns (label, reasoning, tokens, elapsed)."""
    prompt = f"""Classify the sentiment of this movie review as "positive" or "negative".

Review: "{text[:800]}"

Let's think step by step."""

    response, tokens, elapsed = await run_delegate(prompt)
    reasoning = response.strip()
    return normalise_label(reasoning), reasoning, tokens, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, reasoning, tokens, elapsed = await zero_shot_cot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "tokens": tokens,
                "elapsed": elapsed,
                "reasoning": reasoning,
            }
        )
        if i < 3:
            mark = "[ok]" if correct else "[miss]"
            print(f"  Doc {i+1}: pred={pred}, true={true_label} {mark}")
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

# R9A: visual proof — 4-method comparison chart (the full prompting ladder).
# Zero-shot, few-shot and CoT come from YOUR saved runs of 01-03; a rung
# you have not run is reported and left out — never a made-up baseline.
zs_cot_metrics = compute_metrics(zs_cot_results, "ZS-CoT")
save_technique_metrics(zs_cot_metrics)
all_methods = load_technique_metrics(["Zero-Shot", "Few-Shot", "CoT"]) + [
    zs_cot_metrics
]

plot_comparison_bars(
    all_methods,
    title="Prompting Ladder — All 4 Methods Compared",
    filename="ex1_04_method_comparison.png",
)

plot_tokens_vs_accuracy(
    all_methods,
    title="Tokens vs Accuracy — Full Prompting Ladder",
    filename="ex1_04_tokens_vs_accuracy.png",
)

# INTERPRETATION: In the literature ZS-CoT usually lands between zero-shot
# and full CoT on accuracy, with fewer tokens than the full template.
# Check whether YOUR chart agrees — on a 20-doc SST-2 sample the gaps can
# be within noise (one doc = 5 points).
# The 4-method chart is the decision tool: pick the cheapest method (fewest
# tokens, lowest latency) that clears your accuracy bar.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Delivery-Complaint Triage at a National Postal Operator
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a national postal operator receives around
# 6,000 delivery-complaint messages per day via its mobile app. Each
# message must be tagged "urgent" (missed delivery, wrong address,
# damaged parcel) or "informational" (status query, general feedback).
# Urgent messages feed a priority queue the dispatch team works every
# 30 minutes.
#
# Why zero-shot CoT fits:
#   - At this volume latency matters — full CoT's longer outputs would
#     let messages pile up faster than the team can process them
#   - The task is moderately ambiguous — "my parcel is late" could be
#     urgent (2 days overdue) or informational (asking for an ETA)
#   - There is no regulatory audit requirement — unlike the clinical
#     triage case in Ex 1.3
#
# BUSINESS IMPACT (illustrative figures): suppose each urgent message
# triaged within 30 minutes (instead of a 4-hour baseline) saves S$22
# in re-delivery and goodwill credits. With ~8% urgent (480/day), a
# 6-point lift over zero-shot catches ~29 more urgent messages/day =
# ~S$640/day ≈ S$230K/year. Compare that with the extra tokens per
# message your ZS-CoT run used over zero-shot, priced at the reference
# rate from 01 — then decide whether the lift on YOUR chart is real.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Triggered reasoning with one sentence instead of a 4-step template
  [x] Understood why "Let's think step by step" generalises across tasks
  [x] Compared the tokens/accuracy trade-off vs zero-shot and full CoT
  [x] Applied it to a high-volume, moderately-ambiguous triage task

  KEY INSIGHT: The trigger phrase is a cultural shortcut. The LLM knows
  what "think step by step" SHOULD look like because it saw a million
  tutors write that phrase before careful explanations.

  Next: 05_self_consistency.py — when one reasoning path isn't enough,
  sample N of them and vote.
"""
)
