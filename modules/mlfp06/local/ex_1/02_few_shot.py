# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.2: Few-Shot Prompting with Curated Examples
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Construct a few-shot prompt with 3-8 curated examples
#   - Understand why examples steer output format AND decision boundary
#   - Trade longer prompts for better consistency and accuracy
#   - Compare few-shot against the zero-shot baseline
#
# PREREQUISITES: 01_zero_shot.py (baseline metrics for comparison)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — what examples do to LLM behaviour
#   2. Build — prompt with curated positive/negative exemplars
#   3. Train — evaluate on SST-2 eval docs
#   4. Visualise — few-shot metrics vs your measured zero-shot run
#   5. Apply — supervisory incident-report triage at a financial regulator
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
    print_summary,
    run_delegate,
    save_technique_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — What Examples Do
# ════════════════════════════════════════════════════════════════════════
# In-context learning: the LLM reads examples in the prompt and patterns
# its response on them. Examples (a) demonstrate output format, (b) narrow
# the decision boundary, (c) anchor category names. 3-8 examples is the
# sweet spot; diverse, balanced, representative.


# TODO: Define FEW_SHOT_EXAMPLES as a list of 4 dicts with keys "text"
# and "category". Use 2 positive + 2 negative, all distinct styles.
FEW_SHOT_EXAMPLES = ____


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the few-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def few_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with 4 curated examples. Returns (label, tokens, elapsed)."""
    # TODO: Build an examples_text string by joining each example as:
    #   Review: "<text>"
    #   Sentiment: <category>
    examples_text = ____

    # TODO: Build the full prompt: intro line, examples_text, "Now classify:",
    # the new review (truncated 800 chars), and "Sentiment:" trailing.
    prompt = ____

    # TODO: run_delegate, normalise, return
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    # TODO: Loop over docs, call few_shot_classify, build the results list
    # with the same keys as 01_zero_shot.py. Print the first 5.
    ____
    return results


print("\n" + "=" * 70)
print("  Few-Shot Classification on SST-2 (4 exemplars)")
print("=" * 70)
few_shot_results = asyncio.run(evaluate())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(few_shot_results) > 0, "Task 3: few-shot should produce results"
assert all(
    r["pred"] in CATEGORIES or r["pred"] == "unknown" for r in few_shot_results
), "Predictions must be in CATEGORIES or 'unknown'"
print("\n[ok] Checkpoint passed — few-shot evaluation complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE
# ════════════════════════════════════════════════════════════════════════
print_summary(few_shot_results, "Few-Shot (4 examples)")

# R9A: visual proof — few-shot vs zero-shot accuracy comparison.
# The zero-shot bar comes from YOUR run of 01_zero_shot.py (saved to
# outputs/ex1_prompting/technique_metrics.json). If you have not run it
# yet, the chart shows only the few-shot bar — no invented baseline.
few_shot_metrics = compute_metrics(few_shot_results, "Few-Shot")
save_technique_metrics(few_shot_metrics)
plot_comparison_bars(
    load_technique_metrics(["Zero-Shot"]) + [few_shot_metrics],
    title="Few-Shot vs Zero-Shot — Accuracy / Tokens / Latency",
    filename="ex1_02_few_shot_comparison.png",
)

# INTERPRETATION: Few-shot usually buys a few percentage points over
# zero-shot in exchange for several times the input tokens (the four
# examples are re-sent on every call). Compare the two bars from YOUR
# run: did accuracy rise, and by how much did tokens grow? On a 20-doc
# sample one doc is 5 points, so small differences are within noise.
# The bar chart makes the trade-off visible: how many extra tokens does
# each percentage point of accuracy cost you?


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Supervisory Incident-Report Triage at a Financial Regulator
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a financial regulator receives several hundred
# incident reports per week from the institutions it supervises. Each
# report needs tagging as "material" or "routine" so senior examiners
# look at the material ones first.
#
# Zero-shot struggles because:
#   - The domain is specialised (banking operational-risk vocabulary)
#   - "Material" is defined by the regulator's own supervisory policy,
#     not by a textbook — the model's pre-training prior is the wrong one
#   - Examiner time is expensive; misclassification costs senior time
#
# Few-shot fits because:
#   - The supervision team can supply 6 examples from its historical log
#     that encode ITS definition of "material"
#   - The LLM mimics those examples instead of relying on its prior
#   - The extra cost is just the examples' tokens on every call —
#     measured below from your two runs
zero_shot_saved = load_technique_metrics(["Zero-Shot"])
if zero_shot_saved:
    extra_per_call = (
        few_shot_metrics["total_tokens"] / max(few_shot_metrics["n"], 1)
        - zero_shot_saved[0]["total_tokens"] / max(zero_shot_saved[0]["n"], 1)
    )
    print(f"\n  Extra tokens per call paid for the examples: {extra_per_call:,.0f}")
#
# BUSINESS IMPACT (illustrative figures): suppose a senior examiner's
# time costs S$250/hour and each mis-routed report wastes 30 minutes
# (S$125). At 800 reports/week, a 5-point routing improvement is
# 40 reports/week = S$5,000/week ≈ S$260,000/year of reclaimed
# capacity. Price the extra example tokens with the reference price
# from 01 and compare — the examples are almost always the cheaper side.
#
# OPERATIONAL NOTE: Store the examples in a version-controlled repo
# (not in the Python file). When the supervisory definition evolves,
# the compliance team updates the examples without touching code.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built a few-shot prompt with curated positive/negative examples
  [x] Understood in-context learning — LLMs learn patterns from the prompt
  [x] Traded longer prompts for better accuracy and output consistency
  [x] Sized the approach against a supervisory-report triage use case
  [x] Compared the extra example tokens against the accuracy they buy

  KEY INSIGHT: Examples are the cheapest form of "training" an LLM.
  You don't update weights — you update the prompt. When the
  definition changes, you edit the examples, not retrain the model.

  Next: 03_chain_of_thought.py — make the model show its reasoning
  before answering, and watch accuracy climb on ambiguous inputs.
"""
)
