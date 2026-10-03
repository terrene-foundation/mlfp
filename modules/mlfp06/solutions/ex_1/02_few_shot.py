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
# A few-shot prompt places examples INSIDE the prompt, before the real
# question. The LLM reads the examples the same way it reads any other
# context: as a pattern to continue. Because autoregressive LLMs are
# trained to predict the next token given everything before it, they
# effectively "learn" the task from the in-context examples without any
# weight updates (this is called in-context learning).
#
# Examples do three things at once:
#   1. DEMONSTRATE the output format — the LLM mimics "Sentiment: positive"
#      rather than "I think it's positive because..."
#   2. NARROW the decision boundary — ambiguous cases pattern-match against
#      the closest example
#   3. ANCHOR the category names — prevents the LLM from inventing categories
#      (e.g. "mixed", "neutral", "tentatively positive")
#
# SELECTION TIPS:
#   - 3-8 examples is the usual sweet spot; more is diminishing returns
#   - DIVERSE examples (don't use 4 similar ones)
#   - BALANCED classes (2 pos + 2 neg for binary)
#   - REPRESENTATIVE of the distribution you'll see at test time
#   - ORDER MATTERS but there's no universal best order — experiment


FEW_SHOT_EXAMPLES = [
    {
        "text": "an absolute masterpiece of storytelling and visual style.",
        "category": "positive",
    },
    {
        "text": "a tedious and predictable mess from start to finish.",
        "category": "negative",
    },
    {
        "text": "delightfully clever, with performances that elevate every scene.",
        "category": "positive",
    },
    {
        "text": "fails to land a single emotional beat in over two hours.",
        "category": "negative",
    },
]


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the few-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def few_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with 4 curated examples. Returns (label, tokens, elapsed)."""
    examples_text = "\n".join(
        f'Review: "{ex["text"]}"\nSentiment: {ex["category"]}\n'
        for ex in FEW_SHOT_EXAMPLES
    )
    prompt = f"""Classify movie review snippets by sentiment.

{examples_text}
Now classify:
Review: "{text[:800]}"
Sentiment:"""

    response, tokens, elapsed = await run_delegate(prompt)
    return normalise_label(response), tokens, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, tokens, elapsed = await few_shot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "tokens": tokens,
                "elapsed": elapsed,
            }
        )
        if i < 5:
            mark = "[ok]" if correct else "[miss]"
            print(f"  Doc {i+1}: pred={pred}, true={true_label} {mark}")
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
