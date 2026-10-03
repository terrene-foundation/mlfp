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
#   - Measure accuracy, tokens, and latency across a sample (plus an
#     illustrative hosted-price estimate from the token count)
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
#   5. Apply — multilingual app-review triage at a Singapore retail bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from shared.mlfp06.ex_1 import (
    CATEGORIES,
    REFERENCE_PRICE_USD_PER_MTOK,
    compute_metrics,
    get_eval_docs,
    normalise_label,
    plot_accuracy_bars,
    print_summary,
    reference_cost_usd,
    run_delegate,
    save_technique_metrics,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Zero-Shot Works
# ════════════════════════════════════════════════════════════════════════
# A large LLM has been pre-trained on trillions of tokens that include
# book reviews, product reviews, news commentary, and movie criticism.
# Every one of those documents contains sentiment signals ("wonderful",
# "dreadful", "tedious", "masterpiece"). The LLM has already learned the
# mapping from language -> sentiment WITHOUT ever being told that task.
#
# Zero-shot prompting exploits that: you describe the task in plain
# English, name the categories, and ask for the label. No examples, no
# fine-tuning, no training data. The LLM generalises from its pre-training
# distribution to your task in a single forward pass.
#
# TOKEN / QUALITY TRADE-OFF:
#   + cheapest (shortest prompt = fewest input tokens)
#   + fastest (single LLM call per item, no reasoning)
#   - inconsistent output format (may return "Positive" vs "positive")
#   - lowest accuracy on ambiguous or domain-shifted inputs
#   - no examples to steer the model's output style
#
# WHEN TO USE: Well-known tasks, large-capability models, low-stakes
# classification, high-volume cheap triage.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the zero-shot classifier
# ════════════════════════════════════════════════════════════════════════


async def zero_shot_classify(text: str) -> tuple[str, float, float]:
    """Classify sentiment with zero examples. Returns (label, total_tokens, elapsed_s)."""
    prompt = f"""Classify the sentiment of the following movie review snippet
into exactly one category.

Categories: {', '.join(CATEGORIES)}

Review: "{text[:800]}"

Respond with ONLY the category name, nothing else."""

    response, tokens, elapsed = await run_delegate(prompt)
    return normalise_label(response), tokens, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate across SST-2 eval docs)
# ════════════════════════════════════════════════════════════════════════
# Zero-shot has no training loop. Instead we run the classifier over a
# held-out slice of SST-2 and measure accuracy/tokens/latency directly.


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    texts = docs["text"].to_list()
    labels = docs["label"].to_list()
    for i, (text, true_label) in enumerate(zip(texts, labels)):
        pred, tokens, elapsed = await zero_shot_classify(text)
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
# TASK 4 — VISUALISE — headline metrics + per-category accuracy chart
# ════════════════════════════════════════════════════════════════════════
print_summary(zero_shot_results, "Zero-Shot")

# Persist the MEASURED metrics so 02-05 can compare against your own run
zero_shot_metrics = compute_metrics(zero_shot_results, "Zero-Shot")
save_technique_metrics(zero_shot_metrics)

# R9A: visual proof — per-category accuracy bar chart
plot_accuracy_bars(
    zero_shot_results,
    CATEGORIES,
    title="Zero-Shot Accuracy by Category (SST-2)",
    filename="ex1_01_zero_shot_accuracy.png",
)

# INTERPRETATION: Zero-shot gives you a baseline with zero engineering
# effort. If the number here is "good enough" for your use case, STOP —
# every technique below this one costs more tokens, more latency, and
# more prompt-engineering effort. Only move up the ladder when zero-shot
# fails your accuracy bar.
# The bar chart reveals whether errors are SYMMETRIC (equal miss rate on
# both categories) or SKEWED (e.g. the model defaults to "positive" on
# ambiguous inputs). Skewed errors signal that zero-shot's prior from
# pre-training is biased — few-shot examples (Exercise 1.2) can fix it.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Multilingual App-Review Triage at a Singapore Retail Bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank receives tens of
# thousands of app-store reviews per month across English, Mandarin,
# Malay, and Tamil. The CX team wants every review tagged
# positive/negative within 10 minutes of posting so complaints can be
# routed to the on-call support lead.
#
# Why zero-shot is the right tool here:
#   - The LLM already understands sentiment across all four languages
#   - No labelled training data exists for Malay/Tamil reviews
#   - Volume is high, so tokens per review drive the running cost
#   - The downstream action (route to an agent) is reversible, so
#     occasional misclassifications are recoverable
#
# The sizing below uses YOUR measured tokens-per-review, an assumed
# monthly volume, and the illustrative reference price — swap in your
# own volume and your provider's real rate when you size a deployment.
MONTHLY_REVIEWS = 40_000  # illustrative volume, not a real bank's figure
tokens_per_review = zero_shot_metrics["total_tokens"] / max(zero_shot_metrics["n"], 1)
monthly_tokens = int(tokens_per_review * MONTHLY_REVIEWS)
print("\n  Illustrative sizing for the review-triage scenario:")
print(f"    measured tokens/review : {tokens_per_review:,.0f}")
print(f"    assumed reviews/month  : {MONTHLY_REVIEWS:,}")
print(
    f"    tokens/month           : {monthly_tokens:,} "
    f"≈ ${reference_cost_usd(monthly_tokens):,.2f}/month at the "
    f"${REFERENCE_PRICE_USD_PER_MTOK:.2f}/Mtok reference price "
    f"(self-hosted Ollama: hardware cost only)"
)
print(
    f"    measured accuracy      : {zero_shot_metrics['accuracy']:.0%} on "
    f"{zero_shot_metrics['n']} SST-2 docs (1 - accuracy = misrouting rate)"
)
#
# BUSINESS IMPACT: the value side depends on what a missed complaint
# costs the bank. If, say, catching a frustrated customer before they
# escalate on social media avoids S$1,000 of remediation effort, then
# 20 extra complaints caught per month is worth S$20,000/month — compare
# that with the token bill printed above. The decision rule
# is the ratio, not the absolute numbers: zero-shot wins whenever its
# accuracy clears the bar at a fraction of the cost of the higher rungs.
#
# LIMITATIONS:
#   - Sarcasm is hard ("wow, another outage, just what I needed")
#   - Mixed reviews (4-star with a complaint) may be misrouted
#   - For these edge cases, Exercise 1.3 (chain-of-thought) does better


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Invoked a local Ollama LLM via Kaizen Delegate (make_delegate)
  [x] Wrote a minimal zero-shot classification prompt
  [x] Normalised free-form LLM output into discrete labels
  [x] Measured accuracy, tokens, and latency on a real SST-2 sample
  [x] Sized a production scenario (bank app-review triage) where
      zero-shot is the economically optimal choice

  KEY INSIGHT: Zero-shot is the first rung of the prompting ladder.
  Every subsequent technique spends more tokens per call — only climb higher
  when the business outcome needs the accuracy.

  Next: 02_few_shot.py — add a handful of examples and watch accuracy
  improve without changing the model or the task.
"""
)
