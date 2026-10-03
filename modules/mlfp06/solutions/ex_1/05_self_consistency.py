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
#   - See the linear token scaling (N samples = N x tokens)
#
# PREREQUISITES: 03_chain_of_thought.py (reuses the CoT classifier)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — why independent samples help
#   2. Build — the sampling loop + vote aggregator
#   3. Train — evaluate on a small subset (N x the tokens per document)
#   4. Visualise — vote distributions + majority outcomes
#   5. Apply — privilege screening of discovery documents at a law firm
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter

from dotenv import load_dotenv

from shared.mlfp06.diagnostics import LLMObservatory
from shared.mlfp06.ex_1 import (
    CATEGORIES,
    compute_metrics,
    get_eval_docs,
    load_technique_metrics,
    normalise_label,
    plot_tokens_vs_accuracy,
    plot_vote_agreement,
    print_summary,
    run_delegate,
    save_technique_metrics,
)

load_dotenv()

N_SAMPLES = 3  # independent CoT paths per query; production uses 5-9
N_DOCS = 10  # subset — self-consistency spends N_SAMPLES x the tokens per doc


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Independent Samples Help
# ════════════════════════════════════════════════════════════════════════
# LLM generation is stochastic at nonzero temperature — sampling the
# same prompt twice produces different reasoning traces. Self-consistency
# (Wang et al. 2023) exploits that: run the CoT prompt N times, collect
# N answers, return the majority vote.
#
# Intuition: if the reasoning is correct, most paths converge on the
# same answer. If the reasoning is noisy, the votes spread across
# categories and the majority still lands on the most-likely-correct
# label. This is the LLM equivalent of an ensemble model.
#
# WHEN IT HELPS:
#   - Ambiguous inputs where a single CoT gets "talked into" a wrong
#     answer by a single bad reasoning step
#   - Tasks with skewed error distribution (most paths are right; a
#     few confidently wrong paths would sink a single-sample baseline)
#
# COST: N times everything — N times input tokens, N times output
# tokens, N times latency (unless you parallelise the calls). Beyond
# N=5 returns diminish for binary classification; N=7-9 for multi-class.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD
# ════════════════════════════════════════════════════════════════════════


async def cot_once(text: str) -> tuple[str, str, float, float]:
    """One CoT sample -> (label, raw_response, tokens, elapsed).

    (We inline the prompt so this file is independently runnable without
    importing from 03_chain_of_thought.py.)"""
    prompt = f"""Classify the sentiment of this movie review as positive or negative.

Think step by step about the opinion words, tone, and any sarcasm.
End with your final classification as exactly "positive" or "negative".

Review: "{text[:800]}"

Step-by-step reasoning:"""
    response, tokens, elapsed = await run_delegate(prompt)
    return normalise_label(response), response, tokens, elapsed


async def self_consistency_classify(
    text: str,
) -> tuple[str, list[str], list[str], float, float]:
    """Sample N_SAMPLES CoT paths in parallel, return majority vote.

    Returns (majority_label, votes, raw_responses, total_tokens, max_elapsed).
    """
    tasks = [cot_once(text) for _ in range(N_SAMPLES)]
    results = await asyncio.gather(*tasks)
    votes = [r[0] for r in results]
    responses = [r[1] for r in results]
    total_tokens = sum(r[2] for r in results)
    # Parallel max latency, not sum — gather runs concurrently
    max_elapsed = max(r[3] for r in results)
    majority = Counter(votes).most_common(1)[0][0]
    return majority, votes, responses, total_tokens, max_elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate on a small subset — N x the tokens per document)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs().head(N_DOCS)
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, votes, responses, tokens, elapsed = await self_consistency_classify(
            text
        )
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "tokens": tokens,
                "elapsed": elapsed,
                "votes": votes,
                "responses": responses,
            }
        )
        if i < 5:
            mark = "[ok]" if correct else "[miss]"
            print(
                f"  Doc {i+1}: votes={votes} -> majority={pred}, "
                f"true={true_label} {mark}"
            )
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

# Distribution of vote agreement — how often did all N paths agree?
unanimous = sum(1 for r in sc_results if len(set(r["votes"])) == 1)
split = len(sc_results) - unanimous
print(f"\n  Vote agreement: {unanimous}/{len(sc_results)} unanimous, {split} split")
print(
    "  Unanimous = high-confidence prediction; split = hard case where "
    "the majority vote saved us from a bad single-sample answer."
)

# R9A: visual proof — vote agreement histogram + tokens-vs-accuracy scatter
plot_vote_agreement(
    sc_results,
    N_SAMPLES,
    title=f"Self-Consistency Vote Agreement (N={N_SAMPLES})",
    filename="ex1_05_vote_agreement.png",
)

# Tokens vs accuracy across the ladder. Self-consistency only ran on the
# first N_DOCS documents, so the earlier rungs (from YOUR saved runs of
# 01-04) are re-scored on those SAME documents for a fair comparison.
sc_metrics = compute_metrics(sc_results, f"SC (N={N_SAMPLES})")
save_technique_metrics(sc_metrics)
ladder = load_technique_metrics(
    ["Zero-Shot", "Few-Shot", "CoT", "ZS-CoT"], n_docs=N_DOCS
) + [sc_metrics]
plot_tokens_vs_accuracy(
    ladder,
    title=f"Tokens vs Accuracy — Self-Consistency vs Single-Path (first {N_DOCS} docs)",
    filename="ex1_05_tokens_vs_accuracy.png",
)

# Output lens of the LLM Observatory: how much do the N raw reasoning
# paths for one document agree with each other? This runs locally on the
# responses you already collected — no extra LLM calls.
obs = LLMObservatory(run_id="ex_1_5_self_consistency")
hardest = min(sc_results, key=lambda r: max(r["votes"].count(v) for v in r["votes"]))
consistency_df = obs.output.self_consistency(
    hardest["responses"], prompt=hardest["text"][:120]
)
print("\n  Observatory — path agreement on the least-unanimous document:")
print(consistency_df)

# INTERPRETATION: The vote-agreement histogram is the confidence signal.
# Unanimous votes (1 distinct label) are high confidence. Split votes
# (2+ labels) flag hard cases. In production, route split-vote items
# to a human reviewer — the model is telling you it's unsure.
# The Observatory table scores each reasoning path's textual agreement with
# the other paths; an "is_outlier" path is the one that argued differently.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Privilege Screening of Discovery Documents at a Law Firm
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore law firm uses an LLM to screen
# discovery documents for "potentially privileged" content. Each
# document passes through a CoT classifier. Misclassification is
# expensive on both sides:
#   - False negative: privileged material disclosed to opposing counsel
#     (malpractice exposure)
#   - False positive: non-privileged material withheld (court sanctions
#     and re-review cost)
#
# Why self-consistency fits here:
#   - The stakes are lopsided — one kind of error is far costlier
#   - Split votes are a built-in "unsure" flag: route them to a lawyer,
#     auto-process only the unanimous ones
#   - The firm's governance policy can REQUIRE multi-sample consensus
#     for any action above a risk threshold (Exercise 7 shows how PACT
#     encodes that kind of rule)
#
# BUSINESS IMPACT (illustrative figures): suppose a single CoT path is
# wrong on 3% of documents and majority-of-7 brings that to 0.5%. At
# 36,000 documents/month that is 1,080 - 180 = ~900 fewer errors. Even
# at a blended S$5,000 expected cost per error, that is ~S$4.5M/month
# of avoided exposure against 7x the single-path token bill. Your own
# split-vote rate above tells you how many documents would go to a
# human reviewer instead.
#
# Note: the extra tokens are only justified because the downside is
# severe. For the postal triage task (Ex 1.4), self-consistency would
# spend ~N x the tokens for little accuracy gain.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Sampled N independent CoT paths and aggregated with majority vote
  [x] Parallelised sampling with asyncio.gather (N x tokens, ~1x latency)
  [x] Observed vote agreement as a confidence signal
  [x] Sized the N x tokens against a high-downside legal scenario

  KEY INSIGHT: Self-consistency is the ensemble method for LLMs. Like
  every ensemble, the right question is: "does the downside of being
  wrong justify N times the cost of being more right?" For most tasks,
  no. For high-stakes tasks, absolutely.

  Next: 06_structured_output.py — ditch free-form text and get
  type-safe structured responses from the LLM.
"""
)
