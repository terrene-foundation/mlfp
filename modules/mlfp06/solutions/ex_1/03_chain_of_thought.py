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
    compute_metrics,
    get_eval_docs,
    normalise_label,
    plot_comparison_bars,
    plot_cost_vs_accuracy,
    print_summary,
    run_delegate,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why CoT Helps
# ════════════════════════════════════════════════════════════════════════
# Autoregressive LLMs generate tokens one at a time, and every token is
# conditioned on ALL prior tokens in the response. When you force the
# model to write out reasoning steps BEFORE committing to an answer, each
# reasoning token acts as a commitment the final answer must be consistent
# with. The model effectively uses its own output as a scratchpad.
#
# The paper that popularised this (Wei et al. 2022, "Chain-of-Thought
# Prompting Elicits Reasoning in LLMs") showed that on arithmetic and
# commonsense benchmarks, CoT can flip a model from near-random to near-
# SOTA without any change to the weights.
#
# COT FOR CLASSIFICATION:
#   - The model identifies evidence (opinion words, tone cues)
#   - It weighs competing signals (one negative line in a positive review)
#   - It considers corner cases (sarcasm, irony)
#   - THEN it commits to a label
#
# COSTS:
#   - Output tokens: 5-10x more than zero-shot (each reasoning step costs tokens)
#   - Latency: 3-5x slower (generation is sequential)
#   - Parsing: the final label lives deep in the response; extraction is fragile


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the CoT classifier
# ════════════════════════════════════════════════════════════════════════


async def cot_classify(text: str) -> tuple[str, str, float, float]:
    """Classify with an explicit 4-step reasoning template.

    Returns (label, reasoning_trace, cost_usd, elapsed_s).
    """
    prompt = f"""Classify the sentiment of this movie review as positive or negative.

Think step by step:
1. Identify the key opinion words and their valence
2. Assess whether the overall tone is favourable or unfavourable
3. Consider any sarcasm or irony
4. State your final classification as exactly "positive" or "negative"

Review: "{text[:800]}"

Step-by-step reasoning:"""

    response, cost, elapsed = await run_delegate(prompt)
    reasoning = response.strip()
    return normalise_label(reasoning), reasoning, cost, elapsed


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (evaluate)
# ════════════════════════════════════════════════════════════════════════


async def evaluate() -> list[dict]:
    docs = get_eval_docs()
    results: list[dict] = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, reasoning, cost, elapsed = await cot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "text": text,
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "cost": cost,
                "elapsed": elapsed,
                "reasoning": reasoning,
            }
        )
        if i < 3:
            print(f"  Doc {i+1} reasoning (excerpt): {reasoning[:180]}...")
            mark = "[ok]" if correct else "[miss]"
            print(f"    final={pred}, true={true_label} {mark}")
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

# R9A: visual proof — CoT vs zero-shot/few-shot accuracy + cost-vs-accuracy scatter
# Expected baselines (R10: independently runnable, no cross-file imports)
zero_shot_expected = {
    "strategy": "Zero-Shot",
    "accuracy": 0.80,
    "total_cost": 0.002,
    "avg_latency_s": 1.0,
    "n": 20,
}
few_shot_expected = {
    "strategy": "Few-Shot",
    "accuracy": 0.85,
    "total_cost": 0.006,
    "avg_latency_s": 1.2,
    "n": 20,
}
cot_metrics = compute_metrics(cot_results, "CoT")
all_methods = [zero_shot_expected, few_shot_expected, cot_metrics]

plot_comparison_bars(
    all_methods,
    title="CoT vs Prior Methods — Accuracy / Cost / Latency",
    filename="ex1_03_cot_comparison.png",
)

plot_cost_vs_accuracy(
    all_methods,
    title="Cost vs Accuracy — Prompting Ladder So Far",
    filename="ex1_03_cost_vs_accuracy.png",
)

# INTERPRETATION: CoT is the first technique that noticeably slows things
# down. The reasoning trace is AUDITABLE — you can read WHY the model
# chose a label, which matters for regulated industries (healthcare,
# finance, legal). For simple tasks, the cost is hard to justify; for
# ambiguous/high-stakes tasks, the auditability alone is worth it.
# The scatter plot shows the cost-accuracy Pareto frontier — each method
# buys accuracy with more tokens. The slope of the line tells you the
# marginal cost of each accuracy point.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore General Hospital Clinical Triage
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: Singapore General Hospital (SGH) runs a pilot where the
# triage nurse dictates a 2-3 sentence assessment at intake. An LLM
# classifies the note as "ambulatory" (non-urgent) or "resuscitation"
# (urgent). Misclassification in either direction is costly:
#   - Under-triage (urgent -> non-urgent): life-threatening delay
#   - Over-triage (non-urgent -> urgent): wastes resus bay capacity
#
# Why CoT is mandatory here:
#   - Every decision must be AUDITABLE. A year later, when a case is
#     reviewed, the clinician must see the LLM's reasoning, not just
#     its output. Zero-shot "label only" responses fail this bar.
#   - Notes contain COMPETING signals ("alert but diaphoretic", "stable
#     vitals but known MI history"). CoT forces the model to weigh them.
#   - The reasoning trace is used to TRAIN triage nurses — they review
#     disagreements between the LLM and the nurse to refine their own
#     decision-making.
#
# BUSINESS IMPACT: SGH's resus bay capacity is ~30 patients/day. Each
# false-positive admission displaces ~1.4 true emergencies (measured
# against NHGP baseline). Each false-negative is a near-miss with a
# mean incident-reporting cost of S$8,500 (staff time + root cause
# analysis + risk committee). Moving from zero-shot (~85% acc) to CoT
# (~92% acc) on 200 intakes/day avoids ~14 false-negatives/day and
# ~9 false-positive admissions/day. Annualised avoided cost: S$4.6M.
# LLM inference cost at CoT rate: ~S$110K/year. 42x ROI.
#
# AUDIT TRAIL: Every CoT response is persisted to the SGH clinical
# governance store alongside the patient MRN. Access is RBAC-gated
# via PACT (see Exercise 6) and reviewed quarterly.


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
  [x] Parsed a discrete label out of a multi-line response
  [x] Understood the cost/latency penalty of reasoning tokens
  [x] Recognised CoT's role in regulated, auditability-critical settings

  KEY INSIGHT: CoT's biggest value in production isn't accuracy — it's
  AUDITABILITY. The reasoning trace is the artefact that survives the
  quarterly compliance review.

  Next: 04_zero_shot_cot.py — skip the hand-crafted template and use
  one magic phrase instead.
"""
)
