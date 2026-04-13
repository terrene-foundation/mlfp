# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1: LLM Fundamentals, Prompt Engineering, and
#                       Structured Output
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain how LLMs are pre-trained (next-token prediction vs masked LM)
#     and why scaling laws matter (parameters, data, compute)
#   - Apply 6 prompt engineering techniques: zero-shot, few-shot, CoT,
#     zero-shot CoT, self-consistency (majority vote), and structured
#     output specification
#   - Use Kaizen Delegate with streaming, events, and cost tracking
#   - Define typed Signatures with InputField / OutputField for structured
#     LLM output extraction
#   - Compare prompting strategies quantitatively (accuracy, cost, latency)
#     and explain when each technique helps
#   - Describe inference-time optimisations: KV-cache, speculative decoding,
#     continuous batching
#
# PREREQUISITES:
#   M5 complete (transformers, attention, positional encoding from M5.4).
#   Understanding that LLMs predict the next token — prompts shift which
#   tokens are likely.  No fine-tuning required; prompting is zero-cost
#   adaptation.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. LLM foundations recap (pre-training, scaling laws)
#    2. Zero-shot classification with Delegate
#    3. Few-shot with curated example selection
#    4. Chain-of-thought (CoT) prompting
#    5. Zero-shot CoT ("Let's think step by step")
#    6. Self-consistency (sample multiple CoT paths, majority vote)
#    7. Structured prompting with explicit JSON output format
#    8. Kaizen Signature for type-safe structured extraction
#    9. Quantitative comparison across all prompting strategies
#   10. Inference optimisations (KV-cache, speculative decoding, batching)
#
# DATASET: SST-2 Sentiment (stanfordnlp/sst2 on HuggingFace)
#   The Stanford Sentiment Treebank — real movie review snippets labelled
#   positive (1) or negative (0).  Standard NLP benchmark for evaluating
#   transformer language models.  We use a 200-row subsample for fast
#   prompt-engineering experiments.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
if not model:
    raise EnvironmentError("Set DEFAULT_LLM_MODEL or OPENAI_PROD_MODEL in .env")
print(f"LLM Model: {model}")

# ── Data Loading (SST-2 sentiment from HuggingFace) ─────────────────────

CACHE_DIR = Path("data/mlfp06/sst2")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "sst2_200.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached SST-2 from {CACHE_FILE}")
    sst2 = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading stanfordnlp/sst2 from HuggingFace (first run)...")
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/sst2", split="train")
    ds = ds.shuffle(seed=42).select(range(min(200, len(ds))))

    label_names = {0: "negative", 1: "positive"}
    rows = [
        {
            "text": row["sentence"],
            "label": label_names[row["label"]],
            "label_id": row["label"],
        }
        for row in ds
    ]
    sst2 = pl.DataFrame(rows)
    sst2.write_parquet(CACHE_FILE)
    print(f"Cached {sst2.height} SST-2 rows to {CACHE_FILE}")

eval_docs = sst2.head(20)  # first 20 for evaluation across strategies
print(f"Loaded {sst2.height:,} sentences for classification")
print(f"Label distribution: {dict(sst2['label'].value_counts().iter_rows())}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: LLM Foundations Recap
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: LLM Foundations Recap")
print("=" * 70)

print(
    """
Pre-training objectives:
  GPT family  — next-token prediction (autoregressive, left-to-right)
                P(x_t | x_1, ..., x_{t-1})
  BERT family — masked language modelling (bidirectional)
                Randomly mask 15% of tokens, predict them from context

Scaling laws (Chinchilla, 2022):
  Performance improves predictably with three knobs:
    N — number of parameters
    D — size of training data (tokens)
    C — compute budget (FLOPs)
  Rule of thumb: N and D should scale proportionally.
  Doubling parameters without doubling data → overfitting.

Notable model families (no vendor preference — Foundation independence):
  Autoregressive:  GPT, Claude, Gemini, Llama, Phi, Mistral, Gemma
  Encoder-only:    BERT, RoBERTa, DeBERTa
  Encoder-decoder: T5, BART, Flan-T5

RLHF overview (connecting M5.8 PPO to alignment):
  1. Pre-train on next-token prediction (unsupervised)
  2. Fine-tune on instructions (SFT — Exercise 2)
  3. Train reward model on human preferences
  4. Optimise policy with PPO against reward model
  DPO (Exercise 3) bypasses step 3 entirely.
"""
)


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Zero-shot classification with Delegate
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 2: Zero-Shot Classification")
print("=" * 70)

CATEGORIES = ["positive", "negative"]


async def zero_shot_classify(text: str) -> tuple[str, float]:
    """Classify sentiment using zero-shot prompting.  Returns (label, cost)."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)

    prompt = f"""Classify the sentiment of the following movie review snippet
into exactly one category.

Categories: {', '.join(CATEGORIES)}

Review: "{text[:800]}"

Respond with ONLY the category name, nothing else."""

    response = ""
    cost = 0.0
    t0 = time.perf_counter()
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
        if hasattr(event, "cost"):
            cost = event.cost
    elapsed = time.perf_counter() - t0
    label = response.strip().lower()
    # normalise to known category
    if "positive" in label:
        label = "positive"
    elif "negative" in label:
        label = "negative"
    return label, cost, elapsed


async def run_zero_shot(docs: pl.DataFrame) -> list[dict]:
    results = []
    texts = docs["text"].to_list()
    labels = docs["label"].to_list()
    for i, (text, true_label) in enumerate(zip(texts, labels)):
        pred, cost, elapsed = await zero_shot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "cost": cost,
                "elapsed": elapsed,
            }
        )
        if i < 5:
            print(
                f"  Doc {i+1}: pred={pred}, true={true_label} {'✓' if correct else '✗'}"
            )
    acc = sum(r["correct"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / len(results)
    print(
        f"  Accuracy: {acc:.0%} | Cost: ${total_cost:.4f} | Avg latency: {avg_latency:.2f}s"
    )
    return results


print("\n=== Zero-Shot Classification ===")
zero_shot_results = asyncio.run(run_zero_shot(eval_docs))

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(zero_shot_results) > 0, "Task 2: zero-shot should produce results"
assert all(
    r["pred"] in CATEGORIES for r in zero_shot_results
), "Predictions must be valid categories"
print("✓ Checkpoint 1 passed — zero-shot classification complete\n")

# INTERPRETATION: Zero-shot asks the model to classify without examples.
# Performance depends on how well the categories align with the model's
# pre-training distribution.  Strengths: fast, cheap, no example curation.
# Weaknesses: may hallucinate categories, inconsistent formatting.


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Few-shot with curated example selection
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Few-Shot Classification")
print("=" * 70)

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


async def few_shot_classify(text: str) -> tuple[str, float, float]:
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    examples_text = "\n".join(
        f'Review: "{ex["text"]}"\nSentiment: {ex["category"]}\n'
        for ex in FEW_SHOT_EXAMPLES
    )
    prompt = f"""Classify movie review snippets by sentiment.

{examples_text}
Now classify:
Review: "{text[:800]}"
Sentiment:"""

    response, cost = "", 0.0
    t0 = time.perf_counter()
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
        if hasattr(event, "cost"):
            cost = event.cost
    elapsed = time.perf_counter() - t0
    label = response.strip().lower()
    if "positive" in label:
        label = "positive"
    elif "negative" in label:
        label = "negative"
    return label, cost, elapsed


async def run_few_shot(docs: pl.DataFrame) -> list[dict]:
    results = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, cost, elapsed = await few_shot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "cost": cost,
                "elapsed": elapsed,
            }
        )
        if i < 5:
            print(
                f"  Doc {i+1}: pred={pred}, true={true_label} {'✓' if correct else '✗'}"
            )
    acc = sum(r["correct"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / len(results)
    print(
        f"  Accuracy: {acc:.0%} | Cost: ${total_cost:.4f} | Avg latency: {avg_latency:.2f}s"
    )
    return results


print("\n=== Few-Shot Classification (4 examples) ===")
few_shot_results = asyncio.run(run_few_shot(eval_docs))

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(few_shot_results) > 0, "Task 3: few-shot should produce results"
print("✓ Checkpoint 2 passed — few-shot classification complete\n")

# INTERPRETATION: Few-shot provides examples of the desired behaviour.
# The model sees input->output pairs and applies the pattern to new inputs.
# Key decisions: how many examples (3-8 typically), selection (diverse,
# representative), ordering (experiment — no universal best order).


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Chain-of-Thought (CoT) Prompting
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Chain-of-Thought Prompting")
print("=" * 70)


async def cot_classify(text: str) -> tuple[str, str, float, float]:
    """CoT: explicit step-by-step reasoning before classification."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    prompt = f"""Classify the sentiment of this movie review as positive or negative.

Think step by step:
1. Identify the key opinion words and their valence
2. Assess whether the overall tone is favourable or unfavourable
3. Consider any sarcasm or irony
4. State your final classification as exactly "positive" or "negative"

Review: "{text[:800]}"

Step-by-step reasoning:"""

    response, cost = "", 0.0
    t0 = time.perf_counter()
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
        if hasattr(event, "cost"):
            cost = event.cost
    elapsed = time.perf_counter() - t0

    reasoning = response.strip()
    # extract final label from the last line
    lower = reasoning.lower()
    if "negative" in lower.split("\n")[-1]:
        label = "negative"
    elif "positive" in lower.split("\n")[-1]:
        label = "positive"
    elif "negative" in lower:
        label = "negative"
    else:
        label = "positive"
    return label, reasoning, cost, elapsed


async def run_cot(docs: pl.DataFrame) -> list[dict]:
    results = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, reasoning, cost, elapsed = await cot_classify(text)
        correct = pred == true_label
        results.append(
            {
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
            print(f"    Final: {pred}, true: {true_label} {'✓' if correct else '✗'}")
    acc = sum(r["correct"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    avg_latency = sum(r["elapsed"] for r in results) / len(results)
    print(
        f"  Accuracy: {acc:.0%} | Cost: ${total_cost:.4f} | Avg latency: {avg_latency:.2f}s"
    )
    return results


print("\n=== Chain-of-Thought Classification ===")
cot_results = asyncio.run(run_cot(eval_docs))

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(cot_results) > 0, "Task 4: CoT should produce results"
assert all(
    "reasoning" in r for r in cot_results
), "Each CoT result should contain reasoning"
print("✓ Checkpoint 3 passed — chain-of-thought classification complete\n")

# INTERPRETATION: CoT forces step-by-step reasoning before answering.
# Trade-off: more tokens (higher cost, higher latency), but better for
# ambiguous cases.  The reasoning trace is auditable.


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Zero-Shot CoT ("Let's think step by step")
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Zero-Shot CoT")
print("=" * 70)


async def zero_shot_cot_classify(text: str) -> tuple[str, str, float, float]:
    """Append 'Let's think step by step' without any explicit reasoning template."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    prompt = f"""Classify the sentiment of this movie review as "positive" or "negative".

Review: "{text[:800]}"

Let's think step by step."""

    response, cost = "", 0.0
    t0 = time.perf_counter()
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
        if hasattr(event, "cost"):
            cost = event.cost
    elapsed = time.perf_counter() - t0
    reasoning = response.strip()
    lower = reasoning.lower()
    if "negative" in lower.split("\n")[-1]:
        label = "negative"
    elif "positive" in lower.split("\n")[-1]:
        label = "positive"
    elif lower.count("negative") > lower.count("positive"):
        label = "negative"
    else:
        label = "positive"
    return label, reasoning, cost, elapsed


async def run_zero_shot_cot(docs: pl.DataFrame) -> list[dict]:
    results = []
    for i, (text, true_label) in enumerate(
        zip(docs["text"].to_list(), docs["label"].to_list())
    ):
        pred, reasoning, cost, elapsed = await zero_shot_cot_classify(text)
        correct = pred == true_label
        results.append(
            {
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "cost": cost,
                "elapsed": elapsed,
            }
        )
        if i < 3:
            print(
                f"  Doc {i+1}: pred={pred}, true={true_label} {'✓' if correct else '✗'}"
            )
    acc = sum(r["correct"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    print(f"  Accuracy: {acc:.0%} | Cost: ${total_cost:.4f}")
    return results


print("\n=== Zero-Shot CoT ('Let's think step by step') ===")
zs_cot_results = asyncio.run(run_zero_shot_cot(eval_docs))

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert len(zs_cot_results) > 0, "Task 5: zero-shot CoT should produce results"
print("✓ Checkpoint 4 passed — zero-shot CoT classification complete\n")

# INTERPRETATION: "Let's think step by step" is the simplest CoT trigger.
# No manually crafted reasoning template, yet it often improves accuracy
# over plain zero-shot — the model generates its own chain of thought.


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Self-Consistency (sample multiple CoT paths, majority vote)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Self-Consistency (Majority Vote)")
print("=" * 70)

N_SAMPLES = 3  # Number of independent CoT samples per query


async def self_consistency_classify(text: str) -> tuple[str, list[str], float, float]:
    """Sample N_SAMPLES CoT paths and take majority vote."""
    votes: list[str] = []
    total_cost = 0.0
    t0 = time.perf_counter()
    for _ in range(N_SAMPLES):
        label, _, cost, _ = await cot_classify(text)
        votes.append(label)
        total_cost += cost
    elapsed = time.perf_counter() - t0
    counter = Counter(votes)
    majority = counter.most_common(1)[0][0]
    return majority, votes, total_cost, elapsed


async def run_self_consistency(docs: pl.DataFrame) -> list[dict]:
    # Use a subset for cost reasons
    subset = docs.head(10)
    results = []
    for i, (text, true_label) in enumerate(
        zip(subset["text"].to_list(), subset["label"].to_list())
    ):
        pred, votes, cost, elapsed = await self_consistency_classify(text)
        correct = pred == true_label
        results.append(
            {
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "cost": cost,
                "elapsed": elapsed,
                "votes": votes,
            }
        )
        if i < 5:
            print(
                f"  Doc {i+1}: votes={votes} -> majority={pred}, "
                f"true={true_label} {'✓' if correct else '✗'}"
            )
    acc = sum(r["correct"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    print(f"  Accuracy: {acc:.0%} | Cost: ${total_cost:.4f} (N={N_SAMPLES} samples)")
    return results


print(f"\n=== Self-Consistency ({N_SAMPLES} CoT samples, majority vote) ===")
sc_results = asyncio.run(run_self_consistency(eval_docs))

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert len(sc_results) > 0, "Task 6: self-consistency should produce results"
assert all(
    "votes" in r for r in sc_results
), "Each result should record individual votes"
print("✓ Checkpoint 5 passed — self-consistency classification complete\n")

# INTERPRETATION: Self-consistency samples multiple INDEPENDENT reasoning
# paths and uses majority vote.  It combats the randomness of any single
# CoT sample.  Cost scales linearly with N_SAMPLES.  Diminishing returns
# beyond N=5 for binary classification, N=7-9 for multi-class.


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Structured Prompting (explicit JSON output format)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Structured Prompting (JSON Output Spec)")
print("=" * 70)


async def structured_prompt_classify(text: str) -> tuple[dict, float, float]:
    """Ask the model to return a JSON object with specific fields."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)
    prompt = f"""Analyse the sentiment of the following movie review.
Return a JSON object with exactly these fields:
  "sentiment": "positive" or "negative"
  "confidence": a float between 0.0 and 1.0
  "key_words": a list of up to 5 words that signal the sentiment
  "reasoning": a one-sentence explanation

Review: "{text[:800]}"

JSON:"""

    response, cost = "", 0.0
    t0 = time.perf_counter()
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text
        if hasattr(event, "cost"):
            cost = event.cost
    elapsed = time.perf_counter() - t0

    # Parse JSON from response
    try:
        # find the first { ... } block
        start = response.index("{")
        end = response.rindex("}") + 1
        parsed = json.loads(response[start:end])
    except (ValueError, json.JSONDecodeError):
        parsed = {
            "sentiment": "unknown",
            "confidence": 0.0,
            "key_words": [],
            "reasoning": response[:100],
        }
    return parsed, cost, elapsed


async def run_structured_prompt(docs: pl.DataFrame) -> list[dict]:
    subset = docs.head(10)
    results = []
    for i, (text, true_label) in enumerate(
        zip(subset["text"].to_list(), subset["label"].to_list())
    ):
        parsed, cost, elapsed = await structured_prompt_classify(text)
        pred = parsed.get("sentiment", "unknown").lower()
        if "positive" in pred:
            pred = "positive"
        elif "negative" in pred:
            pred = "negative"
        correct = pred == true_label
        results.append(
            {
                "pred": pred,
                "true": true_label,
                "correct": correct,
                "cost": cost,
                "elapsed": elapsed,
                "parsed": parsed,
            }
        )
        if i < 3:
            print(f"  Doc {i+1}: {json.dumps(parsed, indent=2)[:200]}")
    acc = sum(r["correct"] for r in results) / len(results)
    total_cost = sum(r["cost"] for r in results)
    print(f"  Accuracy: {acc:.0%} | Cost: ${total_cost:.4f}")
    return results


print("\n=== Structured Prompting (JSON output) ===")
structured_prompt_results = asyncio.run(run_structured_prompt(eval_docs))

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert (
    len(structured_prompt_results) > 0
), "Task 7: structured prompting should produce results"
assert all(
    "parsed" in r for r in structured_prompt_results
), "Results should contain parsed JSON"
print("✓ Checkpoint 6 passed — structured prompting complete\n")

# INTERPRETATION: Structured prompting specifies the output format explicitly.
# It's a halfway house between free-form text and Signature enforcement:
# the model usually complies, but there's no type system guarantee.
# JSON parsing can fail on malformed output.  Use Signatures for production.


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: Kaizen Signature for Type-Safe Structured Extraction
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: Kaizen Signature — Type-Safe Structured Extraction")
print("=" * 70)


class ReviewExtraction(Signature):
    """Extract structured information from a movie review snippet."""

    review_text: str = InputField(description="Movie review snippet to analyse")

    sentiment: str = OutputField(description="Exactly one of: positive, negative")
    confidence: float = OutputField(description="Classification confidence 0.0 to 1.0")
    key_phrases: list[str] = OutputField(
        description="Up to 5 phrases that signal sentiment (e.g. 'a masterpiece', 'tedious')"
    )
    targets: list[str] = OutputField(
        description="Aspects of the film evaluated (acting, plot, visuals, pacing)"
    )
    tone: str = OutputField(
        description="Emotional tone: enthusiastic, measured, disappointed, angry, sarcastic"
    )


async def run_signature_extraction(docs: pl.DataFrame) -> list:
    agent = SimpleQAAgent(
        signature=ReviewExtraction,
        model=model,
        max_llm_cost_usd=1.0,
    )
    subset = docs.head(10)
    results = []
    for i, text in enumerate(subset["text"].to_list()):
        result = await agent.run(review_text=text[:800])
        results.append(result)
        if i < 3:
            print(f"\n  Review {i+1}:")
            print(f"    Sentiment:  {result.sentiment}")
            print(f"    Confidence: {result.confidence:.2f}")
            print(f"    Key phrases: {result.key_phrases[:3]}")
            print(f"    Targets:    {result.targets[:3]}")
            print(f"    Tone:       {result.tone}")
    return results


print("\n=== Signature Extraction ===")
signature_results = asyncio.run(run_signature_extraction(eval_docs))

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert len(signature_results) > 0, "Task 8: Signature extraction should produce results"
sample = signature_results[0]
assert hasattr(sample, "sentiment"), "Result should have 'sentiment' field"
assert hasattr(sample, "confidence"), "Result should have 'confidence' field"
assert 0 <= sample.confidence <= 1, "Confidence should be in [0, 1]"
assert hasattr(sample, "tone"), "Result should have 'tone' field"
print(
    "✓ Checkpoint 7 passed — Signature extraction: "
    f"sentiment='{sample.sentiment}', confidence={sample.confidence:.2f}\n"
)

# INTERPRETATION: Signatures guarantee structure.  Unlike structured
# prompting (Task 7), the output is typed — downstream code accesses
# result.sentiment, not result["sentiment"].  If the model returns
# something malformed, Kaizen retries or raises a typed error.


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Quantitative Comparison Across All Strategies
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Quantitative Strategy Comparison")
print("=" * 70)


def compute_metrics(results: list[dict], name: str) -> dict:
    """Compute accuracy, total cost, and average latency from results."""
    n = len(results)
    acc = sum(r["correct"] for r in results) / n if n else 0.0
    total_cost = sum(r.get("cost", 0.0) for r in results)
    avg_latency = sum(r.get("elapsed", 0.0) for r in results) / n if n else 0.0
    return {
        "strategy": name,
        "n": n,
        "accuracy": acc,
        "total_cost": total_cost,
        "avg_latency_s": avg_latency,
    }


# Build comparison table
strategy_metrics = [
    compute_metrics(zero_shot_results, "Zero-Shot"),
    compute_metrics(few_shot_results, "Few-Shot (4 examples)"),
    compute_metrics(cot_results, "Chain-of-Thought"),
    compute_metrics(zs_cot_results, "Zero-Shot CoT"),
    compute_metrics(sc_results, f"Self-Consistency (N={N_SAMPLES})"),
    compute_metrics(structured_prompt_results, "Structured JSON"),
]

comparison_df = pl.DataFrame(strategy_metrics)
print("\n" + str(comparison_df))

print("\nKey insights:")
print("  - Zero-shot: cheapest, fastest, but may hallucinate categories")
print("  - Few-shot: better consistency via examples; cost of longer prompt")
print("  - CoT: forces reasoning; best for ambiguous cases; highest latency")
print("  - Zero-shot CoT: lightweight reasoning trigger; good cost/quality ratio")
print(
    f"  - Self-consistency: majority vote over N={N_SAMPLES} paths; highest accuracy, N× cost"
)
print("  - Structured JSON: format control, but fragile (parsing failures)")
print("  - Signature: type-safe, retries on failure — production standard")

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert comparison_df.height >= 6, "Task 9: comparison should cover all strategies"
print("\n✓ Checkpoint 8 passed — strategy comparison table generated\n")

# INTERPRETATION: The comparison reveals the cost-quality Pareto frontier.
# For production: Signature (type safety) + few-shot (consistency).
# For research: self-consistency (highest accuracy) at N× cost.
# For speed: zero-shot (cheapest, fastest, good enough for simple tasks).


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Inference Optimisations
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Inference Optimisations")
print("=" * 70)

print(
    """
KV-Cache (Key-Value Cache):
  During autoregressive generation, each new token attends to ALL prior
  tokens.  Without caching, the K and V projections for all prior tokens
  are recomputed at every step — O(n^2) total compute for n tokens.
  KV-cache stores computed K/V tensors and reuses them:
    Step 1: compute K,V for tokens [1..t], cache them
    Step 2: compute K,V for token [t+1] only, append to cache
    Reduces generation from O(n^2) to O(n) total compute.
  Memory cost: O(n * d_model * n_layers) — can be large for long contexts.

Speculative Decoding:
  Use a SMALL draft model to generate K candidate tokens cheaply.
  Then verify ALL K tokens in ONE forward pass of the LARGE target model.
  If the large model agrees with the draft, all K tokens are accepted
  (wallclock speedup ≈ K×).  Rejected tokens are re-sampled from the
  target distribution.  Net: 2-3× speedup with zero quality degradation.

Continuous Batching (vLLM, TGI):
  Traditional: wait for all requests in a batch to finish before serving.
  Continuous: as one request finishes, immediately fill its slot with a
  new request.  GPU utilisation stays near 100% even with variable-length
  requests.  Key enabler: paged attention (vLLM) — KV-cache allocated
  in fixed-size pages instead of contiguous blocks.

Flash Attention:
  Fuses the attention computation (Q @ K^T / sqrt(d), softmax, @ V) into
  a single GPU kernel.  Avoids materialising the n×n attention matrix in
  HBM.  Memory: O(n) instead of O(n^2).  Speed: 2-4× faster.
"""
)

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 9 passed — inference optimisations explained\n")

# INTERPRETATION: These optimisations are invisible to the prompt engineer
# but critical to production deployment:
# KV-cache:           every serving framework uses it; 10× throughput
# Speculative decode: 2-3× latency reduction with zero quality loss
# Continuous batching: keeps GPU utilisation >90% in production
# Flash attention:     enables long context (128K+) on consumer hardware


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ LLM foundations: pre-training (next-token, masked LM), scaling laws,
    RLHF pipeline (SFT → reward model → PPO)
  ✓ Zero-shot: task description only; cheapest, fastest, least consistent
  ✓ Few-shot: examples steer the model; better consistency and formatting
  ✓ Chain-of-thought: "Think step by step" elicits intermediate reasoning
  ✓ Zero-shot CoT: append trigger phrase; lightweight reasoning at low cost
  ✓ Self-consistency: majority vote over N independent CoT paths; highest
    accuracy for ambiguous tasks; N× cost
  ✓ Structured prompting: JSON output spec; convenient but fragile parsing
  ✓ Kaizen Signature: typed, validated, retryable — production standard
  ✓ Kaizen Delegate: streaming LLM calls with cost budget enforcement
  ✓ Inference optimisations: KV-cache, speculative decoding, continuous
    batching, flash attention

  Prompting cost vs quality hierarchy:
    Zero-shot (cheapest) < Few-shot < Zero-shot CoT < CoT
    < Self-consistency (N×) < Signature (type-safe, production)

  NEXT: Exercise 2 (Fine-Tuning) goes beyond prompting.  Instead of
  giving the model better instructions, you change the model weights
  using low-rank matrix decomposition (LoRA) and adapter layers — both
  implemented from scratch.
"""
)
