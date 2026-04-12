# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1: Prompt Engineering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Apply zero-shot, few-shot, and chain-of-thought prompting techniques
#   - Explain when each prompting strategy outperforms the others
#   - Use Kaizen Delegate with streaming and cost tracking
#   - Define a typed Signature for structured LLM output extraction
#   - Compare prompting strategies quantitatively on the same task
#
# PREREQUISITES:
#   M5 complete (transformers and LLM architecture). Understanding that
#   LLMs are trained to predict the next token — prompts shift which
#   tokens are likely. No fine-tuning required; prompting is zero-cost.
#
# ESTIMATED TIME: 45-75 minutes
#
# TASKS:
#   1. Zero-shot classification with Delegate
#   2. Few-shot with example selection
#   3. Chain-of-thought prompting
#   4. Build a custom Signature for structured extraction
#   5. Compare accuracy across prompting strategies
#
# DATASET: Singapore company reports (10-document sample from a larger corpus)
#   Columns: text (report excerpt), likely metadata columns
#   Classification target: Financial / Technology / Healthcare /
#     Real Estate / Manufacturing
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = MLFPDataLoader()
reports = loader.load("mlfp06", "sg_company_reports.parquet")

sample_docs = reports.head(10)
print(f"Loaded {reports.height:,} documents for classification")
print(f"Columns: {reports.columns}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Zero-shot classification with Delegate
# ══════════════════════════════════════════════════════════════════════

CATEGORIES = ["Financial", "Technology", "Healthcare", "Real Estate", "Manufacturing"]


async def zero_shot_classify(text: str) -> str:
    """Classify a document using zero-shot prompting."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)

    prompt = f"""Classify the following Singapore company report into exactly one category.

Categories: {', '.join(CATEGORIES)}

Report excerpt:
{text[:800]}

Respond with ONLY the category name, nothing else."""

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text

    return response.strip()


async def run_zero_shot():
    """Run zero-shot classification on sample documents."""
    print(f"\n=== Zero-Shot Classification ===")
    results = []
    texts = sample_docs.select("text").to_series().to_list()
    for i, text in enumerate(texts[:5]):
        category = await zero_shot_classify(text)
        print(f"  Doc {i+1}: {category}")
        results.append(category)
    return results


zero_shot_results = asyncio.run(run_zero_shot())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Few-shot with example selection
# ══════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "text": "Revenue increased 15% driven by strong loan growth and net interest margin expansion.",
        "category": "Financial",
    },
    {
        "text": "The company launched its new cloud-native SaaS platform serving enterprise clients across APAC.",
        "category": "Technology",
    },
    {
        "text": "Clinical trials for the new oncology drug showed 40% improvement in patient outcomes.",
        "category": "Healthcare",
    },
    {
        "text": "The integrated township development in Jurong added 2,500 residential units to the portfolio.",
        "category": "Real Estate",
    },
    {
        "text": "Factory automation reduced production cycle time by 30% at the Tuas semiconductor fab.",
        "category": "Manufacturing",
    },
]


async def few_shot_classify(text: str) -> str:
    """Classify a document using few-shot prompting with examples."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)

    examples_text = "\n".join(
        f'Text: "{ex["text"]}"\nCategory: {ex["category"]}\n'
        for ex in FEW_SHOT_EXAMPLES
    )

    prompt = f"""Classify Singapore company reports into categories. Here are examples:

{examples_text}
Now classify this report:
Text: "{text[:800]}"
Category:"""

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text

    return response.strip()


async def run_few_shot():
    """Run few-shot classification on sample documents."""
    print(f"\n=== Few-Shot Classification ===")
    results = []
    texts = sample_docs.select("text").to_series().to_list()
    for i, text in enumerate(texts[:5]):
        category = await few_shot_classify(text)
        print(f"  Doc {i+1}: {category}")
        results.append(category)
    return results


few_shot_results = asyncio.run(run_few_shot())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Chain-of-thought prompting
# ══════════════════════════════════════════════════════════════════════


async def cot_classify(text: str) -> str:
    """Classify using chain-of-thought reasoning."""
    delegate = Delegate(model=model, max_llm_cost_usd=0.5)

    prompt = f"""Classify this Singapore company report into one category: {', '.join(CATEGORIES)}.

Think step by step:
1. Identify key terms and topics in the text
2. Match those terms to the most relevant category
3. State your final classification

Report excerpt:
{text[:800]}

Step-by-step reasoning:"""

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text

    # Extract final category from reasoning
    lines = response.strip().split("\n")
    final_category = lines[-1].strip() if lines else "Unknown"
    return response.strip(), final_category


async def run_cot():
    """Run chain-of-thought classification."""
    print(f"\n=== Chain-of-Thought Classification ===")
    results = []
    texts = sample_docs.select("text").to_series().to_list()
    for i, text in enumerate(texts[:3]):
        reasoning, category = await cot_classify(text)
        print(f"\n  Doc {i+1} reasoning (excerpt): {reasoning[:200]}...")
        print(f"  Final: {category}")
        results.append(category)
    return results


cot_results = asyncio.run(run_cot())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Custom Signature for structured extraction
# ══════════════════════════════════════════════════════════════════════


class ReportExtraction(Signature):
    """Extract structured information from a company report."""

    report_text: str = InputField(description="Company report text excerpt")

    category: str = OutputField(
        description="One of: Financial, Technology, Healthcare, Real Estate, Manufacturing"
    )
    key_entities: list[str] = OutputField(
        description="Named entities mentioned (companies, products, locations)"
    )
    financial_metrics: list[str] = OutputField(
        description="Any financial figures or percentages mentioned"
    )
    sentiment: str = OutputField(
        description="Overall sentiment: positive, negative, or neutral"
    )
    confidence: float = OutputField(description="Classification confidence 0-1")


async def structured_extract():
    """Use SimpleQAAgent for structured extraction."""
    agent = SimpleQAAgent(
        signature=ReportExtraction,
        model=model,
        max_llm_cost_usd=1.0,
    )

    print(f"\n=== Structured Extraction (Signature) ===")
    texts = sample_docs.select("text").to_series().to_list()
    results = []
    for i, text in enumerate(texts[:3]):
        result = await agent.run(report_text=text[:800])
        print(f"\n  Doc {i+1}:")
        print(f"    Category: {result.category}")
        print(f"    Entities: {result.key_entities[:5]}")
        print(f"    Metrics: {result.financial_metrics[:3]}")
        print(f"    Sentiment: {result.sentiment}")
        print(f"    Confidence: {result.confidence}")
        results.append(result)
    return results


structured_results = asyncio.run(structured_extract())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare accuracy across prompting strategies
# ══════════════════════════════════════════════════════════════════════

comparison = pl.DataFrame(
    {
        "strategy": ["Zero-Shot", "Few-Shot", "Chain-of-Thought"],
        "doc_1": [
            zero_shot_results[0] if zero_shot_results else "N/A",
            few_shot_results[0] if few_shot_results else "N/A",
            cot_results[0] if cot_results else "N/A",
        ],
        "doc_2": [
            zero_shot_results[1] if len(zero_shot_results) > 1 else "N/A",
            few_shot_results[1] if len(few_shot_results) > 1 else "N/A",
            cot_results[1] if len(cot_results) > 1 else "N/A",
        ],
        "doc_3": [
            zero_shot_results[2] if len(zero_shot_results) > 2 else "N/A",
            few_shot_results[2] if len(few_shot_results) > 2 else "N/A",
            cot_results[2] if len(cot_results) > 2 else "N/A",
        ],
    }
)

print(f"\n=== Strategy Comparison ===")
print(comparison)
print(f"\nKey insights:")
print(f"  Zero-shot: fast, no examples needed, may hallucinate categories")
print(f"  Few-shot: more consistent, requires curated examples")
print(f"  CoT: best for ambiguous cases, slower and more expensive")
print(f"  Signature: guarantees structure, best for pipelines")

print("=" * 60)
print("  MLFP06 Exercise 1: Prompt Engineering")
print("=" * 60)
print(f"\n  Zero-shot, few-shot, and CoT prompting demonstrated.\n")

# ── Checkpoint 1: Zero-shot ────────────────────────────────────────────
assert len(zero_shot_results) > 0, "Zero-shot should produce at least one result"
assert all(r is not None for r in zero_shot_results), "Results should not be None"
print(f"✓ Checkpoint 1 passed — zero-shot: {len(zero_shot_results)} classifications\n")

# INTERPRETATION: Zero-shot prompting asks the model to classify without any
# examples. The model uses its training knowledge about what "Financial",
# "Technology", etc. mean. Performance depends on how well the categories
# align with the model's pre-training distribution.
# When zero-shot fails: the task definition is ambiguous, the domain is
# very specialised, or the output format needs to be exact.

# ── Checkpoint 2: Few-shot ────────────────────────────────────────────
assert len(few_shot_results) > 0, "Few-shot should produce results"
print(f"✓ Checkpoint 2 passed — few-shot: {len(few_shot_results)} classifications\n")

# INTERPRETATION: Few-shot prompting provides examples of correct classifications.
# The model sees the pattern (input -> output) and applies it to new inputs.
# Key decisions: how many examples (3-8 typically), how to select them
# (diverse, representative), and how to order them (no clear rule, experiment).
# Few-shot typically improves consistency and reduces hallucination of
# categories that aren't in the allowed list.

# ── Checkpoint 3: Chain-of-thought ────────────────────────────────────
assert len(cot_results) > 0, "CoT should produce results"
print(f"✓ Checkpoint 3 passed — chain-of-thought: {len(cot_results)} classifications\n")

# INTERPRETATION: CoT forces the model to reason step-by-step before answering.
# "Think step by step" elicits intermediate reasoning that:
# 1. Commits the model to a logical path before stating the answer
# 2. Allows verification of the reasoning (interpretability)
# 3. Improves accuracy on complex, multi-step tasks
# Trade-off: CoT uses more tokens (higher cost) and is slower.
# Best for: ambiguous cases where the right answer needs reasoning.
# Not needed for: simple pattern matching where zero-shot or few-shot suffices.

# ── Checkpoint 4: Structured extraction ──────────────────────────────
assert len(structured_results) > 0, "Structured extraction should produce results"
sample_result = structured_results[0]
assert hasattr(sample_result, "category"), "Result should have category field"
assert hasattr(sample_result, "confidence"), "Result should have confidence field"
assert 0 <= sample_result.confidence <= 1, "Confidence should be in [0, 1]"
print(f"✓ Checkpoint 4 passed — Signature extraction: "
      f"category='{sample_result.category}', "
      f"confidence={sample_result.confidence:.2f}\n")

# INTERPRETATION: Kaizen Signatures enforce type-safe structured output.
# Instead of parsing free-form text (fragile), the model is constrained to
# produce a specific JSON schema. This is the production approach:
# - category: str (validated against allowed values)
# - confidence: float (0-1, not a vague "high/medium/low")
# - key_entities: list[str] (structured, not comma-separated prose)
# Use Signatures whenever downstream code needs to process LLM outputs.

# ── Checkpoint 5: Strategy comparison ────────────────────────────────
assert comparison is not None, "Comparison DataFrame should be created"
print(f"✓ Checkpoint 5 passed — strategy comparison table generated\n")

# INTERPRETATION: The comparison reveals where strategies agree and disagree.
# Agreement = higher confidence in the classification.
# Disagreement = the document is ambiguous or lies at a category boundary.
# For production deployment:
# - Use zero-shot for speed-sensitive, simple classification
# - Use few-shot when consistency across runs matters
# - Use CoT when audit trail / explainability is required
# - Use Signature always when the output feeds into code


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print("""
  ✓ Zero-shot: task description only, relies on model's pre-training
  ✓ Few-shot: provide examples, improves consistency and format adherence
  ✓ Chain-of-thought: "Think step by step" elicits intermediate reasoning,
    best for complex tasks where the correct answer requires multi-step logic
  ✓ Kaizen Delegate: streaming LLM calls with cost budget enforcement
  ✓ Kaizen Signature: typed structured output (no fragile string parsing)

  Prompting cost vs quality hierarchy:
    Zero-shot (cheapest) < Few-shot < CoT < Signature (safest for pipelines)
  Accuracy hierarchy (typically):
    Zero-shot < Few-shot < CoT ≈ Signature (depends on task complexity)

  NEXT: Exercise 2 (LoRA Fine-Tuning) goes beyond prompting.
  Instead of giving the model better instructions, you change the model
  weights using low-rank matrix decomposition — adapting it to your
  domain with <1% of full fine-tuning parameter cost.
""")
