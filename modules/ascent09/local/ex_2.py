# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 2: Prompt Engineering
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Master zero-shot, few-shot, and chain-of-thought prompting
#   via Delegate, comparing structured output quality across strategies.
#
# TASKS:
#   1. Zero-shot classification with Delegate
#   2. Few-shot with example selection
#   3. Chain-of-thought prompting
#   4. Build a custom Signature for structured extraction
#   5. Compare accuracy across prompting strategies
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen import Signature, InputField, OutputField
from kaizen_agents.agents.specialized.simple_qa import SimpleQAAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))
print(f"LLM Model: {model}")

# ── Data Loading ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

# TODO: Select first 10 documents as sample.
# Hint: reports.head(10)
sample_docs = ____
print(f"Loaded {reports.height:,} documents for classification")
print(f"Columns: {reports.columns}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Zero-shot classification with Delegate
# ══════════════════════════════════════════════════════════════════════

CATEGORIES = ["Financial", "Technology", "Healthcare", "Real Estate", "Manufacturing"]


async def zero_shot_classify(text: str) -> str:
    """Classify a document using zero-shot prompting."""
    # TODO: Create a Delegate with model and cost budget.
    # Hint: Delegate(model=model, max_llm_cost_usd=0.5)
    delegate = ____

    # TODO: Build a zero-shot classification prompt with categories and text.
    # Hint: f-string with CATEGORIES, text[:800], and instruction to respond with only the category name
    prompt = ____

    response = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            response += event.text

    return response.strip()


async def run_zero_shot():
    """Run zero-shot classification on sample documents."""
    print(f"\n=== Zero-Shot Classification ===")
    results = []
    # TODO: Extract text column as a Python list.
    # Hint: sample_docs.select("text").to_series().to_list()
    texts = ____
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
    # TODO: Create a Delegate with model and cost budget.
    # Hint: Delegate(model=model, max_llm_cost_usd=0.5)
    delegate = ____

    # TODO: Build examples_text by joining each example's text and category.
    # Hint: "\n".join(f'Text: "{ex["text"]}"\nCategory: {ex["category"]}\n' for ex in FEW_SHOT_EXAMPLES)
    examples_text = ____

    prompt = f"""Classify Singapore company reports into categories. Here are examples:

{examples_text}
Now classify this report:
Text: "{text[:800]}"
Category:"""

    response = ""
    # TODO: Stream the response from delegate.run(prompt).
    # Hint: async for event in delegate.run(prompt): if hasattr(event, "text"): response += event.text
    ____

    return response.strip()


async def run_few_shot():
    """Run few-shot classification on sample documents."""
    print(f"\n=== Few-Shot Classification ===")
    results = []
    # TODO: Extract text column as a Python list.
    # Hint: sample_docs.select("text").to_series().to_list()
    texts = ____
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
    # TODO: Create a Delegate with model and cost budget.
    # Hint: Delegate(model=model, max_llm_cost_usd=0.5)
    delegate = ____

    # TODO: Write a CoT prompt that instructs step-by-step reasoning before classification.
    # Hint: Include CATEGORIES, steps (identify key terms, match to category, state final classification), and text[:800]
    prompt = ____

    response = ""
    # TODO: Stream the response from delegate.run(prompt).
    # Hint: async for event in delegate.run(prompt): if hasattr(event, "text"): response += event.text
    ____

    # Extract final category from reasoning
    lines = response.strip().split("\n")
    final_category = lines[-1].strip() if lines else "Unknown"
    return response.strip(), final_category


async def run_cot():
    """Run chain-of-thought classification."""
    print(f"\n=== Chain-of-Thought Classification ===")
    results = []
    # TODO: Extract text column as a Python list.
    # Hint: sample_docs.select("text").to_series().to_list()
    texts = ____
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

    # TODO: Define OutputField for category (one of the 5 categories)
    category: str = ____
    # TODO: Define OutputField for key_entities (list of named entities)
    key_entities: list[str] = ____
    # TODO: Define OutputField for financial_metrics (list of figures/percentages)
    financial_metrics: list[str] = ____
    # TODO: Define OutputField for sentiment (positive/negative/neutral)
    sentiment: str = ____
    # TODO: Define OutputField for confidence (0-1 float)
    # Hint: OutputField(description="Classification confidence 0-1")
    confidence: float = ____


async def structured_extract():
    """Use SimpleQAAgent for structured extraction."""
    # TODO: Create SimpleQAAgent with ReportExtraction signature, model, and cost budget.
    # Hint: SimpleQAAgent(signature=ReportExtraction, model=model, max_llm_cost_usd=1.0)
    agent = ____

    print(f"\n=== Structured Extraction (Signature) ===")
    # TODO: Extract text column as a Python list.
    # Hint: sample_docs.select("text").to_series().to_list()
    texts = ____
    results = []
    for i, text in enumerate(texts[:3]):
        # TODO: Run the agent with report_text input.
        # Hint: await agent.run(report_text=text[:800])
        result = ____
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

print("\n✓ Exercise 2 complete — prompt engineering strategies compared")
