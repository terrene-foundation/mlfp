# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 1: Delegate + SimpleQAAgent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use Delegate for autonomous data analysis Q&A and build a
#   SimpleQAAgent with custom Signature for structured answers. Set
#   max_llm_cost_usd from Exercise 1 — mandatory for all M5 exercises.
#
# TASKS:
#   1. Set up Delegate with cost budget governance
#   2. Run Delegate on data analysis questions
#   3. Build SimpleQAAgent with custom Signature (InputField/OutputField)
#   4. Compare Delegate vs SimpleQAAgent outputs
#   5. Track LLM costs and demonstrate budget enforcement
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

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
customers = loader.load("ascent04", "ecommerce_customers.parquet")

# Prepare a text summary of the data for the agents
data_summary = f"""
E-commerce Customer Dataset:
- {customers.height:,} customers
- Columns: {', '.join(customers.columns)}
- Revenue range: ${customers['total_revenue'].min():.0f} - ${customers['total_revenue'].max():.0f}
- Average orders: {customers['order_count'].mean():.1f}
"""

print(f"=== Data Context ===")
print(data_summary)


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Delegate with cost budget
# ══════════════════════════════════════════════════════════════════════


async def delegate_analysis():
    """Use Delegate for autonomous data analysis Q&A."""

    # TODO: Create a Delegate with model and a hard cost budget of $2.00
    # Hint: Delegate(model=____, max_llm_cost_usd=____)
    delegate = ____

    questions = [
        "Based on this customer data, what are the top 3 customer segments you'd recommend for targeted marketing?",
        "What metrics should we track to predict customer churn from this dataset?",
        "If we wanted to build a recommendation system, which features would be most important?",
    ]

    print(f"\n=== Delegate Analysis (budget: $2.00) ===")
    results = []
    for i, question in enumerate(questions):
        prompt = f"{data_summary}\n\nQuestion: {question}"
        print(f"\nQ{i+1}: {question}")

        # TODO: Stream the Delegate response using async for
        # Hint: async for event in delegate.run(____):
        #           if hasattr(event, "____"):
        #               response_text += event.____
        response_text = ""
        # ... your streaming loop here ...

        print(f"A{i+1}: {response_text[:300]}...")
        results.append({"question": question, "answer": response_text})

    return delegate, results


delegate, delegate_results = asyncio.run(delegate_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Custom Signature for structured output
# ══════════════════════════════════════════════════════════════════════


class CustomerSegmentAnalysis(Signature):
    """Analyse customer data and return structured segment recommendations."""

    # TODO: Define two InputFields: data_context (dataset summary) and question (analysis question)
    # Hint: field_name: type = InputField(description="____")
    data_context: str = ____
    question: str = ____

    # TODO: Define four OutputFields: segments (list[str]), reasoning (str),
    #       confidence (float 0-1), next_steps (list[str])
    # Hint: field_name: type = OutputField(description="____")
    segments: list[str] = ____
    reasoning: str = ____
    confidence: float = ____
    next_steps: list[str] = ____


class ChurnPrediction(Signature):
    """Predict churn risk factors from customer data context."""

    # TODO: Define one InputField: data_context (dataset summary)
    data_context: str = ____

    # TODO: Define three OutputFields: risk_factors (list[str]),
    #       key_metrics (list[str]), model_recommendation (str)
    risk_factors: list[str] = ____
    key_metrics: list[str] = ____
    model_recommendation: str = ____


# ══════════════════════════════════════════════════════════════════════
# TASK 3: SimpleQAAgent with custom Signature
# ══════════════════════════════════════════════════════════════════════


async def structured_analysis():
    """Use SimpleQAAgent for structured output."""

    # TODO: Create a SimpleQAAgent using CustomerSegmentAnalysis signature
    # Hint: SimpleQAAgent(signature=____, model=____, max_llm_cost_usd=____)
    segment_agent = ____

    # TODO: Run the segment_agent with data_context and question keyword arguments
    # Hint: await segment_agent.run(data_context=____, question=____)
    segment_result = await segment_agent.run(
        data_context=____,
        question="What customer segments should we target for a premium loyalty programme?",
    )

    print(f"\n=== Structured Segment Analysis ===")
    print(f"Segments: {segment_result.segments}")
    print(f"Confidence: {segment_result.confidence}")
    print(f"Reasoning: {segment_result.reasoning[:200]}...")
    print(f"Next steps: {segment_result.next_steps}")

    # TODO: Create a SimpleQAAgent using ChurnPrediction signature with budget $1.00
    churn_agent = ____

    # TODO: Run the churn_agent with only data_context
    churn_result = await churn_agent.run(
        data_context=____,
    )

    print(f"\n=== Structured Churn Analysis ===")
    print(f"Risk factors: {churn_result.risk_factors}")
    print(f"Key metrics: {churn_result.key_metrics}")
    print(f"Model recommendation: {churn_result.model_recommendation}")

    return segment_result, churn_result


segment_result, churn_result = asyncio.run(structured_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare outputs
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Delegate vs SimpleQAAgent ===")
print(f"Delegate: free-form text, flexible, unpredictable structure")
print(f"SimpleQA: typed Signature, structured output, guaranteed fields")
print(f"\nWhen to use each:")
print(f"  Delegate: exploratory analysis, open-ended questions")
print(f"  SimpleQA: production pipelines requiring structured data")
print(f"  → Signature = contract. Like ModelSignature for models, but for agents.")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Cost tracking
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== LLM Cost Governance ===")
print(f"max_llm_cost_usd is MANDATORY for all M5 exercises.")
print(f"CO methodology: human-on-the-loop, not in-the-loop.")
print(f"The budget cap ensures agents cannot run away with API costs.")
print(f"\nIn production:")
print(f"  1. Set max_llm_cost_usd per agent based on expected task complexity")
print(f"  2. Monitor actual spend vs budget")
print(f"  3. Alert if spend approaches limit")
print(f"  4. Module 6: PACT GovernanceEngine formalises cost envelopes")

print("\n✓ Exercise 1 complete — Delegate + SimpleQAAgent with cost governance")
