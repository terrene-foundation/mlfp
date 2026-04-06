# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT05 — Exercise 1: LLM Fundamentals and Kaizen
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

    delegate = Delegate(
        model=model,
        max_llm_cost_usd=2.0,  # Hard budget cap — MANDATORY for all M5 exercises
    )

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

        response_text = ""
        async for event in delegate.run(prompt):
            if hasattr(event, "text"):
                response_text += event.text

        print(f"A{i+1}: {response_text[:300]}...")
        results.append({"question": question, "answer": response_text})

    return delegate, results


delegate, delegate_results = asyncio.run(delegate_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Custom Signature for structured output
# ══════════════════════════════════════════════════════════════════════


class CustomerSegmentAnalysis(Signature):
    """Analyse customer data and return structured segment recommendations."""

    data_context: str = InputField(description="Summary of the customer dataset")
    question: str = InputField(description="Analysis question to answer")

    segments: list[str] = OutputField(
        description="List of recommended customer segments"
    )
    reasoning: str = OutputField(
        description="Step-by-step reasoning for the segmentation"
    )
    confidence: float = OutputField(
        description="Confidence score 0-1 for the recommendation"
    )
    next_steps: list[str] = OutputField(description="Recommended next actions")


class ChurnPrediction(Signature):
    """Predict churn risk factors from customer data context."""

    data_context: str = InputField(description="Summary of the customer dataset")

    risk_factors: list[str] = OutputField(description="Top churn risk factors")
    key_metrics: list[str] = OutputField(
        description="Metrics to track for churn prediction"
    )
    model_recommendation: str = OutputField(description="Recommended ML approach")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: SimpleQAAgent with custom Signature
# ══════════════════════════════════════════════════════════════════════


async def structured_analysis():
    """Use SimpleQAAgent for structured output."""

    # Segment analysis
    segment_agent = SimpleQAAgent(
        signature=CustomerSegmentAnalysis,
        model=model,
        max_llm_cost_usd=1.0,
    )

    segment_result = await segment_agent.run(
        data_context=data_summary,
        question="What customer segments should we target for a premium loyalty programme?",
    )

    print(f"\n=== Structured Segment Analysis ===")
    print(f"Segments: {segment_result.segments}")
    print(f"Confidence: {segment_result.confidence}")
    print(f"Reasoning: {segment_result.reasoning[:200]}...")
    print(f"Next steps: {segment_result.next_steps}")

    # Churn analysis
    churn_agent = SimpleQAAgent(
        signature=ChurnPrediction,
        model=model,
        max_llm_cost_usd=1.0,
    )

    churn_result = await churn_agent.run(
        data_context=data_summary,
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
