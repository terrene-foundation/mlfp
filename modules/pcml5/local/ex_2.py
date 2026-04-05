# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT5 — Exercise 2: ChainOfThoughtAgent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a ChainOfThoughtAgent that reasons step-by-step about
#   Module 4 clustering results. Compare CoT vs direct answering.
#
# TASKS:
#   1. Build ChainOfThoughtAgent for cluster interpretation
#   2. Provide clustering results as context
#   3. Agent explains WHY segments formed, not just WHAT
#   4. Compare CoT vs direct answer quality
#   5. Evaluate reasoning chain quality
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import Signature, InputField, OutputField
from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.chain_of_thought import ChainOfThoughtAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Prepare clustering context from M4 ───────────────────────────────

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

# Simulate clustering results (from M4 Exercise 1)
cluster_summary = """
Clustering Results (K-means, K=4, Silhouette=0.42):

Cluster 0 (n=12,500 — "Casual Browsers"):
  avg_revenue: $45, avg_orders: 1.2, avg_session_duration: 3.5min
  ▼ revenue, ▼ orders, ─ sessions

Cluster 1 (n=8,200 — "Power Shoppers"):
  avg_revenue: $890, avg_orders: 15.3, avg_session_duration: 12.1min
  ▲ revenue, ▲ orders, ▲ sessions

Cluster 2 (n=18,000 — "Window Shoppers"):
  avg_revenue: $120, avg_orders: 3.1, avg_session_duration: 22.5min
  ─ revenue, ─ orders, ▲ sessions (high browse, low buy)

Cluster 3 (n=11,300 — "Bargain Hunters"):
  avg_revenue: $210, avg_orders: 8.7, avg_session_duration: 8.3min
  ─ revenue, ▲ orders (high frequency, low basket)
"""


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define Signature for cluster interpretation
# ══════════════════════════════════════════════════════════════════════


class ClusterInterpretation(Signature):
    """Interpret clustering results with step-by-step reasoning."""

    # TODO: Define two InputFields: cluster_data (clustering results summary)
    #       and question (interpretation question)
    # Hint: field_name: type = InputField(description="____")
    cluster_data: str = ____
    question: str = ____

    # TODO: Define four OutputFields: reasoning_steps (list[str]),
    #       interpretation (str), actionable_insights (list[str]), confidence (float)
    # Hint: field_name: type = OutputField(description="____")
    reasoning_steps: list[str] = ____
    interpretation: str = ____
    actionable_insights: list[str] = ____
    confidence: float = ____


# ══════════════════════════════════════════════════════════════════════
# TASK 2: ChainOfThoughtAgent
# ══════════════════════════════════════════════════════════════════════


async def cot_analysis():
    # TODO: Create a ChainOfThoughtAgent with ClusterInterpretation signature
    # Hint: ChainOfThoughtAgent(signature=____, model=____, max_llm_cost_usd=____)
    cot_agent = ____

    questions = [
        "Why did Cluster 2 form separately from Cluster 0? Both have low revenue.",
        "Which cluster represents the highest ROI opportunity for marketing spend?",
        "If we could only target one cluster for a retention campaign, which one and why?",
    ]

    results = []
    for q in questions:
        # TODO: Run the cot_agent passing cluster_data and question keyword arguments
        # Hint: await cot_agent.run(cluster_data=____, question=____)
        result = await cot_agent.run(
            cluster_data=____,
            question=____,
        )

        print(f"\n=== Q: {q} ===")
        print(f"Reasoning steps ({len(result.reasoning_steps)}):")
        for i, step in enumerate(result.reasoning_steps):
            print(f"  {i+1}. {step}")
        print(f"\nInterpretation: {result.interpretation}")
        print(f"Confidence: {result.confidence}")
        print(f"Actions: {result.actionable_insights}")
        results.append(result)

    return results


cot_results = asyncio.run(cot_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Compare with direct answering (no CoT)
# ══════════════════════════════════════════════════════════════════════


async def direct_analysis():
    # TODO: Create a Delegate for direct (non-CoT) answering with budget $1.00
    # Hint: Delegate(model=____, max_llm_cost_usd=____)
    delegate = ____

    question = (
        "Why did Cluster 2 form separately from Cluster 0? Both have low revenue."
    )
    prompt = (
        f"{cluster_summary}\n\nAnswer directly (no step-by-step reasoning): {question}"
    )

    # TODO: Stream the Delegate response into direct_answer
    # Hint: async for event in delegate.run(____):
    #           if hasattr(event, "text"):
    #               direct_answer += event.text
    direct_answer = ""
    # ... your streaming loop here ...

    print(f"\n=== Direct Answer (no CoT) ===")
    print(direct_answer[:500])

    return direct_answer


direct_answer = asyncio.run(direct_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Evaluate reasoning quality
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== CoT vs Direct Comparison ===")
print(f"CoT reasoning steps: {len(cot_results[0].reasoning_steps)}")
print(f"CoT has structured output: segments, confidence, actions")
print(f"Direct is free-form text: no guaranteed structure")
print()
print("Key insight: CoT forces the model to SHOW its work.")
print("This is critical for:")
print("  1. Debugging: when the agent is wrong, you can see WHERE")
print("  2. Trust: stakeholders can verify the reasoning chain")
print("  3. Reproducibility: reasoning steps are logged and auditable")
print()
print("In Module 6, CoT reasoning chains become governance artifacts —")
print("PACT's AuditChain records every step for regulatory compliance.")

print("\n✓ Exercise 2 complete — ChainOfThoughtAgent with reasoning evaluation")
