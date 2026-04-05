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

    cluster_data: str = InputField(description="Clustering results summary")
    question: str = InputField(description="Interpretation question")

    reasoning_steps: list[str] = OutputField(description="Step-by-step reasoning chain")
    interpretation: str = OutputField(description="Final interpretation")
    actionable_insights: list[str] = OutputField(description="Business actions to take")
    confidence: float = OutputField(description="Confidence 0-1 in the interpretation")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: ChainOfThoughtAgent
# ══════════════════════════════════════════════════════════════════════


async def cot_analysis():
    cot_agent = ChainOfThoughtAgent(
        signature=ClusterInterpretation,
        model=model,
        max_llm_cost_usd=2.0,
    )

    questions = [
        "Why did Cluster 2 form separately from Cluster 0? Both have low revenue.",
        "Which cluster represents the highest ROI opportunity for marketing spend?",
        "If we could only target one cluster for a retention campaign, which one and why?",
    ]

    results = []
    for q in questions:
        result = await cot_agent.run(
            cluster_data=cluster_summary,
            question=q,
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
    delegate = Delegate(model=model, max_llm_cost_usd=1.0)

    question = (
        "Why did Cluster 2 form separately from Cluster 0? Both have low revenue."
    )
    prompt = (
        f"{cluster_summary}\n\nAnswer directly (no step-by-step reasoning): {question}"
    )

    direct_answer = ""
    async for event in delegate.run(prompt):
        if hasattr(event, "text"):
            direct_answer += event.text

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
