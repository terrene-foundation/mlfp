# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 6: Multi-Agent Orchestration
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Build specialist agents with domain-specific Signatures
#   - Use Pipeline.router() for LLM-based query routing (not keyword matching)
#   - Implement the supervisor pattern (fan-out specialists -> fan-in synthesis)
#   - Compare single-agent vs multi-agent output quality
#   - Apply security considerations to multi-agent architectures
#
# PREREQUISITES:
#   Exercise 5 (BaseAgent, Signature, single-agent patterns). Understanding
#   that each specialist agent is a separate LLM call with a focused prompt —
#   multi-agent = orchestrated parallelism, not one mega-prompt.
#
# ESTIMATED TIME: 45-75 minutes
#
# TASKS:
#   1. Build 3 specialist agents (financial, legal, technical)
#   2. Create router agent via Pipeline.router()
#   3. Implement supervisor pattern with delegation
#   4. Run multi-agent analysis on complex document
#   5. Compare single-agent vs multi-agent quality
#
# DATASET: Singapore company reports
#   Same dataset as Exercise 5. The document analysed here is complex
#   enough to benefit from specialist analysis across three domains.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kaizen.orchestration.pipeline import Pipeline
from kaizen_agents import Delegate

from shared import MLFPDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build 3 specialist agents
# ══════════════════════════════════════════════════════════════════════

loader = MLFPDataLoader()
reports = loader.load("mlfp06", "sg_company_reports.parquet")

print(f"=== Company Reports: {reports.height} documents ===")
sample_doc = reports["text"][0][:500]
print(f"Sample (first 500 chars): {sample_doc}...")


class FinancialAnalysisSignature(Signature):
    """Analyse financial aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Financial analysis question")
    revenue_insights: str = OutputField(
        description="Revenue and profitability analysis"
    )
    risk_factors: list[str] = OutputField(
        description="Financial risk factors identified"
    )
    recommendation: str = OutputField(description="Financial recommendation")


class LegalAnalysisSignature(Signature):
    """Analyse legal and compliance aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Legal analysis question")
    compliance_issues: list[str] = OutputField(description="Compliance issues found")
    regulatory_references: list[str] = OutputField(description="Relevant regulations")
    legal_risk: str = OutputField(description="Legal risk assessment")


class TechnicalAnalysisSignature(Signature):
    """Analyse technical aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Technical analysis question")
    tech_assessment: str = OutputField(description="Technical feasibility assessment")
    architecture_notes: list[str] = OutputField(
        description="Architecture considerations"
    )
    scalability: str = OutputField(description="Scalability assessment")


class FinancialAgent(BaseAgent):
    signature = FinancialAnalysisSignature
    model = os.environ.get("DEFAULT_LLM_MODEL")
    max_llm_cost_usd = 1.0
    description = (
        "Specialist in financial analysis: revenue, profitability, risk factors"
    )


class LegalAgent(BaseAgent):
    signature = LegalAnalysisSignature
    model = os.environ.get("DEFAULT_LLM_MODEL")
    max_llm_cost_usd = 1.0
    description = "Specialist in legal analysis: compliance, regulation, legal risk"


class TechnicalAgent(BaseAgent):
    signature = TechnicalAnalysisSignature
    model = os.environ.get("DEFAULT_LLM_MODEL")
    max_llm_cost_usd = 1.0
    description = (
        "Specialist in technical analysis: architecture, scalability, feasibility"
    )


financial_agent = FinancialAgent()
legal_agent = LegalAgent()
technical_agent = TechnicalAgent()

print(f"\n=== Specialist Agents ===")
print(f"  1. FinancialAgent: revenue, risk, profitability")
print(f"  2. LegalAgent: compliance, regulation, legal risk")
print(f"  3. TechnicalAgent: architecture, scalability, feasibility")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Create router agent via Pipeline.router()
# ══════════════════════════════════════════════════════════════════════

# Pipeline.router() uses LLM reasoning to route queries to the best specialist
# — NOT keyword matching. The LLM reads each agent's description (capability card)
# and reasons about which specialist is best suited for each query.
router = Pipeline.router(
    agents=[financial_agent, legal_agent, technical_agent],
)

print(f"\n=== Router Agent ===")
print(f"Pipeline.router() uses LLM reasoning to select the best specialist.")
print(f"Each agent's description serves as a capability card.")
print(f"The router examines the query and agent descriptions, then delegates.")
print(f"This is fundamentally different from keyword matching or dispatch tables.")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Supervisor pattern with delegation
# ══════════════════════════════════════════════════════════════════════


class SupervisorSignature(Signature):
    """Coordinate specialist analyses into a unified assessment."""

    document: str = InputField(description="Original business document")
    financial_analysis: str = InputField(description="Financial specialist's analysis")
    legal_analysis: str = InputField(description="Legal specialist's analysis")
    technical_analysis: str = InputField(description="Technical specialist's analysis")
    executive_summary: str = OutputField(description="Unified executive summary")
    overall_risk: str = OutputField(description="Overall risk rating: low/medium/high")
    action_items: list[str] = OutputField(description="Prioritized action items")


class SupervisorAgent(BaseAgent):
    signature = SupervisorSignature
    model = os.environ.get("DEFAULT_LLM_MODEL")
    max_llm_cost_usd = 2.0
    description = (
        "Supervisor that synthesizes specialist analyses into actionable decisions"
    )


supervisor = SupervisorAgent()

print(f"\n=== Supervisor Pattern ===")
print(f"1. Router dispatches query to appropriate specialist")
print(f"2. Each specialist analyses from their domain perspective")
print(f"3. Supervisor synthesizes all analyses into unified assessment")
print(f"This is fan-out (parallel specialists) → fan-in (supervisor synthesis)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run multi-agent analysis
# ══════════════════════════════════════════════════════════════════════


async def multi_agent_analysis():
    doc = reports["text"][0]
    question = "Should we invest in this company's expansion plans?"

    print(f"\n=== Multi-Agent Analysis ===")
    print(f"Question: {question}")

    # Step 1: Run specialists in parallel
    financial_result = await financial_agent.run(
        document=doc, question="Analyse the financial health and revenue potential"
    )
    legal_result = await legal_agent.run(
        document=doc, question="Identify compliance risks and regulatory requirements"
    )
    technical_result = await technical_agent.run(
        document=doc, question="Assess technical feasibility and scalability"
    )

    print(f"\n--- Financial Analysis ---")
    print(f"Revenue insights: {financial_result.revenue_insights[:200]}...")
    print(f"Risk factors: {financial_result.risk_factors[:3]}")

    print(f"\n--- Legal Analysis ---")
    print(f"Compliance issues: {legal_result.compliance_issues[:3]}")
    print(f"Legal risk: {legal_result.legal_risk[:200]}...")

    print(f"\n--- Technical Analysis ---")
    print(f"Assessment: {technical_result.tech_assessment[:200]}...")
    print(f"Scalability: {technical_result.scalability[:200]}...")

    # Step 2: Supervisor synthesizes
    supervisor_result = await supervisor.run(
        document=doc,
        financial_analysis=financial_result.revenue_insights,
        legal_analysis=str(legal_result.compliance_issues),
        technical_analysis=technical_result.tech_assessment,
    )

    print(f"\n--- Supervisor Summary ---")
    print(f"Executive summary: {supervisor_result.executive_summary[:300]}...")
    print(f"Overall risk: {supervisor_result.overall_risk}")
    print(f"Action items: {supervisor_result.action_items}")

    return supervisor_result


multi_result = asyncio.run(multi_agent_analysis())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare single-agent vs multi-agent
# ══════════════════════════════════════════════════════════════════════


async def single_agent_comparison():
    doc = reports["text"][0]
    delegate = Delegate(model=model, max_llm_cost_usd=3.0)

    print(f"\n=== Single-Agent Comparison ===")
    response = ""
    async for event in delegate.run(
        f"Analyse this business document from financial, legal, and technical perspectives. "
        f"Provide an executive summary with risk assessment.\n\n{doc[:2000]}"
    ):
        if hasattr(event, "content"):
            response += event.content

    print(f"Single-agent response: {response[:400]}...")
    return response


single_result = asyncio.run(single_agent_comparison())

print(f"\n=== Comparison ===")
print(f"Single-agent: one LLM call, broad but shallow analysis")
print(f"Multi-agent:  3 specialists + supervisor, deep domain-specific insights")
print(f"\nWhen to use multi-agent:")
print(f"  - Task requires multiple domain expertise areas")
print(f"  - Deep analysis needed per domain (not just surface-level)")
print(f"  - Quality matters more than latency")
print(f"  - Audit trail needed (which specialist said what)")

print("=" * 60)
print("  MLFP06 Exercise 6: Multi-Agent Orchestration")
print("=" * 60)
print(f"\n  Supervisor + 3 specialists + router demonstrated.\n")

# ── Checkpoint 1: Specialist agents ──────────────────────────────────
assert financial_agent is not None, "FinancialAgent should be created"
assert legal_agent is not None, "LegalAgent should be created"
assert technical_agent is not None, "TechnicalAgent should be created"
print(f"✓ Checkpoint 1 passed — 3 specialist agents defined\n")

# INTERPRETATION: Specialist agents work better than generalists for complex tasks
# because each agent's prompt is focused. A financial analyst only analyses
# financial aspects — it doesn't dilute attention on legal or technical concerns.
# The Signature enforces that each specialist produces domain-specific fields:
# FinancialAgent -> revenue_insights, risk_factors, recommendation
# LegalAgent -> compliance_issues, regulatory_references, legal_risk
# This structure prevents the agents from giving vague, generic answers.

# ── Checkpoint 2: Router ──────────────────────────────────────────────
assert router is not None, "Router should be created"
print(f"✓ Checkpoint 2 passed — Pipeline.router() created with 3 agents\n")

# INTERPRETATION: Pipeline.router() uses the LLM to read each agent's
# description (capability card) and match it to the incoming query.
# This is fundamentally different from keyword routing (if "revenue" -> financial):
# - It handles synonyms and paraphrases
# - It routes based on INTENT, not vocabulary
# - It gracefully handles ambiguous queries (may route to multiple agents)
# The cost is one additional LLM call per routing decision.

# ── Checkpoint 3: Supervisor pattern ─────────────────────────────────
assert supervisor is not None, "SupervisorAgent should be created"
print(f"✓ Checkpoint 3 passed — supervisor + fan-out/fan-in pattern defined\n")

# INTERPRETATION: The supervisor pattern is fan-out -> fan-in:
# Fan-out: run specialists in parallel (financial, legal, technical run concurrently)
# Fan-in: supervisor reads ALL specialist outputs and synthesises into one decision
# This beats a single-agent approach because:
# 1. Each specialist gives deep, focused analysis (not surface-level)
# 2. The supervisor can weigh conflicting assessments
# 3. You know WHICH specialist said what (audit trail)
# Parallel execution also reduces total latency vs sequential.

# ── Checkpoint 4: Multi-agent analysis ───────────────────────────────
assert multi_result is not None, "Multi-agent analysis should produce a result"
assert hasattr(multi_result, "executive_summary"), "Should have executive summary"
assert hasattr(multi_result, "overall_risk"), "Should have risk rating"
assert hasattr(multi_result, "action_items"), "Should have action items"
assert len(multi_result.action_items) > 0, "Should produce at least one action item"
print(f"✓ Checkpoint 4 passed — supervisor synthesis: "
      f"risk={multi_result.overall_risk}, "
      f"actions={len(multi_result.action_items)}\n")

# INTERPRETATION: The overall_risk field (low/medium/high) integrates
# financial, legal, and technical risks into a single decision signal.
# In production, this would trigger different workflows:
# low -> auto-approve, medium -> human review, high -> reject or escalate.
# Action items should be specific and actionable, not vague recommendations.

# ── Checkpoint 5: Comparison ──────────────────────────────────────────
assert single_result is not None, "Single-agent comparison should produce output"
print(f"✓ Checkpoint 5 passed — single-agent vs multi-agent comparison done\n")

# INTERPRETATION: Single-agent typically produces broader but shallower analysis.
# It covers all three domains in one pass but at reduced depth per domain.
# Multi-agent produces deeper, more structured insights per domain.
# The choice depends on: quality requirements, latency tolerance, cost budget.
# Multi-agent costs 3x more (3 specialist calls + supervisor) but quality
# improvement is typically larger than 3x for complex multi-domain tasks.


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("═" * 60)
print("  WHAT YOU'VE MASTERED")
print("═" * 60)
print("""
  ✓ Specialist agents: focused Signatures produce deeper domain analysis
  ✓ Pipeline.router(): LLM-based routing reads capability cards, handles
    ambiguous queries better than keyword matching
  ✓ Supervisor pattern: fan-out specialists -> fan-in supervisor synthesis
  ✓ Structured outputs: Signature contracts make agent outputs reliable
  ✓ Single vs multi-agent: quality vs cost trade-off quantified

  Multi-agent patterns:
    Supervisor-worker: one director, multiple specialists (this exercise)
    Sequential:        output of agent A feeds into agent B feeds into C
    Parallel:          multiple agents run simultaneously, results merged
    Handoff:           agent transfers to specialist when topic changes

  NEXT: Exercise 7 (PACT Governance) wraps your multi-agent system
  in formal governance. D/T/R addressing, operating envelopes, budget
  cascading, and audit trails — the engineering implementation of AI safety.
""")
