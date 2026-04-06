# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT09 — Exercise 6: Multi-Agent Orchestration
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a multi-agent system with delegation — router,
#   specialist agents, supervisor pattern — for complex document analysis.
#
# TASKS:
#   1. Build 3 specialist agents (financial, legal, technical)
#   2. Create router agent via Pipeline.router()
#   3. Implement supervisor pattern with delegation
#   4. Run multi-agent analysis on complex document
#   5. Compare single-agent vs multi-agent quality
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl
from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kaizen.orchestration.pipeline import Pipeline
from kaizen_agents import Delegate

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build 3 specialist agents
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
reports = loader.load("ascent09", "sg_company_reports.parquet")

print(f"=== Company Reports: {reports.height} documents ===")
sample_doc = reports["text"][0][:500]
print(f"Sample (first 500 chars): {sample_doc}...")


class FinancialAnalysisSignature(Signature):
    """Analyse financial aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Financial analysis question")
    # TODO: Define OutputField for revenue_insights (revenue and profitability analysis)
    revenue_insights: str = ____
    # TODO: Define OutputField for risk_factors (list of financial risk factors)
    risk_factors: list[str] = ____
    recommendation: str = OutputField(description="Financial recommendation")


class LegalAnalysisSignature(Signature):
    """Analyse legal and compliance aspects of a business document."""

    document: str = InputField(description="Business document text")
    question: str = InputField(description="Legal analysis question")
    # TODO: Define OutputField for compliance_issues (list of issues found)
    compliance_issues: list[str] = ____
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
# TODO: Create a Pipeline.router() with the three specialist agents
router = Pipeline.router(____)

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
    # TODO: Define OutputField for executive_summary
    executive_summary: str = ____
    # TODO: Define OutputField for overall_risk (low/medium/high)
    overall_risk: str = ____
    # TODO: Define OutputField for action_items (prioritized list)
    action_items: list[str] = ____


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
    # TODO: Run supervisor with document, financial, legal, and technical analyses
    supervisor_result = await supervisor.run(____)

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

print("\n✓ Exercise 6 complete — multi-agent orchestration with Pipeline.router()")
