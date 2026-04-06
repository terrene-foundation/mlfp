# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 7: Governed Agents with PactGovernedAgent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Wrap agents with PactGovernedAgent — budget cascading, tool
#   restrictions, clearance levels, audit chain — for compliant ML ops.
#
# TASKS:
#   1. Build base ReActAgent for data analysis
#   2. Wrap with PactGovernedAgent (role, budget, tools)
#   3. Test budget enforcement (request exceeding budget)
#   4. Test tool restriction (request using forbidden tool)
#   5. Extract and analyze audit trail
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kaizen_agents import Delegate
from kaizen_agents.agents.specialized.react import ReActAgent
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Build base ReActAgent for data analysis
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
reports = loader.load("ascent10", "sg_company_reports.parquet")

print(f"=== Company Reports: {reports.height} documents ===")


# Define tools for the agent
async def read_data(dataset: str) -> str:
    """Read a dataset and return summary statistics."""
    df = loader.load("ascent10", f"{dataset}.parquet")
    return f"Dataset: {dataset}, Shape: {df.shape}, Columns: {df.columns}"


async def analyze_text(text: str) -> str:
    """Analyze text content and return key themes."""
    words = text.lower().split()
    return f"Text length: {len(words)} words, Sample: {' '.join(words[:20])}..."


async def deploy_model(model_name: str) -> str:
    """Deploy a model to production (restricted operation)."""
    return f"Model {model_name} deployed to production endpoint"


async def delete_data(dataset: str) -> str:
    """Delete a dataset (dangerous operation)."""
    return f"Dataset {dataset} deleted permanently"


tools = [read_data, analyze_text, deploy_model, delete_data]

base_agent = ReActAgent(
    model=model,
    tools=tools,
    max_llm_cost_usd=10.0,  # Unrestricted budget on base agent
)

print(f"Base agent has {len(tools)} tools (including dangerous ones)")
print(f"Without governance: agent can deploy models, delete data, spend $10")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Wrap with PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════

engine = GovernanceEngine()

governed_analyst = PactGovernedAgent(
    agent=base_agent,
    governance_engine=engine,
    role="analyst",
    max_budget_usd=2.0,  # Tight budget
    allowed_tools=["read_data", "analyze_text"],  # Read-only
    clearance_level="internal",  # No access to restricted data
)

governed_deployer = PactGovernedAgent(
    agent=base_agent,
    governance_engine=engine,
    role="operator",
    max_budget_usd=5.0,
    allowed_tools=["read_data", "deploy_model"],  # Can deploy but not delete
    clearance_level="confidential",
)

print(f"\n=== Governed Agents ===")
print(f"governed_analyst:")
print(f"  Budget: $2.00 (vs $10 base)")
print(f"  Tools: read_data, analyze_text (no deploy, no delete)")
print(f"  Clearance: internal")
print(f"\ngoverned_deployer:")
print(f"  Budget: $5.00")
print(f"  Tools: read_data, deploy_model (no delete)")
print(f"  Clearance: confidential")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Test budget enforcement
# ══════════════════════════════════════════════════════════════════════


async def test_budget():
    print(f"\n=== Budget Enforcement ===")

    # Normal request within budget
    result = await governed_analyst.run(
        "Read the sg_company_reports dataset and summarize the key themes."
    )
    print(f"Normal request: SUCCESS")
    print(f"  Result: {str(result)[:200]}...")

    # Check remaining budget
    budget_info = governed_analyst.get_budget_status()
    print(f"  Budget spent: ${budget_info.get('spent', 0):.4f}")
    print(f"  Budget remaining: ${budget_info.get('remaining', 0):.4f}")

    # Expensive request that exceeds budget
    print(f"\nAttempting expensive multi-step analysis (likely exceeds $2 budget)...")
    try:
        result = await governed_analyst.run(
            "Perform a comprehensive analysis of all documents: "
            "read each one, analyze themes, cross-reference findings, "
            "and generate a 500-word executive summary."
        )
        print(f"  Result: {str(result)[:200]}...")
    except Exception as e:
        print(f"  BLOCKED: {e}")
        print(f"  Budget enforcement prevented overspending.")

    return budget_info


budget_info = asyncio.run(test_budget())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Test tool restriction
# ══════════════════════════════════════════════════════════════════════


async def test_tool_restrictions():
    print(f"\n=== Tool Restriction Enforcement ===")

    # Analyst tries to deploy (not in allowed_tools)
    print(f"1. Analyst attempts to deploy model:")
    try:
        result = await governed_analyst.run("Deploy the sentiment model to production.")
        print(f"   Result: {str(result)[:200]}...")
    except Exception as e:
        print(f"   BLOCKED: {e}")
        print(f"   Analyst cannot deploy — not in allowed_tools")

    # Analyst tries to delete (not in allowed_tools)
    print(f"\n2. Analyst attempts to delete data:")
    try:
        result = await governed_analyst.run("Delete the sg_company_reports dataset.")
        print(f"   Result: {str(result)[:200]}...")
    except Exception as e:
        print(f"   BLOCKED: {e}")
        print(f"   Analyst cannot delete — not in allowed_tools")

    # Deployer can deploy but not delete
    print(f"\n3. Deployer attempts to deploy:")
    result = await governed_deployer.run("Deploy the latest model to production.")
    print(f"   ALLOWED: {str(result)[:200]}...")

    print(f"\n4. Deployer attempts to delete:")
    try:
        result = await governed_deployer.run("Delete the old training dataset.")
        print(f"   Result: {str(result)[:200]}...")
    except Exception as e:
        print(f"   BLOCKED: {e}")
        print(f"   Deployer cannot delete — not in allowed_tools")


asyncio.run(test_tool_restrictions())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Extract and analyze audit trail
# ══════════════════════════════════════════════════════════════════════


async def analyze_audit():
    print(f"\n=== Audit Trail ===")

    analyst_audit = governed_analyst.get_audit_trail()
    deployer_audit = governed_deployer.get_audit_trail()

    print(f"Analyst audit entries: {len(analyst_audit)}")
    for entry in analyst_audit:
        print(
            f"  [{entry.get('timestamp', '?')}] {entry.get('action', '?')}: "
            f"{entry.get('status', '?')} — {entry.get('reason', '')[:80]}"
        )

    print(f"\nDeployer audit entries: {len(deployer_audit)}")
    for entry in deployer_audit:
        print(
            f"  [{entry.get('timestamp', '?')}] {entry.get('action', '?')}: "
            f"{entry.get('status', '?')} — {entry.get('reason', '')[:80]}"
        )

    print(f"\n=== Audit Trail Analysis ===")
    total_blocked = sum(
        1 for e in analyst_audit + deployer_audit if e.get("status") == "blocked"
    )
    total_allowed = sum(
        1 for e in analyst_audit + deployer_audit if e.get("status") == "allowed"
    )
    print(f"Total actions: {len(analyst_audit) + len(deployer_audit)}")
    print(f"Allowed: {total_allowed}")
    print(f"Blocked: {total_blocked}")
    print(f"\nEvery action — allowed or blocked — is logged immutably.")
    print(f"This satisfies EU AI Act Art. 12 (record-keeping) and")
    print(f"MAS TRM 7.5 (audit trail requirements).")

    return analyst_audit, deployer_audit


analyst_audit, deployer_audit = asyncio.run(analyze_audit())

print(f"\n=== PactGovernedAgent Summary ===")
print(f"Governance layer adds three guarantees:")
print(f"  1. Budget cascade: agent cannot exceed allocated budget")
print(f"  2. Tool restriction: agent can only use approved tools")
print(f"  3. Clearance levels: agent cannot access data above clearance")
print(f"All enforcement is immutable, auditable, and fail-closed.")

print("\n✓ Exercise 7 complete — PactGovernedAgent with budget, tools, and audit")
