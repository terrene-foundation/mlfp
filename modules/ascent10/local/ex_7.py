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
from dotenv import load_dotenv

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

# TODO: Create a ReActAgent with all tools and an unrestricted budget.
# Hint: ReActAgent(model=model, tools=tools, max_llm_cost_usd=10.0)
base_agent = ____

print(f"Base agent has {len(tools)} tools (including dangerous ones)")
print(f"Without governance: agent can deploy models, delete data, spend $10")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Wrap with PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════

# TODO: Create GovernanceEngine.
# Hint: GovernanceEngine()
engine = ____

# TODO: Create governed analyst with tight budget and read-only tools.
# Hint: PactGovernedAgent(agent=..., governance_engine=..., role=...,
#        max_budget_usd=..., allowed_tools=..., clearance_level=...)
governed_analyst = ____

# TODO: Create governed deployer with deploy permissions but no delete.
# Hint: PactGovernedAgent(agent=..., role="operator", allowed_tools=["read_data", "deploy_model"], ...)
governed_deployer = ____

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

    # TODO: Run a normal request within budget.
    # Hint: await governed_analyst.run("Read the sg_company_reports dataset...")
    result = ____
    print(f"Normal request: SUCCESS")
    print(f"  Result: {str(result)[:200]}...")

    # TODO: Check remaining budget.
    # Hint: governed_analyst.get_budget_status()
    budget_info = ____
    print(f"  Budget spent: ${budget_info.get('spent', 0):.4f}")
    print(f"  Budget remaining: ${budget_info.get('remaining', 0):.4f}")

    # TODO: Attempt an expensive request that exceeds budget.
    # Hint: try/except around governed_analyst.run(...) with a complex prompt.
    print(f"\nAttempting expensive multi-step analysis (likely exceeds $2 budget)...")
    try:
        result = ____
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

    # TODO: Test analyst trying to deploy (should be blocked).
    # Hint: try/except around governed_analyst.run("Deploy the sentiment model...")
    print(f"1. Analyst attempts to deploy model:")
    try:
        result = ____
        print(f"   Result: {str(result)[:200]}...")
    except Exception as e:
        print(f"   BLOCKED: {e}")
        print(f"   Analyst cannot deploy — not in allowed_tools")

    # TODO: Test analyst trying to delete (should be blocked).
    print(f"\n2. Analyst attempts to delete data:")
    try:
        result = ____
        print(f"   Result: {str(result)[:200]}...")
    except Exception as e:
        print(f"   BLOCKED: {e}")
        print(f"   Analyst cannot delete — not in allowed_tools")

    # TODO: Test deployer deploying (should be allowed).
    print(f"\n3. Deployer attempts to deploy:")
    result = ____
    print(f"   ALLOWED: {str(result)[:200]}...")

    # TODO: Test deployer deleting (should be blocked).
    print(f"\n4. Deployer attempts to delete:")
    try:
        result = ____
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

    # TODO: Extract audit trails from both governed agents.
    # Hint: governed_analyst.get_audit_trail(), governed_deployer.get_audit_trail()
    analyst_audit = ____
    deployer_audit = ____

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

    # TODO: Count blocked and allowed actions across both audit trails.
    # Hint: Sum entries where status == "blocked" or "allowed".
    total_blocked = ____
    total_allowed = ____
    print(f"\n=== Audit Trail Analysis ===")
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
