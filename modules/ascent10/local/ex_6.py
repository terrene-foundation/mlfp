# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 6: AI Governance with PACT
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define an organization in YAML with D/T/R roles, compile
#   with GovernanceEngine, enforce operating envelopes, and verify access.
#
# TASKS:
#   1. Write YAML organization definition (departments, roles, agents)
#   2. Compile with GovernanceEngine
#   3. Define operating envelopes (budget, tool access, data clearance)
#   4. Test access control decisions
#   5. Generate governance report with decision explanations
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile

from dotenv import load_dotenv

from kailash_pact import GovernanceEngine

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Write YAML organization definition
# ══════════════════════════════════════════════════════════════════════

# TODO: Write a PACT organization YAML definition.
# Include: organization (name, jurisdiction), departments with agents (id, role, clearance),
# delegations with D/T/R chains and envelopes, and global operating_envelopes.
# Hint: Define at least 3 departments, 5+ agents, 4+ delegations with budget/tool/clearance limits.
org_yaml = """
____
"""

# Write YAML to temp file
org_yaml_path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
with open(org_yaml_path, "w") as f:
    f.write(org_yaml)

print(f"=== Organization Definition ===")
print(f"Organization: SG FinTech AI Division")
print(f"Departments: ML Engineering, Risk & Compliance, Customer Intelligence")
print(f"Agents: 5 (analyst, engineer, operator, 2 auditors, customer agent)")
print(f"Delegations: 4 D/T/R chains")
print(f"YAML written to: {org_yaml_path}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Compile with GovernanceEngine
# ══════════════════════════════════════════════════════════════════════


async def compile_org():
    # TODO: Create GovernanceEngine and compile the org YAML.
    # Hint: GovernanceEngine(), engine.compile_org(org_yaml_path)
    engine = ____
    org = ____

    print(f"\n=== Compiled Organization ===")
    print(f"Agents registered: {org.n_agents}")
    print(f"Delegations: {org.n_delegations}")
    print(f"Departments: {org.n_departments}")
    print(f"Compilation validates:")
    print(f"  - Every agent has a responsible delegation chain")
    print(f"  - No circular delegation (A→B→A)")
    print(f"  - Clearance levels are monotonically decreasing down chains")
    print(f"  - Budget envelopes don't exceed parent limits")

    return engine, org


engine, org = asyncio.run(compile_org())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Define operating envelopes
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Operating Envelopes ===")
print(f"Each agent operates within strict bounds:")
print(f"\n  data_analyst:")
print(f"    Budget: N/A (no delegation for spending)")
print(f"    Tools: read-only data access")
print(f"    Clearance: internal (no PII)")
print(f"\n  model_trainer:")
print(f"    Budget: $100/task")
print(f"    Tools: train, evaluate, read")
print(f"    Clearance: confidential")
print(f"\n  risk_assessor:")
print(f"    Budget: $200/task")
print(f"    Tools: read, audit, report")
print(f"    Clearance: restricted (highest)")
print(f"\n  customer_agent:")
print(f"    Budget: $5/task")
print(f"    Tools: answer, search")
print(f"    Clearance: public (lowest)")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Test access control decisions
# ══════════════════════════════════════════════════════════════════════


async def test_access():
    print(f"\n=== Access Control Tests ===")

    test_cases = [
        # (agent, resource, action, expected_allowed)
        ("model_trainer", "training_data", "read", True),
        ("model_trainer", "production_model", "deploy", False),  # Not in allowed_tools
        ("customer_agent", "customer_faq", "search_faq", True),
        ("customer_agent", "training_data", "read_data", False),  # Clearance too low
        ("risk_assessor", "model_audit_log", "audit_model", True),
        (
            "risk_assessor",
            "production_model",
            "deploy_model",
            False,
        ),  # Not in allowed_tools
        ("model_deployer", "production_model", "deploy_model", True),
        ("data_analyst", "restricted_data", "read", False),  # Clearance too low
    ]

    for agent_id, resource, action, expected in test_cases:
        # TODO: Check access using the governance engine.
        # Hint: engine.check_access(agent_id=..., resource=..., action=...)
        decision = ____
        status = "ALLOWED" if decision.allowed else "DENIED"
        match = "✓" if decision.allowed == expected else "✗ UNEXPECTED"
        print(f"  {match} {agent_id} → {action}({resource}): {status}")
        if not decision.allowed:
            print(f"      Reason: {decision.reason}")

    return True


asyncio.run(test_access())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Generate governance report
# ══════════════════════════════════════════════════════════════════════


async def generate_report():
    print(f"\n=== Governance Report ===")

    # TODO: Check a specific access decision and trace the decision chain.
    # Hint: engine.check_access(agent_id="model_trainer", resource="training_data", action="read")
    decision = ____

    print(f"Decision trace for model_trainer → read(training_data):")
    print(f"  1. Agent: model_trainer (role=engineer, clearance=confidential)")
    print(f"  2. Delegation: chief_ml_officer → model_training → model_trainer")
    print(f"  3. Envelope check:")
    print(f"     - Tool 'read_data' in allowed_tools: YES")
    print(f"     - Data clearance 'confidential' ≤ allowed 'confidential': YES")
    print(f"     - Budget consumed < $100 limit: YES")
    print(f"  4. Decision: {decision.allowed}")
    print(f"  5. Audit: logged to immutable audit chain")

    print(f"\nD/T/R Accountability Grammar:")
    print(f"  D (Delegator): chief_ml_officer — authorizes the task")
    print(f"  T (Task): model_training — the bounded scope of work")
    print(f"  R (Responsible): model_trainer — executes within envelope")
    print(f"  If model_trainer exceeds envelope → GovernanceEngine blocks")
    print(f"  If task fails → accountability traces to chief_ml_officer")

    print(f"\nRegulatory mapping:")
    print(f"  EU AI Act: operating envelopes satisfy Art. 9 (risk management)")
    print(f"  AI Verify: D/T/R chains satisfy accountability principle")
    print(f"  MAS TRM: audit trails satisfy record-keeping requirements")

    return decision


asyncio.run(generate_report())

print("\n✓ Exercise 6 complete — PACT governance with D/T/R and operating envelopes")
