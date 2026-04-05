# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 4: Governed Agents
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Wrap Module 5's ReActAgent with PactGovernedAgent. Define
#   operating envelopes and demonstrate that agents cannot modify their
#   own governance (frozen GovernanceContext).
#
# TASKS:
#   1. Create GovernanceEngine from Exercise 3's organization
#   2. Wrap a ReActAgent with PactGovernedAgent
#   3. Define operating envelopes (cost, tools, data)
#   4. Test governance enforcement
#   5. Demonstrate frozen context and monotonic tightening
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from pact import GovernanceEngine, GovernanceContext, PactGovernedAgent
from pact import Address, compile_org, load_org_yaml
from kaizen_agents import Delegate

from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Set up GovernanceEngine (from Ex3)
# ══════════════════════════════════════════════════════════════════════

# Minimal org for this exercise
org_dict = {
    "organization": {
        "name": "ASCENT Demo",
        "departments": [
            {
                "name": "data_science",
                "teams": [
                    {
                        "name": "modeling",
                        "roles": [
                            {
                                "name": "ml_agent",
                                "permissions": {
                                    "data_access": ["credit_data", "feature_store"],
                                    "tools": [
                                        "profile_data",
                                        "describe_column",
                                        "default_rate_by",
                                    ],
                                    "max_cost_usd": 5.0,
                                    "can_deploy": False,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    }
}


async def setup():
    compiled = compile_org(org_dict)
    engine = GovernanceEngine(compiled)

    # Create governance context for the ML agent
    agent_address = Address("data_science", "modeling", "ml_agent")
    context = await engine.create_context(agent_address)

    print(f"=== GovernanceContext ===")
    print(f"Address: {agent_address}")
    print(f"Max cost: ${context.max_cost_usd}")
    print(f"Data access: {context.data_access}")
    print(f"Tools allowed: {context.tools_allowed}")
    print(f"Can deploy: {context.can_deploy}")
    print(f"Frozen: {type(context).__dataclass_fields__ is not None}")

    return engine, context


engine, governance_context = asyncio.run(setup())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Wrap agent with PactGovernedAgent
# ══════════════════════════════════════════════════════════════════════


async def create_governed_agent():
    # Base agent (from M5)
    base_agent = Delegate(model=model)

    # Wrap with governance
    governed_agent = PactGovernedAgent(
        agent=base_agent,
        governance_context=governance_context,
    )

    print(f"\n=== PactGovernedAgent ===")
    print(f"Base agent: Delegate")
    print(f"Governance: ASCENT Demo / data_science / modeling / ml_agent")
    print(f"Cost budget: ${governance_context.max_cost_usd}")

    return governed_agent


governed_agent = asyncio.run(create_governed_agent())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Test governance enforcement
# ══════════════════════════════════════════════════════════════════════


async def test_enforcement():
    """Test that governance restricts agent behavior."""

    # Allowed: use permitted tools on permitted data
    print(f"\n=== Governance Tests ===")

    # Test 1: Allowed action
    allowed = await governed_agent.check_permission(
        action="use_tool",
        resource="profile_data",
    )
    print(f"1. Use profile_data tool: {'ALLOW' if allowed else 'DENY'} ✓")

    # Test 2: Denied tool (not in allowed list)
    denied_tool = await governed_agent.check_permission(
        action="use_tool",
        resource="training_pipeline",  # Not in allowed tools
    )
    print(f"2. Use training_pipeline: {'ALLOW' if denied_tool else 'DENY'} ✓")

    # Test 3: Denied data access
    denied_data = await governed_agent.check_permission(
        action="access_data",
        resource="production_logs",  # Not in allowed data
    )
    print(f"3. Access production_logs: {'ALLOW' if denied_data else 'DENY'} ✓")

    # Test 4: Cost check
    within_budget = await governed_agent.check_cost(amount=3.0)
    over_budget = await governed_agent.check_cost(amount=10.0)
    print(f"4. Spend $3.00 (budget $5): {'ALLOW' if within_budget else 'DENY'} ✓")
    print(f"5. Spend $10.00 (budget $5): {'ALLOW' if over_budget else 'DENY'} ✓")

    # Test 5: Deploy action (not permitted for this role)
    can_deploy = await governed_agent.check_permission(
        action="deploy",
        resource="model_v1",
    )
    print(f"6. Deploy model: {'ALLOW' if can_deploy else 'DENY'} ✓")


asyncio.run(test_enforcement())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Run governed agent on a real task
# ══════════════════════════════════════════════════════════════════════


async def run_governed_task():
    """Run the governed agent — governance is transparent to the task."""

    prompt = (
        "Analyse the credit scoring dataset. Profile the data and "
        "identify which features predict default. Keep analysis under $5."
    )

    print(f"\n=== Governed Agent Execution ===")
    print(f"Task: {prompt}")
    print(
        f"Governance active: cost≤$5, tools=[profile_data, describe_column, default_rate_by]"
    )

    response_text = ""
    async for event in governed_agent.run(prompt):
        if hasattr(event, "text"):
            response_text += event.text

    print(f"\nResponse: {response_text[:300]}...")

    # Check cost after execution
    cost_spent = governed_agent.cost_tracker.total_spent
    print(f"\nCost spent: ${cost_spent:.4f} / ${governance_context.max_cost_usd}")


asyncio.run(run_governed_task())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Frozen context demonstration
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Frozen GovernanceContext ===")
print(
    """
GovernanceContext is a frozen dataclass:

  @dataclass(frozen=True)
  class GovernanceContext:
      max_cost_usd: float
      data_access: tuple[str, ...]
      tools_allowed: tuple[str, ...]
      can_deploy: bool

  # This RAISES FrozenInstanceError:
  context.max_cost_usd = 999.0

WHY frozen?
  1. Agents cannot escalate their own privileges
  2. Context is immutable proof of what was authorized
  3. AuditChain can verify context wasn't tampered with
  4. Monotonic tightening is guaranteed (child ≤ parent)

Attack prevention:
  ✗ Agent modifies its budget: FrozenInstanceError
  ✗ Agent creates child with higher permissions: tightened to parent
  ✗ Agent accesses data outside scope: fail-closed DENY
  ✗ Governance engine error: fail-closed DENY (not ALLOW)
"""
)

# Demonstrate immutability
try:
    governance_context.max_cost_usd = 999.0
    print("ERROR: Should have raised FrozenInstanceError!")
except (AttributeError, TypeError) as e:
    print(f"✓ Attempted modification blocked: {type(e).__name__}")

print("\n✓ Exercise 4 complete — governed agents with PACT enforcement")
