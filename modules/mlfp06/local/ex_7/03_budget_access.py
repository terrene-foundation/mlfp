# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.3: Budget Cascading & Access Control Decisions
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Cascade budget from parent agents to children
#   - Detect overspend attempts before they reach the LLM
#   - Exercise check_access() on allow AND deny paths
#   - Understand why deny-path coverage is non-negotiable
#
# PREREQUISITES: 02_envelopes.py
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Allocate & consume budget across a 3-agent hierarchy
#   2. Attempt an overspend and verify it is denied
#   3. Visualise spend vs. allocation
#   4. Run 10 check_access() tests — both allow and deny
#   5. Apply — MAS TRM cost controls
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from shared.mlfp06.ex_7 import BudgetTracker, compile_governance

engine, org = compile_governance()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Budget as a Governance Primitive
# ════════════════════════════════════════════════════════════════════════
# A bug or prompt injection can drain your API credits in minutes.
# The first public "agent spent $10K in an afternoon" incident was a
# missing budget envelope. Hard budget cascading is the structural fix.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build the budget hierarchy
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Build Budget Hierarchy")
print("=" * 70)

# TODO: Create a BudgetTracker with total_budget=500.0
tracker = ____

# TODO: Allocate $20 to data_analyst, $100 to model_trainer, $50 to model_deployer
____
____
____

# TODO: Simulate a day's work — three model_trainer tasks at
#       $30, $30, $25; one data_analyst at $8; one model_deployer at $15.
____
____
____
____
____

print(tracker.summary())

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert tracker.remaining("model_trainer") == 15.0
print("\n[x] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Overspend must be denied
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Overspend Attempt")
print("=" * 70)

# model_trainer has $15 remaining; try to spend $50.
# TODO: Call tracker.spend("model_trainer", 50.0) and bind to overspend_ok.
overspend_ok = ____

print(f"  Overspend $50: {'ALLOWED (bug!)' if overspend_ok else 'DENIED (correct)'}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert not overspend_ok
assert tracker.remaining("model_trainer") == 15.0
print("\n[x] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise spend vs. allocation
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visual Proof")
print("=" * 70)

# TODO: For each row in tracker.summary(), print an ASCII bar of
#       length 40 proportional to spent/allocated.
summary = tracker.summary()
print(f"  {'agent':<18} {'alloc':>8} {'spent':>8}  bar")
print("  " + "-" * 55)
for row in summary.iter_rows(named=True):
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Exercise check_access() (allow + deny cases)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 4: Access Control Decisions")
print("=" * 70)


async def test_access_control():
    test_cases = [
        ("model_trainer", "training_data", "read", True),
        ("model_trainer", "production_model", "deploy", False),
        ("customer_agent", "customer_faq", "search_faq", True),
        ("customer_agent", "training_data", "read_data", False),
        ("risk_assessor", "model_audit_log", "audit_model", True),
        ("risk_assessor", "production_model", "deploy_model", False),
        ("model_deployer", "production_model", "deploy_model", True),
        ("data_analyst", "restricted_data", "read", False),
        ("bias_checker", "model_fairness", "run_fairness_check", True),
        ("customer_agent", "internal_data", "read_data", False),
    ]

    results = []
    for agent_id, resource, action, expected in test_cases:
        # TODO: Call engine.check_access(agent_id=..., resource=..., action=...)
        decision = ____
        actual = decision.allowed
        results.append({"agent": agent_id, "expected": expected, "actual": actual})
        status = "[ok]" if actual == expected else "[FAIL]"
        print(
            f"  {status} {agent_id:<14} {action:<18} "
            f"{'ALLOW' if expected else 'DENY':<7} "
            f"{'ALLOW' if actual else 'DENY':<7}"
        )
    return results


access_results = asyncio.run(test_access_control())

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert len(access_results) >= 10
print(f"\n[x] Checkpoint 4 passed — {len(access_results)} tests\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS TRM Cost Controls
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore digital bank's 24/7 loan triage agent must
# have a hard cap on per-decision compute spend under MAS TRM 5.8.
# Cascading budgets in a compiled envelope make the cap structural
# instead of config-file-editable.
#
# BUSINESS IMPACT: A runaway agent can easily produce a S$50K–S$200K
# API bill. Hard envelopes close the blast radius to the per-task cap.

print("=" * 70)
print("  KEY TAKEAWAY: Budget + Access Control Together")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Cascaded budget across a hierarchy
  [x] Caught an overspend attempt
  [x] Ran 10 allow+deny access tests
  [x] Mapped budgets to MAS TRM cost controls

  Next: 04_runtime_audit.py wires PactGovernedAgent into a live
  LLM workflow and tests it against real adversarial prompts.
"""
)
