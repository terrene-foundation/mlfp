# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.2: Operating Envelopes & Monotonic Tightening
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define operating envelopes (task, role, budget, tool, clearance)
#   - Express the monotonic-tightening rule in code
#   - Detect privilege-escalation attempts
#   - Visualise the clearance lattice
#
# PREREQUISITES: 01_org_compile.py
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Inspect each agent's operating envelope
#   2. Verify monotonic tightening for every chain
#   3. Simulate a privilege-escalation attempt
#   4. Apply — IMDA AI Verify self-assessment
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared.mlfp06.ex_7 import CLEARANCE_LEVELS, compile_governance

engine, org = compile_governance()


# ════════════════════════════════════════════════════════════════════════
# THEORY — The 5 Dimensions of an Operating Envelope
# ════════════════════════════════════════════════════════════════════════
# Task, Role, Budget, Tool, Clearance. The envelope ONLY restricts —
# it never grants. Clearance hierarchy is:
#   public < internal < confidential < restricted


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Inspect the per-agent envelopes
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Operating Envelopes")
print("=" * 70)

# TODO: Build a polars DataFrame with columns
#       Agent, Clearance, Budget, Tools, Role
#       listing the 6 agents: data_analyst, model_trainer, model_deployer,
#       risk_assessor, bias_checker, customer_agent.
envelopes = ____

print(envelopes)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert envelopes.height == 6
print("\n[x] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Verify monotonic tightening for every chain
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Monotonic Tightening")
print("=" * 70)

delegation_chains = [
    ("chief_ml_officer", "restricted", "model_trainer", "confidential"),
    ("chief_ml_officer", "restricted", "data_analyst", "internal"),
    ("chief_ml_officer", "restricted", "model_deployer", "confidential"),
    ("chief_risk_officer", "restricted", "risk_assessor", "restricted"),
    ("chief_risk_officer", "restricted", "bias_checker", "confidential"),
    ("vp_customer", "internal", "customer_agent", "public"),
]

# TODO: For each chain, check CLEARANCE_LEVELS[agent_clearance] <=
#       CLEARANCE_LEVELS[delegator_clearance] and print a status line.
for delegator, del_clearance, agent, agent_clearance in delegation_chains:
    ____

# ── Checkpoint 2 ────────────────────────────────────────────────────────
# TODO: Assert that every chain is tightening.
assert ____
print("\n[x] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Simulate a privilege-escalation attempt
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Privilege-Escalation Attempt")
print("=" * 70)

# Rogue attempt: vp_customer (internal) -> customer_agent (restricted)
attempt = ("vp_customer", "internal", "customer_agent", "restricted")
delegator, del_cl, agent, agent_cl = attempt

# TODO: Compute is_tighter the same way as Task 2.
is_tighter = ____

print(f"  Attempt: {delegator}({del_cl}) -> {agent}({agent_cl})")
print(f"  Monotonic tightening check: {'ACCEPTED' if is_tighter else 'REJECTED'}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert not is_tighter
print("\n[x] Checkpoint 3 passed — privilege escalation caught\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: IMDA AI Verify Self-Assessment
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore e-commerce platform preparing an IMDA AI Verify
# self-assessment must evidence that every AI agent has a least-
# privilege envelope that cannot be escalated. With compiled envelopes
# the evidence is automated; without, it drifts from code within weeks.
#
# BUSINESS IMPACT: IMDA AI Verify is increasingly a procurement filter
# for Singapore government and GLC-linked tenders worth S$1M–S$10M
# annually. A platform that cannot produce the evidence is excluded.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Envelopes = Structural Least-Privilege")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Enumerated the 5 envelope dimensions
  [x] Verified monotonic tightening
  [x] Caught a privilege-escalation attempt
  [x] Mapped envelopes to IMDA AI Verify evidence

  Next: 03_budget_access.py combines budget cascading with access control.
"""
)
