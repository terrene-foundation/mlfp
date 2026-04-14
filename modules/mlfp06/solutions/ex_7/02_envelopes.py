# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.2: Operating Envelopes & Monotonic Tightening
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define operating envelopes (task, role, budget, tool, clearance)
#   - Express the monotonic-tightening rule: envelopes only get stricter
#   - Detect privilege-escalation attempts in a delegation chain
#   - Visualise the clearance lattice per agent
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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from shared.mlfp06.ex_7 import CLEARANCE_LEVELS, compile_governance

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Compile once so this technique file is runnable in isolation
engine, org = compile_governance()
print("\n--- GovernanceEngine compiled ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — What an Operating Envelope Is
# ════════════════════════════════════════════════════════════════════════
# An operating envelope is the multi-dimensional boundary PACT draws
# around an agent. Five dimensions matter for this exercise:
#
#   Task envelope      — which task types the agent may perform
#                        (model_trainer can train+evaluate, not deploy)
#   Role envelope      — role-based constraints
#                        (auditors read + audit, never mutate)
#   Budget envelope    — max cost per task execution
#                        (customer_agent limited to $5 per interaction)
#   Tool envelope      — whitelist of permitted tool calls
#                        (deployer: deploy + monitor + rollback only)
#   Clearance envelope — highest data classification accessible
#                        (public < internal < confidential < restricted)
#
# Analogy: A security badge at a research hospital. The badge lists
# which floors you may enter (task), which rooms on those floors
# (tool), what patient data you may read (clearance), your role
# (doctor vs. cleaner), and a daily budget for PET scans you may
# order (budget). The badge NEVER grants you more; it only restricts.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Inspect Operating Envelopes Per Agent
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Operating Envelopes")
print("=" * 70)

envelopes = pl.DataFrame(
    {
        "Agent": [
            "data_analyst",
            "model_trainer",
            "model_deployer",
            "risk_assessor",
            "bias_checker",
            "customer_agent",
        ],
        "Clearance": [
            "internal",
            "confidential",
            "confidential",
            "restricted",
            "confidential",
            "public",
        ],
        "Budget": ["$20", "$100", "$50", "$200", "$75", "$5"],
        "Tools": [
            "read,summarise,report",
            "train,evaluate,read",
            "deploy,monitor,rollback",
            "read,audit,report,log",
            "read,audit,fairness",
            "answer,search",
        ],
        "Role": ["analyst", "engineer", "operator", "auditor", "auditor", "analyst"],
    }
)
print(envelopes)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert envelopes.height == 6, "Task 1: should have 6 agent envelopes"
print("\n[x] Checkpoint 1 passed — 6 agent envelopes loaded\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Verify Monotonic Tightening
# ════════════════════════════════════════════════════════════════════════
#
# Monotonic tightening: an agent's envelope is ALWAYS <= the envelope
# of its delegator. Clearance cannot go up as you walk down the chain,
# and neither can budget or tool scope.
#
#   Example:
#     chief_ml_officer (restricted clearance, $500 budget)
#       ├─ model_trainer (confidential, $100)  — tighter
#       ├─ data_analyst  (internal,     $20)   — tighter
#       └─ INVALID: any child with "restricted" — not tighter, BLOCKED
#
# This is the structural prevention of privilege escalation.

print("=" * 70)
print("TASK 2: Monotonic Tightening Verification")
print("=" * 70)

delegation_chains = [
    ("chief_ml_officer", "restricted", "model_trainer", "confidential"),
    ("chief_ml_officer", "restricted", "data_analyst", "internal"),
    ("chief_ml_officer", "restricted", "model_deployer", "confidential"),
    ("chief_risk_officer", "restricted", "risk_assessor", "restricted"),
    ("chief_risk_officer", "restricted", "bias_checker", "confidential"),
    ("vp_customer", "internal", "customer_agent", "public"),
]

for delegator, del_clearance, agent, agent_clearance in delegation_chains:
    is_tighter = CLEARANCE_LEVELS[agent_clearance] <= CLEARANCE_LEVELS[del_clearance]
    status = "[ok]" if is_tighter else "[VIOLATION]"
    print(f"  {status} {delegator}({del_clearance}) -> {agent}({agent_clearance})")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
all_valid = all(
    CLEARANCE_LEVELS[ac] <= CLEARANCE_LEVELS[dc] for _, dc, _, ac in delegation_chains
)
assert all_valid, "Task 2: every chain must be monotonically tightening"
print("\n[x] Checkpoint 2 passed — monotonic tightening verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visual Proof: Simulate a Privilege Escalation Attempt
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Privilege-Escalation Attempt (Visual Proof)")
print("=" * 70)

# Hypothetical: what if the customer_agent (public clearance) tried to
# be re-delegated with "restricted" clearance via a rogue YAML edit?
attempt = ("vp_customer", "internal", "customer_agent", "restricted")
delegator, del_cl, agent, agent_cl = attempt
is_tighter = CLEARANCE_LEVELS[agent_cl] <= CLEARANCE_LEVELS[del_cl]
status = "ACCEPTED" if is_tighter else "REJECTED"
print(f"  Attempt: {delegator}({del_cl}) -> {agent}({agent_cl})")
print(f"  Monotonic tightening check: {status}")
print("  (PACT would refuse to compile this org YAML.)")

# Visualise the clearance lattice as a lattice-like plot
print("\n  Clearance lattice (higher = more access):")
for level_name, level in sorted(CLEARANCE_LEVELS.items(), key=lambda x: -x[1]):
    agents_at_level = envelopes.filter(pl.col("Clearance") == level_name)[
        "Agent"
    ].to_list()
    bar = "#" * (level + 1)
    print(f"    {level_name:<13} {bar:<5} {agents_at_level}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert not is_tighter, "Task 3: the attempt should be rejected"
print("\n[x] Checkpoint 3 passed — privilege escalation caught\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Envelope dimension radar chart
# ════════════════════════════════════════════════════════════════════════
# Each agent's operating envelope spans five dimensions. The radar chart
# shows at a glance how "wide" each agent's authority is — a public
# customer agent has a tiny footprint, while the risk assessor has
# broad reach. This is the visual proof of least-privilege.

dimensions = ["Clearance", "Budget", "Tools", "Role\nScope", "Data\nAccess"]

# Normalise each dimension to 0-1 for the radar
agent_data = {
    "data_analyst": [1 / 3, 20 / 200, 3 / 6, 0.4, 0.33],
    "model_trainer": [2 / 3, 100 / 200, 3 / 6, 0.6, 0.66],
    "risk_assessor": [3 / 3, 200 / 200, 4 / 6, 0.8, 1.0],
    "customer_agent": [0 / 3, 5 / 200, 2 / 6, 0.3, 0.1],
}

angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_radar = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

for (agent_name, values), color in zip(agent_data.items(), colors_radar):
    vals = values + values[:1]
    ax.plot(angles, vals, "o-", linewidth=2, label=agent_name, color=color)
    ax.fill(angles, vals, alpha=0.1, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(dimensions, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title(
    "Operating Envelope Radar — Per-Agent Dimensions", fontweight="bold", pad=20
)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
plt.tight_layout()
fname = OUTPUT_DIR / "ex7_envelope_radar.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: IMDA AI Verify Self-Assessment
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore e-commerce platform is preparing an IMDA AI
# Verify self-assessment (Singapore's voluntary AI governance testing
# framework). One of the required controls is "Access Control and
# Authorisation": show that every AI agent has an explicit least-
# privilege envelope and that no agent can escalate beyond its
# delegator.
#
# Without structural monotonic tightening, the answer is a spreadsheet
# that the compliance team updates by hand and that drifts from the
# code within weeks. With compiled envelopes, the answer is the YAML
# file that CI runs `compile_org()` against on every PR. The self-
# assessment evidence is automated.
#
# BUSINESS IMPACT: IMDA AI Verify is increasingly used as a
# procurement filter by Singapore government agencies and
# GLC-linked companies. A platform that cannot produce the
# required evidence is excluded from tenders worth S$1M–S$10M
# annually. Getting the evidence right is not a nice-to-have.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Envelopes are the Structural Least-Privilege Gate")
print("=" * 70)
print("  Monotonic tightening makes privilege escalation")
print("  impossible at compile time — not 'unlikely at runtime'.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Enumerated the five dimensions of an operating envelope
  [x] Verified monotonic tightening for every chain in the org
  [x] Simulated a privilege-escalation attempt and caught it
  [x] Mapped envelopes to IMDA AI Verify self-assessment evidence

  KEY INSIGHT: 'Least privilege' is a slogan until you can express
  it as a structural property of a compiled graph. Monotonic
  tightening is that structural property.

  Next: 03_budget_access.py combines budget cascading with the
  access-control decision function.
"""
)
