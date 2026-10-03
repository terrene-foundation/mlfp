# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.2: Operating Envelopes & Monotonic Tightening
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build `ConstraintEnvelopeConfig` objects across all 5 canonical
#     dimensions (Financial, Operational, Temporal, Data Access,
#     Communication)
#   - Express the monotonic-tightening rule structurally via
#     `RoleEnvelope.validate_tightening()` — no hand-rolled integer
#     comparisons
#   - Detect privilege-escalation attempts at envelope-compile time
#   - Visualise the clearance lattice and per-agent envelope footprint
#
# PREREQUISITES: 01_org_compile.py
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Build each agent's full 5-dimension operating envelope
#   2. Verify monotonic tightening for every delegation chain
#   3. Simulate a privilege-escalation attempt (caught structurally)
#   4. Apply — IMDA AI Verify self-assessment
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from kailash.trust.pact.envelopes import MonotonicTighteningError
from pact import (
    CommunicationConstraintConfig,
    ConfidentialityLevel,
    ConstraintEnvelopeConfig,
    DataAccessConstraintConfig,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
    RoleEnvelope,
    TemporalConstraintConfig,
)

from shared.mlfp06.ex_7 import CLEARANCE_LEVELS, compile_governance

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Compile once so this technique file is runnable in isolation
engine, org = compile_governance()
print("\n--- GovernanceEngine compiled ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — The Five Canonical Envelope Dimensions
# ════════════════════════════════════════════════════════════════════════
# An operating envelope is the multi-dimensional boundary PACT draws
# around a role. Canonical PACT names FIVE dimensions, and this
# exercise builds each one structurally — not as a narrated slide:
#
#   Financial     — max spend per invocation (the dollar ceiling)
#   Operational   — allowed/blocked actions + rate limits
#   Temporal      — active hours, blackout windows (off-hours freeze)
#   Data Access   — readable/writable paths, blocked data types
#   Communication — allowed channels, external-call approval gates
#
# The clearance level rides on top as a confidentiality classifier.
# Together these make up a `ConstraintEnvelopeConfig` — the object
# that PACT's engine checks on every `verify_action()` call.
#
# Analogy: A security badge at a research hospital. Budget = daily
# PET-scan allowance (Financial). Which procedures you may run =
# Operational. Which hours the badge is active = Temporal. Which
# patient files you may open = Data Access. Whether you may share
# findings outside the hospital = Communication. The badge NEVER
# grants more — it ONLY restricts.
#
# ── Sidebar: PACT's clearance ladder ───────────────────────────────
# pact orders clearances, lowest to highest:
#     PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET
# "restricted" is the SECOND-LOWEST rung, just above public — it is
# NOT the most privileged level. kaizen_agents also accepts the string
# "internal" as an alias of RESTRICTED. In this org the department
# heads hold SECRET and every agent sits at or below its head.
# ────────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build Each Agent's Full 5-Dimension Envelope
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Construct ConstraintEnvelopeConfig (all 5 dimensions)")
print("=" * 70)

# Address map (Shard 3 convention — dash-delimited D/T/R positions).
# Department heads: "D<n>-R<n>" (2 segments).
# Agent roles (team heads): "D<n>-R<n>-T<n>-R<n>" (4 segments).
AGENT_ADDRESSES: dict[str, str] = {
    "data_analyst": "D1-R1-T1-R1",
    "model_trainer": "D1-R1-T2-R1",
    "model_deployer": "D1-R1-T3-R1",
    "risk_assessor": "D2-R1-T1-R1",
    "bias_checker": "D2-R1-T2-R1",
    "customer_agent": "D3-R1-T1-R1",
}
DELEGATOR_ADDRESSES: dict[str, str] = {
    "chief_ml_officer": "D1-R1",
    "chief_risk_officer": "D2-R1",
    "vp_customer": "D3-R1",
}


def make_envelope(
    *,
    envelope_id: str,
    description: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
    active_hours: tuple[str, str] | None = None,
    read_paths: list[str],
    write_paths: list[str],
    allowed_channels: list[str],
) -> ConstraintEnvelopeConfig:
    """Build a 5-dimension envelope.

    Every dimension is populated explicitly so the structural
    guarantee is visible at the call site. Missing a dimension would
    allow a later refactor to silently widen the envelope.
    """
    start, end = active_hours if active_hours else (None, None)
    # TODO: Return a ConstraintEnvelopeConfig with id, description,
    #       confidentiality_clearance, max_delegation_depth=3 and ALL FIVE
    #       dimensions: financial, operational, temporal, data_access,
    #       communication.
    # Hint: one *ConstraintConfig class per dimension (see the imports)
    return ____


# Build the 6 agent envelopes. Each call populates ALL five dimensions.
# Every child envelope sits inside its department head's envelope (see
# `head_envelopes` below). Clearances follow pact's ladder
# (public < restricted < confidential < secret): the ML and risk heads
# carry SECRET, the customer head CONFIDENTIAL, and each agent is at or
# below its head.
envelopes_by_role: dict[str, ConstraintEnvelopeConfig] = {
    "data_analyst": make_envelope(
        envelope_id="data_analyst_envelope",
        description="Data analyst — read-only data work",
        clearance=ConfidentialityLevel.RESTRICTED,
        max_spend_usd=20.0,
        allowed_actions=["read_data", "summarise_data", "generate_report"],
        read_paths=["/data/raw/*", "/data/curated/*"],
        write_paths=["/reports/analyst/*"],
        allowed_channels=["internal"],
    ),
    "model_trainer": make_envelope(
        envelope_id="model_trainer_envelope",
        description="Model trainer — training + evaluation",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=100.0,
        allowed_actions=["train_model", "evaluate_model", "read_data"],
        read_paths=["/data/raw/*", "/data/curated/*"],
        write_paths=["/models/staging/*"],
        allowed_channels=["internal"],
    ),
    "model_deployer": make_envelope(
        envelope_id="model_deployer_envelope",
        description="Model deployer — deploy + monitor + rollback",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=50.0,
        allowed_actions=["deploy_model", "monitor_model", "rollback_model"],
        read_paths=["/models/staging/*", "/models/prod/*"],
        write_paths=["/models/prod/*"],
        allowed_channels=["internal", "pagerduty"],
    ),
    "risk_assessor": make_envelope(
        envelope_id="risk_assessor_envelope",
        description="Risk assessor — audit-read + report",
        clearance=ConfidentialityLevel.SECRET,
        max_spend_usd=200.0,
        allowed_actions=[
            "read_data",
            "audit_model",
            "generate_report",
            "access_audit_log",
        ],
        read_paths=["/data/raw/*", "/data/curated/*", "/models/prod/*", "/audit/*"],
        write_paths=["/reports/risk/*"],
        allowed_channels=["internal", "compliance"],
    ),
    "bias_checker": make_envelope(
        envelope_id="bias_checker_envelope",
        description="Bias checker — fairness audit only",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=75.0,
        allowed_actions=["read_data", "audit_model", "run_fairness_check"],
        read_paths=["/data/curated/*", "/models/prod/*"],
        write_paths=["/reports/bias/*"],
        allowed_channels=["internal"],
    ),
    "customer_agent": make_envelope(
        envelope_id="customer_agent_envelope",
        description="Customer agent — public FAQ answers",
        clearance=ConfidentialityLevel.PUBLIC,
        max_spend_usd=5.0,
        allowed_actions=["answer_question", "search_faq"],
        active_hours=("00:00", "23:59"),
        read_paths=["/faq/*"],
        write_paths=[],
        allowed_channels=["customer_chat"],
    ),
}

# Attach each envelope to its role so the engine enforces it on
# subsequent verify_action() calls. The defining role is the department
# head; the target role is the agent. set_role_envelope() REPLACES the
# envelope compile_governance() applied from the YAML for that role.
ROLE_TO_DELEGATOR: dict[str, str] = {
    "data_analyst": "chief_ml_officer",
    "model_trainer": "chief_ml_officer",
    "model_deployer": "chief_ml_officer",
    "risk_assessor": "chief_risk_officer",
    "bias_checker": "chief_risk_officer",
    "customer_agent": "vp_customer",
}
for role_id, env in envelopes_by_role.items():
    # TODO: Wrap env in a RoleEnvelope (defining role = the head's
    #       address, target role = the agent's address) and attach it.
    # Hint: RoleEnvelope(id=..., defining_role_address=...,
    #       target_role_address=..., envelope=...); engine.set_role_envelope
    ____

envelope_table = pl.DataFrame(
    {
        "Agent": list(envelopes_by_role.keys()),
        "Clearance": [
            env.confidentiality_clearance.name.lower()
            for env in envelopes_by_role.values()
        ],
        "Max $": [env.financial.max_spend_usd for env in envelopes_by_role.values()],
        "Allowed Actions": [
            ",".join(env.operational.allowed_actions)[:40]
            for env in envelopes_by_role.values()
        ],
        "Channels": [
            ",".join(env.communication.allowed_channels)
            for env in envelopes_by_role.values()
        ],
    }
)
print(envelope_table)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert envelope_table.height == 6, "Task 1: should have 6 agent envelopes"
for env in envelopes_by_role.values():
    assert env.financial is not None, "Financial dimension required"
    assert env.operational is not None, "Operational dimension required"
    assert env.temporal is not None, "Temporal dimension required"
    assert env.data_access is not None, "Data Access dimension required"
    assert env.communication is not None, "Communication dimension required"
print("\n[x] Checkpoint 1 passed — 6 envelopes, each with all 5 dimensions\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Verify Monotonic Tightening Structurally
# ════════════════════════════════════════════════════════════════════════
#
# Monotonic tightening: a child envelope is ALWAYS <= the envelope of
# its delegator on every dimension. Budget down, allowed-action set
# subset, clearance down, data paths subset, communication channels
# subset. PACT's `RoleEnvelope.validate_tightening()` checks EVERY
# dimension in one call — no hand-rolled integer comparisons.
#
# We build a synthetic "parent" envelope for each department head
# (the widest possible across its children) and verify each child
# chain tightens against it.

print("=" * 70)
print("TASK 2: Monotonic Tightening via RoleEnvelope.validate_tightening()")
print("=" * 70)

# The three department-head envelopes — permissive, but structurally
# bounded. Each one is the union (or superset) of its children's
# needs on every dimension: budget, actions, data paths, channels.
head_envelopes: dict[str, ConstraintEnvelopeConfig] = {
    "chief_ml_officer": make_envelope(
        envelope_id="chief_ml_officer_envelope",
        description="Chief ML Officer — ML department head",
        clearance=ConfidentialityLevel.SECRET,
        max_spend_usd=500.0,
        allowed_actions=[
            "read_data",
            "summarise_data",
            "generate_report",
            "train_model",
            "evaluate_model",
            "deploy_model",
            "monitor_model",
            "rollback_model",
        ],
        read_paths=[
            "/data/raw/*",
            "/data/curated/*",
            "/models/staging/*",
            "/models/prod/*",
        ],
        write_paths=[
            "/reports/analyst/*",
            "/models/staging/*",
            "/models/prod/*",
        ],
        allowed_channels=["internal", "pagerduty"],
    ),
    "chief_risk_officer": make_envelope(
        envelope_id="chief_risk_officer_envelope",
        description="Chief Risk Officer — risk department head",
        clearance=ConfidentialityLevel.SECRET,
        max_spend_usd=500.0,
        allowed_actions=[
            "read_data",
            "audit_model",
            "generate_report",
            "access_audit_log",
            "run_fairness_check",
        ],
        read_paths=[
            "/data/raw/*",
            "/data/curated/*",
            "/models/prod/*",
            "/audit/*",
        ],
        write_paths=["/reports/risk/*", "/reports/bias/*"],
        allowed_channels=["internal", "compliance"],
    ),
    "vp_customer": make_envelope(
        envelope_id="vp_customer_envelope",
        description="VP Customer — customer intelligence head",
        clearance=ConfidentialityLevel.CONFIDENTIAL,
        max_spend_usd=50.0,
        allowed_actions=["answer_question", "search_faq"],
        active_hours=("00:00", "23:59"),
        read_paths=["/faq/*"],
        write_paths=[],
        allowed_channels=["customer_chat"],
    ),
}

delegation_chains: list[tuple[str, str]] = [
    ("chief_ml_officer", "data_analyst"),
    ("chief_ml_officer", "model_trainer"),
    ("chief_ml_officer", "model_deployer"),
    ("chief_risk_officer", "risk_assessor"),
    ("chief_risk_officer", "bias_checker"),
    ("vp_customer", "customer_agent"),
]

all_valid = True
for delegator, agent in delegation_chains:
    try:
        # TODO: Validate the child envelope against its head's envelope.
        # Hint: RoleEnvelope.validate_tightening takes KEYWORD arguments
        ____
        print(f"  [ok] {delegator} -> {agent}")
    except MonotonicTighteningError as exc:
        all_valid = False
        print(f"  [VIOLATION] {delegator} -> {agent}: {exc}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert all_valid, "Task 2: every chain must tighten on every dimension"
print("\n[x] Checkpoint 2 passed — all 6 chains tighten structurally\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Simulate a Privilege-Escalation Attempt
# ════════════════════════════════════════════════════════════════════════
#
# Hypothetical: an operator tries to re-delegate the customer_agent
# with elevated authority — higher budget, broader allowed actions,
# and SECRET clearance (above vp_customer's CONFIDENTIAL). Under PACT,
# this attempt is caught structurally by `validate_tightening()` — not
# by a runtime integer comparison that a refactor can silently drop.

print("=" * 70)
print("TASK 3: Privilege-Escalation Attempt (caught at envelope time)")
print("=" * 70)

rogue_child = make_envelope(
    envelope_id="customer_agent_rogue_envelope",
    description="Rogue escalation — secret clearance, high budget",
    clearance=ConfidentialityLevel.SECRET,  # above parent's CONFIDENTIAL
    max_spend_usd=1000.0,  # 200x the legit budget
    allowed_actions=[
        "answer_question",
        "search_faq",
        "read_data",  # not in parent!
        "deploy_model",  # not in parent!
    ],
    read_paths=["/data/*", "/models/prod/*"],  # widened
    write_paths=["/models/prod/*"],  # widened
    allowed_channels=["customer_chat", "external_email"],  # widened
)

escalation_caught = False
violation_reason: str | None = None
try:
    # TODO: Validate rogue_child against vp_customer's envelope.
    ____
except MonotonicTighteningError as exc:
    escalation_caught = True
    violation_reason = str(exc)

print(f"  Attempt: vp_customer -> customer_agent (ROGUE)")
print(f"  Result:  {'REJECTED' if escalation_caught else 'ACCEPTED (bug!)'}")
if violation_reason:
    # One line per violated dimension
    for part in violation_reason.split("; "):
        print(f"  Reason:  {part[:160]}")
print("\n  PACT catches this at envelope construction, not at runtime.")

# Print pact's clearance ladder with the agents at each rung, so
# students can see where the rogue SECRET request would have landed.
print("\n  Clearance ladder (higher = more access):")
for level_name, level in sorted(CLEARANCE_LEVELS.items(), key=lambda x: -x[1]):
    agents_at_level = [
        role
        for role, env in envelopes_by_role.items()
        if env.confidentiality_clearance.value == level_name
    ]
    bar = "#" * (level + 1)
    print(f"    {level_name:<13} {bar:<5} {agents_at_level}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert escalation_caught, "Task 3: the escalation must be rejected"
assert "Confidentiality" in violation_reason, "Task 3: clearance escalation caught"
print("\n[x] Checkpoint 3 passed — privilege escalation caught structurally\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Envelope Dimension Radar Chart
# ════════════════════════════════════════════════════════════════════════
# Each agent's operating envelope spans five dimensions. The radar
# chart shows at a glance how "wide" each agent's authority is — a
# public customer agent has a tiny footprint, while the risk assessor
# has broad reach. Every value is READ FROM the envelope objects built in
# Task 1, normalised by the maximum across agents (clearance by the top
# of pact's ladder).

dimensions = ["Clearance", "Budget", "Actions", "Read\npaths", "Channels"]
max_level = max(CLEARANCE_LEVELS.values())
max_budget = max(e.financial.max_spend_usd for e in envelopes_by_role.values())
max_actions = max(len(e.operational.allowed_actions) for e in envelopes_by_role.values())
max_paths = max(len(e.data_access.read_paths) for e in envelopes_by_role.values())
max_channels = max(
    len(e.communication.allowed_channels) for e in envelopes_by_role.values()
)
# TODO: Build {role: [clearance, budget, actions, read_paths, channels]}
#       with each value normalised by the max above (clearance via
#       CLEARANCE_LEVELS[env.confidentiality_clearance.value]).
agent_data = ____

angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors_radar = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#7f8c8d"]

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
# code within weeks. With `ConstraintEnvelopeConfig` + `RoleEnvelope.
# validate_tightening()`, the answer is the YAML file that CI runs
# `compile_governance()` against on every PR — and the validation is
# structural, not narrative.
#
# BUSINESS IMPACT (illustrative figures): buyers increasingly ask AI
# vendors for governance evidence during procurement. If a platform
# that cannot produce it loses even one S$1M contract a year, the
# envelope work above is cheap by comparison.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Envelopes are the Structural Least-Privilege Gate")
print("=" * 70)
print("  Monotonic tightening makes privilege escalation")
print("  impossible at envelope time — not 'unlikely at runtime'.")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens
# ══════════════════════════════════════════════════════════════════
# Negative drills against the envelopes attached in Task 1: each agent
# asks for an action that belongs to a DIFFERENT agent's envelope. Every
# drill runs through engine.verify_action() — no LLM call is needed.
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=engine, run_id="ex_7_2_envelopes")
# TODO: Run the cross-envelope drills through the governance lens.
# Hint: obs.governance.negative_drills([...scenario dicts...])
drills = ____(
    [
        {
            "label": f"{role} -> deploy_model",
            "role_address": AGENT_ADDRESSES[role],
            "action": "deploy_model",
            "context": {"cost": 0.10},
        }
        for role in envelopes_by_role
        if "deploy_model" not in envelopes_by_role[role].operational.allowed_actions
    ]
)
print("\n── LLM Observatory: cross-envelope drills ──")
print(drills.select("scenario", "verdict"))
print(obs.governance.report())
# INTERPRETATION: every agent without deploy_model in its envelope is
# blocked from deploying. If any row reads auto_approved, that role has
# no envelope attached — fix the attachment, not the drill.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Built ConstraintEnvelopeConfig across all 5 canonical dimensions
  [x] Verified monotonic tightening via RoleEnvelope.validate_tightening
  [x] Simulated a privilege-escalation attempt and caught it structurally
  [x] Mapped envelopes to IMDA AI Verify self-assessment evidence

  KEY INSIGHT: 'Least privilege' is a slogan until you can express
  it as a structural property of a compiled graph. Monotonic
  tightening across five dimensions is that structural property.

  Next: 03_budget_access.py combines budget cascading with the
  access-control decision function.
"""
)
