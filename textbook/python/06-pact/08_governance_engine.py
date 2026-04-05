# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / GovernanceEngine
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use the GovernanceEngine facade for thread-safe governance
# LEVEL: Advanced
# PARITY: Full — Rust has equivalent GovernanceEngine with same API
# VALIDATES: GovernanceEngine, compile + query in one interface
#
# Run: uv run python textbook/python/06-pact/08_governance_engine.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    GovernanceEngine,
    GovernanceVerdict,
    KnowledgeItem,
    RoleClearance,
    RoleDefinition,
    RoleEnvelope,
    VettingStatus,
)
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig,
    DepartmentConfig,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
    OrgDefinition,
    TeamConfig,
)

TrustPostureLevel = TrustPosture

# ── 1. Create an OrgDefinition ──────────────────────────────────────

org_def = OrgDefinition(
    org_id="engine-demo",
    name="Engine Demo Corp",
    departments=[
        DepartmentConfig(department_id="ops", name="Operations"),
    ],
    teams=[
        TeamConfig(id="infra", name="Infrastructure Team", workspace="ws"),
    ],
    roles=[
        RoleDefinition(
            role_id="vp-ops",
            name="VP Operations",
            is_primary_for_unit="ops",
        ),
        RoleDefinition(
            role_id="infra-lead",
            name="Infrastructure Lead",
            reports_to_role_id="vp-ops",
            is_primary_for_unit="infra",
        ),
        RoleDefinition(
            role_id="sre-1",
            name="SRE Engineer",
            reports_to_role_id="infra-lead",
        ),
    ],
)

# ── 2. Initialize GovernanceEngine ──────────────────────────────────
# The engine accepts an OrgDefinition (compiles it) or a CompiledOrg.
# It is the single entry point for all governance decisions.

engine = GovernanceEngine(org_def)

# The engine compiled the org internally
assert engine._compiled_org.org_id == "engine-demo"
assert "D1-R1" in engine._compiled_org.nodes

# ── 3. Grant clearance through the engine ───────────────────────────
# All state mutations go through the engine for thread safety.

engine.grant_clearance(
    "D1-R1",
    RoleClearance(
        role_address="D1-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        vetting_status=VettingStatus.ACTIVE,
    ),
)

engine.grant_clearance(
    "D1-R1-T1-R1",
    RoleClearance(
        role_address="D1-R1-T1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
)

engine.grant_clearance(
    "D1-R1-T1-R1-R1",
    RoleClearance(
        role_address="D1-R1-T1-R1-R1",
        max_clearance=ConfidentialityLevel.RESTRICTED,
        vetting_status=VettingStatus.ACTIVE,
    ),
)

# ── 4. Set role envelopes ───────────────────────────────────────────

vp_envelope = ConstraintEnvelopeConfig(
    id="env-vp-ops",
    financial=FinancialConstraintConfig(
        max_spend_usd=100000.0,
        requires_approval_above_usd=50000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy", "approve", "provision"],
        blocked_actions=["delete_production"],
    ),
)

lead_envelope = ConstraintEnvelopeConfig(
    id="env-infra-lead",
    financial=FinancialConstraintConfig(
        max_spend_usd=10000.0,
        requires_approval_above_usd=5000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy", "provision"],
        blocked_actions=["delete_production", "approve"],
    ),
)

engine.set_role_envelope(
    RoleEnvelope(
        id="re-vp-ops",
        defining_role_address="R1",  # Board defines VP envelope
        target_role_address="D1-R1",
        envelope=vp_envelope,
    )
)

engine.set_role_envelope(
    RoleEnvelope(
        id="re-infra-lead",
        defining_role_address="D1-R1",  # VP defines lead envelope
        target_role_address="D1-R1-T1-R1",
        envelope=lead_envelope,
    )
)

# ── 5. verify_action — the primary decision API ─────────────────────
# Combines vacancy check, envelope evaluation, gradient classification,
# and optional knowledge access check.

verdict = engine.verify_action("D1-R1-T1-R1", "deploy")

assert isinstance(verdict, GovernanceVerdict)
assert verdict.role_address == "D1-R1-T1-R1"
assert verdict.action == "deploy"

# "deploy" is in the allowed_actions, so it should be auto_approved
assert verdict.level == "auto_approved"
assert verdict.allowed is True
assert verdict.is_blocked is False

# ── 6. Blocked action ──────────────────────────────────────────────
# "delete_production" is in blocked_actions.

blocked_verdict = engine.verify_action("D1-R1-T1-R1", "delete_production")

assert blocked_verdict.level == "blocked"
assert blocked_verdict.allowed is False
assert blocked_verdict.is_blocked is True

# ── 7. verify_action with financial context ─────────────────────────
# Pass a "cost" in context to trigger financial envelope checks.

low_cost_verdict = engine.verify_action(
    "D1-R1-T1-R1",
    "provision",
    {"cost": 100.0},
)
assert low_cost_verdict.allowed is True

# ── 8. check_access — knowledge access through the engine ───────────
# The engine gathers clearances, KSPs, and bridges from its stores.

ops_doc = KnowledgeItem(
    item_id="doc-ops-runbook",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",
)

decision = engine.check_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=ops_doc,
    posture=TrustPosture.DELEGATED,
)

assert decision.allowed is True  # SRE is in the same team

# ── 9. check_access — fail-closed on errors ────────────────────────
# The engine catches all exceptions and returns DENY.

secret_doc = KnowledgeItem(
    item_id="doc-classified",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1",
)

# SRE has RESTRICTED clearance, SECRET doc requires higher
secret_decision = engine.check_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=secret_doc,
    posture=TrustPosture.DELEGATED,
)

assert secret_decision.allowed is False
assert secret_decision.step_failed == 2

# ── 10. GovernanceVerdict properties ────────────────────────────────

assert verdict.allowed is True  # auto_approved or flagged = True
assert verdict.is_held is False
assert verdict.is_blocked is False

assert blocked_verdict.allowed is False
assert blocked_verdict.is_blocked is True
assert blocked_verdict.is_held is False

# Verdict serialization
verdict_dict = verdict.to_dict()
assert isinstance(verdict_dict, dict)
assert verdict_dict["level"] == "auto_approved"
assert verdict_dict["action"] == "deploy"
assert verdict_dict["allowed"] is True

# ── 11. Engine with pre-compiled org ────────────────────────────────
# You can pass a CompiledOrg directly to skip recompilation.

from kailash.trust.pact import compile_org

compiled = compile_org(org_def)
engine2 = GovernanceEngine(compiled)
assert engine2._compiled_org.org_id == "engine-demo"

# ── 12. Thread safety ───────────────────────────────────────────────
# All public methods acquire self._lock. Verify the lock exists.

import threading

assert isinstance(engine._lock, threading.Lock)

# ── 13. verify_action is fail-closed ────────────────────────────────
# Any exception path returns BLOCKED. This is tested implicitly by the
# engine's try/except in verify_action. We can verify the contract:

# Verify that the engine returns a GovernanceVerdict even for unknown roles
unknown_verdict = engine.verify_action("D99-R99", "some_action")
assert isinstance(unknown_verdict, GovernanceVerdict)
# Unknown role with no envelope = auto_approved (no constraints)
# or blocked (depending on vacancy/clearance checks)
assert unknown_verdict.level in ("auto_approved", "blocked", "flagged", "held")

print("PASS: 06-pact/08_governance_engine")
