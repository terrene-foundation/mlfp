# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / Governed Agent & GovernanceContext
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Wrap agents with PACT governance using GovernanceContext
# LEVEL: Advanced
# PARITY: Equivalent — Rust has GovernanceContext with same frozen semantics
# VALIDATES: GovernanceContext (frozen), GovernanceBlockedError
#
# Run: uv run python textbook/python/06-pact/09_governed_agent.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import pickle

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    GovernanceBlockedError,
    GovernanceContext,
    GovernanceEngine,
    GovernanceVerdict,
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

# ── Setup: Build org and engine ──────────────────────────────────────

org_def = OrgDefinition(
    org_id="agent-demo",
    name="Agent Demo Corp",
    departments=[
        DepartmentConfig(department_id="ops", name="Operations"),
    ],
    teams=[
        TeamConfig(id="deploy", name="Deployment Team", workspace="ws"),
    ],
    roles=[
        RoleDefinition(
            role_id="vp-ops",
            name="VP Operations",
            is_primary_for_unit="ops",
        ),
        RoleDefinition(
            role_id="deploy-lead",
            name="Deployment Lead",
            reports_to_role_id="vp-ops",
            is_primary_for_unit="deploy",
        ),
        RoleDefinition(
            role_id="deploy-agent",
            name="Deploy Agent",
            reports_to_role_id="deploy-lead",
            agent_id="agent-deploy-01",
        ),
    ],
)

engine = GovernanceEngine(org_def)

# Grant clearance
engine.grant_clearance(
    "D1-R1-T1-R1-R1",
    RoleClearance(
        role_address="D1-R1-T1-R1-R1",
        max_clearance=ConfidentialityLevel.RESTRICTED,
        compartments=frozenset({"deployment-configs"}),
        vetting_status=VettingStatus.ACTIVE,
    ),
)

# Set envelope
engine.set_role_envelope(
    RoleEnvelope(
        id="re-deploy-agent",
        defining_role_address="D1-R1-T1-R1",
        target_role_address="D1-R1-T1-R1-R1",
        envelope=ConstraintEnvelopeConfig(
            id="env-deploy-agent",
            confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
            financial=FinancialConstraintConfig(
                max_spend_usd=100.0,
            ),
            operational=OperationalConstraintConfig(
                allowed_actions=["read", "deploy", "rollback"],
                blocked_actions=["delete_production", "approve"],
            ),
        ),
    )
)

# ── 1. Get a GovernanceContext — the frozen snapshot ─────────────────
# Agents receive GovernanceContext, NOT GovernanceEngine.
# This is the anti-self-modification defense.

ctx = engine.get_context(
    role_address="D1-R1-T1-R1-R1",
    posture=TrustPosture.SUPERVISED,
)

assert isinstance(ctx, GovernanceContext)
assert ctx.role_address == "D1-R1-T1-R1-R1"
assert ctx.posture == TrustPosture.SUPERVISED
assert ctx.org_id == "agent-demo"

# ── 2. GovernanceContext is frozen (immutable) ──────────────────────
# frozen=True means no attribute can be mutated after creation.

try:
    ctx.posture = TrustPosture.DELEGATED  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen=True prevents mutation

try:
    ctx.role_address = "D1-R1"  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: cannot change role address

# ── 3. Context carries effective envelope snapshot ──────────────────

assert ctx.effective_envelope is not None
assert ctx.effective_envelope.financial is not None
assert ctx.effective_envelope.financial.max_spend_usd == 100.0

# Allowed actions from the operational dimension
assert "read" in ctx.allowed_actions
assert "deploy" in ctx.allowed_actions
assert "rollback" in ctx.allowed_actions
assert "delete_production" not in ctx.allowed_actions

# ── 4. Context carries clearance information ────────────────────────

assert ctx.clearance is not None
assert ctx.clearance.max_clearance == ConfidentialityLevel.RESTRICTED

# Effective clearance is posture-capped: min(RESTRICTED, SUPERVISED ceiling)
# SUPERVISED ceiling = RESTRICTED, so effective = RESTRICTED
assert ctx.effective_clearance_level == ConfidentialityLevel.RESTRICTED

# Compartments from clearance
assert "deployment-configs" in ctx.compartments

# ── 5. Context at different posture levels ──────────────────────────

delegated_ctx = engine.get_context(
    role_address="D1-R1-T1-R1-R1",
    posture=TrustPosture.DELEGATED,
)

# At DELEGATED, ceiling = TOP_SECRET, so effective = RESTRICTED (role max)
assert delegated_ctx.effective_clearance_level == ConfidentialityLevel.RESTRICTED

pseudo_ctx = engine.get_context(
    role_address="D1-R1-T1-R1-R1",
    posture=TrustPosture.PSEUDO_AGENT,
)

# At PSEUDO_AGENT, ceiling = PUBLIC, so effective = PUBLIC
assert pseudo_ctx.effective_clearance_level == ConfidentialityLevel.PUBLIC

# ── 6. GovernanceContext CANNOT be pickled — security invariant ─────
# Pickle deserialization would allow forged context injection.

try:
    pickle.dumps(ctx)
    assert False, "Should raise TypeError"
except TypeError as e:
    assert "cannot be pickled" in str(e).lower()

# ── 7. Simulate agent governance check before action ────────────────
# The pattern: agent checks context before performing any action.


def agent_perform_action(
    context: GovernanceContext,
    engine_ref: GovernanceEngine,
    action: str,
) -> str:
    """Simulate an agent checking governance before acting.

    In production, PactGovernedAgent does this automatically.
    Here we show the manual pattern for understanding.
    """
    # Step 1: Quick check against context (no engine call needed)
    if action not in context.allowed_actions:
        raise GovernanceBlockedError(
            GovernanceVerdict(
                level="blocked",
                reason=f"Action '{action}' not in allowed_actions",
                role_address=context.role_address,
                action=action,
            )
        )

    # Step 2: Full verification through engine (thread-safe, audited)
    verdict = engine_ref.verify_action(context.role_address, action)
    if verdict.is_blocked:
        raise GovernanceBlockedError(verdict)

    return f"Action '{action}' executed successfully"


# Allowed action
result = agent_perform_action(ctx, engine, "deploy")
assert "deploy" in result
assert "successfully" in result

# Blocked action
try:
    agent_perform_action(ctx, engine, "delete_production")
    assert False, "Should raise GovernanceBlockedError"
except GovernanceBlockedError as e:
    assert "delete_production" in str(e)
    assert e.verdict.level == "blocked"

# ── 8. GovernanceBlockedError carries the verdict ───────────────────

try:
    agent_perform_action(ctx, engine, "approve")
    assert False, "Should raise GovernanceBlockedError"
except GovernanceBlockedError as e:
    assert isinstance(e.verdict, GovernanceVerdict)
    assert e.verdict.level == "blocked"
    assert e.verdict.action == "approve" or "approve" in str(e)

# ── 9. Context serialization for transport ──────────────────────────
# to_dict() is safe for logging and transport. from_dict() emits a warning.

ctx_dict = ctx.to_dict()

assert isinstance(ctx_dict, dict)
assert ctx_dict["role_address"] == "D1-R1-T1-R1-R1"
assert ctx_dict["posture"] == "supervised"
assert ctx_dict["org_id"] == "agent-demo"
assert isinstance(ctx_dict["allowed_actions"], list)
assert isinstance(ctx_dict["compartments"], list)

# ── 10. from_dict() emits a security warning ────────────────────────
# The authoritative construction path is GovernanceEngine.get_context().
# from_dict() is for display/audit only, not for access control.

import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    restored = GovernanceContext.from_dict(ctx_dict)

assert len(caught) >= 1
assert "unverified" in str(caught[0].message).lower()

# The restored context has the same data
assert restored.role_address == ctx.role_address
assert restored.posture == ctx.posture

# ── 11. Context without clearance ───────────────────────────────────
# A role with no clearance gets a context with None clearance fields.

no_clearance_ctx = engine.get_context(
    role_address="D1-R1",  # VP Ops — we did not grant clearance
    posture=TrustPosture.SUPERVISED,
)

assert no_clearance_ctx.clearance is None
assert no_clearance_ctx.effective_clearance_level is None
assert len(no_clearance_ctx.compartments) == 0

# ── 12. The key security principle ──────────────────────────────────
# Agents get GovernanceContext (frozen, read-only snapshot).
# Agents NEVER get GovernanceEngine (mutable, state-changing).
# This prevents agents from:
# - Escalating their own clearance (grant_clearance)
# - Widening their envelope (set_role_envelope)
# - Modifying organizational structure
# - Approving their own bridges

# Verify the context has no reference to the engine
assert not hasattr(ctx, "engine")
assert not hasattr(ctx, "_engine")
assert not hasattr(ctx, "grant_clearance")
assert not hasattr(ctx, "set_role_envelope")
assert not hasattr(ctx, "verify_action")

print("PASS: 06-pact/09_governed_agent")
