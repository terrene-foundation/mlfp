# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / Operating Envelopes
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Create and intersect operating envelopes with monotonic tightening
# LEVEL: Intermediate
# PARITY: Full — Rust has equivalent envelope intersection semantics
# VALIDATES: RoleEnvelope, TaskEnvelope, intersect_envelopes,
#            compute_effective_envelope, MonotonicTighteningError
#
# Run: uv run python textbook/python/06-pact/04_envelopes.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kailash.trust import ConfidentialityLevel
from kailash.trust.pact import (
    MonotonicTighteningError,
    RoleEnvelope,
    TaskEnvelope,
    compute_effective_envelope,
    intersect_envelopes,
)
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig,
    DataAccessConstraintConfig,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
)

# ── 1. Create a ConstraintEnvelopeConfig ─────────────────────────────
# The envelope has five CARE dimensions: financial, operational, temporal,
# data_access, communication. Plus confidentiality ceiling.

cto_envelope = ConstraintEnvelopeConfig(
    id="env-cto",
    description="CTO operating envelope",
    confidentiality_clearance=ConfidentialityLevel.SECRET,
    financial=FinancialConstraintConfig(
        max_spend_usd=50000.0,
        api_cost_budget_usd=5000.0,
        requires_approval_above_usd=10000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy", "approve"],
        blocked_actions=["delete_production"],
    ),
    data_access=DataAccessConstraintConfig(
        read_paths=["/engineering/**", "/shared/**"],
        write_paths=["/engineering/**"],
    ),
)

assert cto_envelope.id == "env-cto"
assert cto_envelope.financial is not None
assert cto_envelope.financial.max_spend_usd == 50000.0
assert "deploy" in cto_envelope.operational.allowed_actions
assert "delete_production" in cto_envelope.operational.blocked_actions

# ── 2. Create a tighter envelope for a direct report ─────────────────

lead_envelope = ConstraintEnvelopeConfig(
    id="env-lead",
    description="Backend Lead operating envelope",
    confidentiality_clearance=ConfidentialityLevel.CONFIDENTIAL,
    financial=FinancialConstraintConfig(
        max_spend_usd=5000.0,
        api_cost_budget_usd=1000.0,
        requires_approval_above_usd=2000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy"],
        blocked_actions=["delete_production", "approve"],
    ),
    data_access=DataAccessConstraintConfig(
        read_paths=["/engineering/**"],
        write_paths=["/engineering/backend/**"],
    ),
)

# Lead has lower spend, fewer actions, narrower paths — tighter in every dimension.

# ── 3. Intersect two envelopes ───────────────────────────────────────
# intersect_envelopes() computes the most restrictive combination.
# Financial: min() of numeric limits
# Operational: intersection of allowed, union of blocked
# Data access: intersection of paths

intersection = intersect_envelopes(cto_envelope, lead_envelope)

# Financial: min of spend limits
assert intersection.financial is not None
assert intersection.financial.max_spend_usd == 5000.0  # min(50000, 5000)
assert intersection.financial.api_cost_budget_usd == 1000.0  # min(5000, 1000)

# Operational: intersection of allowed, union of blocked
assert "read" in intersection.operational.allowed_actions
assert "write" in intersection.operational.allowed_actions
# "deploy" is in both allowed sets, so it stays
assert "deploy" in intersection.operational.allowed_actions
# "approve" is blocked by lead, so removed from allowed and added to blocked
assert "approve" not in intersection.operational.allowed_actions
assert "approve" in intersection.operational.blocked_actions
assert "delete_production" in intersection.operational.blocked_actions

# Data access: intersection of paths
assert "/engineering/**" in intersection.data_access.read_paths
# "/shared/**" only in CTO, not in lead, so excluded from intersection
assert "/shared/**" not in intersection.data_access.read_paths

# Confidentiality: min of levels
assert intersection.confidentiality_clearance == ConfidentialityLevel.CONFIDENTIAL

# ── 4. RoleEnvelope — standing boundary from supervisor ──────────────

role_env = RoleEnvelope(
    id="re-backend-lead",
    defining_role_address="D1-R1",  # CTO defines it
    target_role_address="D1-R1-T1-R1",  # for Backend Lead
    envelope=lead_envelope,
)

assert role_env.defining_role_address == "D1-R1"
assert role_env.target_role_address == "D1-R1-T1-R1"
assert role_env.version == 1

# RoleEnvelope is frozen
try:
    role_env.version = 2  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen=True

# ── 5. Monotonic tightening validation ───────────────────────────────
# Child envelopes must be at most as permissive as parent envelopes.

# This should pass — lead_envelope is tighter than cto_envelope
RoleEnvelope.validate_tightening(
    parent_envelope=cto_envelope,
    child_envelope=lead_envelope,
)

# This should fail — trying to give child more spend than parent
wider_envelope = ConstraintEnvelopeConfig(
    id="env-too-wide",
    financial=FinancialConstraintConfig(
        max_spend_usd=100000.0,  # More than CTO's 50000
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read"],
    ),
)

try:
    RoleEnvelope.validate_tightening(
        parent_envelope=cto_envelope,
        child_envelope=wider_envelope,
    )
    assert False, "Should raise MonotonicTighteningError"
except MonotonicTighteningError:
    pass  # Expected: child max_spend exceeds parent

# ── 6. TaskEnvelope — ephemeral narrowing for a task ─────────────────

task_env = TaskEnvelope(
    id="te-deploy-v2",
    task_id="task-deploy-v2",
    parent_envelope_id="re-backend-lead",
    envelope=ConstraintEnvelopeConfig(
        id="env-deploy-task",
        confidentiality_clearance=ConfidentialityLevel.RESTRICTED,
        financial=FinancialConstraintConfig(
            max_spend_usd=500.0,  # Much tighter for this specific task
        ),
        operational=OperationalConstraintConfig(
            allowed_actions=["read", "deploy"],
        ),
        data_access=DataAccessConstraintConfig(
            read_paths=["/engineering/backend/**"],
            write_paths=["/engineering/backend/deployments/**"],
        ),
    ),
    expires_at=datetime.now(UTC) + timedelta(hours=4),
)

assert task_env.task_id == "task-deploy-v2"
assert task_env.is_expired is False

# Expired task envelope
expired_task = TaskEnvelope(
    id="te-old",
    task_id="task-old",
    parent_envelope_id="re-backend-lead",
    envelope=ConstraintEnvelopeConfig(
        id="env-old-task",
        operational=OperationalConstraintConfig(allowed_actions=["read"]),
    ),
    expires_at=datetime.now(UTC) - timedelta(hours=1),  # Already expired
)

assert expired_task.is_expired is True

# ── 7. compute_effective_envelope — three-layer model ────────────────
# Effective = intersection of all ancestor RoleEnvelopes + TaskEnvelope.
#
# RoleEnvelope (standing)
#   intersection (monotonic tightening)
# TaskEnvelope (ephemeral)
#   =
# EffectiveEnvelope (computed)

# Create supervisor envelope
supervisor_role_env = RoleEnvelope(
    id="re-cto",
    defining_role_address="R1",  # Board defines CTO envelope
    target_role_address="D1-R1",
    envelope=cto_envelope,
)

# Create direct report envelope
report_role_env = RoleEnvelope(
    id="re-lead",
    defining_role_address="D1-R1",  # CTO defines lead envelope
    target_role_address="D1-R1-T1-R1",
    envelope=lead_envelope,
)

# Map envelopes by target_role_address
role_envelopes = {
    "D1-R1": supervisor_role_env,
    "D1-R1-T1-R1": report_role_env,
}

# Compute effective envelope for the Backend Lead
effective = compute_effective_envelope(
    role_address="D1-R1-T1-R1",
    role_envelopes=role_envelopes,
    task_envelope=None,
)

assert effective is not None
# The effective envelope is the intersection of CTO + Lead envelopes
assert effective.financial is not None
assert effective.financial.max_spend_usd == 5000.0  # min(50000, 5000)

# ── 8. Effective envelope with task envelope ─────────────────────────

effective_with_task = compute_effective_envelope(
    role_address="D1-R1-T1-R1",
    role_envelopes=role_envelopes,
    task_envelope=task_env,
)

assert effective_with_task is not None
assert effective_with_task.financial is not None
assert effective_with_task.financial.max_spend_usd == 500.0  # min(5000, 500)

# ── 9. Expired task envelopes are ignored ────────────────────────────

effective_expired_task = compute_effective_envelope(
    role_address="D1-R1-T1-R1",
    role_envelopes=role_envelopes,
    task_envelope=expired_task,
)

assert effective_expired_task is not None
# Expired task is skipped, so result equals role-only effective
assert effective_expired_task.financial is not None
assert effective_expired_task.financial.max_spend_usd == 5000.0

# ── 10. No envelopes means maximally permissive ─────────────────────

no_envelope = compute_effective_envelope(
    role_address="D1-R1",
    role_envelopes={},
    task_envelope=None,
)
assert no_envelope is None  # None = no constraints (maximally permissive)

# ── 11. NaN/Inf validation — security invariant ─────────────────────
# NaN bypasses all numeric comparisons. Inf bypasses budget checks.
# The SDK rejects both explicitly.

try:
    FinancialConstraintConfig(
        max_spend_usd=float("nan"),
    )
    assert False, "Should reject NaN"
except ValueError:
    pass  # Expected: NaN values bypass governance checks

try:
    FinancialConstraintConfig(
        max_spend_usd=float("inf"),
    )
    assert False, "Should reject Inf"
except ValueError:
    pass  # Expected: Inf values bypass governance checks

# ── 12. Dimension scope for intersections ────────────────────────────
# When dimension_scope is provided, only those dimensions are intersected
# from the second envelope. Others are taken from the first unchanged.

base = ConstraintEnvelopeConfig(
    id="env-base",
    financial=FinancialConstraintConfig(max_spend_usd=10000.0),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy"],
    ),
)

override = ConstraintEnvelopeConfig(
    id="env-override",
    financial=FinancialConstraintConfig(max_spend_usd=1000.0),
    operational=OperationalConstraintConfig(
        allowed_actions=["read"],
    ),
)

# Only intersect financial, keep operational from base
scoped = intersect_envelopes(
    base,
    override,
    dimension_scope=frozenset({"financial"}),
)

assert scoped.financial is not None
assert scoped.financial.max_spend_usd == 1000.0  # Intersected
# Operational is preserved from base (not intersected)
assert "deploy" in scoped.operational.allowed_actions

print("PASS: 06-pact/04_envelopes")
