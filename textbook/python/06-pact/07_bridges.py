# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / Cross-Functional Bridges
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Configure cross-boundary access bridges between roles
# LEVEL: Advanced
# PARITY: Python-only — bridge configuration details are Python-specific
# VALIDATES: PactBridge, bridge configuration, bilateral/unilateral access
#
# Run: uv run python textbook/python/06-pact/07_bridges.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    KnowledgeItem,
    KnowledgeSharePolicy,
    PactBridge,
    RoleClearance,
    RoleDefinition,
    VettingStatus,
    can_access,
    compile_org,
)
from kailash.trust.pact.config import DepartmentConfig, OrgDefinition, TeamConfig

TrustPostureLevel = TrustPosture

# ── Setup: Two-department org ────────────────────────────────────────

org_def = OrgDefinition(
    org_id="bridge-demo",
    name="Bridge Demo Corp",
    departments=[
        DepartmentConfig(department_id="engineering", name="Engineering"),
        DepartmentConfig(department_id="security", name="Security"),
    ],
    teams=[
        TeamConfig(id="backend", name="Backend Team", workspace="ws"),
        TeamConfig(id="soc", name="SOC Team", workspace="ws"),
    ],
    roles=[
        RoleDefinition(
            role_id="cto",
            name="CTO",
            is_primary_for_unit="engineering",
        ),
        RoleDefinition(
            role_id="backend-lead",
            name="Backend Lead",
            reports_to_role_id="cto",
            is_primary_for_unit="backend",
        ),
        RoleDefinition(
            role_id="ciso",
            name="CISO",
            is_primary_for_unit="security",
        ),
        RoleDefinition(
            role_id="soc-lead",
            name="SOC Lead",
            reports_to_role_id="ciso",
            is_primary_for_unit="soc",
        ),
    ],
)

compiled = compile_org(org_def)

# Verify structure
assert compiled.get_node("D1-R1").name == "CTO"
assert compiled.get_node("D1-R1-T1-R1").name == "Backend Lead"
assert compiled.get_node("D2-R1").name == "CISO"
assert compiled.get_node("D2-R1-T1-R1").name == "SOC Lead"

# Clearances for all roles
clearances: dict[str, RoleClearance] = {
    "D1-R1": RoleClearance(
        role_address="D1-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D1-R1-T1-R1": RoleClearance(
        role_address="D1-R1-T1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D2-R1": RoleClearance(
        role_address="D2-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D2-R1-T1-R1": RoleClearance(
        role_address="D2-R1-T1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
}

# ── 1. Create a PactBridge ──────────────────────────────────────────
# Bridges grant role-level access paths across organizational boundaries.
# Three types: "standing" (permanent), "scoped" (operation-limited),
# "ad_hoc" (temporary).

standing_bridge = PactBridge(
    id="bridge-eng-sec",
    role_a_address="D1-R1-T1-R1",  # Backend Lead
    role_b_address="D2-R1-T1-R1",  # SOC Lead
    bridge_type="standing",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    bilateral=True,
    active=True,
)

assert standing_bridge.id == "bridge-eng-sec"
assert standing_bridge.bridge_type == "standing"
assert standing_bridge.bilateral is True
assert standing_bridge.active is True
assert standing_bridge.expires_at is None

# ── 2. PactBridge is frozen ─────────────────────────────────────────

try:
    standing_bridge.active = False  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen=True

# ── 3. Bilateral bridge — both sides can access ────────────────────

eng_doc = KnowledgeItem(
    item_id="doc-backend-architecture",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",  # Backend Team
)

sec_doc = KnowledgeItem(
    item_id="doc-incident-report",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D2-R1-T1",  # SOC Team
)

# Without bridge: SOC Lead cannot access Backend Team data
without_bridge = can_access(
    role_address="D2-R1-T1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert without_bridge.allowed is False
assert without_bridge.step_failed == 5

# With bilateral bridge: SOC Lead CAN access Backend Team data
with_bridge = can_access(
    role_address="D2-R1-T1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[standing_bridge],
)

assert with_bridge.allowed is True
assert with_bridge.audit_details.get("step") == "4e"
assert with_bridge.audit_details.get("access_path") == "bridge"

# Bilateral: Backend Lead can also access SOC Team data
reverse_access = can_access(
    role_address="D1-R1-T1-R1",
    knowledge_item=sec_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[standing_bridge],
)

assert reverse_access.allowed is True

# ── 4. Unilateral bridge — one direction only ───────────────────────
# bilateral=False means only role_a can access role_b's data.
# role_b cannot access role_a's data through this bridge.

unilateral_bridge = PactBridge(
    id="bridge-sec-reads-eng",
    role_a_address="D2-R1-T1-R1",  # SOC Lead (reader)
    role_b_address="D1-R1-T1-R1",  # Backend Lead (data owner side)
    bridge_type="standing",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    bilateral=False,
    active=True,
)

# A->B direction (SOC reads Eng): ALLOWED
a_to_b = can_access(
    role_address="D2-R1-T1-R1",  # role_a
    knowledge_item=eng_doc,  # owned by role_b's domain
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[unilateral_bridge],
)

assert a_to_b.allowed is True

# B->A direction (Eng reads SOC): DENIED (unilateral)
b_to_a = can_access(
    role_address="D1-R1-T1-R1",  # role_b
    knowledge_item=sec_doc,  # owned by role_a's domain
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[unilateral_bridge],
)

assert b_to_a.allowed is False
assert b_to_a.step_failed == 5

# ── 5. Bridge classification cap ────────────────────────────────────
# Bridges have a max_classification that limits what can be accessed.

confidential_doc = KnowledgeItem(
    item_id="doc-confidential-eng",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D1-R1-T1",
)

restricted_bridge = PactBridge(
    id="bridge-restricted",
    role_a_address="D2-R1-T1-R1",
    role_b_address="D1-R1-T1-R1",
    bridge_type="standing",
    max_classification=ConfidentialityLevel.RESTRICTED,  # Only RESTRICTED
    bilateral=True,
    active=True,
)

# CONFIDENTIAL doc through RESTRICTED-capped bridge: DENIED
capped_access = can_access(
    role_address="D2-R1-T1-R1",
    knowledge_item=confidential_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[restricted_bridge],
)

assert capped_access.allowed is False
assert capped_access.step_failed == 5  # No path (bridge cap exceeded)

# ── 6. Scoped bridge — limited to specific operations ───────────────

scoped_bridge = PactBridge(
    id="bridge-scoped",
    role_a_address="D2-R1-T1-R1",
    role_b_address="D1-R1-T1-R1",
    bridge_type="scoped",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    operational_scope=("incident_response", "security_audit"),
    bilateral=False,
    active=True,
)

assert scoped_bridge.bridge_type == "scoped"
assert "incident_response" in scoped_bridge.operational_scope
assert "security_audit" in scoped_bridge.operational_scope

# ── 7. Ad-hoc bridge with expiry ────────────────────────────────────

ad_hoc_bridge = PactBridge(
    id="bridge-adhoc-001",
    role_a_address="D1-R1-T1-R1",
    role_b_address="D2-R1-T1-R1",
    bridge_type="ad_hoc",
    max_classification=ConfidentialityLevel.RESTRICTED,
    bilateral=True,
    active=True,
    expires_at=datetime.now(UTC) + timedelta(hours=4),
)

assert ad_hoc_bridge.bridge_type == "ad_hoc"
assert ad_hoc_bridge.expires_at is not None

# Active ad-hoc bridge works
adhoc_access = can_access(
    role_address="D1-R1-T1-R1",
    knowledge_item=KnowledgeItem(
        item_id="doc-quick-share",
        classification=ConfidentialityLevel.RESTRICTED,
        owning_unit_address="D2-R1-T1",
    ),
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[ad_hoc_bridge],
)

assert adhoc_access.allowed is True

# ── 8. Expired bridge is treated as non-existent ────────────────────

expired_bridge = PactBridge(
    id="bridge-expired",
    role_a_address="D2-R1-T1-R1",
    role_b_address="D1-R1-T1-R1",
    bridge_type="ad_hoc",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    bilateral=True,
    active=True,
    expires_at=datetime.now(UTC) - timedelta(hours=1),  # Already expired
)

expired_access = can_access(
    role_address="D2-R1-T1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[expired_bridge],
)

assert expired_access.allowed is False
assert expired_access.step_failed == 5

# ── 9. Inactive bridge is skipped ───────────────────────────────────

inactive_bridge = PactBridge(
    id="bridge-inactive",
    role_a_address="D2-R1-T1-R1",
    role_b_address="D1-R1-T1-R1",
    bridge_type="standing",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    bilateral=True,
    active=False,  # Deactivated
)

inactive_access = can_access(
    role_address="D2-R1-T1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[inactive_bridge],
)

assert inactive_access.allowed is False

# ── 10. Bridges are role-level, not inherited by descendants ────────
# A bridge to Backend Lead does NOT give Developer 1 access.
# Descendant access is governed by KSPs (unit-level), not bridges.

# Backend Lead has a bridge, but a hypothetical dev under them does not.
# (We did not add a dev role here, but the principle is demonstrated
# by the bridge's _role_matches_bridge_side using exact match.)

# Verify bridge structure
assert standing_bridge.role_a_address == "D1-R1-T1-R1"
assert standing_bridge.role_b_address == "D2-R1-T1-R1"

# CTO (D1-R1) is NOT a bridge endpoint, so the bridge does not help them
# access security data through the bridge.
cto_via_bridge = can_access(
    role_address="D1-R1",
    knowledge_item=sec_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[standing_bridge],
)

# CTO may get access via downward visibility (4b) to their own domain,
# but for cross-department security data, the bridge is role-specific.
# Since CTO is not an endpoint of this bridge, access depends on
# structural path only.
assert cto_via_bridge.step_failed == 5 or not cto_via_bridge.allowed

print("PASS: 06-pact/07_bridges")
