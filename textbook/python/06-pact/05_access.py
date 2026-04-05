# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / 5-Step Access Enforcement
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Enforce 5-step access control decisions
# LEVEL: Intermediate
# PARITY: Full — Rust has equivalent can_access() with identical steps
# VALIDATES: can_access(), AccessDecision, explain_access()
#
# Run: uv run python textbook/python/06-pact/05_access.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    AccessDecision,
    KnowledgeItem,
    KnowledgeSharePolicy,
    PactBridge,
    RoleClearance,
    RoleDefinition,
    VettingStatus,
    can_access,
    compile_org,
    explain_access,
)
from kailash.trust.pact.config import DepartmentConfig, OrgDefinition, TeamConfig

TrustPostureLevel = TrustPosture

# ── Setup: Build a realistic org for access testing ──────────────────
# Two departments: Engineering and Finance. Engineering has a Backend team.
# This creates cross-boundary scenarios for all 5 access steps.

org_def = OrgDefinition(
    org_id="access-demo",
    name="Access Demo Corp",
    departments=[
        DepartmentConfig(department_id="engineering", name="Engineering"),
        DepartmentConfig(department_id="finance", name="Finance"),
    ],
    teams=[
        TeamConfig(id="backend", name="Backend Team", workspace="ws"),
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
            role_id="dev-1",
            name="Developer 1",
            reports_to_role_id="backend-lead",
        ),
        RoleDefinition(
            role_id="cfo",
            name="CFO",
            is_primary_for_unit="finance",
        ),
        RoleDefinition(
            role_id="accountant",
            name="Accountant",
            reports_to_role_id="cfo",
        ),
    ],
)

compiled = compile_org(org_def)

# Verify addresses: D1=Engineering, D2=Finance
assert compiled.get_node("D1").name == "Engineering"
assert compiled.get_node("D1-R1").name == "CTO"
assert compiled.get_node("D1-R1-T1").name == "Backend Team"
assert compiled.get_node("D1-R1-T1-R1").name == "Backend Lead"
assert compiled.get_node("D1-R1-T1-R1-R1").name == "Developer 1"
assert compiled.get_node("D2").name == "Finance"
assert compiled.get_node("D2-R1").name == "CFO"
assert compiled.get_node("D2-R1-R1").name == "Accountant"

# ── Setup: Clearances ────────────────────────────────────────────────

clearances: dict[str, RoleClearance] = {
    "D1-R1": RoleClearance(
        role_address="D1-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        compartments=frozenset({"engineering-internal"}),
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D1-R1-T1-R1": RoleClearance(
        role_address="D1-R1-T1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D1-R1-T1-R1-R1": RoleClearance(
        role_address="D1-R1-T1-R1-R1",
        max_clearance=ConfidentialityLevel.RESTRICTED,
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D2-R1": RoleClearance(
        role_address="D2-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        compartments=frozenset({"financial-records"}),
        vetting_status=VettingStatus.ACTIVE,
    ),
    "D2-R1-R1": RoleClearance(
        role_address="D2-R1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
}

# ── Setup: Knowledge items ───────────────────────────────────────────

eng_doc = KnowledgeItem(
    item_id="doc-arch-review",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",  # Owned by Backend Team
    description="Architecture review document",
)

finance_report = KnowledgeItem(
    item_id="doc-quarterly-report",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D2",  # Owned by Finance department
    description="Quarterly financial report",
)

secret_doc = KnowledgeItem(
    item_id="doc-secret-plan",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1",  # Owned by Engineering department
    compartments=frozenset({"engineering-internal"}),
    description="Secret engineering plan",
)

# ── 1. Step 1: Missing clearance -> DENY ────────────────────────────

no_clearance_decision = can_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances={},  # Empty — no clearances at all
    ksps=[],
    bridges=[],
)

assert no_clearance_decision.allowed is False
assert no_clearance_decision.step_failed == 1
assert "No clearance found" in no_clearance_decision.reason

# ── 2. Step 1: Non-ACTIVE vetting -> DENY ───────────────────────────

expired_clearances = {
    "D1-R1-T1-R1-R1": RoleClearance(
        role_address="D1-R1-T1-R1-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        vetting_status=VettingStatus.EXPIRED,
    ),
}

expired_decision = can_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=expired_clearances,
    ksps=[],
    bridges=[],
)

assert expired_decision.allowed is False
assert expired_decision.step_failed == 1
assert "ACTIVE" in expired_decision.reason

# ── 3. Step 2: Classification exceeds clearance -> DENY ─────────────
# Developer has RESTRICTED clearance, finance report is CONFIDENTIAL.

classification_decision = can_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=finance_report,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert classification_decision.allowed is False
assert classification_decision.step_failed == 2
assert "below" in classification_decision.reason.lower()

# ── 4. Step 2: Posture caps clearance ───────────────────────────────
# CTO has SECRET clearance, but at SUPERVISED posture the ceiling
# is RESTRICTED. A CONFIDENTIAL item is denied.

posture_decision = can_access(
    role_address="D1-R1",
    knowledge_item=finance_report,  # CONFIDENTIAL
    posture=TrustPosture.SUPERVISED,  # Ceiling = RESTRICTED
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert posture_decision.allowed is False
assert posture_decision.step_failed == 2

# ── 5. Step 3: Missing compartments -> DENY ─────────────────────────
# CFO has SECRET clearance with "financial-records" compartment, but
# the secret doc requires "engineering-internal" compartment.

compartment_decision = can_access(
    role_address="D2-R1",
    knowledge_item=secret_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert compartment_decision.allowed is False
assert compartment_decision.step_failed == 3
assert "compartment" in compartment_decision.reason.lower()

# ── 6. Step 4a: Same unit -> ALLOW ──────────────────────────────────
# Developer accessing a document owned by their own team.

same_unit_decision = can_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=eng_doc,  # Owned by D1-R1-T1 (Backend Team)
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert same_unit_decision.allowed is True
assert same_unit_decision.step_failed is None
assert "same unit" in same_unit_decision.reason.lower() or "4a" in str(
    same_unit_decision.audit_details.get("step", "")
)

# ── 7. Step 4b: Downward visibility -> ALLOW ────────────────────────
# CTO at D1-R1 has downward visibility to everything under D1-R1-*.

downward_decision = can_access(
    role_address="D1-R1",
    knowledge_item=eng_doc,  # Owned by D1-R1-T1
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert downward_decision.allowed is True

# ── 8. Step 5: No path found -> DENY (fail-closed) ──────────────────
# CFO trying to access engineering doc — different department, no KSP,
# no bridge.

cross_dept_decision = can_access(
    role_address="D2-R1",
    knowledge_item=eng_doc,  # Owned by D1-R1-T1 (Engineering)
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert cross_dept_decision.allowed is False
assert cross_dept_decision.step_failed == 5
assert "no access path" in cross_dept_decision.reason.lower()

# ── 9. AccessDecision structure ──────────────────────────────────────

assert isinstance(same_unit_decision, AccessDecision)
assert isinstance(same_unit_decision.audit_details, dict)
assert "role_address" in same_unit_decision.audit_details
assert "item_id" in same_unit_decision.audit_details

# ── 10. explain_access — human-readable trace ────────────────────────

trace = explain_access(
    role_address="D1-R1-T1-R1-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert isinstance(trace, str)
assert "Step 1" in trace
assert "Step 2" in trace
assert "PASS" in trace or "ALLOWED" in trace

# Trace for a denied access
denied_trace = explain_access(
    role_address="D2-R1",
    knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert "DENIED" in denied_trace

print("PASS: 06-pact/05_access")
