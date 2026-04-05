# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ══════════════════════��═════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / Knowledge Items & Share Policies
# ════════════════════��═══════════════════════════════════════════════════
# OBJECTIVE: Create and classify knowledge items with compartments
# LEVEL: Advanced
# PARITY: Python-only — KSP integration is Python-specific
# VALIDATES: KnowledgeItem, KnowledgeSharePolicy
#
# Run: uv run python textbook/python/06-pact/06_knowledge.py
# ══════════���════════════════��════════════════════════════════════════════
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    AccessDecision,
    KnowledgeItem,
    KnowledgeSharePolicy,
    RoleClearance,
    RoleDefinition,
    VettingStatus,
    can_access,
    compile_org,
)
from kailash.trust.pact.config import DepartmentConfig, OrgDefinition, TeamConfig

TrustPostureLevel = TrustPosture

# ── 1. Create a KnowledgeItem ───────���────────────────────────────────
# KnowledgeItem is the target object in access decisions. It carries a
# classification level, owning unit, and optional compartments.

public_doc = KnowledgeItem(
    item_id="doc-company-handbook",
    classification=ConfidentialityLevel.PUBLIC,
    owning_unit_address="D1",
    description="Employee handbook - public information",
)

assert public_doc.item_id == "doc-company-handbook"
assert public_doc.classification == ConfidentialityLevel.PUBLIC
assert public_doc.owning_unit_address == "D1"
assert len(public_doc.compartments) == 0

# ── 2. KnowledgeItem is frozen ───────���───────────────────────────────

try:
    public_doc.classification = ConfidentialityLevel.SECRET  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen=True prevents mutation

# ��─ 3. Classified items with compartments ──────��─────────────────────

secret_investigation = KnowledgeItem(
    item_id="case-2026-001",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1-R1-T1",
    compartments=frozenset({"aml-cases", "sanctions"}),
    description="Active AML investigation case file",
)

assert secret_investigation.classification == ConfidentialityLevel.SECRET
assert "aml-cases" in secret_investigation.compartments
assert "sanctions" in secret_investigation.compartments
assert len(secret_investigation.compartments) == 2

# ── 4. Classification levels for different data types ────────────────

items_by_level = {
    "public": KnowledgeItem(
        item_id="press-release",
        classification=ConfidentialityLevel.PUBLIC,
        owning_unit_address="D1",
    ),
    "restricted": KnowledgeItem(
        item_id="internal-memo",
        classification=ConfidentialityLevel.RESTRICTED,
        owning_unit_address="D1",
    ),
    "confidential": KnowledgeItem(
        item_id="salary-data",
        classification=ConfidentialityLevel.CONFIDENTIAL,
        owning_unit_address="D2",
    ),
    "secret": KnowledgeItem(
        item_id="merger-plans",
        classification=ConfidentialityLevel.SECRET,
        owning_unit_address="D1",
        compartments=frozenset({"board-only"}),
    ),
    "top_secret": KnowledgeItem(
        item_id="security-keys",
        classification=ConfidentialityLevel.TOP_SECRET,
        owning_unit_address="D1",
        compartments=frozenset({"infosec", "key-management"}),
    ),
}

assert len(items_by_level) == 5

# ── 5. KnowledgeSharePolicy (KSP) — cross-unit sharing ──────────────
# KSPs grant one organizational unit read access to another's knowledge.
# They are directional: source shares WITH target.

ksp = KnowledgeSharePolicy(
    id="ksp-eng-to-finance",
    source_unit_address="D1",  # Engineering shares
    target_unit_address="D2",  # Finance receives
    max_classification=ConfidentialityLevel.RESTRICTED,
    created_by_role_address="D1-R1",
    active=True,
)

assert ksp.source_unit_address == "D1"
assert ksp.target_unit_address == "D2"
assert ksp.max_classification == ConfidentialityLevel.RESTRICTED
assert ksp.active is True
assert ksp.expires_at is None  # No expiry

# ── 6. KSP is frozen ────────���───────────────────────────────────────

try:
    ksp.active = False  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen=True

# ── 7. KSP with compartment restrictions ────────────────────────────
# Compartments on a KSP restrict which compartments can be shared.

compartmented_ksp = KnowledgeSharePolicy(
    id="ksp-compliance-share",
    source_unit_address="D1-R1-T1",
    target_unit_address="D2",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    compartments=frozenset({"compliance-reports"}),
    created_by_role_address="D1-R1",
    active=True,
)

assert "compliance-reports" in compartmented_ksp.compartments

# ── 8. KSP with expiry ──────────────────────────────────────────────

expiring_ksp = KnowledgeSharePolicy(
    id="ksp-temp",
    source_unit_address="D1",
    target_unit_address="D2",
    max_classification=ConfidentialityLevel.RESTRICTED,
    created_by_role_address="D1-R1",
    active=True,
    expires_at=datetime.now(UTC) + timedelta(days=30),
)

assert expiring_ksp.expires_at is not None

# ── 9. KSP in action — cross-department access ─────���────────────────
# Build a small org to demonstrate KSP-granted access.

org_def = OrgDefinition(
    org_id="ksp-demo",
    name="KSP Demo",
    departments=[
        DepartmentConfig(department_id="eng", name="Engineering"),
        DepartmentConfig(department_id="fin", name="Finance"),
    ],
    roles=[
        RoleDefinition(role_id="cto", name="CTO", is_primary_for_unit="eng"),
        RoleDefinition(role_id="cfo", name="CFO", is_primary_for_unit="fin"),
        RoleDefinition(
            role_id="analyst",
            name="Financial Analyst",
            reports_to_role_id="cfo",
        ),
    ],
)

compiled = compile_org(org_def)

clearances = {
    "D2-R1-R1": RoleClearance(
        role_address="D2-R1-R1",
        max_clearance=ConfidentialityLevel.RESTRICTED,
        vetting_status=VettingStatus.ACTIVE,
    ),
}

eng_memo = KnowledgeItem(
    item_id="eng-status-update",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1",  # Owned by Engineering
)

# Without KSP: Finance analyst cannot access Engineering data
without_ksp = can_access(
    role_address="D2-R1-R1",
    knowledge_item=eng_memo,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[],
    bridges=[],
)

assert without_ksp.allowed is False
assert without_ksp.step_failed == 5

# With KSP: Finance analyst gets cross-department access
cross_dept_ksp = KnowledgeSharePolicy(
    id="ksp-eng-fin",
    source_unit_address="D1",
    target_unit_address="D2",
    max_classification=ConfidentialityLevel.RESTRICTED,
    created_by_role_address="D1-R1",
    active=True,
)

with_ksp = can_access(
    role_address="D2-R1-R1",
    knowledge_item=eng_memo,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances,
    ksps=[cross_dept_ksp],
    bridges=[],
)

assert with_ksp.allowed is True
assert with_ksp.audit_details.get("step") == "4d"
assert with_ksp.audit_details.get("access_path") == "ksp"

# ── 10. KSP classification cap ──────────────────────────────────────
# KSP max_classification limits what can be shared. A CONFIDENTIAL item
# cannot be shared through a RESTRICTED-capped KSP.

confidential_eng_doc = KnowledgeItem(
    item_id="eng-confidential",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D1",
)

# Upgrade analyst clearance for this test
clearances_upgraded = {
    "D2-R1-R1": RoleClearance(
        role_address="D2-R1-R1",
        max_clearance=ConfidentialityLevel.CONFIDENTIAL,
        vetting_status=VettingStatus.ACTIVE,
    ),
}

# KSP only allows RESTRICTED, so CONFIDENTIAL item is blocked
capped = can_access(
    role_address="D2-R1-R1",
    knowledge_item=confidential_eng_doc,
    posture=TrustPosture.DELEGATED,
    compiled_org=compiled,
    clearances=clearances_upgraded,
    ksps=[cross_dept_ksp],  # max_classification=RESTRICTED
    bridges=[],
)

assert capped.allowed is False
assert capped.step_failed == 5  # No access path (KSP cap exceeded)

# ── 11. Owning unit semantics ───────────────────────────────────────
# The owning_unit_address determines containment. It should be a D or T
# prefix, not a role address.

team_owned = KnowledgeItem(
    item_id="team-retro-notes",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",  # Owned by a team under Engineering
)

dept_owned = KnowledgeItem(
    item_id="dept-budget",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1",  # Owned by the department itself
)

# Both have different containment scope but use the same pattern
assert team_owned.owning_unit_address.startswith("D1")
assert dept_owned.owning_unit_address == "D1"

print("PASS: 06-pact/06_knowledge")
