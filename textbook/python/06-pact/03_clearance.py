# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / Knowledge Clearance
# ════════���═══════════════════════════════════════════════════════════════
# OBJECTIVE: Define and check classification levels and role clearance
# LEVEL: Intermediate
# PARITY: Full — Rust has equivalent clearance types and posture ceiling
# VALIDATES: RoleClearance, VettingStatus, effective_clearance, POSTURE_CEILING
#
# Run: uv run python textbook/python/06-pact/03_clearance.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash.trust import ConfidentialityLevel, TrustPosture
from kailash.trust.pact import (
    POSTURE_CEILING,
    RoleClearance,
    VettingStatus,
    effective_clearance,
)

# Backward-compatible alias used throughout PACT
TrustPostureLevel = TrustPosture

# ── 1. ConfidentialityLevel ordering ─────────────────────────────────
# PUBLIC < RESTRICTED < CONFIDENTIAL < SECRET < TOP_SECRET

assert ConfidentialityLevel.PUBLIC.value == "public"
assert ConfidentialityLevel.RESTRICTED.value == "restricted"
assert ConfidentialityLevel.CONFIDENTIAL.value == "confidential"
assert ConfidentialityLevel.SECRET.value == "secret"
assert ConfidentialityLevel.TOP_SECRET.value == "top_secret"

# ── 2. TrustPosture levels ──────────────────────────────────────────
# Five graduated posture levels from most restricted to most autonomous.

assert TrustPosture.PSEUDO_AGENT.value == "pseudo_agent"
assert TrustPosture.SUPERVISED.value == "supervised"
assert TrustPosture.SHARED_PLANNING.value == "shared_planning"
assert TrustPosture.CONTINUOUS_INSIGHT.value == "continuous_insight"
assert TrustPosture.DELEGATED.value == "delegated"

# ── 3. POSTURE_CEILING mapping ──────────────────────────────────────
# Each posture has a ceiling on what classification level a role can access.
# Even a TOP_SECRET-cleared role cannot access SECRET data at SUPERVISED posture.

assert POSTURE_CEILING[TrustPosture.PSEUDO_AGENT] == ConfidentialityLevel.PUBLIC
assert POSTURE_CEILING[TrustPosture.SUPERVISED] == ConfidentialityLevel.RESTRICTED
assert (
    POSTURE_CEILING[TrustPosture.SHARED_PLANNING] == ConfidentialityLevel.CONFIDENTIAL
)
assert POSTURE_CEILING[TrustPosture.CONTINUOUS_INSIGHT] == ConfidentialityLevel.SECRET
assert POSTURE_CEILING[TrustPosture.DELEGATED] == ConfidentialityLevel.TOP_SECRET

# ── 4. VettingStatus enum ───────────────────────────────────────────
# Clearance grants have lifecycle states. Only ACTIVE clearances are valid.

assert VettingStatus.PENDING.value == "pending"
assert VettingStatus.ACTIVE.value == "active"
assert VettingStatus.EXPIRED.value == "expired"
assert VettingStatus.REVOKED.value == "revoked"

# ── 5. Create a RoleClearance ────────────────────────────────────────
# RoleClearance is independent of authority — a junior role can hold
# higher clearance than a senior role if the knowledge domain requires it.

clearance = RoleClearance(
    role_address="D1-R1-T1-R1",
    max_clearance=ConfidentialityLevel.SECRET,
    compartments=frozenset({"aml-investigations", "sanctions"}),
    granted_by_role_address="D1-R1",
    vetting_status=VettingStatus.ACTIVE,
    nda_signed=True,
)

assert clearance.role_address == "D1-R1-T1-R1"
assert clearance.max_clearance == ConfidentialityLevel.SECRET
assert "aml-investigations" in clearance.compartments
assert clearance.vetting_status == VettingStatus.ACTIVE

# ── 6. RoleClearance is frozen (immutable) ───────────────────────────

try:
    clearance.max_clearance = ConfidentialityLevel.TOP_SECRET  # type: ignore[misc]
    assert False, "Should raise FrozenInstanceError"
except AttributeError:
    pass  # Expected: frozen=True prevents mutation

# ── 7. effective_clearance computation ───────────────────────────────
# effective = min(role.max_clearance, POSTURE_CEILING[posture])
# This is the core clearance formula in PACT.

# At DELEGATED posture (ceiling = TOP_SECRET), SECRET clearance is uncapped
eff = effective_clearance(clearance, TrustPosture.DELEGATED)
assert eff == ConfidentialityLevel.SECRET, "min(SECRET, TOP_SECRET) = SECRET"

# At CONTINUOUS_INSIGHT (ceiling = SECRET), SECRET clearance is still SECRET
eff = effective_clearance(clearance, TrustPosture.CONTINUOUS_INSIGHT)
assert eff == ConfidentialityLevel.SECRET, "min(SECRET, SECRET) = SECRET"

# At SHARED_PLANNING (ceiling = CONFIDENTIAL), SECRET clearance is capped
eff = effective_clearance(clearance, TrustPosture.SHARED_PLANNING)
assert (
    eff == ConfidentialityLevel.CONFIDENTIAL
), "min(SECRET, CONFIDENTIAL) = CONFIDENTIAL"

# At SUPERVISED (ceiling = RESTRICTED), SECRET clearance is severely capped
eff = effective_clearance(clearance, TrustPosture.SUPERVISED)
assert eff == ConfidentialityLevel.RESTRICTED, "min(SECRET, RESTRICTED) = RESTRICTED"

# At PSEUDO_AGENT (ceiling = PUBLIC), any clearance is capped to PUBLIC
eff = effective_clearance(clearance, TrustPosture.PSEUDO_AGENT)
assert eff == ConfidentialityLevel.PUBLIC, "min(SECRET, PUBLIC) = PUBLIC"

# ── 8. Clearance with different max levels ───────────────────────────

public_clearance = RoleClearance(
    role_address="D1-R1-R1",
    max_clearance=ConfidentialityLevel.PUBLIC,
    vetting_status=VettingStatus.ACTIVE,
)

# Even at DELEGATED posture, PUBLIC clearance stays PUBLIC
eff = effective_clearance(public_clearance, TrustPosture.DELEGATED)
assert eff == ConfidentialityLevel.PUBLIC, "min(PUBLIC, TOP_SECRET) = PUBLIC"

top_secret_clearance = RoleClearance(
    role_address="D1-R1",
    max_clearance=ConfidentialityLevel.TOP_SECRET,
    vetting_status=VettingStatus.ACTIVE,
)

# TOP_SECRET clearance at DELEGATED posture gives full access
eff = effective_clearance(top_secret_clearance, TrustPosture.DELEGATED)
assert eff == ConfidentialityLevel.TOP_SECRET

# But at SUPERVISED, even TOP_SECRET is capped to RESTRICTED
eff = effective_clearance(top_secret_clearance, TrustPosture.SUPERVISED)
assert eff == ConfidentialityLevel.RESTRICTED

# ── 9. Non-ACTIVE clearance ─────────────────────────────────────────
# Clearances with non-ACTIVE vetting are rejected at Step 1 of access
# enforcement. The effective_clearance function itself does not check
# vetting status — that is done in the 5-step access algorithm.

expired_clearance = RoleClearance(
    role_address="D1-R1-T1-R1-R1",
    max_clearance=ConfidentialityLevel.SECRET,
    vetting_status=VettingStatus.EXPIRED,
)

# effective_clearance still computes a value (it does not enforce vetting)
eff = effective_clearance(expired_clearance, TrustPosture.DELEGATED)
assert eff == ConfidentialityLevel.SECRET

# But access enforcement (covered in 05_access.py) will reject this
# at Step 1 because vetting_status != ACTIVE.

# ── 10. Compartments ────────────────────────────────────────────────
# Compartments are used for SECRET and TOP_SECRET data. A role must
# hold ALL compartments an item belongs to.

compartmented_clearance = RoleClearance(
    role_address="D1-R1-T1-R1",
    max_clearance=ConfidentialityLevel.TOP_SECRET,
    compartments=frozenset({"nuclear", "cyber", "humint"}),
    vetting_status=VettingStatus.ACTIVE,
    nda_signed=True,
)

assert len(compartmented_clearance.compartments) == 3
assert "nuclear" in compartmented_clearance.compartments
assert "cyber" in compartmented_clearance.compartments

# Default clearance has empty compartments
basic_clearance = RoleClearance(
    role_address="D1-R1-R1",
    max_clearance=ConfidentialityLevel.RESTRICTED,
)
assert len(basic_clearance.compartments) == 0
assert basic_clearance.vetting_status == VettingStatus.ACTIVE  # Default

print("PASS: 06-pact/03_clearance")
