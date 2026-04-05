# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / D/T/R Addressing
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Parse and validate D/T/R positional addresses
# LEVEL: Basic
# PARITY: Full — Rust has Address::parse() with identical grammar
# VALIDATES: Address, Address.parse(), AddressSegment, NodeType, depth()
#
# Run: uv run python textbook/python/06-pact/01_addressing.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash.trust.pact import (
    Address,
    AddressError,
    AddressSegment,
    GrammarError,
    NodeType,
)

# ── 1. NodeType enum ───────────────────────────────────────────────────
# PACT has three node types: Department (D), Team (T), Role (R).

assert NodeType.DEPARTMENT.value == "D"
assert NodeType.TEAM.value == "T"
assert NodeType.ROLE.value == "R"

# ── 2. Parse a single AddressSegment ──────────────────────────────────
# Segments look like "D1", "R2", "T3" — a type char plus a 1-based index.

seg = AddressSegment.parse("D1")
assert seg.node_type == NodeType.DEPARTMENT
assert seg.sequence == 1
assert str(seg) == "D1"

seg_r = AddressSegment.parse("R3")
assert seg_r.node_type == NodeType.ROLE
assert seg_r.sequence == 3

# Case-insensitive parsing
seg_lower = AddressSegment.parse("t2")
assert seg_lower.node_type == NodeType.TEAM
assert seg_lower.sequence == 2

# ── 3. Parse a full Address ───────────────────────────────────────────
# Addresses are hyphen-separated segments: D1-R1-D2-R1-T1-R1
# Grammar rule: every D or T must be immediately followed by exactly one R.

addr = Address.parse("D1-R1-D2-R1-T1-R1")
assert len(addr.segments) == 6
assert addr.depth == 6
assert str(addr) == "D1-R1-D2-R1-T1-R1"

# ── 4. Segments are frozen dataclasses ────────────────────────────────

assert addr.segments[0] == AddressSegment(node_type=NodeType.DEPARTMENT, sequence=1)
assert addr.segments[1] == AddressSegment(node_type=NodeType.ROLE, sequence=1)

# ── 5. last_segment property ──────────────────────────────────────────

assert addr.last_segment.node_type == NodeType.ROLE
assert addr.last_segment.sequence == 1

# ── 6. parent property ───────────────────────────────────────────────

parent = addr.parent
assert parent is not None
assert str(parent) == "D1-R1-D2-R1-T1"

# Root-level address has no parent
root = Address.parse("D1-R1")
assert root.parent is not None  # D1 is the structural parent
assert root.parent.parent is None  # D1 has no parent

# ── 7. containment_unit — nearest D or T ancestor ────────────────────

cu = addr.containment_unit
assert cu is not None
assert str(cu) == "D1-R1-D2-R1-T1"

# A role directly under a department
dept_role = Address.parse("D1-R1")
assert dept_role.containment_unit is not None
assert str(dept_role.containment_unit) == "D1"

# ── 8. accountability_chain — all R segments ─────────────────────────
# The chain of accountable people from root to leaf.

chain = addr.accountability_chain
assert len(chain) == 3  # D1-R1, D1-R1-D2-R1, D1-R1-D2-R1-T1-R1
assert str(chain[0]) == "D1-R1"
assert str(chain[1]) == "D1-R1-D2-R1"
assert str(chain[2]) == "D1-R1-D2-R1-T1-R1"

# ── 9. is_prefix_of and is_ancestor_of ───────────────────────────────

prefix = Address.parse("D1-R1")
full = Address.parse("D1-R1-D2-R1")

assert prefix.is_prefix_of(full), "D1-R1 is a proper prefix of D1-R1-D2-R1"
assert not full.is_prefix_of(prefix), "Not a prefix in reverse"
assert not prefix.is_prefix_of(prefix), "is_prefix_of is strict (not reflexive)"

assert prefix.is_ancestor_of(full), "is_ancestor_of is reflexive prefix check"
assert prefix.is_ancestor_of(prefix), "is_ancestor_of includes self"

# ── 10. ancestors() ──────────────────────────────────────────────────

anc = full.ancestors()
assert len(anc) == 3  # D1, D1-R1, D1-R1-D2
assert str(anc[0]) == "D1"
assert str(anc[1]) == "D1-R1"
assert str(anc[2]) == "D1-R1-D2"

# ── 11. lowest_common_ancestor ───────────────────────────────────────

addr_a = Address.parse("D1-R1-D2-R1-T1-R1")
addr_b = Address.parse("D1-R1-D2-R1-T2-R1")
lca = Address.lowest_common_ancestor(addr_a, addr_b)
assert lca is not None
assert str(lca) == "D1-R1-D2-R1"

# Disjoint trees have no common ancestor
disjoint_a = Address.parse("D1-R1")
disjoint_b = Address.parse("D2-R1")
assert Address.lowest_common_ancestor(disjoint_a, disjoint_b) is None

# ── 12. from_segments constructor ────────────────────────────────────

constructed = Address.from_segments(
    AddressSegment(NodeType.DEPARTMENT, 1),
    AddressSegment(NodeType.ROLE, 1),
)
assert str(constructed) == "D1-R1"

# ── 13. Grammar violations ──────────────────────────────────────────
# Every D or T must be immediately followed by R.

# D without trailing R
try:
    Address.parse("D1")
    assert False, "Should raise GrammarError"
except GrammarError:
    pass  # Expected: D1 has no trailing Role

# Two departments in a row (D-D without R between)
try:
    Address.parse("D1-D2-R1")
    assert False, "Should raise GrammarError"
except GrammarError:
    pass  # Expected: D1 not followed by R

# Team without trailing R
try:
    Address.parse("D1-R1-T1")
    assert False, "Should raise GrammarError"
except GrammarError:
    pass  # Expected: T1 has no trailing Role

# ── 14. AddressError for malformed segments ──────────────────────────

# Invalid type character
try:
    AddressSegment.parse("X1")
    assert False, "Should raise AddressError"
except AddressError:
    pass  # Expected: 'X' is not D, T, or R

# Empty address string
try:
    Address.parse("")
    assert False, "Should raise AddressError"
except AddressError:
    pass  # Expected: empty string

# Non-positive sequence
try:
    AddressSegment.parse("D0")
    assert False, "Should raise AddressError"
except AddressError:
    pass  # Expected: sequence must be >= 1

print("PASS: 06-pact/01_addressing")
