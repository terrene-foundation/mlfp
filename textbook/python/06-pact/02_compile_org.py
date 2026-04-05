# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — PACT / Organization Compilation
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Compile an organization definition into a CompiledOrg
# LEVEL: Basic
# PARITY: Full — Rust has compile_org() with identical semantics
# VALIDATES: compile_org(), OrgDefinition input types, CompiledOrg output
#
# Run: uv run python textbook/python/06-pact/02_compile_org.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kailash.trust.pact import (
    Address,
    CompilationError,
    CompiledOrg,
    NodeType,
    OrgNode,
    RoleDefinition,
    compile_org,
)
from kailash.trust.pact.config import DepartmentConfig, OrgDefinition, TeamConfig

# ── 1. Define a minimal organization ─────────────────────────────────
# OrgDefinition is the declarative input; compile_org() transforms it
# into address-indexed CompiledOrg with O(1) lookups.

org_def = OrgDefinition(
    org_id="acme-corp",
    name="Acme Corporation",
    departments=[
        DepartmentConfig(
            department_id="engineering",
            name="Engineering",
        ),
    ],
    teams=[
        TeamConfig(
            id="backend",
            name="Backend Team",
            workspace="ws-eng",
        ),
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
    ],
)

# ── 2. Compile the organization ──────────────────────────────────────

compiled = compile_org(org_def)

assert isinstance(compiled, CompiledOrg)
assert compiled.org_id == "acme-corp"

# ── 3. Inspect compiled nodes ────────────────────────────────────────
# The compiler assigns positional D/T/R addresses to every node.

# The Engineering department gets D1
eng_dept = compiled.get_node("D1")
assert eng_dept.node_type == NodeType.DEPARTMENT
assert eng_dept.name == "Engineering"
assert eng_dept.node_id == "engineering"

# CTO heads the department at D1-R1
cto_node = compiled.get_node("D1-R1")
assert cto_node.node_type == NodeType.ROLE
assert cto_node.name == "CTO"
assert cto_node.role_definition is not None
assert cto_node.role_definition.role_id == "cto"

# ── 4. Address assignment follows the hierarchy ──────────────────────
# Backend team is under CTO, so it gets D1-R1-T1
# Backend Lead heads the team at D1-R1-T1-R1

backend_team = compiled.get_node("D1-R1-T1")
assert backend_team.node_type == NodeType.TEAM
assert backend_team.name == "Backend Team"

backend_lead = compiled.get_node("D1-R1-T1-R1")
assert backend_lead.name == "Backend Lead"

# Developer 1 reports to Backend Lead, gets D1-R1-T1-R1-R1
dev1 = compiled.get_node("D1-R1-T1-R1-R1")
assert dev1.name == "Developer 1"
assert dev1.parent_address == "D1-R1-T1-R1"

# ── 5. Node count ────────────────────────────────────────────────────
# D1, D1-R1, D1-R1-T1, D1-R1-T1-R1, D1-R1-T1-R1-R1 = 5 nodes

assert len(compiled.nodes) == 5, f"Expected 5 nodes, got {len(compiled.nodes)}"

# ── 6. get_node_by_role_id ───────────────────────────────────────────
# Look up a node by its original role_id rather than by address.

found = compiled.get_node_by_role_id("cto")
assert found is not None
assert found.address == "D1-R1"

not_found = compiled.get_node_by_role_id("nonexistent")
assert not_found is None

# ── 7. query_by_prefix ───────────────────────────────────────────────
# Returns all nodes whose address starts with a given prefix.

eng_subtree = compiled.query_by_prefix("D1")
assert len(eng_subtree) == 5  # All nodes are under D1

team_subtree = compiled.query_by_prefix("D1-R1-T1")
assert len(team_subtree) == 3  # T1, T1-R1, T1-R1-R1

# ── 8. get_subtree ───────────────────────────────────────────────────
# Returns the node and all descendants. Raises KeyError if not found.

subtree = compiled.get_subtree("D1-R1-T1-R1")
assert len(subtree) == 2  # T1-R1 and T1-R1-R1

try:
    compiled.get_subtree("D99")
    assert False, "Should raise KeyError"
except KeyError:
    pass  # Expected: no node at D99

# ── 9. root_roles property ───────────────────────────────────────────
# Root-level nodes are those with no parent_address.

roots = compiled.root_roles
# D1 is the top-level department (parent_address=None)
root_addrs = [r.address for r in roots]
assert "D1" in root_addrs

# ── 10. Vacancy status ──────────────────────────────────────────────

status = compiled.get_vacancy_status("D1-R1")
assert status.address == "D1-R1"
assert status.role_id == "cto"
assert status.is_vacant is False

# ── 11. Nodes are immutable after compilation ────────────────────────
# CompiledOrg.nodes is wrapped in MappingProxyType to prevent mutation.

try:
    compiled.nodes["INJECTED"] = OrgNode(  # type: ignore[index]
        address="INJECTED",
        node_type=NodeType.ROLE,
        name="Malicious",
        node_id="evil",
    )
    assert False, "Should raise TypeError (read-only)"
except TypeError:
    pass  # Expected: MappingProxyType prevents insertion

# ── 12. parent_address chain ─────────────────────────────────────────
# Walk from leaf to root using parent_address.

leaf = compiled.get_node("D1-R1-T1-R1-R1")
parent_chain = []
current = leaf.parent_address
while current is not None:
    parent_node = compiled.get_node(current)
    parent_chain.append(parent_node.address)
    current = parent_node.parent_address

# D1-R1-T1-R1 -> D1-R1-T1 -> D1-R1 -> D1
assert parent_chain == ["D1-R1-T1-R1", "D1-R1-T1", "D1-R1", "D1"]

# ── 13. children_addresses ───────────────────────────────────────────

dept_node = compiled.get_node("D1")
assert "D1-R1" in dept_node.children_addresses

cto = compiled.get_node("D1-R1")
assert "D1-R1-T1" in cto.children_addresses

# ── 14. Auto-created vacant head roles ───────────────────────────────
# If a department or team has no head role, PACT auto-creates a vacant one.

headless_org = OrgDefinition(
    org_id="headless",
    name="Headless Org",
    departments=[
        DepartmentConfig(department_id="sales", name="Sales"),
    ],
    roles=[],  # No roles at all
)

headless_compiled = compile_org(headless_org)
# Auto-created vacant head: "sales-head-vacant" at D1-R1
sales_head = headless_compiled.get_node("D1-R1")
assert sales_head.is_vacant is True
assert sales_head.role_definition is not None
assert "vacant" in sales_head.role_definition.role_id.lower()

# ── 15. Compilation errors ───────────────────────────────────────────
# Duplicate role IDs cause CompilationError.

try:
    bad_org = OrgDefinition(
        org_id="bad",
        name="Bad Org",
        departments=[],
        roles=[
            RoleDefinition(role_id="dup", name="First"),
            RoleDefinition(role_id="dup", name="Second"),
        ],
    )
    compile_org(bad_org)
    assert False, "Should raise CompilationError"
except CompilationError:
    pass  # Expected: duplicate role IDs

# Dangling reports_to reference
try:
    bad_org2 = OrgDefinition(
        org_id="bad2",
        name="Bad Org 2",
        departments=[],
        roles=[
            RoleDefinition(
                role_id="orphan",
                name="Orphan",
                reports_to_role_id="nonexistent",
            ),
        ],
    )
    compile_org(bad_org2)
    assert False, "Should raise CompilationError"
except CompilationError:
    pass  # Expected: dangling reference

# ── 16. Empty org compiles successfully ──────────────────────────────

empty_org = OrgDefinition(org_id="empty", name="Empty")
empty_compiled = compile_org(empty_org)
assert len(empty_compiled.nodes) == 0

print("PASS: 06-pact/02_compile_org")
