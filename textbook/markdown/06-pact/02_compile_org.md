# Chapter 2: Organization Compilation

## Overview

Raw organization definitions are declarative -- they describe departments, teams, and roles with human-readable IDs. Before PACT can make governance decisions, this definition must be **compiled** into an address-indexed structure with O(1) lookups. This chapter teaches you how to define organizations, compile them, and query the resulting structure.

## Prerequisites

- [Chapter 1: D/T/R Addressing](01_addressing.md)
- Understanding of tree data structures

## Concepts

### What is Organization Compilation?

Compilation transforms a declarative `OrgDefinition` (departments, teams, roles with string IDs and reporting relationships) into a `CompiledOrg` where every node has a positional D/T/R address. This is analogous to compiling source code: the input is human-readable, the output is machine-optimized.

### Why Compile?

Access decisions happen on every request. Walking a tree of string-keyed relationships for each decision would be too slow. The compiled form provides O(1) address lookups, pre-computed parent chains, and immutable node maps that prevent runtime tampering.

### How Compilation Works

1. Departments get `D1`, `D2`, ... in declaration order
2. Each department's primary role gets `Dn-R1`
3. Teams under a role get `Dn-Rm-T1`, `Dn-Rm-T2`, ...
4. Each team's primary role gets `Dn-Rm-Tk-R1`
5. Subordinate roles get appended: `Dn-Rm-Tk-Rp-R1`
6. If a department or team has no head role, a vacant one is auto-created

### When to Use

Use `compile_org()` at application startup or when the organization structure changes. The compiled result is then passed to `GovernanceEngine` or used directly for access checks.

## Key API

| Class / Function                       | Purpose                                                          |
| -------------------------------------- | ---------------------------------------------------------------- |
| `OrgDefinition`                        | Declarative input: org_id, name, departments, teams, roles       |
| `DepartmentConfig`                     | Department declaration with department_id and name               |
| `TeamConfig`                           | Team declaration with id, name, workspace                        |
| `RoleDefinition`                       | Role with role_id, name, reports_to_role_id, is_primary_for_unit |
| `compile_org(org_def)`                 | Transform OrgDefinition into CompiledOrg                         |
| `CompiledOrg`                          | Address-indexed organization with O(1) lookups                   |
| `CompiledOrg.get_node(addr)`           | Look up a node by its D/T/R address                              |
| `CompiledOrg.get_node_by_role_id(id)`  | Look up a node by its original role_id                           |
| `CompiledOrg.query_by_prefix(prefix)`  | All nodes under a given address prefix                           |
| `CompiledOrg.get_subtree(addr)`        | Node and all its descendants                                     |
| `CompiledOrg.root_roles`               | Top-level nodes with no parent                                   |
| `CompiledOrg.get_vacancy_status(addr)` | Check if a role position is filled                               |
| `CompilationError`                     | Raised for invalid org definitions                               |

## Code Walkthrough

### Step 1: Define the Organization

Create an `OrgDefinition` with departments, teams, and roles:

```python
from kailash.trust.pact import compile_org, RoleDefinition
from kailash.trust.pact.config import OrgDefinition, DepartmentConfig, TeamConfig

org_def = OrgDefinition(
    org_id="acme-corp",
    name="Acme Corporation",
    departments=[
        DepartmentConfig(department_id="engineering", name="Engineering"),
    ],
    teams=[
        TeamConfig(id="backend", name="Backend Team", workspace="ws-eng"),
    ],
    roles=[
        RoleDefinition(
            role_id="cto", name="CTO",
            is_primary_for_unit="engineering",
        ),
        RoleDefinition(
            role_id="backend-lead", name="Backend Lead",
            reports_to_role_id="cto",
            is_primary_for_unit="backend",
        ),
        RoleDefinition(
            role_id="dev-1", name="Developer 1",
            reports_to_role_id="backend-lead",
        ),
    ],
)
```

Key relationships: `is_primary_for_unit` links a role as the head of a department or team. `reports_to_role_id` establishes the reporting chain.

### Step 2: Compile

```python
compiled = compile_org(org_def)
assert compiled.org_id == "acme-corp"
```

### Step 3: Inspect Compiled Nodes

The compiler assigns positional addresses automatically:

```python
# Engineering department -> D1
eng = compiled.get_node("D1")
assert eng.name == "Engineering"

# CTO heads Engineering -> D1-R1
cto = compiled.get_node("D1-R1")
assert cto.name == "CTO"

# Backend Team under CTO -> D1-R1-T1
backend = compiled.get_node("D1-R1-T1")
assert backend.name == "Backend Team"

# Backend Lead heads the team -> D1-R1-T1-R1
lead = compiled.get_node("D1-R1-T1-R1")
assert lead.name == "Backend Lead"

# Developer 1 reports to Backend Lead -> D1-R1-T1-R1-R1
dev = compiled.get_node("D1-R1-T1-R1-R1")
assert dev.name == "Developer 1"
```

### Step 4: Query by Prefix and Subtree

```python
# All nodes under Engineering
eng_subtree = compiled.query_by_prefix("D1")
assert len(eng_subtree) == 5  # D1, D1-R1, D1-R1-T1, D1-R1-T1-R1, D1-R1-T1-R1-R1

# Subtree from Backend Lead down
subtree = compiled.get_subtree("D1-R1-T1-R1")
assert len(subtree) == 2  # T1-R1 and T1-R1-R1
```

### Step 5: Walk the Parent Chain

```python
leaf = compiled.get_node("D1-R1-T1-R1-R1")
chain = []
current = leaf.parent_address
while current is not None:
    node = compiled.get_node(current)
    chain.append(node.address)
    current = node.parent_address

assert chain == ["D1-R1-T1-R1", "D1-R1-T1", "D1-R1", "D1"]
```

### Step 6: Auto-Created Vacant Roles

If a department has no head role defined, PACT auto-creates a vacant one:

```python
headless_org = OrgDefinition(
    org_id="headless", name="Headless Org",
    departments=[DepartmentConfig(department_id="sales", name="Sales")],
    roles=[],
)
headless = compile_org(headless_org)
sales_head = headless.get_node("D1-R1")
assert sales_head.is_vacant is True
```

### Step 7: Immutability

The compiled structure is wrapped in `MappingProxyType` to prevent mutation:

```python
try:
    compiled.nodes["INJECTED"] = some_node
except TypeError:
    pass  # Read-only: prevents runtime tampering
```

### Step 8: Compilation Errors

Invalid definitions are caught at compile time:

```python
from kailash.trust.pact import CompilationError

# Duplicate role IDs
try:
    compile_org(OrgDefinition(
        org_id="bad", name="Bad",
        departments=[], roles=[
            RoleDefinition(role_id="dup", name="First"),
            RoleDefinition(role_id="dup", name="Second"),
        ],
    ))
except CompilationError:
    pass  # Duplicate role IDs

# Dangling reports_to reference
try:
    compile_org(OrgDefinition(
        org_id="bad2", name="Bad2",
        departments=[], roles=[
            RoleDefinition(role_id="orphan", name="Orphan",
                           reports_to_role_id="nonexistent"),
        ],
    ))
except CompilationError:
    pass  # Dangling reference
```

## Common Mistakes

1. **Forgetting `is_primary_for_unit`** -- Without it, the role will not be assigned as the head of the department/team, resulting in an auto-created vacant head instead.

2. **Circular `reports_to_role_id`** -- Role A reports to B, B reports to A. This will cause a `CompilationError`.

3. **Assuming mutable nodes** -- `CompiledOrg.nodes` is read-only. If you need to change the org, recompile from a modified `OrgDefinition`.

## Exercises

1. **Two-Department Org**: Define an organization with Engineering and Finance departments, each with a head role and one subordinate. Compile it and verify all addresses.

2. **Vacancy Detection**: Create an org where one department has no roles defined. After compilation, use `get_vacancy_status()` to find the auto-created vacant role.

3. **Subtree Query**: Build a 3-level org and use `query_by_prefix()` to extract just the nodes under a specific team.

## Key Takeaways

- `compile_org()` transforms declarative definitions into address-indexed structures
- Address assignment follows declaration order and reporting relationships
- Headless departments/teams get auto-created vacant roles
- Compiled orgs are immutable -- no runtime tampering
- Compilation catches errors early: duplicate IDs, dangling references

## Next Chapter

[Chapter 3: Knowledge Clearance](03_clearance.md) -- Define classification levels and role clearance for controlling access to sensitive data.
