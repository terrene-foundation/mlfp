# Chapter 1: D/T/R Addressing

## Overview

Every governance decision in PACT starts with a question: _who is asking?_ The D/T/R addressing system answers this by assigning a unique positional address to every node in the organization -- departments, teams, and roles. This chapter teaches you how to parse, construct, and navigate these addresses using the Kailash SDK.

## Prerequisites

- Python fundamentals (dataclasses, enums, string parsing)
- Kailash SDK installed (`pip install kailash`)
- Familiarity with organizational hierarchies

## Concepts

### What is D/T/R Addressing?

D/T/R stands for **Department / Team / Role** -- the three building blocks of organizational structure in PACT. Every entity in an organization gets a positional address built from these three node types:

- **D** (Department) -- a top-level organizational unit
- **T** (Team) -- a functional group within a department
- **R** (Role) -- a person or agent within a department or team

### Why Positional Addresses?

Positional addresses encode _where_ a role sits in the hierarchy, not _who_ fills it. This separation is critical: governance rules attach to positions, not people. When someone leaves and a replacement fills the role, all access rights, clearance levels, and operating envelopes transfer automatically.

### How Addresses Work

Addresses are hyphen-separated segments like `D1-R1-D2-R1-T1-R1`. Each segment has a type character (`D`, `T`, or `R`) and a 1-based sequence number. The grammar enforces one rule: **every D or T must be immediately followed by exactly one R**. This ensures every structural unit has an accountable head.

### When to Use Addressing

Use D/T/R addressing whenever you need to:

- Identify who is requesting access to knowledge
- Build an accountability chain from leaf role to root
- Determine containment (which department/team owns a resource)
- Check ancestor/descendant relationships between roles

## Key API

| Class / Function                       | Purpose                                          |
| -------------------------------------- | ------------------------------------------------ |
| `NodeType`                             | Enum with values `DEPARTMENT`, `TEAM`, `ROLE`    |
| `AddressSegment`                       | A single segment like `D1` or `R3`               |
| `AddressSegment.parse(s)`              | Parse a string into a segment (case-insensitive) |
| `Address.parse(s)`                     | Parse a full hyphenated address string           |
| `Address.from_segments(*segs)`         | Construct an address from segment objects        |
| `Address.depth`                        | Number of segments in the address                |
| `Address.parent`                       | The address with the last segment removed        |
| `Address.last_segment`                 | The final segment of the address                 |
| `Address.containment_unit`             | Nearest D or T ancestor                          |
| `Address.accountability_chain`         | All R-type addresses from root to leaf           |
| `Address.ancestors()`                  | All proper prefixes of the address               |
| `Address.is_prefix_of(other)`          | Strict prefix check                              |
| `Address.is_ancestor_of(other)`        | Reflexive prefix check (includes self)           |
| `Address.lowest_common_ancestor(a, b)` | Deepest shared prefix of two addresses           |

## Code Walkthrough

### Step 1: Understanding NodeType

The `NodeType` enum defines the three fundamental node types:

```python
from kailash.trust.pact import NodeType

assert NodeType.DEPARTMENT.value == "D"
assert NodeType.TEAM.value == "T"
assert NodeType.ROLE.value == "R"
```

These single-character values appear in address strings and are used throughout the PACT system.

### Step 2: Parsing Address Segments

An `AddressSegment` represents one piece of an address. Parse it from a string like `"D1"`:

```python
from kailash.trust.pact import AddressSegment

seg = AddressSegment.parse("D1")
assert seg.node_type == NodeType.DEPARTMENT
assert seg.sequence == 1
assert str(seg) == "D1"

# Case-insensitive: "t2" works the same as "T2"
seg_lower = AddressSegment.parse("t2")
assert seg_lower.node_type == NodeType.TEAM
assert seg_lower.sequence == 2
```

Segments are frozen dataclasses -- once created, they cannot be mutated.

### Step 3: Parsing Full Addresses

A full `Address` is a hyphen-separated chain of segments. The grammar rule is strict: every D or T must be immediately followed by exactly one R.

```python
from kailash.trust.pact import Address

addr = Address.parse("D1-R1-D2-R1-T1-R1")
assert len(addr.segments) == 6
assert addr.depth == 6
assert str(addr) == "D1-R1-D2-R1-T1-R1"
```

### Step 4: Navigating the Hierarchy

Addresses support rich navigation for walking the organizational tree:

```python
# Parent: remove the last segment
parent = addr.parent
assert str(parent) == "D1-R1-D2-R1-T1"

# Last segment
assert addr.last_segment.node_type == NodeType.ROLE

# Containment unit: nearest D or T ancestor
cu = addr.containment_unit
assert str(cu) == "D1-R1-D2-R1-T1"
```

### Step 5: Accountability Chains

The accountability chain extracts all R-type addresses, giving you the chain of responsible people from root to leaf:

```python
chain = addr.accountability_chain
assert len(chain) == 3
assert str(chain[0]) == "D1-R1"        # Department head
assert str(chain[1]) == "D1-R1-D2-R1"  # Sub-department head
assert str(chain[2]) == "D1-R1-D2-R1-T1-R1"  # Team lead
```

### Step 6: Prefix and Ancestor Checks

Two related but distinct operations:

```python
prefix = Address.parse("D1-R1")
full = Address.parse("D1-R1-D2-R1")

# is_prefix_of is strict (not reflexive)
assert prefix.is_prefix_of(full)
assert not prefix.is_prefix_of(prefix)

# is_ancestor_of includes self
assert prefix.is_ancestor_of(full)
assert prefix.is_ancestor_of(prefix)
```

### Step 7: Lowest Common Ancestor

Find the deepest shared prefix between two addresses:

```python
addr_a = Address.parse("D1-R1-D2-R1-T1-R1")
addr_b = Address.parse("D1-R1-D2-R1-T2-R1")
lca = Address.lowest_common_ancestor(addr_a, addr_b)
assert str(lca) == "D1-R1-D2-R1"

# Disjoint trees return None
disjoint_a = Address.parse("D1-R1")
disjoint_b = Address.parse("D2-R1")
assert Address.lowest_common_ancestor(disjoint_a, disjoint_b) is None
```

### Step 8: Constructing Addresses Programmatically

Instead of parsing strings, you can build addresses from segment objects:

```python
constructed = Address.from_segments(
    AddressSegment(NodeType.DEPARTMENT, 1),
    AddressSegment(NodeType.ROLE, 1),
)
assert str(constructed) == "D1-R1"
```

### Step 9: Grammar Violations

The parser enforces that every D or T is immediately followed by R:

```python
from kailash.trust.pact import GrammarError, AddressError

# D without trailing R
try:
    Address.parse("D1")
except GrammarError:
    pass  # D1 has no trailing Role

# Two departments in a row
try:
    Address.parse("D1-D2-R1")
except GrammarError:
    pass  # D1 not followed by R

# Invalid type character
try:
    AddressSegment.parse("X1")
except AddressError:
    pass  # 'X' is not D, T, or R
```

## Common Mistakes

1. **Forgetting the trailing R** -- `D1-T1` is invalid because T1 has no role. Every structural unit needs a head: `D1-R1-T1-R1`.

2. **Using 0-based sequences** -- Sequences are 1-based. `D0` raises `AddressError`.

3. **Confusing `is_prefix_of` and `is_ancestor_of`** -- `is_prefix_of` is strict (does not include self), while `is_ancestor_of` is reflexive (includes self). Use `is_ancestor_of` when you mean "is this address the same as or above the other?"

## Exercises

1. **Parse and Navigate**: Given the address `D1-R1-T1-R1-R1`, find its parent, containment unit, and accountability chain. Verify each programmatically.

2. **Build from Scratch**: Construct the address `D1-R1-D2-R1-T1-R1` using `Address.from_segments()` and verify it equals `Address.parse("D1-R1-D2-R1-T1-R1")`.

3. **Error Handling**: Write a function that takes a string, attempts to parse it as an `Address`, and returns either the valid address or a descriptive error message explaining why it failed.

## Key Takeaways

- D/T/R addresses encode organizational position, not identity
- The grammar rule "every D or T must be followed by R" ensures accountability
- Addresses support rich navigation: parent, ancestors, containment unit, accountability chain
- `lowest_common_ancestor` finds the deepest shared reporting line between two roles
- All address types are immutable (frozen dataclasses)

## Next Chapter

[Chapter 2: Organization Compilation](02_compile_org.md) -- Transform a declarative organization definition into an address-indexed compiled structure with O(1) lookups.
