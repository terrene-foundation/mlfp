# Chapter 7: Cross-Functional Bridges

## Overview

KSPs share at the unit level -- every role in the target unit gets access. Sometimes you need surgical precision: one specific role accessing another specific role's data. **PactBridge** provides role-to-role access paths across organizational boundaries. This chapter teaches you how to create, configure, and use bridges for standing, scoped, and ad-hoc cross-functional access.

## Prerequisites

- [Chapter 6: Knowledge Items & Share Policies](06_knowledge.md)

## Concepts

### What is a Bridge?

A `PactBridge` grants one role access to another role's domain. Unlike KSPs (unit-level), bridges are role-level: they connect two specific positions in the org chart. Bridges do not cascade to subordinates.

### Three Bridge Types

| Type       | Lifetime                        | Use Case                               |
| ---------- | ------------------------------- | -------------------------------------- |
| `standing` | Permanent                       | Ongoing cross-functional collaboration |
| `scoped`   | Permanent but operation-limited | Access only during specific operations |
| `ad_hoc`   | Temporary (with expiry)         | One-off collaboration                  |

### Bilateral vs Unilateral

- **Bilateral** (`bilateral=True`): Both roles can access each other's data
- **Unilateral** (`bilateral=False`): Only role_a can access role_b's data; not the reverse

### When to Use Bridges vs KSPs

- **Bridge**: Backend Lead needs to see SOC Lead's incident reports (role-to-role)
- **KSP**: All of Engineering needs access to Finance's quarterly reports (unit-to-unit)

## Key API

| Class / Function  | Purpose                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| `PactBridge`      | Frozen dataclass: role_a, role_b, type, classification cap, bilateral flag |
| `can_access(...)` | Evaluates bridges at Step 4e                                               |

## Code Walkthrough

### Step 1: Create a Bilateral Standing Bridge

```python
from kailash.trust import ConfidentialityLevel
from kailash.trust.pact import PactBridge

standing_bridge = PactBridge(
    id="bridge-eng-sec",
    role_a_address="D1-R1-T1-R1",  # Backend Lead
    role_b_address="D2-R1-T1-R1",  # SOC Lead
    bridge_type="standing",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    bilateral=True,
    active=True,
)
```

### Step 2: Test Bilateral Access

```python
from kailash.trust.pact import can_access, KnowledgeItem
from kailash.trust import TrustPosture

eng_doc = KnowledgeItem(
    item_id="doc-backend-architecture",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",
)

# Without bridge: SOC Lead CANNOT access Backend Team data
without = can_access(
    role_address="D2-R1-T1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert without.allowed is False

# With bridge: SOC Lead CAN access Backend Team data
with_bridge = can_access(
    role_address="D2-R1-T1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[standing_bridge],
)
assert with_bridge.allowed is True
assert with_bridge.audit_details.get("access_path") == "bridge"
```

### Step 3: Unilateral Bridge

```python
unilateral = PactBridge(
    id="bridge-sec-reads-eng",
    role_a_address="D2-R1-T1-R1",  # SOC Lead (reader)
    role_b_address="D1-R1-T1-R1",  # Backend Lead (data owner)
    bridge_type="standing",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    bilateral=False,
    active=True,
)

# A->B direction (SOC reads Eng): ALLOWED
a_to_b = can_access(
    role_address="D2-R1-T1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[unilateral],
)
assert a_to_b.allowed is True

# B->A direction (Eng reads SOC): DENIED
b_to_a = can_access(
    role_address="D1-R1-T1-R1", knowledge_item=sec_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[unilateral],
)
assert b_to_a.allowed is False
```

### Step 4: Bridge Classification Cap

```python
restricted_bridge = PactBridge(
    id="bridge-restricted",
    role_a_address="D2-R1-T1-R1",
    role_b_address="D1-R1-T1-R1",
    bridge_type="standing",
    max_classification=ConfidentialityLevel.RESTRICTED,
    bilateral=True, active=True,
)

confidential_doc = KnowledgeItem(
    item_id="doc-confidential",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D1-R1-T1",
)

# CONFIDENTIAL doc through RESTRICTED bridge: DENIED
capped = can_access(
    role_address="D2-R1-T1-R1", knowledge_item=confidential_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[restricted_bridge],
)
assert capped.allowed is False
```

### Step 5: Ad-Hoc Bridge with Expiry

```python
from datetime import UTC, datetime, timedelta

ad_hoc = PactBridge(
    id="bridge-adhoc-001",
    role_a_address="D1-R1-T1-R1",
    role_b_address="D2-R1-T1-R1",
    bridge_type="ad_hoc",
    max_classification=ConfidentialityLevel.RESTRICTED,
    bilateral=True, active=True,
    expires_at=datetime.now(UTC) + timedelta(hours=4),
)

# Active: works
# After expiry: treated as non-existent
```

### Step 6: Inactive and Expired Bridges

Both inactive (`active=False`) and expired bridges are silently skipped during access evaluation. Access fails at Step 5 (no path found).

### Step 7: Bridges Are Role-Level

A bridge between Backend Lead and SOC Lead does NOT give the CTO access through the bridge. Bridges connect specific positions, not hierarchies.

## Common Mistakes

1. **Assuming inheritance** -- A bridge to a manager does not give their subordinates access. Use KSPs for unit-level sharing.

2. **Forgetting to set `active=True`** -- Inactive bridges are skipped silently.

3. **Confusing bridge direction** -- In a unilateral bridge, `role_a` is the reader and `role_b` is the data owner's side. Get the direction right.

## Exercises

1. **Bilateral vs Unilateral**: Create one of each type. Verify that bilateral allows both directions while unilateral blocks the reverse.

2. **Scoped Bridge**: Create a scoped bridge with `operational_scope=("incident_response",)` and explain when this would be checked.

3. **Expiry Behavior**: Create a bridge that expired 1 minute ago. Verify that `can_access()` returns denied with `step_failed == 5`.

## Key Takeaways

- Bridges provide role-to-role access paths (more surgical than KSPs)
- Three types: standing (permanent), scoped (operation-limited), ad_hoc (temporary)
- Bilateral bridges allow both directions; unilateral is one-way
- Bridges have classification caps, just like KSPs
- Bridges do not cascade to subordinates
- Expired and inactive bridges are silently ignored

## Next Chapter

[Chapter 8: GovernanceEngine](08_governance_engine.md) -- The unified facade that combines compilation, clearance, envelopes, and access checks in one thread-safe interface.
