# Chapter 6: Knowledge Items & Share Policies

## Overview

Knowledge items are the _objects_ in access decisions -- documents, datasets, reports, and secrets that carry classification levels and compartments. Knowledge Share Policies (KSPs) are the mechanism for controlled cross-department sharing. This chapter teaches you how to create classified items and establish sharing agreements between organizational units.

## Prerequisites

- [Chapter 5: 5-Step Access Enforcement](05_access.md)

## Concepts

### What is a Knowledge Item?

A `KnowledgeItem` represents any piece of information that needs access control. It carries:

- A **classification level** (PUBLIC through TOP_SECRET)
- An **owning unit** (the department or team responsible for it)
- Optional **compartments** (need-to-know groups for SECRET+ items)

### What is a Knowledge Share Policy?

A KSP is a directional sharing agreement: the source unit shares data _with_ the target unit, up to a maximum classification level. KSPs enable Step 4d in the access algorithm -- without them, cross-department access is denied.

### When to Use KSPs vs Bridges

- **KSPs** share at the _unit_ level -- all roles in the target unit gain access
- **Bridges** (Chapter 7) share at the _role_ level -- only specific roles gain access
- Use KSPs for systematic cross-department collaboration
- Use bridges for specific person-to-person working relationships

## Key API

| Class / Function       | Purpose                                                                      |
| ---------------------- | ---------------------------------------------------------------------------- |
| `KnowledgeItem`        | Frozen dataclass: item_id, classification, owning_unit_address, compartments |
| `KnowledgeSharePolicy` | Directional sharing agreement between units                                  |
| `can_access(...)`      | Access check that evaluates KSPs at Step 4d                                  |

## Code Walkthrough

### Step 1: Create Knowledge Items

```python
from kailash.trust import ConfidentialityLevel
from kailash.trust.pact import KnowledgeItem

public_doc = KnowledgeItem(
    item_id="doc-company-handbook",
    classification=ConfidentialityLevel.PUBLIC,
    owning_unit_address="D1",
    description="Employee handbook - public information",
)

secret_investigation = KnowledgeItem(
    item_id="case-2026-001",
    classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1-R1-T1",
    compartments=frozenset({"aml-cases", "sanctions"}),
    description="Active AML investigation case file",
)
```

Knowledge items are frozen -- classification cannot be downgraded after creation.

### Step 2: Create a Knowledge Share Policy

```python
from kailash.trust.pact import KnowledgeSharePolicy

ksp = KnowledgeSharePolicy(
    id="ksp-eng-to-finance",
    source_unit_address="D1",      # Engineering shares
    target_unit_address="D2",      # Finance receives
    max_classification=ConfidentialityLevel.RESTRICTED,
    created_by_role_address="D1-R1",
    active=True,
)
```

The KSP is directional: Engineering shares _with_ Finance, not the other way around.

### Step 3: KSP with Compartment Restrictions

KSPs can restrict which compartments are shareable:

```python
compartmented_ksp = KnowledgeSharePolicy(
    id="ksp-compliance",
    source_unit_address="D1-R1-T1",
    target_unit_address="D2",
    max_classification=ConfidentialityLevel.CONFIDENTIAL,
    compartments=frozenset({"compliance-reports"}),
    created_by_role_address="D1-R1",
    active=True,
)
```

### Step 4: KSP with Expiry

```python
from datetime import UTC, datetime, timedelta

expiring_ksp = KnowledgeSharePolicy(
    id="ksp-temp",
    source_unit_address="D1",
    target_unit_address="D2",
    max_classification=ConfidentialityLevel.RESTRICTED,
    created_by_role_address="D1-R1",
    active=True,
    expires_at=datetime.now(UTC) + timedelta(days=30),
)
```

### Step 5: KSP in Action

```python
from kailash.trust.pact import can_access, compile_org, RoleClearance, VettingStatus
from kailash.trust import TrustPosture

# Without KSP: Finance analyst CANNOT access Engineering data
without_ksp = can_access(
    role_address="D2-R1-R1", knowledge_item=eng_memo,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert without_ksp.allowed is False

# With KSP: Finance analyst CAN access Engineering data
with_ksp = can_access(
    role_address="D2-R1-R1", knowledge_item=eng_memo,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[cross_dept_ksp], bridges=[],
)
assert with_ksp.allowed is True
assert with_ksp.audit_details.get("access_path") == "ksp"
```

### Step 6: KSP Classification Cap

A KSP with `max_classification=RESTRICTED` cannot share CONFIDENTIAL items, even if the requesting role has CONFIDENTIAL clearance:

```python
confidential_doc = KnowledgeItem(
    item_id="eng-confidential",
    classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D1",
)

capped = can_access(
    role_address="D2-R1-R1", knowledge_item=confidential_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances_upgraded,
    ksps=[cross_dept_ksp],  # max_classification=RESTRICTED
    bridges=[],
)
assert capped.allowed is False  # KSP cap exceeded
```

## Common Mistakes

1. **Bidirectional assumption** -- KSPs are directional. Engineering sharing with Finance does not mean Finance shares with Engineering. Create two KSPs for bidirectional access.

2. **Ignoring the classification cap** -- A KSP's `max_classification` is an absolute ceiling. Even if both the role and the item allow higher levels, the KSP blocks it.

3. **Confusing owning_unit_address with role_address** -- The owning unit should be a D or T address (structural unit), not an R address (role).

## Exercises

1. **Cross-Department Sharing**: Create two departments. Establish a KSP that allows the second department to read RESTRICTED items from the first. Verify with `can_access()`.

2. **Classification Ceiling**: Create a CONFIDENTIAL item and a RESTRICTED-capped KSP. Verify that the KSP blocks access even when the role has sufficient clearance.

3. **Expiring KSP**: Create a KSP that expires in 1 hour and another that expired 1 hour ago. Test how each affects `can_access()`.

## Key Takeaways

- Knowledge items carry classification, ownership, and optional compartments
- KSPs enable controlled cross-department sharing at Step 4d
- KSPs are directional: source shares with target
- KSPs have their own classification ceiling that cannot be exceeded
- All knowledge and KSP objects are frozen (immutable)

## Next Chapter

[Chapter 7: Cross-Functional Bridges](07_bridges.md) -- Role-level access paths for specific cross-boundary working relationships.
