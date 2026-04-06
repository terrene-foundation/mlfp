# Chapter 5: 5-Step Access Enforcement

## Overview

All the pieces -- addresses, clearances, postures, compartments, envelopes -- come together in the **5-step access algorithm**. This is the single function that decides whether a role can access a knowledge item. It is fail-closed: if any step fails or any error occurs, access is denied. This chapter walks through every step with concrete examples.

## Prerequisites

- [Chapter 3: Knowledge Clearance](03_clearance.md)
- [Chapter 4: Operating Envelopes](04_envelopes.md)

## Concepts

### The Five Steps

| Step | Check                                      | Failure Means                                  |
| ---- | ------------------------------------------ | ---------------------------------------------- |
| 1    | Clearance exists and vetting is ACTIVE     | No valid clearance for this role               |
| 2    | Effective clearance >= item classification | Role cannot see data at this level             |
| 3    | Role holds all required compartments       | Missing need-to-know compartment               |
| 4a-e | Structural or policy path exists           | Check same-unit, downward, upward, KSP, bridge |
| 5    | Fail-closed default                        | No access path found                           |

### Fail-Closed Design

If no step explicitly grants access, the answer is DENY. There is no "default allow." This is a security-critical design choice: bugs in access logic result in denied access (safe) rather than granted access (dangerous).

### Access Paths (Step 4)

Step 4 checks multiple access paths in order:

- **4a**: Same unit -- role is in the same department/team as the item
- **4b**: Downward visibility -- role is above the item's unit in the hierarchy
- **4c**: Upward visibility -- role is below but has explicit upward access
- **4d**: Knowledge Share Policy (KSP) -- cross-unit sharing agreement
- **4e**: Bridge -- role-to-role cross-boundary access

## Key API

| Class / Function      | Purpose                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `can_access(...)`     | The 5-step access decision function                                    |
| `AccessDecision`      | Result with `allowed`, `step_failed`, `reason`, `audit_details`        |
| `explain_access(...)` | Human-readable trace of all 5 steps                                    |
| `KnowledgeItem`       | The target object (item_id, classification, owning_unit, compartments) |

## Code Walkthrough

### Step 1: Set Up a Realistic Organization

```python
from kailash.trust.pact import compile_org, RoleDefinition, RoleClearance, VettingStatus
from kailash.trust.pact.config import OrgDefinition, DepartmentConfig, TeamConfig
from kailash.trust import ConfidentialityLevel, TrustPosture

org_def = OrgDefinition(
    org_id="access-demo", name="Access Demo Corp",
    departments=[
        DepartmentConfig(department_id="engineering", name="Engineering"),
        DepartmentConfig(department_id="finance", name="Finance"),
    ],
    teams=[TeamConfig(id="backend", name="Backend Team", workspace="ws")],
    roles=[
        RoleDefinition(role_id="cto", name="CTO", is_primary_for_unit="engineering"),
        RoleDefinition(role_id="backend-lead", name="Backend Lead",
                       reports_to_role_id="cto", is_primary_for_unit="backend"),
        RoleDefinition(role_id="dev-1", name="Developer 1",
                       reports_to_role_id="backend-lead"),
        RoleDefinition(role_id="cfo", name="CFO", is_primary_for_unit="finance"),
        RoleDefinition(role_id="accountant", name="Accountant",
                       reports_to_role_id="cfo"),
    ],
)
compiled = compile_org(org_def)
```

### Step 2: Define Clearances and Knowledge Items

```python
from kailash.trust.pact import KnowledgeItem, can_access

clearances = {
    "D1-R1": RoleClearance(role_address="D1-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        compartments=frozenset({"engineering-internal"}),
        vetting_status=VettingStatus.ACTIVE),
    "D1-R1-T1-R1-R1": RoleClearance(role_address="D1-R1-T1-R1-R1",
        max_clearance=ConfidentialityLevel.RESTRICTED,
        vetting_status=VettingStatus.ACTIVE),
    "D2-R1": RoleClearance(role_address="D2-R1",
        max_clearance=ConfidentialityLevel.SECRET,
        compartments=frozenset({"financial-records"}),
        vetting_status=VettingStatus.ACTIVE),
}

eng_doc = KnowledgeItem(
    item_id="doc-arch-review",
    classification=ConfidentialityLevel.RESTRICTED,
    owning_unit_address="D1-R1-T1",
    description="Architecture review document",
)
```

### Step 3: Step 1 Failure -- No Clearance

```python
decision = can_access(
    role_address="D1-R1-T1-R1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances={}, ksps=[], bridges=[],  # Empty clearances
)
assert decision.allowed is False
assert decision.step_failed == 1
```

### Step 4: Step 2 Failure -- Classification Exceeds Clearance

```python
finance_report = KnowledgeItem(
    item_id="quarterly", classification=ConfidentialityLevel.CONFIDENTIAL,
    owning_unit_address="D2",
)

# Developer has RESTRICTED clearance, report is CONFIDENTIAL
decision = can_access(
    role_address="D1-R1-T1-R1-R1", knowledge_item=finance_report,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert decision.allowed is False
assert decision.step_failed == 2
```

### Step 5: Step 3 Failure -- Missing Compartments

```python
secret_doc = KnowledgeItem(
    item_id="secret-plan", classification=ConfidentialityLevel.SECRET,
    owning_unit_address="D1",
    compartments=frozenset({"engineering-internal"}),
)

# CFO has SECRET but compartment "financial-records", not "engineering-internal"
decision = can_access(
    role_address="D2-R1", knowledge_item=secret_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert decision.allowed is False
assert decision.step_failed == 3
```

### Step 6: Step 4a -- Same Unit Access

```python
# Developer accessing a document owned by their own team
decision = can_access(
    role_address="D1-R1-T1-R1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert decision.allowed is True
```

### Step 7: Step 4b -- Downward Visibility

```python
# CTO at D1-R1 can see everything under D1-R1-*
decision = can_access(
    role_address="D1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert decision.allowed is True
```

### Step 8: Step 5 -- Fail-Closed Default

```python
# CFO trying to access engineering doc -- different department, no policy
decision = can_access(
    role_address="D2-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
assert decision.allowed is False
assert decision.step_failed == 5  # No access path found
```

### Step 9: Explain Access

```python
from kailash.trust.pact import explain_access

trace = explain_access(
    role_address="D1-R1-T1-R1-R1", knowledge_item=eng_doc,
    posture=TrustPosture.DELEGATED, compiled_org=compiled,
    clearances=clearances, ksps=[], bridges=[],
)
print(trace)
# Shows Step 1: PASS, Step 2: PASS, ... Step 4a: ALLOWED
```

## Common Mistakes

1. **Passing empty clearances** -- The most common integration bug. If the clearances dict is empty, every access check fails at Step 1.

2. **Confusing posture with clearance** -- Posture is system-wide and dynamic. Clearance is per-role and static. Both must pass for access.

3. **Not checking `step_failed`** -- The `AccessDecision` tells you exactly which step failed. Use this for debugging, not just checking `allowed`.

## Exercises

1. **All Five Steps**: Create scenarios that trigger failure at each of the five steps. Verify with `assert decision.step_failed == N`.

2. **Posture Sweep**: Take a single access request and run it at all five posture levels. Document which postures allow access and which deny it.

3. **Audit Trail**: Use `explain_access()` to generate a human-readable trace. Parse the output and verify it mentions each step.

## Key Takeaways

- The 5-step algorithm is the single access control decision point in PACT
- Steps are evaluated in order; the first failure short-circuits
- Step 4 checks multiple structural and policy paths (same unit, downward, KSP, bridge)
- Step 5 is the fail-closed default: no path means no access
- `explain_access()` provides a debuggable trace of every step

## Next Chapter

[Chapter 6: Knowledge Items & Share Policies](06_knowledge.md) -- Create classified knowledge items and cross-unit sharing agreements.
