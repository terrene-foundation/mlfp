# Chapter 3: Knowledge Clearance

## Overview

Not every role should access every piece of information. PACT implements a **dual-layer clearance system**: each role has a maximum clearance level, and each trust posture imposes a ceiling. The effective clearance is always the _minimum_ of the two. This chapter teaches you how classification levels, trust postures, and clearance grants work together.

## Prerequisites

- [Chapter 1: D/T/R Addressing](01_addressing.md)
- [Chapter 2: Organization Compilation](02_compile_org.md)

## Concepts

### What is Knowledge Clearance?

Knowledge clearance determines what classification level of information a role can access. It combines two independent dimensions:

1. **Role clearance** -- the maximum level granted to a specific role (e.g., SECRET)
2. **Trust posture** -- the system-wide operating mode that caps all clearances

### Why Two Layers?

A single clearance level would be static. Trust posture adds a dynamic dimension: during an incident, you can tighten the posture to restrict access system-wide without revoking individual clearances. When the incident is resolved, the posture relaxes and access returns to normal.

### The Five Classification Levels

From least to most sensitive:

| Level        | Value            | Example          |
| ------------ | ---------------- | ---------------- |
| PUBLIC       | `"public"`       | Company handbook |
| RESTRICTED   | `"restricted"`   | Internal memos   |
| CONFIDENTIAL | `"confidential"` | Salary data      |
| SECRET       | `"secret"`       | M&A plans        |
| TOP_SECRET   | `"top_secret"`   | Security keys    |

### The Five Trust Postures

From most restricted to most autonomous:

| Posture            | Ceiling      | Meaning                        |
| ------------------ | ------------ | ------------------------------ |
| PSEUDO_AGENT       | PUBLIC       | Agent can only see public data |
| SUPERVISED         | RESTRICTED   | Human watching every action    |
| SHARED_PLANNING    | CONFIDENTIAL | Human and agent plan together  |
| CONTINUOUS_INSIGHT | SECRET       | Agent reports, human reviews   |
| DELEGATED          | TOP_SECRET   | Full autonomy within envelope  |

### The Clearance Formula

```
effective_clearance = min(role.max_clearance, POSTURE_CEILING[posture])
```

This single formula is the core of PACT clearance. Even a TOP_SECRET-cleared role cannot access SECRET data at SUPERVISED posture.

## Key API

| Class / Function                          | Purpose                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| `ConfidentialityLevel`                    | Enum: PUBLIC, RESTRICTED, CONFIDENTIAL, SECRET, TOP_SECRET                     |
| `TrustPosture`                            | Enum: PSEUDO_AGENT, SUPERVISED, SHARED_PLANNING, CONTINUOUS_INSIGHT, DELEGATED |
| `POSTURE_CEILING`                         | Dict mapping each posture to its maximum classification level                  |
| `VettingStatus`                           | Enum: PENDING, ACTIVE, EXPIRED, REVOKED                                        |
| `RoleClearance`                           | Frozen dataclass: role_address, max_clearance, compartments, vetting_status    |
| `effective_clearance(clearance, posture)` | Compute the effective clearance level                                          |

## Code Walkthrough

### Step 1: Classification Levels

```python
from kailash.trust import ConfidentialityLevel

assert ConfidentialityLevel.PUBLIC.value == "public"
assert ConfidentialityLevel.SECRET.value == "secret"
assert ConfidentialityLevel.TOP_SECRET.value == "top_secret"
```

### Step 2: Trust Posture Levels

```python
from kailash.trust import TrustPosture

assert TrustPosture.PSEUDO_AGENT.value == "pseudo_agent"
assert TrustPosture.DELEGATED.value == "delegated"
```

### Step 3: The Posture Ceiling Map

```python
from kailash.trust.pact import POSTURE_CEILING

assert POSTURE_CEILING[TrustPosture.PSEUDO_AGENT] == ConfidentialityLevel.PUBLIC
assert POSTURE_CEILING[TrustPosture.SUPERVISED] == ConfidentialityLevel.RESTRICTED
assert POSTURE_CEILING[TrustPosture.DELEGATED] == ConfidentialityLevel.TOP_SECRET
```

### Step 4: Create a Role Clearance

```python
from kailash.trust.pact import RoleClearance, VettingStatus

clearance = RoleClearance(
    role_address="D1-R1-T1-R1",
    max_clearance=ConfidentialityLevel.SECRET,
    compartments=frozenset({"aml-investigations", "sanctions"}),
    granted_by_role_address="D1-R1",
    vetting_status=VettingStatus.ACTIVE,
    nda_signed=True,
)

assert clearance.max_clearance == ConfidentialityLevel.SECRET
assert "aml-investigations" in clearance.compartments
```

RoleClearance is frozen -- once created, it cannot be modified.

### Step 5: Compute Effective Clearance

```python
from kailash.trust.pact import effective_clearance

# SECRET clearance at DELEGATED posture (ceiling=TOP_SECRET): uncapped
eff = effective_clearance(clearance, TrustPosture.DELEGATED)
assert eff == ConfidentialityLevel.SECRET  # min(SECRET, TOP_SECRET)

# SECRET clearance at SHARED_PLANNING posture (ceiling=CONFIDENTIAL): capped
eff = effective_clearance(clearance, TrustPosture.SHARED_PLANNING)
assert eff == ConfidentialityLevel.CONFIDENTIAL  # min(SECRET, CONFIDENTIAL)

# SECRET clearance at PSEUDO_AGENT posture (ceiling=PUBLIC): severely capped
eff = effective_clearance(clearance, TrustPosture.PSEUDO_AGENT)
assert eff == ConfidentialityLevel.PUBLIC  # min(SECRET, PUBLIC)
```

### Step 6: Vetting Status Lifecycle

Only ACTIVE clearances are valid for access. The vetting status tracks the clearance lifecycle:

```python
assert VettingStatus.PENDING.value == "pending"
assert VettingStatus.ACTIVE.value == "active"
assert VettingStatus.EXPIRED.value == "expired"
assert VettingStatus.REVOKED.value == "revoked"
```

Note: `effective_clearance()` does not check vetting status -- that is done in the 5-step access algorithm (Chapter 5).

### Step 7: Compartments

Compartments provide need-to-know control for SECRET and TOP_SECRET data. A role must hold ALL compartments that an item belongs to:

```python
compartmented = RoleClearance(
    role_address="D1-R1-T1-R1",
    max_clearance=ConfidentialityLevel.TOP_SECRET,
    compartments=frozenset({"nuclear", "cyber", "humint"}),
    vetting_status=VettingStatus.ACTIVE,
)
assert len(compartmented.compartments) == 3
```

## Common Mistakes

1. **Ignoring posture ceiling** -- A SECRET-cleared role cannot access SECRET data at SUPERVISED posture. Always check effective clearance, not max clearance.

2. **Forgetting vetting status** -- `effective_clearance()` computes a value even for EXPIRED clearances. The access algorithm (Chapter 5) rejects non-ACTIVE clearances at Step 1.

3. **Assuming clearance equals authority** -- Clearance is independent of position. A junior analyst can hold higher clearance than a senior manager if the knowledge domain requires it.

## Exercises

1. **Clearance Table**: Create clearances for five roles at different levels. Compute effective clearance for each at all five postures. Present the results as a 5x5 table.

2. **Posture Simulation**: Write a function that takes a list of `RoleClearance` objects and a target `ConfidentialityLevel`, and returns which posture is the minimum required for each role to access data at that level.

3. **Compartment Check**: Given a `RoleClearance` with compartments `{"alpha", "beta"}` and a knowledge item requiring `{"alpha", "gamma"}`, explain why access would be denied and what compartment is missing.

## Key Takeaways

- Effective clearance = min(role max clearance, posture ceiling)
- Trust posture provides a dynamic, system-wide cap on all access
- Compartments enforce need-to-know beyond classification level
- Only ACTIVE vetting status grants access (enforced in the access algorithm)
- Clearance is independent of organizational authority

## Next Chapter

[Chapter 4: Operating Envelopes](04_envelopes.md) -- Define financial, operational, and data boundaries for roles and tasks.
