# Chapter 4: Operating Envelopes

## Overview

Clearance controls _what data_ a role can see. Operating envelopes control _what a role can do_ -- spending limits, allowed actions, data paths, and time windows. Envelopes follow a strict rule: **monotonic tightening**. A child envelope can never be more permissive than its parent. This chapter teaches you how to create, intersect, and compute effective envelopes.

## Prerequisites

- [Chapter 3: Knowledge Clearance](03_clearance.md)
- Understanding of set intersection operations

## Concepts

### What are Operating Envelopes?

An operating envelope is a multi-dimensional boundary that constrains what a role can do. It covers:

- **Financial** -- spending limits, API cost budgets, approval thresholds
- **Operational** -- allowed/blocked actions
- **Data access** -- read/write path patterns
- **Confidentiality** -- classification ceiling

### Why Monotonic Tightening?

A manager cannot grant their subordinate more authority than they themselves have. This is monotonic tightening: each layer down the hierarchy can only narrow the envelope, never widen it. This prevents privilege escalation through delegation.

### Three Envelope Layers

| Layer             | Type      | Lifetime                | Purpose                                     |
| ----------------- | --------- | ----------------------- | ------------------------------------------- |
| RoleEnvelope      | Standing  | Permanent until changed | Supervisor defines subordinate's boundaries |
| TaskEnvelope      | Ephemeral | Expires after task      | Narrow the envelope for a specific task     |
| EffectiveEnvelope | Computed  | Per-request             | Intersection of all layers                  |

### How Intersection Works

When two envelopes are intersected:

- **Financial**: minimum of numeric limits
- **Operational**: intersection of allowed actions, union of blocked actions
- **Data access**: intersection of path patterns
- **Confidentiality**: minimum of levels

## Key API

| Class / Function                                  | Purpose                                          |
| ------------------------------------------------- | ------------------------------------------------ |
| `ConstraintEnvelopeConfig`                        | Multi-dimensional constraint specification       |
| `FinancialConstraintConfig`                       | Spending limits and approval thresholds          |
| `OperationalConstraintConfig`                     | Allowed and blocked action lists                 |
| `DataAccessConstraintConfig`                      | Read and write path patterns                     |
| `RoleEnvelope`                                    | Standing envelope from supervisor to subordinate |
| `TaskEnvelope`                                    | Ephemeral narrowing for a specific task          |
| `intersect_envelopes(a, b)`                       | Compute most restrictive combination             |
| `compute_effective_envelope(...)`                 | Three-layer intersection                         |
| `RoleEnvelope.validate_tightening(parent, child)` | Verify monotonic tightening                      |
| `MonotonicTighteningError`                        | Raised when child exceeds parent                 |

## Code Walkthrough

### Step 1: Create an Envelope

```python
from kailash.trust import ConfidentialityLevel
from kailash.trust.pact.config import (
    ConstraintEnvelopeConfig, FinancialConstraintConfig,
    OperationalConstraintConfig, DataAccessConstraintConfig,
)

cto_envelope = ConstraintEnvelopeConfig(
    id="env-cto",
    description="CTO operating envelope",
    confidentiality_clearance=ConfidentialityLevel.SECRET,
    financial=FinancialConstraintConfig(
        max_spend_usd=50000.0,
        api_cost_budget_usd=5000.0,
        requires_approval_above_usd=10000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy", "approve"],
        blocked_actions=["delete_production"],
    ),
    data_access=DataAccessConstraintConfig(
        read_paths=["/engineering/**", "/shared/**"],
        write_paths=["/engineering/**", "/shared/**"],
    ),
)
```

### Step 2: Create a Tighter Subordinate Envelope

```python
lead_envelope = ConstraintEnvelopeConfig(
    id="env-lead",
    confidentiality_clearance=ConfidentialityLevel.CONFIDENTIAL,
    financial=FinancialConstraintConfig(
        max_spend_usd=5000.0,
        api_cost_budget_usd=1000.0,
    ),
    operational=OperationalConstraintConfig(
        allowed_actions=["read", "write", "deploy"],
        blocked_actions=["delete_production", "approve"],
    ),
    data_access=DataAccessConstraintConfig(
        read_paths=["/engineering/**"],
        write_paths=["/engineering/**"],
    ),
)
```

### Step 3: Intersect Envelopes

```python
from kailash.trust.pact import intersect_envelopes

intersection = intersect_envelopes(cto_envelope, lead_envelope)

# Financial: min of limits
assert intersection.financial.max_spend_usd == 5000.0     # min(50000, 5000)
assert intersection.financial.api_cost_budget_usd == 1000.0  # min(5000, 1000)

# Operational: intersection of allowed, union of blocked
assert "deploy" in intersection.operational.allowed_actions
assert "approve" not in intersection.operational.allowed_actions  # blocked by lead
assert "approve" in intersection.operational.blocked_actions

# Data access: intersection of paths
assert "/engineering/**" in intersection.data_access.read_paths
assert "/shared/**" not in intersection.data_access.read_paths  # only in CTO

# Confidentiality: min of levels
assert intersection.confidentiality_clearance == ConfidentialityLevel.CONFIDENTIAL
```

### Step 4: Validate Monotonic Tightening

```python
from kailash.trust.pact import MonotonicTighteningError, RoleEnvelope

# This passes: lead is tighter than CTO
RoleEnvelope.validate_tightening(parent_envelope=cto_envelope, child_envelope=lead_envelope)

# This fails: child has higher spend than parent
try:
    wider = ConstraintEnvelopeConfig(
        id="too-wide",
        financial=FinancialConstraintConfig(max_spend_usd=100000.0),
        operational=OperationalConstraintConfig(allowed_actions=["read"]),
    )
    RoleEnvelope.validate_tightening(parent_envelope=cto_envelope, child_envelope=wider)
except MonotonicTighteningError:
    pass  # Child max_spend exceeds parent
```

### Step 5: Task Envelopes

Task envelopes are ephemeral -- they narrow the standing envelope for a specific task and expire automatically:

```python
from datetime import UTC, datetime, timedelta
from kailash.trust.pact import TaskEnvelope

task_env = TaskEnvelope(
    id="te-deploy-v2",
    task_id="task-deploy-v2",
    parent_envelope_id="re-backend-lead",
    envelope=ConstraintEnvelopeConfig(
        id="env-deploy-task",
        financial=FinancialConstraintConfig(max_spend_usd=500.0),
        operational=OperationalConstraintConfig(allowed_actions=["read", "deploy"]),
    ),
    expires_at=datetime.now(UTC) + timedelta(hours=4),
)

assert task_env.is_expired is False
```

### Step 6: Compute Effective Envelope

The effective envelope is the intersection of all ancestor RoleEnvelopes plus any TaskEnvelope:

```python
from kailash.trust.pact import compute_effective_envelope

effective = compute_effective_envelope(
    role_address="D1-R1-T1-R1",
    role_envelopes=role_envelopes,
    task_envelope=task_env,
)
assert effective.financial.max_spend_usd == 500.0  # min(5000, 500)
```

Expired task envelopes are ignored. No envelopes means maximally permissive (returns `None`).

### Step 7: Security Invariants

The SDK rejects NaN and Inf values that would bypass numeric comparisons:

```python
try:
    FinancialConstraintConfig(max_spend_usd=float("nan"))
except ValueError:
    pass  # NaN bypasses all comparisons

try:
    FinancialConstraintConfig(max_spend_usd=float("inf"))
except ValueError:
    pass  # Inf bypasses budget checks
```

## Common Mistakes

1. **Widening instead of tightening** -- Attempting to give a subordinate more permissive limits than the supervisor. Always validate with `RoleEnvelope.validate_tightening()`.

2. **Forgetting task expiry** -- Task envelopes expire silently. After expiry, the effective envelope reverts to the standing RoleEnvelope.

3. **Ignoring the intersection semantics** -- An action blocked in ANY layer is blocked in the intersection. An action must be allowed in ALL layers to be effective.

## Exercises

1. **Three-Layer Stack**: Create a VP envelope, a Manager envelope, and a Developer envelope. Verify that each layer tightens monotonically.

2. **Task Scoping**: Create a task envelope that restricts a developer to only `["read"]` actions for 2 hours. Compute the effective envelope and verify.

3. **Dimension Scope**: Use the `dimension_scope` parameter of `intersect_envelopes()` to intersect only the financial dimension while preserving other dimensions unchanged.

## Key Takeaways

- Operating envelopes constrain what a role can do across financial, operational, and data dimensions
- Monotonic tightening prevents privilege escalation through delegation
- Intersection computes the most restrictive combination: min for numbers, intersection for sets
- Task envelopes provide ephemeral narrowing that expires automatically
- NaN and Inf are explicitly rejected as security invariants

## Next Chapter

[Chapter 5: 5-Step Access Enforcement](05_access.md) -- The complete algorithm that combines clearance, posture, compartments, containment, and cross-boundary paths into a single access decision.
