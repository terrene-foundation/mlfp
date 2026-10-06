# MLFP06 — Task 1: Operating Envelopes and Deny-Paths for the FinTech Org

**Weight**: 30 marks · **Framework**: PACT (`GovernanceEngine`,
`RoleEnvelope`, `ConstraintEnvelopeConfig`) · **Outcomes assessed**:
operating envelopes, deny-path testing, monotonic tightening (6.7)

## Scenario

The Singapore FinTech AI division from Exercise 7 (`shared.mlfp06.ex_7`) is
going to production. Six agent roles run under three human department heads.
Until now the org compiled with **no envelopes attached** — and in the
installed kailash-pact, a role with no envelope is **auto-approved** for
everything (`verify_action` returns `allowed=True`, `level="auto_approved"`).
Deny-paths exist only where an envelope is attached.

Your job: attach least-privilege operating envelopes to the four agent roles
in the table below, and implement the monotonic-tightening check the platform
team uses to review envelope changes.

The grader probes **your returned engine** directly — with actions and costs
it chooses itself at grading time (some within the envelope, some outside,
some over budget, some from addresses that do not exist). A submission that
hard-codes verdicts, or one that never actually attaches the envelopes,
fails.

## Interfaces

```python
def solve() -> dict: ...
def validate_child(parent, child) -> bool: ...
```

`solve()` returns `{"engine": engine}` — the canonical org compiled with
`compile_governance(apply_specs=False)` from `shared.mlfp06.ex_7`, plus your
four envelopes attached with `engine.set_role_envelope(...)`. Compiling with
`apply_specs=False` keeps the YAML envelope block as inert metadata: the only
governance in force on your engine must be the envelopes **you** attach.
Each envelope is a full five-dimension `ConstraintEnvelopeConfig` (Financial,
Operational, Temporal, Data Access, Communication) wrapped in a `RoleEnvelope`
whose `defining_role_address` is the role's department head.

`validate_child(parent, child)` takes two `ConstraintEnvelopeConfig` objects
and returns True iff `child` is a legal tightening of `parent` — equal or
more restrictive on every dimension. Use the framework's structural check;
do not hand-compare numbers.

## The envelope table

| Role             | Address       | Defined by | Clearance  | Cap (USD) | Allowed actions                                                   |
| ---------------- | ------------- | ---------- | ---------- | --------- | ----------------------------------------------------------------- |
| `data_analyst`   | `D1-R1-T1-R1` | `D1-R1`    | restricted | 20.00     | `read_data`, `summarise_data`, `generate_report`                  |
| `model_trainer`  | `D1-R1-T2-R1` | `D1-R1`    | restricted | 100.00    | `train_model`, `evaluate_model`, `read_data`                      |
| `risk_assessor`  | `D2-R1-T1-R1` | `D2-R1`    | restricted | 200.00    | `read_data`, `audit_model`, `generate_report`, `access_audit_log` |
| `customer_agent` | `D3-R1-T1-R1` | `D3-R1`    | public     | 5.00      | `answer_question`, `search_faq`                                   |

Leave `model_deployer` (`D1-R1-T3-R1`) and `bias_checker` (`D2-R1-T2-R1`)
**without** envelopes — the fail-open default for envelope-less roles is part
of what the grader pins.

A probe carries a context `{"cost": dollars}`. An action outside the role's
allowed set must be **blocked**; a cost above the cap must be **blocked**;
within both, **allowed**.

## PACT's clearance ladder (as installed)

`public < restricted < confidential < secret < top_secret` — `restricted`
is the **second-lowest** rung, just above public. A `restricted` child under
a `confidential` parent is a legal tightening; a `secret` child under a
`confidential` parent is an escalation and must be rejected.

## Acceptance criteria (what the grader measures)

| #   | Check                                                                          |
| --- | ------------------------------------------------------------------------------ |
| 1   | `solve()` returns a dict whose engine answers `verify_action` (gate)           |
| 2   | Within-envelope probes on all four roles are **allowed**                       |
| 3   | Action-outside-envelope probes are **blocked**                                 |
| 4   | Over-budget probes (cost = cap × 1.5–3, grader-drawn) are **blocked**          |
| 5   | Under-budget probes (cost = cap × 0.3–0.8, grader-drawn) are **allowed**       |
| 6   | Unknown address `D99-R99-T99-R99` is **auto-approved** (the installed default) |
| 7   | An envelope-less known role is **auto-approved** (fail-open pinned)            |
| 8   | `validate_child`: a genuinely tighter child passes                             |
| 9   | `validate_child`: `secret` child under a `confidential` parent is rejected     |
| 10  | `validate_child`: a budget-widening child is rejected                          |
| 11  | `validate_child`: an action-widening child is rejected                         |
| 12  | `validate_child`: a `restricted` child under a `confidential` parent passes    |

Marks = 30 × (non-gate checks passed / 11). If the gate fails, the task
scores 0.

## Rules

- No LLM calls. Everything here is in-process governance.
- Build envelopes with the PACT classes (`FinancialConstraintConfig`,
  `OperationalConstraintConfig`, ...), one per dimension.
- Deterministic: no randomness in your code — the grader owns the probes.
- Self-check: run `starter.py`; it prints the verdicts for a fixed probe list.
