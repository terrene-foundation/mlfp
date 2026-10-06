# MLFP06 — Task 4: Design the Governance Org for a Bank's AI Office

**Weight**: 20 marks · **Framework**: PACT (`load_org_yaml`,
`GovernanceEngine`, `apply_governance_specs`) · **Outcomes assessed**: D/T/R
org design, clearance ladders, envelope authoring (6.7)

## Scenario

A regional bank stands up an AI office. You are handed the organisational
brief below and asked to express it as a PACT governance definition (the flat
YAML schema used in Exercise 7) that compiles, applies, and **enforces**.

The grader does not trust your compiled engine: it re-loads **your YAML**,
rebuilds the engine itself, applies the specs, and probes the rebuilt engine
with its own requests. The YAML is the artefact under assessment.

## The organisational brief

**Departments** (2): `model_development`, `operations`.
**Teams** (3): `research_team` and `deployment_team` under model development;
`support_team` under operations.

**Roles** (5):

| Role id              | Heads / reports to                                       | Clearance      |
| -------------------- | -------------------------------------------------------- | -------------- |
| `chief_data_officer` | heads `model_development`                                | `secret`       |
| `head_of_operations` | heads `operations`                                       | `secret`       |
| `research_scientist` | reports to `chief_data_officer`, heads `research_team`   | `confidential` |
| `ml_engineer`        | reports to `chief_data_officer`, heads `deployment_team` | `confidential` |
| `support_agent`      | reports to `head_of_operations`, heads `support_team`    | `public`       |

Clearances use PACT's ladder as installed: `public < restricted <
confidential < secret < top_secret`. Every agent sits at or below the head it
reports to.

**Envelopes** (3): each defined by the agent's department head, for the agent:

| Target               | Cap (USD) | Allowed actions                                                |
| -------------------- | --------- | -------------------------------------------------------------- |
| `research_scientist` | 50.00     | `read_data`, `run_experiment`, `train_model`                   |
| `ml_engineer`        | 80.00     | `deploy_model`, `monitor_model`, `rollback_model`, `read_data` |
| `support_agent`      | 5.00      | `answer_ticket`, `search_kb`                                   |

The two human heads carry **no** envelopes — under the installed default they
are auto-approved; that is a deliberate choice for this office, and the
grader pins it.

## Interface

```python
def solve() -> dict: ...
```

Returns `{"org_yaml": str, "engine": GovernanceEngine}` — your YAML as a
string, and the engine you get by loading it and applying the specs
(`load_org_yaml` → `GovernanceEngine(loaded.org_definition)` →
`apply_governance_specs(engine, loaded)`).

The flat schema (departments/teams/roles/clearances/envelopes with `heads` /
`reports_to` / `defined_by` / `target`) is the one in Exercise 7;
`shared.mlfp06.ex_7.ORG_YAML` is a working example of it for a different org.

## Acceptance criteria (what the grader measures)

| #   | Check                                                                          |
| --- | ------------------------------------------------------------------------------ |
| 1   | `org_yaml` is a string and the engine answers `verify_action` (gate)           |
| 2   | The grader can re-load your YAML itself (gate)                                 |
| 3   | Structure: 2 departments, 3 teams, 5 roles                                     |
| 4   | Heads and reporting lines match the brief                                      |
| 5   | Every clearance is a valid pact level, and every agent is at or below its head |
| 6   | Clearances match the brief exactly (heads secret; agents as tabled)            |
| 7   | Envelope targets, caps and action sets match the brief                         |
| 8   | Within-envelope probes on the rebuilt engine are **allowed**                   |
| 9   | Outside-envelope action probes are **blocked**                                 |
| 10  | An over-budget probe (cap × 1.5–3, grader-drawn) is **blocked**                |
| 11  | Unknown address and the envelope-less heads are **auto-approved**              |

Marks = 20 × (non-gate checks passed / 9). If a gate fails, the task scores 0.

## Rules

- No LLM calls. Deterministic YAML: no timestamps or random ids.
- Use the flat schema — nested `head`/`tasks` blocks are not it.
- Self-check: run `starter.py`; it compiles your YAML, applies the specs, and
  prints verdicts for a fixed probe list.
