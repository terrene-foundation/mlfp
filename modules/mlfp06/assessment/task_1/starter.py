# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 1: Operating Envelopes and Deny-Paths

Implement `solve()` and `validate_child()`. problem.md holds the envelope
table, the ladder, and the acceptance criteria. The grader probes your
RETURNED engine with its own actions and costs — hard-coded verdicts fail.

    python starter.py               # (you) attach envelopes + probe them
    python grader.py starter.py     # (instructor) grade an attempt

No LLM is involved in this task.
"""
from __future__ import annotations

# The envelope table from problem.md, as data.
ROLE_ADDRESSES = {
    "data_analyst": "D1-R1-T1-R1",
    "model_trainer": "D1-R1-T2-R1",
    "model_deployer": "D1-R1-T3-R1",  # no envelope — fail-open default
    "risk_assessor": "D2-R1-T1-R1",
    "bias_checker": "D2-R1-T2-R1",  # no envelope — fail-open default
    "customer_agent": "D3-R1-T1-R1",
}
HEAD_ADDRESSES = {
    "chief_ml_officer": "D1-R1",
    "chief_risk_officer": "D2-R1",
    "vp_customer": "D3-R1",
}
ROLE_TO_HEAD = {
    "data_analyst": "chief_ml_officer",
    "model_trainer": "chief_ml_officer",
    "model_deployer": "chief_ml_officer",
    "risk_assessor": "chief_risk_officer",
    "bias_checker": "chief_risk_officer",
    "customer_agent": "vp_customer",
}


def solve() -> dict:
    """Compile the canonical org and attach the four envelopes from
    problem.md's table.

    Returns:
        {"engine": GovernanceEngine} — the engine with the four envelopes
        attached (model_deployer and bias_checker stay envelope-less).
    """
    raise NotImplementedError("Implement solve() — see problem.md")


def validate_child(parent, child) -> bool:
    """True iff `child` is a legal monotonic tightening of `parent`.

    Both arguments are pact ConstraintEnvelopeConfig objects. Use the
    framework's structural check (note: it takes keyword-only arguments).
    """
    raise NotImplementedError("Implement validate_child() — see problem.md")


if __name__ == "__main__":
    out = solve()
    engine = out["engine"]
    probes = [
        ("data_analyst", "read_data", 0.10),
        ("data_analyst", "deploy_model", 0.10),
        ("customer_agent", "answer_question", 100.0),
        ("D99-R99-T99-R99", "read_data", 0.0),  # unknown address
    ]
    for role, action, cost in probes:
        addr = ROLE_ADDRESSES.get(role, role)
        v = engine.verify_action(role_address=addr, action=action, context={"cost": cost})
        print(f"{role:>16} {action:<18} ${cost:>7.2f} -> allowed={v.allowed} level={v.level}")
