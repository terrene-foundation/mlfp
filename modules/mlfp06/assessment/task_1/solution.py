# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 1: Operating Envelopes and Deny-Paths
(Reference Solution)

Withheld from students. Verified to pass grader.py across seeds. No LLM calls.

The deny-paths only exist because envelopes are attached: the installed
kailash-pact auto-approves roles without envelopes (and unknown addresses),
so attaching the four least-privilege envelopes is the whole job. The
tightening check is structural: RoleEnvelope.validate_tightening is
keyword-only and raises MonotonicTighteningError on any widening —
clearance (secret child under confidential parent), budget, or actions.
"""
from __future__ import annotations

from kailash.trust.pact.envelopes import MonotonicTighteningError
from pact import (
    CommunicationConstraintConfig,
    ConfidentialityLevel,
    ConstraintEnvelopeConfig,
    DataAccessConstraintConfig,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
    RoleEnvelope,
    TemporalConstraintConfig,
)

from shared.mlfp06.ex_7 import compile_governance

ROLE_ADDRESSES = {
    "data_analyst": "D1-R1-T1-R1",
    "model_trainer": "D1-R1-T2-R1",
    "risk_assessor": "D2-R1-T1-R1",
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
    "risk_assessor": "chief_risk_officer",
    "customer_agent": "vp_customer",
}

# role -> (clearance, max_spend_usd, allowed_actions) from problem.md.
ENVELOPE_SPEC = {
    "data_analyst": (
        ConfidentialityLevel.RESTRICTED,
        20.0,
        ["read_data", "summarise_data", "generate_report"],
    ),
    "model_trainer": (
        ConfidentialityLevel.RESTRICTED,
        100.0,
        ["train_model", "evaluate_model", "read_data"],
    ),
    "risk_assessor": (
        ConfidentialityLevel.RESTRICTED,
        200.0,
        ["read_data", "audit_model", "generate_report", "access_audit_log"],
    ),
    "customer_agent": (
        ConfidentialityLevel.PUBLIC,
        5.0,
        ["answer_question", "search_faq"],
    ),
}


def _config(
    envelope_id: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
) -> ConstraintEnvelopeConfig:
    """A structurally-complete five-dimension envelope config."""
    return ConstraintEnvelopeConfig(
        id=envelope_id,
        description=envelope_id,
        confidentiality_clearance=clearance,
        financial=FinancialConstraintConfig(max_spend_usd=max_spend_usd),
        operational=OperationalConstraintConfig(
            allowed_actions=list(allowed_actions), blocked_actions=[]
        ),
        temporal=TemporalConstraintConfig(blackout_periods=[]),
        data_access=DataAccessConstraintConfig(
            read_paths=["/*"], write_paths=[], blocked_data_types=[]
        ),
        communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
        max_delegation_depth=3,
    )


def solve() -> dict:
    # Structural compile only: the YAML envelope block stays unapplied, so the
    # ONLY governance in force is what this function attaches.
    engine, _org = compile_governance(apply_specs=False)
    for role, (clearance, cap, actions) in ENVELOPE_SPEC.items():
        engine.set_role_envelope(
            RoleEnvelope(
                id=f"{role}_role_envelope",
                defining_role_address=HEAD_ADDRESSES[ROLE_TO_HEAD[role]],
                target_role_address=ROLE_ADDRESSES[role],
                envelope=_config(f"{role}_envelope", clearance, cap, actions),
            )
        )
    return {"engine": engine}


def validate_child(parent, child) -> bool:
    """True iff `child` is equal-or-tighter than `parent` on every dimension."""
    try:
        RoleEnvelope.validate_tightening(parent_envelope=parent, child_envelope=child)
    except MonotonicTighteningError:
        return False
    return True


if __name__ == "__main__":
    out = solve()
    engine = out["engine"]
    probes = [
        ("D1-R1-T1-R1", "read_data", 0.10),
        ("D1-R1-T1-R1", "deploy_model", 0.10),
        ("D3-R1-T1-R1", "answer_question", 100.0),
        ("D99-R99-T99-R99", "read_data", 0.0),
        ("D1-R1-T3-R1", "deploy_model", 1.0),
    ]
    for addr, action, cost in probes:
        v = engine.verify_action(role_address=addr, action=action, context={"cost": cost})
        print(f"{addr:>16} {action:<18} ${cost:>7.2f} -> allowed={v.allowed} level={v.level}")

    parent = _config("p", ConfidentialityLevel.CONFIDENTIAL, 50.0, ["a", "b"])
    legal = _config("c", ConfidentialityLevel.RESTRICTED, 25.0, ["a"])
    rogue = _config("r", ConfidentialityLevel.SECRET, 25.0, ["a"])
    print("tighter child legal:", validate_child(parent, legal))
    print("escalating child caught:", not validate_child(parent, rogue))
