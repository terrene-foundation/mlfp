# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 4: PACT Governance for a Production Agent Fleet
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
Fully deterministic — no LLM calls.
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

# Dash-delimited D/T/R addresses (Department-R-Team-R for agents).
AGENT_ADDRESSES: dict[str, str] = {
    "data_analyst": "D1-R1-T1-R1",
    "model_trainer": "D1-R1-T2-R1",
    "risk_assessor": "D2-R1-T1-R1",
    "customer_agent": "D3-R1-T1-R1",
}
DELEGATOR_ADDRESSES: dict[str, str] = {
    "chief_ml_officer": "D1-R1",
    "chief_risk_officer": "D2-R1",
    "vp_customer": "D3-R1",
}
ROLE_TO_DELEGATOR: dict[str, str] = {
    "data_analyst": "chief_ml_officer",
    "model_trainer": "chief_ml_officer",
    "risk_assessor": "chief_risk_officer",
    "customer_agent": "vp_customer",
}

# The 10 verify_action probes, in order (role, action, cost).
CASES: list[tuple[str, str, float]] = [
    ("data_analyst", "read_data", 0.10),
    ("data_analyst", "deploy_model", 0.10),
    ("data_analyst", "read_data", 50.0),
    ("model_trainer", "train_model", 5.0),
    ("model_trainer", "deploy_model", 1.0),
    ("risk_assessor", "audit_model", 0.50),
    ("risk_assessor", "access_audit_log", 1.0),
    ("customer_agent", "search_faq", 0.01),
    ("customer_agent", "read_data", 0.10),
    ("customer_agent", "answer_question", 100.0),
]


def _envelope(
    envelope_id: str,
    clearance: ConfidentialityLevel,
    max_spend_usd: float,
    allowed_actions: list[str],
) -> ConstraintEnvelopeConfig:
    """Build a structurally-complete 5-dimension envelope."""
    return ConstraintEnvelopeConfig(
        id=envelope_id,
        description=envelope_id,
        confidentiality_clearance=clearance,
        financial=FinancialConstraintConfig(max_spend_usd=max_spend_usd),
        operational=OperationalConstraintConfig(
            allowed_actions=allowed_actions, blocked_actions=[]
        ),
        temporal=TemporalConstraintConfig(blackout_periods=[]),
        data_access=DataAccessConstraintConfig(
            read_paths=["/*"], write_paths=[], blocked_data_types=[]
        ),
        communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
        max_delegation_depth=3,
    )


def solve() -> dict:
    """Compile the org, attach envelopes, run verify_action, test escalation."""
    engine, org = compile_governance()

    org_stats = {
        "n_agents": org.n_agents,
        "n_delegations": org.n_delegations,
        "n_departments": org.n_departments,
    }

    # Step 2 — attach least-privilege envelopes.
    role_specs: dict[str, tuple[ConfidentialityLevel, float, list[str]]] = {
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
    for role_id, (clearance, cap, actions) in role_specs.items():
        env = _envelope(f"{role_id}_envelope", clearance, cap, actions)
        engine.set_role_envelope(
            RoleEnvelope(
                id=f"{role_id}_role_envelope",
                defining_role_address=DELEGATOR_ADDRESSES[ROLE_TO_DELEGATOR[role_id]],
                target_role_address=AGENT_ADDRESSES[role_id],
                envelope=env,
            )
        )

    # Step 3 — exercise verify_action across allow + deny paths.
    verdicts: list[bool] = []
    for role_id, action, cost in CASES:
        verdict = engine.verify_action(
            role_address=AGENT_ADDRESSES[role_id],
            action=action,
            context={"cost": cost},
        )
        verdicts.append(bool(verdict.allowed))

    # Step 4 — privilege-escalation attempt, caught structurally.
    parent = _envelope(
        "vp_customer_parent",
        ConfidentialityLevel.CONFIDENTIAL,
        50.0,
        ["answer_question", "search_faq"],
    )
    rogue_child = _envelope(
        "customer_agent_rogue",
        ConfidentialityLevel.RESTRICTED,  # escalated clearance
        1000.0,  # escalated budget
        ["answer_question", "search_faq", "read_data", "deploy_model"],  # widened
    )
    escalation_caught = False
    try:
        RoleEnvelope.validate_tightening(
            parent_envelope=parent, child_envelope=rogue_child
        )
    except MonotonicTighteningError:
        escalation_caught = True

    return {
        "org_stats": org_stats,
        "verdicts": verdicts,
        "escalation_caught": escalation_caught,
    }


if __name__ == "__main__":
    out = solve()
    print("org_stats:        ", out["org_stats"])
    print("verdicts:         ", out["verdicts"])
    print("escalation_caught:", out["escalation_caught"])
