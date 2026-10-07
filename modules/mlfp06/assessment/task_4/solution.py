# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 4: Design the Governance Org (Reference Solution)

Withheld from students. Verified to pass grader.py across seeds. No LLM calls.

The brief expressed in pact's flat YAML schema: departments/teams/roles with
`heads` and `reports_to`, clearances on the real ladder (heads secret, agents
at or below), one envelope per agent defined by its department head. Humans
carry no envelopes — the installed auto-approve default applies to them.
"""
from __future__ import annotations

ORG_YAML = """
# Regional Bank — AI Office Governance Definition
org_id: "bank_ai_office"
name: "AI Office"

departments:
  - id: "model_development"
    name: "Model Development"
  - id: "operations"
    name: "Operations"

teams:
  - id: "research_team"
    name: "Research"
  - id: "deployment_team"
    name: "Deployment"
  - id: "support_team"
    name: "Customer Support"

roles:
  - id: "chief_data_officer"
    name: "Chief Data Officer"
    heads: "model_development"
  - id: "head_of_operations"
    name: "Head of Operations"
    heads: "operations"
  - id: "research_scientist"
    name: "Research Scientist"
    reports_to: "chief_data_officer"
    heads: "research_team"
  - id: "ml_engineer"
    name: "ML Engineer"
    reports_to: "chief_data_officer"
    heads: "deployment_team"
  - id: "support_agent"
    name: "Support Agent"
    reports_to: "head_of_operations"
    heads: "support_team"

clearances:
  - role: "chief_data_officer"
    level: "secret"
  - role: "head_of_operations"
    level: "secret"
  - role: "research_scientist"
    level: "confidential"
  - role: "ml_engineer"
    level: "confidential"
  - role: "support_agent"
    level: "public"

envelopes:
  - target: "research_scientist"
    defined_by: "chief_data_officer"
    financial:
      max_spend_usd: 50.0
    operational:
      allowed_actions: ["read_data", "run_experiment", "train_model"]
  - target: "ml_engineer"
    defined_by: "chief_data_officer"
    financial:
      max_spend_usd: 80.0
    operational:
      allowed_actions: ["deploy_model", "monitor_model", "rollback_model", "read_data"]
  - target: "support_agent"
    defined_by: "head_of_operations"
    financial:
      max_spend_usd: 5.0
    operational:
      allowed_actions: ["answer_ticket", "search_kb"]
"""


def solve() -> dict:
    import tempfile

    from kailash.trust.pact.yaml_resolvers import apply_governance_specs
    from pact import GovernanceEngine, load_org_yaml

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(ORG_YAML)
        path = f.name
    loaded = load_org_yaml(path)
    engine = GovernanceEngine(loaded.org_definition)
    apply_governance_specs(engine, loaded)
    return {"org_yaml": ORG_YAML, "engine": engine}


if __name__ == "__main__":
    out = solve()
    engine = out["engine"]
    org = engine.get_org()
    by_role = {
        node.role_definition.role_id: addr
        for addr, node in org.nodes.items()
        if node.role_definition is not None
    }
    probes = [
        ("research_scientist", "run_experiment", 10.0),
        ("research_scientist", "deploy_model", 1.0),
        ("ml_engineer", "deploy_model", 200.0),
        ("support_agent", "answer_ticket", 1.0),
        ("support_agent", "read_data", 0.10),
        ("chief_data_officer", "read_data", 0.0),
    ]
    for role, action, cost in probes:
        v = engine.verify_action(
            role_address=by_role[role], action=action, context={"cost": cost}
        )
        print(f"{role:>20} {action:<16} ${cost:>7.2f} -> allowed={v.allowed} level={v.level}")
    v = engine.verify_action(role_address="D9-R9-T9-R9", action="read_data", context={})
    print(f"{'unknown':>20} {'read_data':<16} {'':>8} -> allowed={v.allowed} level={v.level}")
