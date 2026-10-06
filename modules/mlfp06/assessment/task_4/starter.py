# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP06 — Assessment Task 4: Design the Governance Org for a Bank's AI Office

Implement `solve()`. problem.md holds the organisational brief (departments,
teams, roles, clearances, envelopes). The grader re-loads YOUR yaml and
rebuilds the engine itself before probing — the YAML is the artefact.

    python starter.py               # (you) compile + probe your org
    python grader.py starter.py     # (instructor) grade an attempt

No LLM is involved in this task.
"""
from __future__ import annotations

ORG_YAML = ""  # your governance definition goes here (flat schema — see problem.md)


def solve() -> dict:
    """Compile and apply your ORG_YAML.

    Returns:
        {"org_yaml": str, "engine": GovernanceEngine} — the engine built from
        your YAML with clearances and envelopes applied.
    """
    raise NotImplementedError("Implement solve() — see problem.md")


if __name__ == "__main__":
    out = solve()
    engine = out["engine"]
    org = engine.get_org()
    from pact import NodeType

    counts = {}
    for node in org.nodes.values():
        counts[node.node_type.name] = counts.get(node.node_type.name, 0) + 1
    print("node counts:", counts)
    for addr in sorted(org.nodes):
        node = org.nodes[addr]
        role = node.role_definition.role_id if node.role_definition else ""
        print(f"  {addr:<14} {node.node_type.name:<10} {role}")
