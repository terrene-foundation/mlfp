# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.2: Build a Governed Agent Pipeline with PACT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compile a PACT organisation YAML into a live GovernanceEngine and
#     apply its clearances + envelopes
#   - Wrap a Kaizen BaseAgent in GovernedSupervisor at three trust tiers
#   - Read D/T/R (Department/Team/Role) addresses for the three agent roles
#   - Verify each tier's deny path, and see that a role without an
#     envelope is auto-approved by the installed default
#   - Apply governed agents to a Singapore financial advisory scenario
#
# PREREQUISITES: Exercise 8.1 (adapter loading), MLFP06 Ex 7 (PACT intro)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Compile the PACT org YAML into a GovernanceEngine
#   2. Build the shared capstone stack via ``build_capstone_stack(engine)``
#   3. Verify the deny path on an out-of-envelope action
#   4. Visualise the envelope hierarchy
#   5. Apply to a Singapore wealth-advisory scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
from pact import ConfidentialityLevel, NodeType

from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    build_capstone_stack,
    compile_capstone_governance,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R and Operating Envelopes
# ════════════════════════════════════════════════════════════════════════
# PACT addresses every position with D/T/R = Department / Team / Role:
#
#   D1            — the AI Services department
#   D1-R1         — the role heading it (the ML Director, a human)
#   D1-R1-T1      — the Question Answering team
#   D1-R1-T1-R1   — the role heading that team (the qa agent)
#
# Each agent role gets an operating envelope DEFINED BY its head: a
# budget (max $/request), an allowed-action list, a confidentiality
# clearance and three more dimensions. `engine.verify_action()` checks a
# requested action against the role's attached envelope.
#
# The installed default is NOT fail-closed: a role with no attached
# envelope (here, the ML Director) is auto-approved ("No envelope
# constraints -- action permitted"). Deny paths exist only where an
# envelope is attached, so the capstone attaches one to every agent role
# and TESTS the deny path.
#
# The runtime wrapper is `GovernedSupervisor` from kaizen_agents. It
# takes budget, tools and clearance, and records a hash-chained audit
# trail. `build_capstone_stack` builds the same 3-tier stack for every
# capstone technique file.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Compile the org YAML into a GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

org_path = write_org_yaml()
# load_org_yaml -> GovernanceEngine(org_definition) -> apply_governance_specs
governance_engine, loaded = compile_capstone_governance(org_path)
compiled_org = governance_engine.get_org()
# Agent roles are ROLE nodes heading a team ("-T<n>-R<n>" in the address);
# the department head is a ROLE node directly under the department ("D1-R1").
n_agents = sum(
    1
    for n in compiled_org.nodes.values()
    if n.node_type == NodeType.ROLE and "-T" in n.address and not n.is_vacant
)
n_roles = sum(1 for n in compiled_org.nodes.values() if n.node_type == NodeType.ROLE)
n_departments = sum(
    1 for n in compiled_org.nodes.values() if n.node_type == NodeType.DEPARTMENT
)
print(
    f"Compiled org: {n_departments} department, "
    f"{n_roles} roles total ({n_agents} agent roles), "
    f"{len(loaded.envelopes)} YAML envelopes applied"
)
for addr, node in compiled_org.nodes.items():
    print(f"  {addr:<12} {node.node_type.name:<10} {node.name}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert n_agents == 3, "Task 1: compiled org should carry 3 agent roles"
assert len(loaded.envelopes) == 3, "Task 1: one envelope per agent role"
print("✓ Checkpoint 1 passed — governance compiled\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the shared 3-tier governed stack
# ════════════════════════════════════════════════════════════════════════
#
# `build_capstone_stack(engine)` attaches a full 5-dimension
# `ConstraintEnvelopeConfig` to each agent role (qa / admin / audit) and
# returns a `GovernedSupervisor` for each — plus the tier metadata.

agents_by_role, tiers = build_capstone_stack(governance_engine)
print("Governed agent tiers (from build_capstone_stack):")
for tier in tiers:
    gs = agents_by_role[tier.role]
    env = gs.envelope
    print(
        f"  {tier.role:6s} -> {tier.address:14s}  "
        f"budget=${env.financial.max_spend_usd:>5.1f}  "
        f"clearance={env.confidentiality_clearance.name:<12s}  "
        f"tools={len(env.operational.allowed_actions)}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 2: should build three tiers"
assert agents_by_role["qa"].envelope.financial.max_spend_usd == 1.0
assert agents_by_role["admin"].envelope.financial.max_spend_usd == 10.0
assert agents_by_role["audit"].envelope.financial.max_spend_usd == 50.0
print("✓ Checkpoint 2 passed — three governed tiers wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Verify the deny path on an out-of-envelope action
# ════════════════════════════════════════════════════════════════════════
#
# Every tier has an envelope attached, so ask the engine to verify an
# action the qa tier does NOT have — it MUST be blocked. Then ask the
# same of the ML Director, who has no envelope: the installed default
# auto-approves it. That contrast is why every agent role needs one.

denied = governance_engine.verify_action(
    role_address="D1-R1-T1-R1",  # qa tier
    action="update_model",  # admin-only tool
    context={"cost": 0.10},
)
director = governance_engine.verify_action(
    role_address="D1-R1",  # ML Director — no envelope attached
    action="update_model",
    context={"cost": 0.10},
)
print(f"qa tier asks to update_model:     level={denied.level}")
print(f"  reason: {denied.reason[:100]}")
print(f"ML Director asks to update_model: level={director.level}")
print(f"  reason: {director.reason[:100]}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert not denied.allowed, "Task 3: qa tier MUST NOT be allowed to update_model"
assert denied.level == "blocked"
assert director.level == "auto_approved", "installed default for an envelope-less role"
print("✓ Checkpoint 3 passed — deny path verified where an envelope exists\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the envelope hierarchy
# ════════════════════════════════════════════════════════════════════════

all_tools = sorted({tool for t in tiers for tool in t.tools})
deny_probe = []
for t in tiers:
    # An action this tier does NOT hold — its deny path must block it.
    missing = [a for a in all_tools if a not in t.tools] or ["delete_all_records"]
    v = governance_engine.verify_action(t.address, missing[0], {"cost": 0.10})
    deny_probe.append(f"{missing[0]} -> {v.level}")

envelope_table = pl.DataFrame(
    {
        "Role": [t.role for t in tiers],
        "Address": [t.address for t in tiers],
        "Clearance": [t.clearance for t in tiers],
        "Budget (USD)": [t.budget_usd for t in tiers],
        "Allowed tools": [", ".join(t.tools) for t in tiers],
        "Deny-path probe": deny_probe,
    }
)
envelope_table.write_parquet(OUTPUT_DIR / "governance_envelopes.parquet")
print("\nGovernance envelope hierarchy:")
print(envelope_table)

# INTERPRETATION: the three tiers are SIBLING envelopes, each defined by
# the ML Director and each no wider than the Director's authority. They
# are not a superset chain: audit has access_audit_log and
# generate_report but NOT admin's update_model or monitor_drift. Monotonic
# tightening constrains parent -> child (Director -> each agent), not one
# sibling against another. A different tier is the only path to different
# capability, and each tier keeps its own audit log.

overlap = set(tiers[1].tools) - set(tiers[2].tools)
print(f"\n  admin tools that audit does NOT have: {sorted(overlap)}")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Governance tier comparison chart
# ════════════════════════════════════════════════════════════════════════
# Grouped bars comparing budget, clearance and tool count per tier. The
# clearance bar is the tier's position on pact's ladder
# (public < restricted < confidential < secret < top_secret).

ladder = [level.value for level in ConfidentialityLevel]
tier_names = [t.role for t in tiers]
budgets = [t.budget_usd for t in tiers]
clearance_rank = [ladder.index(t.clearance) for t in tiers]
tool_counts = [len(t.tools) for t in tiers]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(tier_names))
width = 0.25
max_budget = max(budgets)
top_rank = len(ladder) - 1
ax.bar(
    [i - width for i in x],
    [b / max_budget for b in budgets],
    width,
    label="Budget (normalised)",
    color="#3498db",
)
ax.bar(
    x,
    [c / top_rank for c in clearance_rank],
    width,
    label="Clearance (rank on pact ladder)",
    color="#e67e22",
)
ax.bar(
    [i + width for i in x],
    [t / max(tool_counts) for t in tool_counts],
    width,
    label="Tool count (normalised)",
    color="#2ecc71",
)
ax.set_xticks(list(x))
ax.set_xticklabels(tier_names, fontsize=11)
ax.set_ylabel("Normalised value (0-1)")
ax.set_title("Governance Tier Comparison — Sibling Envelopes", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.2)
for i, t in enumerate(tiers):
    ax.text(i - width, budgets[i] / max_budget + 0.03, f"${t.budget_usd:g}", ha="center", fontsize=8)
    ax.text(i, clearance_rank[i] / top_rank + 0.03, t.clearance, ha="center", fontsize=8)
    ax.text(i + width, tool_counts[i] / max(tool_counts) + 0.03, f"{tool_counts[i]}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_governance_tiers.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore Wealth Advisory
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore wealth-management firm operates a retail-facing
# advisory bot (qa tier), an internal portfolio-ops dashboard (admin
# tier), and a compliance audit console (audit tier). Its regulator
# expects every model-produced recommendation that reaches a retail
# customer to be traceable.
#
# BUSINESS IMPACT (illustrative figures): if one untraceable retail
# advisory incident costs ~S$150,000 in external legal and remediation
# work, envelopes that stop the retail tier from invoking portfolio-ops
# tools — and cap its spend per request — are cheap insurance. The audit
# tier gets the access an investigation needs without widening the
# retail or ops tiers.

print("\n" + "=" * 70)
print("  APPLY — Regulated Wealth Advisory")
print("=" * 70)
for t in tiers:
    print(
        f"  {t.role:<6} tier: {len(t.tools)} tools, US${t.budget_usd:g} budget cap, "
        f"{t.clearance.upper()} clearance, deny probe: "
        f"{envelope_table.filter(pl.col('Role') == t.role)['Deny-path probe'][0]}"
    )
print("  Tier-jumping is refused by verify_action on every attached envelope;")
print("  the envelope-less ML Director role is the gap to close next.")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens
# ══════════════════════════════════════════════════════════════════
# Each tier tries every tool it does NOT hold. All must be blocked.
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=governance_engine, run_id="ex_8_2_governance")
drills = obs.governance.negative_drills(
    [
        {
            "label": f"{t.role} -> {tool}",
            "role_address": t.address,
            "action": tool,
            "context": {"cost": 0.10},
        }
        for t in tiers
        for tool in all_tools
        if tool not in t.tools
    ]
)
print("\n── LLM Observatory: cross-tier drills ──")
print(drills.select("scenario", "verdict"))
print(obs.governance.report())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Compiled a PACT org YAML and applied its clearances + envelopes
  [x] Built the shared 3-tier capstone stack via build_capstone_stack
  [x] Verified the deny path where an envelope is attached
  [x] Saw the envelope-less department head auto-approved (installed default)
  [x] Visualised three sibling envelopes on pact's clearance ladder
  [x] Applied governed agents to a regulated wealth-advisory scenario

  KEY INSIGHT: Governance is not a layer you bolt onto a working
  agent — it is a WRAPPER that runs before the agent's first token
  is generated. And it only restricts what you attached an envelope to.

  Next: 03_multichannel_serving.py deploys these tiers via Nexus.
"""
)
