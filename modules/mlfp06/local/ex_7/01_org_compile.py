# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.1: Organisation Definition & GovernanceEngine Compile
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Write a D/T/R (Department/Team/Role) organisation in YAML
#   - Compile that organisation with kailash-pact's GovernanceEngine and
#     apply its clearances + envelopes so they are actually enforced
#   - Understand what compilation checks — and what it does NOT (clearance
#     chains, roles without envelopes, LLM safety)
#   - See the installed engine's real default: roles with no envelope, and
#     unknown addresses, are auto-approved
#
# PREREQUISITES: Exercise 6 (multi-agent systems)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load real adversarial prompts (used by later techniques)
#   2. Write the SG FinTech org YAML to disk
#   3. Compile with GovernanceEngine and inspect the result
#   4. Visualise the org: who heads what, and which envelopes apply
#   5. Apply — why a regulated bank needs compiled governance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from shared.mlfp06.ex_7 import (
    ORG_YAML,
    clearance_chain_violations,
    compile_governance,
    load_adversarial_prompts,
    write_org_yaml,
)

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R: Department / Team / Role
# ════════════════════════════════════════════════════════════════════════
# PACT addresses every position in an organisation with a D/T/R path:
#
#   D (Department): an organisational unit        D1 = ML Engineering
#   T (Team):       a unit inside a department    D1-R1-T1 = Data Analysis
#   R (Role):       a position — human or agent   D1-R1-T1-R1 = data_analyst
#
# Grammar rule: every D or T is immediately followed by the R that heads
# it. "D1-R1-T1-R1" reads "the role heading Team 1, inside the unit headed
# by D1-R1 (the Chief ML Officer)". So every agent role sits under a chain
# of human heads — accountability is structural, not a slide.
#
# DELEGATION is a separate concept: an operating envelope is defined BY
# one role (the human head, `defined_by`) FOR another role (the agent,
# `target`). The envelope is what restricts the agent.
#
# Analogy: A bank manager (head of a branch) tells a teller (a role in a
# team at that branch) "handle deposits up to S$10,000" — that limit is
# the envelope. If the teller approves a S$50,000 transfer, accountability
# traces to the manager who set the envelope, not to "the system".
#
# WHY THIS MATTERS: the EU AI Act (Art. 14, human oversight) and the MAS
# Technology Risk Management Guidelines both expect a named human to be
# accountable for automated decisions. D/T/R makes that a property of the
# compiled organisation.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Adversarial Test Prompts
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load Adversarial Test Prompts")
print("=" * 70)

# TODO: Load 50 adversarial prompts with the shared loader.
# Hint: load_adversarial_prompts(n=...)
adversarial_prompts = ____
print(f"Loaded {adversarial_prompts.height} real adversarial prompts")
print(
    f"Toxicity range: {adversarial_prompts['toxicity_score'].min():.2f} — "
    f"{adversarial_prompts['toxicity_score'].max():.2f}"
)

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert adversarial_prompts.height > 0, "Task 1: adversarial prompts should load"
print("[x] Checkpoint 1 passed — adversarial test data ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Write Organisation YAML
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Write SG FinTech Org YAML (D/T/R)")
print("=" * 70)

# TODO: Materialise ORG_YAML to a temp file and keep the returned path.
org_yaml_path = ____
print(f"YAML path: {org_yaml_path}")
print(f"YAML size: {len(ORG_YAML.splitlines())} lines")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
# The YAML has a flat schema: departments, teams, roles (with `heads` and
# `reports_to`), clearances, and envelopes (one per delegation).
assert "departments" in ORG_YAML and "envelopes" in ORG_YAML
assert org_yaml_path  # path was returned
print("[x] Checkpoint 2 passed — org YAML written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Compile with GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Construct GovernanceEngine from org YAML")
print("=" * 70)

# compile_governance() = load_org_yaml -> GovernanceEngine(org_definition)
# -> apply_governance_specs (grants the YAML clearances and attaches the
# YAML envelopes). Without that last step the engine knows the structure
# but enforces nothing.
# TODO: Compile the YAML and apply its specs. It returns (engine, org).
# Hint: compile_governance(path)
engine, org = ____

print("Compiled organisation:")
print(f"  Agent roles: {org.n_agents}")
print(f"  Delegations: {org.n_delegations}")
print(f"  Departments: {org.n_departments}")
print(f"  Teams:       {org.n_teams}")

structure = pl.DataFrame(
    [
        {"address": addr, "type": node.node_type.name, "name": node.name}
        for addr, node in engine.get_org().nodes.items()
    ]
)
print("\nD/T/R addresses produced by compilation:")
print(structure)

# Compilation does NOT check that a role's clearance is at or below the
# clearance of the role it reports to — we check that ourselves.
# TODO: Check every reporting chain for a clearance above its head's.
# Hint: the shared helper returns a polars DataFrame of violations
violations = ____
print(f"\nClearance-chain violations (child above its head): {violations.height}")
if violations.height:
    print(violations)

# What does the installed engine do with each kind of address?
probes = [
    ("D1-R1-T1-R1", "read_data", "agent WITH an envelope, allowed action"),
    ("D1-R1-T1-R1", "delete_all_records", "agent WITH an envelope, other action"),
    ("D1-R1", "delete_all_records", "department head, NO envelope"),
    ("D99-R99-T99-R99", "read_data", "address not in the org"),
]
probe_rows = []
for address, action, label in probes:
    # TODO: Ask the engine for a verdict on (address, action) with a
    #       context dict carrying cost=1.0.
    verdict = ____
    probe_rows.append(
        {
            "case": label,
            "address": address,
            "action": action,
            "level": verdict.level,
            "allowed": verdict.allowed,
        }
    )
probe_df = pl.DataFrame(probe_rows)
print("\nverify_action() on four kinds of address:")
print(probe_df.select("case", "level", "allowed"))
print(
    """
What compilation checks:
  - every role points at a declared unit (`heads`) and role (`reports_to`)
  - clearance strings are valid pact levels
  - envelope `target` / `defined_by` resolve to real roles
  - an applied envelope may not widen its defining role's own envelope
What it does NOT check:
  - clearance chains (we checked them above)
  - roles with no envelope: the installed default AUTO-APPROVES them
  - addresses that are not in the org: also AUTO-APPROVED
  - content safety of LLM outputs (that needs adversarial testing)
So a deny path exists ONLY where an envelope is attached."""
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert org is not None, "Task 3: compilation should succeed"
assert org.n_agents > 0, "Task 3: org should have agent roles"
assert org.n_delegations > 0, "Task 3: org should have delegations"
assert violations.height == 0, "Task 3: every clearance chain must be monotonic"
levels = dict(zip(probe_df["case"], probe_df["level"]))
assert levels["agent WITH an envelope, other action"] == "blocked"
assert levels["department head, NO envelope"] == "auto_approved"
assert levels["address not in the org"] == "auto_approved"
print(
    f"\n[x] Checkpoint 3 passed — compiled {org.n_agents} agent roles, "
    f"{org.n_delegations} delegations; deny path only where an envelope exists\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the Organisation and its Delegations
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise D/T/R Organisation and Delegations")
print("=" * 70)

nodes = engine.get_org().nodes
team_name_of = {
    addr: node.name for addr, node in nodes.items() if node.node_type.name == "TEAM"
}
delegation_rows = []
for spec in org.envelope_specs:
    # TODO: Build one row per delegation with keys "Defined by (head)",
    #       "Team", "Agent role", "Address", "Budget $", "Clearance".
    # Hint: spec.defined_by / spec.target / spec.financial,
    #       org.address_of(role_id), org.clearances, team_name_of[team address]
    ____
dtr_chains = pl.DataFrame(delegation_rows)
print(dtr_chains)


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Org chart (department heads -> agent roles)
# ════════════════════════════════════════════════════════════════════════
# Visual proof that every agent role sits under a named human head. Built
# from the compiled delegations above, not from a hand-typed list.

heads = list(dict.fromkeys(dtr_chains["Defined by (head)"].to_list()))
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis("off")
ax.set_title(
    "D/T/R Organisation Chart — SG FinTech AI Division", fontweight="bold", fontsize=13
)

head_x = {h: 10 * (i + 0.5) / len(heads) for i, h in enumerate(heads)}
agents = dtr_chains["Agent role"].to_list()
agent_x = {a: 10 * (i + 0.5) / len(agents) for i, a in enumerate(agents)}
for head, hx in head_x.items():
    ax.text(
        hx,
        6,
        f"{head.replace('_', chr(10))}\n[{org.clearances.get(head, '?')}]",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#3498db", edgecolor="white"),
        color="white",
    )
for row in dtr_chains.iter_rows(named=True):
    agent, hx = row["Agent role"], head_x[row["Defined by (head)"]]
    ax_ = agent_x[agent]
    ax.text(
        ax_,
        3.3,
        f"{agent.replace('_', chr(10))}\n[{row['Clearance']}]\n${row['Budget $']:g}",
        ha="center",
        va="center",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#2ecc71", edgecolor="white"),
        color="white",
    )
    ax.annotate(
        "",
        xy=(ax_, 4.1),
        xytext=(hx, 5.3),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5),
    )

ax.text(0.3, 1.5, "Department head (human role)", fontsize=9, color="#3498db")
ax.text(0.3, 0.8, "Team-head agent role", fontsize=9, color="#2ecc71")
ax.text(
    5,
    1.5,
    f"Agent roles: {org.n_agents}  |  Delegations: {org.n_delegations}",
    fontsize=10,
    color="#2c3e50",
)
ax.text(5, 0.8, "arrow = envelope defined_by -> target", fontsize=9, color="#7f8c8d")

plt.tight_layout()
fname = OUTPUT_DIR / "ex7_dtr_org_chart.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Regulated Bank
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore retail bank deploys an AI agent platform for
# customer service, fraud triage, and risk reporting. Its technology-risk
# auditors ask one question: "For every AI-taken action this year, show
# me which human authorised that class of action."
#
# Without compiled D/T/R governance, the bank's answer is "well, the ML
# team wrote the agents, so... them, I guess?" — no specific human is
# accountable for any specific action. With compile_governance() running
# at boot, every agent role has an address under a named head and an
# envelope that head defined; the auditor's question becomes a lookup.
#
# The probe in Task 3 is the second half of the lesson: an agent role
# that was never given an envelope is NOT restricted by default. "Every
# agent has an envelope" must itself be a checked property of your org.
#
# BUSINESS IMPACT (illustrative figures): if remediating one adverse
# audit finding costs on the order of S$500K in consultants, rework and
# management time, a compiled org that removes a whole class of findings
# ("no accountable owner for automated decisions") pays for itself
# quickly.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Compilation Is the Structural Audit Gate")
print("=" * 70)
print(
    f"  {org.n_agents} agent roles, {org.n_delegations} envelopes, "
    f"each defined by a named human head."
)
print("  Compile + apply proves the structure and the envelopes are loaded;")
print("  it does not make envelope-less roles safe — they are auto-approved.")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens
# ══════════════════════════════════════════════════════════════════
# The LLM Observatory's Governance lens runs negative drills — actions
# that SHOULD be denied — straight through engine.verify_action().
# No LLM call is needed.
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=engine, run_id="ex_7_1_org_compile")
# TODO: Run the three deny drills below through the governance lens.
# Hint: obs.governance.negative_drills([...scenario dicts...])
drills = ____(
    [
        {
            "label": "analyst deletes records",
            "role_address": "D1-R1-T1-R1",
            "action": "delete_all_records",
            "context": {"cost": 0.10},
        },
        {
            "label": "customer agent overspends",
            "role_address": "D3-R1-T1-R1",
            "action": "answer_question",
            "context": {"cost": 50.0},
        },
        {
            "label": "head with no envelope deletes",
            "role_address": "D1-R1",
            "action": "delete_all_records",
            "context": {"cost": 0.10},
        },
    ]
)
print("\n── LLM Observatory: governance negative drills ──")
print(drills.select("scenario", "verdict"))
print(obs.governance.report())
# INTERPRETATION: the two drills against envelope-carrying agent roles are
# blocked; the drill against the department head passes, because that
# role has no envelope. That passing drill is the finding to act on.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Wrote a D/T/R (Department/Team/Role) organisation in YAML
  [x] Compiled it and applied its clearances + envelopes to the engine
  [x] Checked what compilation validates — and the clearance chains it doesn't
  [x] Saw the real default: no envelope (or unknown address) = auto-approved
  [x] Mapped the grammar to a Singapore retail bank audit scenario

  KEY INSIGHT: Governance is engineering, not philosophy. If your
  governance story cannot be compiled, applied, and probed with deny
  cases, it is a slide deck, not a control.

  Next: 02_envelopes.py adds operating envelopes + monotonic
  tightening on top of the compiled org.
"""
)
