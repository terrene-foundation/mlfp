# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.2: Build a Governed Agent Pipeline with PACT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Compile a PACT organisation YAML into a live GovernanceEngine
#   - Wrap a Kaizen BaseAgent in PactGovernedAgent at multiple trust tiers
#   - Configure D/T/R (Delegator/Task/Responsible) chains for three roles
#   - Visualise the governance envelope hierarchy
#   - Apply governed agents to a Singapore financial advisory scenario
#
# PREREQUISITES: Exercise 8.1 (adapter loading), MLFP06 Ex 7 (PACT intro)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Compile the PACT org YAML into a GovernanceEngine
#   2. Build the base CapstoneQAAgent
#   3. Wrap in three PactGovernedAgent tiers (qa / admin / audit)
#   4. Visualise the envelope hierarchy
#   5. Apply to a Singapore financial advisory (MAS TRM) scenario
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl

from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.mlfp06.ex_8 import (
    CapstoneQAAgent,
    OUTPUT_DIR,
    run_async,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R and Operating Envelopes
# ════════════════════════════════════════════════════════════════════════
# PACT governance is built on three ideas:
#
#   Delegator (D) — the human or role who authorises a task
#   Task       (T) — what can be done (bounded scope)
#   Responsible (R) — the agent carrying out the task
#
# Every agent action is traceable back to a human Delegator. There is
# no "rogue agent" path. Each D/T/R chain carries an operating envelope:
# a budget (max $/request), an allowed tool list, and a data clearance
# level. The GovernanceEngine's default is fail-closed: if a request
# falls outside any envelope, it is rejected, not "best-effort handled".
#
# This is the difference between "we trust the model" (BLOCKED) and
# "we trust the envelope around the model" (governed).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Compile the org YAML into a GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

governance_engine = GovernanceEngine()
org_path = write_org_yaml()


async def compile_governance() -> object:
    org = governance_engine.compile_org(org_path)
    print(f"Compiled: {org.n_agents} agents, {org.n_delegations} delegations")
    return org


org = run_async(compile_governance())

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert org is not None and org.n_agents >= 3, "Task 1: should compile 3+ agents"
print("\u2713 Checkpoint 1 passed — governance compiled\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the base agent
# ════════════════════════════════════════════════════════════════════════

base_qa = CapstoneQAAgent()
print(f"Base agent class: {type(base_qa).__name__}")
print(f"Signature:        {CapstoneQAAgent.signature.__name__}")
print(f"Model:            {CapstoneQAAgent.model}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert base_qa is not None, "Task 2: base agent should instantiate"
print("\u2713 Checkpoint 2 passed — base agent ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Wrap the agent in three governance tiers
# ════════════════════════════════════════════════════════════════════════

governed_qa = PactGovernedAgent(
    agent=base_qa,
    governance_engine=governance_engine,
    role="responder",
    max_budget_usd=1.0,
    allowed_tools=["generate_answer", "search_context"],
    clearance_level="internal",
)

governed_admin = PactGovernedAgent(
    agent=base_qa,
    governance_engine=governance_engine,
    role="operator",
    max_budget_usd=10.0,
    allowed_tools=[
        "generate_answer",
        "search_context",
        "update_model",
        "view_metrics",
        "monitor_drift",
    ],
    clearance_level="confidential",
)

governed_audit = PactGovernedAgent(
    agent=base_qa,
    governance_engine=governance_engine,
    role="auditor",
    max_budget_usd=50.0,
    allowed_tools=[
        "generate_answer",
        "search_context",
        "view_metrics",
        "access_audit_log",
        "generate_report",
    ],
    clearance_level="restricted",
)

agents_by_role: dict[str, PactGovernedAgent] = {
    "qa": governed_qa,
    "admin": governed_admin,
    "audit": governed_audit,
}

print("Governed agent tiers:")
for role, agent in agents_by_role.items():
    print(f"  {role:6s} -> {type(agent).__name__} (clearance={agent.clearance_level})")

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 3: should wrap three tiers"
print("\u2713 Checkpoint 3 passed — three governed tiers created\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the envelope hierarchy
# ════════════════════════════════════════════════════════════════════════

envelope_table = pl.DataFrame(
    {
        "Role": ["qa", "admin", "audit"],
        "Clearance": ["internal", "confidential", "restricted"],
        "Budget (USD)": [1.0, 10.0, 50.0],
        "Allowed tools": [
            "answer+search",
            "+update+metrics+drift",
            "+audit_log+report",
        ],
        "Fail mode": ["closed", "closed", "closed"],
    }
)
envelope_table.write_parquet(OUTPUT_DIR / "governance_envelopes.parquet")
print("\nGovernance envelope hierarchy:")
print(envelope_table)

# INTERPRETATION: The envelope is monotonic — admin is a superset of
# qa, audit is a superset of admin. Monotonic tightening prevents
# privilege escalation between tiers: no governed agent can expand
# its own envelope. A higher tier is the only path to more capability,
# and that tier has its own audit log.


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Governance tier comparison chart
# ════════════════════════════════════════════════════════════════════════
# Grouped bar chart comparing the three governance tiers across budget,
# clearance level, and tool count. Makes the monotonic escalation
# visible at a glance.

tiers = ["qa", "admin", "audit"]
budgets = [1.0, 10.0, 50.0]
clearance_numeric = [1, 2, 3]  # internal=1, confidential=2, restricted=3
tool_counts = [2, 5, 5]

fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(tiers))
width = 0.25

# Normalise to 0-1 for visual comparison
max_budget = max(budgets)
bars1 = ax.bar(
    [i - width for i in x],
    [b / max_budget for b in budgets],
    width,
    label="Budget (normalised)",
    color="#3498db",
)
bars2 = ax.bar(
    x,
    [c / 3.0 for c in clearance_numeric],
    width,
    label="Clearance level",
    color="#e67e22",
)
bars3 = ax.bar(
    [i + width for i in x],
    [t / max(tool_counts) for t in tool_counts],
    width,
    label="Tool count (normalised)",
    color="#2ecc71",
)

ax.set_xticks(list(x))
ax.set_xticklabels(tiers, fontsize=11)
ax.set_ylabel("Normalised value (0-1)")
ax.set_title("Governance Tier Comparison — Monotonic Escalation", fontweight="bold")
ax.legend(fontsize=9)
ax.set_ylim(0, 1.2)

# Annotate raw values
for i, (b, c, t) in enumerate(zip(budgets, ["int", "conf", "restr"], tool_counts)):
    ax.text(i - width, budgets[i] / max_budget + 0.03, f"${b}", ha="center", fontsize=8)
    ax.text(i, clearance_numeric[i] / 3.0 + 0.03, c, ha="center", fontsize=8)
    ax.text(i + width, t / max(tool_counts) + 0.03, f"{t}", ha="center", fontsize=8)

ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fname = OUTPUT_DIR / "ex8_governance_tiers.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Singapore Financial Advisory (MAS TRM)
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A MAS-regulated wealth management firm operates a
# retail-facing advisory bot (qa tier), an internal portfolio-ops
# dashboard (admin tier), and a compliance audit console (audit tier).
# MAS TRM 7.5 requires full traceability of every model-produced
# recommendation that reaches a retail customer.
#
# BUSINESS IMPACT: One un-logged retail advisory incident triggers a
# MAS investigation costing S$150,000 in external legal fees plus
# potential license suspension. The governed-agent envelope prevents
# the retail tier from invoking portfolio-ops tools (tool restriction)
# and caps retail-facing cost at US$1/request (budget enforcement).
# The audit tier's envelope has the access needed to satisfy a MAS
# Section 27 production order without extending retail or ops
# clearance.

print("\n" + "=" * 70)
print("  APPLY — MAS-Regulated Wealth Advisory")
print("=" * 70)
print(
    """
  qa tier     (retail customers):
    - answer + search only, US$1 budget cap
    - internal clearance (no portfolio PII)
    - fail_mode=closed → no envelope, no response

  admin tier  (portfolio ops):
    - +update_model +view_metrics +monitor_drift
    - US$10 budget; confidential clearance (aggregated positions)

  audit tier  (compliance + MAS liaison):
    - +access_audit_log +generate_report
    - US$50 budget; restricted clearance (individual trades)

  Un-logged advisory incident cost avoided: ~S$150,000
  Tier-jumping blocked structurally (not "hopefully by prompt")
"""
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Compiled a PACT org YAML into a live GovernanceEngine
  [x] Built a shared base agent behind a typed Signature
  [x] Wrapped the base agent in three PactGovernedAgent tiers
  [x] Visualised the monotonic envelope hierarchy
  [x] Applied governed agents to a MAS-regulated scenario

  KEY INSIGHT: Governance is not a layer you bolt onto a working
  agent — it is a WRAPPER that runs before the agent's first token
  is generated. PactGovernedAgent makes the envelope the primary
  interface; the base agent is an implementation detail.

  Next: 03_multichannel_serving.py deploys these tiers via Nexus.
"""
)
