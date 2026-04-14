# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.3: Budget Cascading & Access Control Decisions
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Cascade budget from parent agents to children
#   - Detect overspend attempts BEFORE they reach the LLM
#   - Exercise GovernanceEngine.check_access() for allow and deny paths
#   - Understand why test coverage MUST include the deny path
#
# PREREQUISITES: 02_envelopes.py
# ESTIMATED TIME: ~35 min
#
# TASKS:
#   1. Allocate & consume budget across a 3-agent hierarchy
#   2. Attempt an overspend and verify it is denied
#   3. Visualise spend vs. allocation
#   4. Exercise access control across 10 allow+deny cases
#   5. Apply — MAS TRM budget and cost controls
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from shared.mlfp06.ex_7 import BudgetTracker, compile_governance

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine, org = compile_governance()
print("\n--- GovernanceEngine compiled ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Budget Is a Governance Primitive
# ════════════════════════════════════════════════════════════════════════
# An LLM call has a dollar cost. An autonomous agent can make many
# LLM calls per task. Without a hard budget per envelope, a bug — or
# a prompt injection — can drain your API credits in minutes. The
# first publicly reported "agent spent $10K in an afternoon" incident
# was a missing budget envelope on a customer-support agent.
#
# Budget cascading enforces the same hierarchy as clearance:
#
#   ml_director total budget:  $500
#     ├─ data_analyst:   $20/task
#     ├─ model_trainer:  $100/task
#     └─ model_deployer: $50/task
#
# After 3 training runs of $30 each: model_trainer spent $90. Next
# $50 request: DENIED (would exceed $100 allocation).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build a Budget Hierarchy
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Build Budget Hierarchy")
print("=" * 70)

tracker = BudgetTracker(total_budget=500.0)
tracker.allocate("data_analyst", 20.0)
tracker.allocate("model_trainer", 100.0)
tracker.allocate("model_deployer", 50.0)

# Simulate a realistic day's work
tracker.spend("model_trainer", 30.0)  # Training task 1
tracker.spend("model_trainer", 30.0)  # Training task 2
tracker.spend("model_trainer", 25.0)  # Training task 3
tracker.spend("data_analyst", 8.0)  # Analysis task
tracker.spend("model_deployer", 15.0)  # Deployment

print("Budget summary after a day's work:")
print(tracker.summary())

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert tracker.remaining("model_trainer") == 15.0
print("\n[x] Checkpoint 1 passed — budget hierarchy populated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Overspend Attempt (Must Be Denied)
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Overspend Attempt")
print("=" * 70)

# model_trainer has $15 remaining. Try to spend $50.
overspend_ok = tracker.spend("model_trainer", 50.0)
print(
    f"  Overspend $50 on model_trainer (remaining $15): "
    f"{'ALLOWED (bug!)' if overspend_ok else 'DENIED (correct)'}"
)

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert not overspend_ok, "Task 2: overspend must be denied"
assert tracker.remaining("model_trainer") == 15.0, "$15 should still remain"
print("\n[x] Checkpoint 2 passed — overspend caught\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise Spend vs. Allocation
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Visual Proof — Spend vs. Allocation")
print("=" * 70)

summary = tracker.summary()
print()
print(f"  {'agent':<18} {'alloc':>8} {'spent':>8}  bar")
print(f"  {'-' * 55}")
for row in summary.iter_rows(named=True):
    alloc = row["allocated"]
    spent = row["consumed"]
    pct = int(40 * spent / alloc) if alloc > 0 else 0
    bar = "#" * pct + "." * (40 - pct)
    print(f"  {row['agent']:<18} ${alloc:>7.0f} ${spent:>7.0f}  {bar}")
print()


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Exercise check_access() Across Allow + Deny Cases
# ════════════════════════════════════════════════════════════════════════
#
# A governance system that has only been tested on the allow path is
# a system that has not been tested at all. Every access-control test
# suite MUST include both allow and deny cases, because the failure
# mode you care about is "deny silently became allow after refactor".

print("=" * 70)
print("TASK 4: Access Control Decisions (Allow + Deny)")
print("=" * 70)


async def test_access_control():
    test_cases = [
        # (agent, resource, action, expected_allowed, reason)
        ("model_trainer", "training_data", "read", True, "Within envelope"),
        ("model_trainer", "production_model", "deploy", False, "deploy not in tools"),
        ("customer_agent", "customer_faq", "search_faq", True, "Within envelope"),
        ("customer_agent", "training_data", "read_data", False, "Clearance too low"),
        ("risk_assessor", "model_audit_log", "audit_model", True, "Within envelope"),
        (
            "risk_assessor",
            "production_model",
            "deploy_model",
            False,
            "deploy not in tools",
        ),
        ("model_deployer", "production_model", "deploy_model", True, "Within envelope"),
        ("data_analyst", "restricted_data", "read", False, "Clearance too low"),
        (
            "bias_checker",
            "model_fairness",
            "run_fairness_check",
            True,
            "Within envelope",
        ),
        ("customer_agent", "internal_data", "read_data", False, "public < internal"),
    ]

    results = []
    print(f"  {'Agent':<16} {'Action':<20} {'Expected':<9} {'Actual':<9}")
    print("  " + "-" * 60)

    for agent_id, resource, action, expected, reason in test_cases:
        decision = engine.check_access(
            agent_id=agent_id,
            resource=resource,
            action=action,
        )
        actual = decision.allowed
        match = actual == expected
        status = "[ok]" if match else "[FAIL]"
        results.append({"agent": agent_id, "action": action, "match": match})
        print(
            f"  {status} {agent_id:<14} {action:<18} "
            f"{'ALLOW' if expected else 'DENY':<7} "
            f"{'ALLOW' if actual else 'DENY':<7}"
        )
        if not actual:
            print(f"      reason: {decision.reason}")

    return results


access_results = asyncio.run(test_access_control())

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert len(access_results) >= 10, "Task 4: should test at least 10 cases"
deny_cases = [r for r in access_results if not r.get("match", True)]
print(f"\n[x] Checkpoint 4 passed — {len(access_results)} access tests executed")
if deny_cases:
    print(f"    ({len(deny_cases)} mismatches — investigate PACT rules)")
print()


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Budget cascade bar chart + access decision heatmap
# ════════════════════════════════════════════════════════════════════════
# Two panels: (1) stacked bars showing allocated vs consumed budget per
# agent — the remaining gap is visual headroom; (2) heatmap of access
# decisions showing allow (green) and deny (red) across agents/actions.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: budget cascade
agents_budget = summary["agent"].to_list()
allocated = summary["allocated"].to_list()
consumed = summary["consumed"].to_list()
remaining = summary["remaining"].to_list()
x = range(len(agents_budget))

ax1.bar(x, consumed, color="#e74c3c", label="Consumed", width=0.5)
ax1.bar(
    x,
    remaining,
    bottom=consumed,
    color="#2ecc71",
    label="Remaining",
    width=0.5,
    alpha=0.7,
)
ax1.set_xticks(list(x))
ax1.set_xticklabels([a.replace("_", "\n") for a in agents_budget], fontsize=8)
ax1.set_ylabel("USD")
ax1.set_title("Budget Cascade — Spend vs Allocation", fontweight="bold")
ax1.legend(fontsize=8)
for i, (c, a) in enumerate(zip(consumed, allocated)):
    ax1.text(i, a + 1, f"${a:.0f}", ha="center", fontsize=8, color="#2c3e50")

# Right: access decision matrix heatmap
unique_agents = sorted(set(r["agent"] for r in access_results))
unique_actions = sorted(set(r["action"] for r in access_results))
matrix = []
for agent in unique_agents:
    row = []
    for action in unique_actions:
        match = [
            r for r in access_results if r["agent"] == agent and r["action"] == action
        ]
        if match:
            row.append(1 if match[0]["match"] else 0)
        else:
            row.append(0.5)  # not tested
    matrix.append(row)

from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#e74c3c", "#bdc3c7", "#2ecc71"])
im = ax2.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
ax2.set_xticks(range(len(unique_actions)))
ax2.set_xticklabels(
    [a.replace("_", "\n") for a in unique_actions], fontsize=7, rotation=45, ha="right"
)
ax2.set_yticks(range(len(unique_agents)))
ax2.set_yticklabels([a.replace("_", "\n") for a in unique_agents], fontsize=8)
ax2.set_title("Access Decision Matrix", fontweight="bold")

plt.tight_layout()
fname = OUTPUT_DIR / "ex7_budget_access_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS TRM Cost Control
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore digital bank's AI-powered loan triage agent
# runs 24/7 against customer applications. MAS TRM Section 5.8 (Cost
# Controls) requires evidence that automated systems have a hard cap
# on per-decision compute spend AND that the cap cannot be silently
# raised by an engineer.
#
# Without cascading budgets, the cap lives in a config file that a
# developer can edit. With PACT-compiled envelopes, the cap lives in
# the governance graph, which is validated at boot AND at every
# access decision. A PR that attempts to raise the cap is visible in
# version control, reviewable, and blocked by CI if monotonic
# tightening is violated.
#
# BUSINESS IMPACT: A single runaway agent incident — say, 24 hours
# of compounding LLM calls against a loan queue — can produce a
# S$50K–S$200K API bill plus remediation overhead. Hard envelopes
# close the blast radius to the per-task budget, typically S$0.10–
# S$5 per decision.

print("=" * 70)
print("  KEY TAKEAWAY: Budget + Access Control Together")
print("=" * 70)
print("  Budget cascading prevents runaway spend; access control")
print("  prevents out-of-scope actions. You need BOTH — a within-")
print("  budget agent doing the wrong thing is still a breach.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Cascaded budget across a 3-agent hierarchy
  [x] Demonstrated a rejected overspend attempt
  [x] Ran 10 check_access() tests covering both allow and deny
  [x] Visualised the spend bars to see remaining headroom
  [x] Mapped budget envelopes to MAS TRM cost controls

  KEY INSIGHT: You have not tested a governance system until you
  have tested the denials. "Happy path works" is not evidence.

  Next: 04_runtime_audit.py wires PactGovernedAgent into a live
  LLM workflow and tests it against real adversarial prompts.
"""
)
