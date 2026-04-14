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
#   - Configure D/T/R chains for three roles (qa/admin/audit)
#   - Visualise the governance envelope hierarchy
#   - Apply governed agents to a MAS-regulated advisory scenario
#
# PREREQUISITES: Exercise 8.1, MLFP06 Ex 7 (PACT intro)
# ESTIMATED TIME: ~30 min
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

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
# Delegator / Task / Responsible — every agent action traces back to a
# human Delegator. The operating envelope around each D/T/R chain sets
# budget, allowed tools, and data clearance. Default is fail-closed.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Compile the org YAML into a GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

# TODO: instantiate GovernanceEngine
governance_engine = ____
org_path = write_org_yaml()


async def compile_governance() -> object:
    # TODO: call governance_engine.compile_org(org_path) and return it
    ____


org = run_async(compile_governance())

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert org is not None and org.n_agents >= 3, "Task 1: should compile 3+ agents"
print("\u2713 Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the base agent
# ════════════════════════════════════════════════════════════════════════

# TODO: instantiate CapstoneQAAgent (from shared.mlfp06.ex_8)
base_qa = ____

assert base_qa is not None, "Task 2: base agent should instantiate"
print("\u2713 Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Wrap the agent in three governance tiers
# ════════════════════════════════════════════════════════════════════════
# TODO: Build three PactGovernedAgent instances. For each, pass:
#       agent=base_qa, governance_engine=governance_engine
#   (a) governed_qa    — role="responder",  max_budget_usd=1.0,
#                        allowed_tools=["generate_answer","search_context"],
#                        clearance_level="internal"
#   (b) governed_admin — role="operator",   max_budget_usd=10.0,
#                        allowed_tools=["generate_answer","search_context",
#                                       "update_model","view_metrics",
#                                       "monitor_drift"],
#                        clearance_level="confidential"
#   (c) governed_audit — role="auditor",    max_budget_usd=50.0,
#                        allowed_tools=["generate_answer","search_context",
#                                       "view_metrics","access_audit_log",
#                                       "generate_report"],
#                        clearance_level="restricted"
governed_qa = ____
governed_admin = ____
governed_audit = ____

agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 3: should wrap three tiers"
print("\u2713 Checkpoint 3 passed\n")


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
print(envelope_table)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS-regulated wealth advisory
# ════════════════════════════════════════════════════════════════════════
# One un-logged retail advisory incident triggers a MAS investigation
# costing ~S$150,000 in external legal fees. The envelope prevents
# retail-tier tool-jumping structurally, not "hopefully by prompt".

print("\n" + "=" * 70)
print("  APPLY — MAS Wealth Advisory (tiered envelopes)")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Compiled a PACT org YAML into a GovernanceEngine
  [x] Wrapped the base agent in three PactGovernedAgent tiers
  [x] Visualised the monotonic envelope hierarchy
  [x] Applied governed agents to a MAS-regulated scenario

  Next: 03_multichannel_serving.py deploys these tiers via Nexus.
"""
)
