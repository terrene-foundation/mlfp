# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.1: Organisation Definition & GovernanceEngine Compile
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Write a D/T/R (Delegator/Task/Responsible) organisation in YAML
#   - Compile that organisation with kailash-pact's GovernanceEngine
#   - Understand what compilation validates (chains, cycles, monotonic
#     clearance, budget bounds) — and what it does NOT (LLM safety)
#   - Map the abstract grammar to a real Singapore FinTech scenario
#
# PREREQUISITES: Exercise 6 (multi-agent systems)
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Load real adversarial prompts (used by later techniques)
#   2. Write the SG FinTech org YAML to disk
#   3. Compile with GovernanceEngine and inspect the result
#   4. Visualise the D/T/R chains as a table
#   5. Apply — why a MAS-regulated bank needs compiled governance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from shared.mlfp06.ex_7 import (
    ORG_YAML,
    compile_governance,
    load_adversarial_prompts,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — D/T/R Accountability Grammar
# ════════════════════════════════════════════════════════════════════════
# Every autonomous action an agent takes in production MUST trace back
# to a human Delegator who authorised it. That is the D/T/R grammar:
#
#   D (Delegator):   A named human authority (chief_ml_officer,
#                    chief_risk_officer, vp_customer, ...).
#   T (Task):        A bounded scope of work the agent may perform
#                    (data_analysis, model_training, bias_audit, ...).
#   R (Responsible): The agent that executes within the envelope the
#                    delegator attached to the task.
#
# Analogy: A bank manager (D) tells a teller (R) "handle customer
# deposits up to S$10,000" (T with an envelope). If the teller approves
# a S$50,000 transfer, accountability traces back to the manager who
# set the envelope too wide — not to a nameless "the system".
#
# WHY THIS MATTERS: MAS TRM 7.5 and EU AI Act Art. 14 both require
# human oversight of AI systems. "The agent did it" is never an
# acceptable audit answer. D/T/R makes accountability structural.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load Adversarial Test Prompts
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: Load Adversarial Test Prompts")
print("=" * 70)

adversarial_prompts = load_adversarial_prompts(n=50)
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

org_yaml_path = write_org_yaml()

print(f"Organisation: SG FinTech AI Division")
print(f"Departments:  3 (ML Engineering, Risk & Compliance, Customer Intelligence)")
print(f"Agents:       6")
print(f"Delegations:  6 D/T/R chains")
print(f"YAML path:    {org_yaml_path}")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert "departments" in ORG_YAML and "delegations" in ORG_YAML
assert org_yaml_path  # path was returned
print("[x] Checkpoint 2 passed — org YAML written\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Compile with GovernanceEngine
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: GovernanceEngine.compile_org()")
print("=" * 70)

engine, org = compile_governance(org_yaml_path)

print(f"Compiled organisation:")
print(f"  Agents:      {org.n_agents}")
print(f"  Delegations: {org.n_delegations}")
print(f"  Departments: {org.n_departments}")
print(
    """
Compilation validates:
  - Every agent has a delegation chain to a human Delegator
  - No circular delegations (A -> B -> A)
  - Clearance levels decrease monotonically down chains
  - Budget envelopes don't exceed parent limits
What compilation does NOT validate:
  - Content safety of LLM outputs (that needs adversarial testing)
  - Runtime budget consumption (that needs the runtime wrapper)
"""
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert org is not None, "Task 3: compilation should succeed"
assert org.n_agents > 0, "Task 3: org should have agents"
assert org.n_delegations > 0, "Task 3: org should have delegations"
print(
    f"[x] Checkpoint 3 passed — compiled {org.n_agents} agents, "
    f"{org.n_delegations} delegations\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the D/T/R Chains
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Visualise D/T/R Chains")
print("=" * 70)

dtr_chains = pl.DataFrame(
    {
        "Delegator": [
            "chief_ml_officer",
            "chief_ml_officer",
            "chief_ml_officer",
            "chief_risk_officer",
            "chief_risk_officer",
            "vp_customer",
        ],
        "Task": [
            "data_analysis",
            "model_training",
            "model_deployment",
            "risk_assessment",
            "bias_audit",
            "customer_interaction",
        ],
        "Responsible": [
            "data_analyst",
            "model_trainer",
            "model_deployer",
            "risk_assessor",
            "bias_checker",
            "customer_agent",
        ],
        "Budget": ["$20", "$100", "$50", "$200", "$75", "$5"],
        "Clearance": [
            "internal",
            "confidential",
            "confidential",
            "restricted",
            "confidential",
            "public",
        ],
    }
)
print(dtr_chains)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS-Regulated Bank
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore retail bank deploys an AI agent platform for
# customer service, fraud triage, and risk reporting. MAS TRM 7.5
# mandates an auditable trail for every automated decision that
# affects a customer. The compliance team asks one question during the
# annual audit: "For every AI-taken action this year, show me which
# human authorised that class of action."
#
# Without compiled D/T/R governance, the bank's answer is "well, the
# ML team wrote the agents, so... them, I guess?" — which fails the
# audit because no specific human is accountable for any specific
# action. With GovernanceEngine.compile_org() running on boot, every
# agent action carries a delegation chain. The auditor's question is
# answered by a 1-line query against the audit trail.
#
# BUSINESS IMPACT: A failed MAS TRM audit triggers remediation orders,
# capital add-ons, and in severe cases, a pause on the affected
# business line. Singapore banks regularly face S$500K–S$5M in
# remediation costs per finding. One compiled org YAML eliminates an
# entire class of findings.

print("\n" + "=" * 70)
print("  KEY TAKEAWAY: Compilation Is the Structural Audit Gate")
print("=" * 70)
print(
    f"  {org.n_agents} agents, {org.n_delegations} delegations, "
    f"all traced to named humans."
)
print("  The compilation step is the difference between 'we have")
print("  governance' and 'we can prove governance ran at boot'.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Wrote a D/T/R organisation definition in YAML
  [x] Compiled it with GovernanceEngine.compile_org()
  [x] Understood what compilation validates (and what it doesn't)
  [x] Mapped the grammar to a Singapore retail bank audit scenario

  KEY INSIGHT: Governance is engineering, not philosophy. If your
  governance story cannot be compiled and validated at boot, it is
  a slide deck, not a control.

  Next: 02_envelopes.py adds operating envelopes + monotonic
  tightening on top of the compiled org.
"""
)
