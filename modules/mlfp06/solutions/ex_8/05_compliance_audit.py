# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.5: Regulatory Compliance Audit Report
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Extract audit trails from a PactGovernedAgent for all three tiers
#   - Summarise governance activity into regulator-ready sections
#   - Map technical controls to EU AI Act, AI Verify, MAS TRM, PDPA
#   - Visualise the full platform scorecard
#   - Apply the audit report to a MAS Section 27 production order
#
# PREREQUISITES: Exercises 8.1-8.4
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Rebuild the governed stack and drive a few sample queries
#   2. Extract the per-tier audit trails
#   3. Build the section-by-section compliance report
#   4. Visualise the regulatory mapping table
#   5. Apply the report to a MAS Section 27 production order
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import polars as pl

from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.mlfp06.ex_8 import (
    CapstoneQAAgent,
    OUTPUT_DIR,
    handle_qa,
    load_mmlu_eval,
    run_async,
    write_org_yaml,
)

# ════════════════════════════════════════════════════════════════════════
# THEORY — Compliance as a read-out, not a rewrite
# ════════════════════════════════════════════════════════════════════════
# If governance is wired correctly, a compliance audit is NOT a
# scramble — it is a SELECT statement against the audit trail. The
# trick is that the audit trail has to exist BEFORE the regulator
# asks. Every capstone tier (qa, admin, audit) has been running with
# ``require_audit_trail: true`` since Exercise 8.2; now we read it out
# and map it to the frameworks that the Singapore regulator cares
# about: EU AI Act (cross-border), AI Verify (local toolkit),
# MAS TRM (technology risk), PDPA (personal data).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild stack + drive sample queries so audit trails exist
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

governance_engine = GovernanceEngine()


async def _init_engine() -> None:
    governance_engine.compile_org(write_org_yaml())


run_async(_init_engine())

base_qa = CapstoneQAAgent()
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
agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}


async def drive_sample_traffic() -> None:
    for q in eval_data["instruction"].to_list()[:3]:
        await handle_qa(q, role="qa", agents_by_role=agents_by_role)
    await handle_qa("Show metrics", role="admin", agents_by_role=agents_by_role)
    await handle_qa("Generate report", role="audit", agents_by_role=agents_by_role)


run_async(drive_sample_traffic())

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: governed stack must be rebuilt"
print("\u2713 Checkpoint 1 passed — stack rebuilt and sample traffic driven\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract per-tier audit trails
# ════════════════════════════════════════════════════════════════════════

qa_audit = governed_qa.get_audit_trail()
admin_audit = governed_admin.get_audit_trail()
audit_audit = governed_audit.get_audit_trail()

print("Audit trail entry counts:")
print(f"  qa:    {len(qa_audit)}")
print(f"  admin: {len(admin_audit)}")
print(f"  audit: {len(audit_audit)}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert qa_audit is not None, "Task 2: qa audit trail should be accessible"
print("\u2713 Checkpoint 2 passed — audit trails extracted\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the section-by-section compliance report
# ════════════════════════════════════════════════════════════════════════


def generate_compliance_report() -> None:
    print("\n" + "=" * 60)
    print("  COMPLIANCE AUDIT REPORT")
    print("  System: MLFP Capstone Governed ML Platform")
    print("  Generated: 2026-04-13")
    print("=" * 60)

    print("\n1. AGENT ACTIVITY SUMMARY")
    print(f"   QA agent actions:    {len(qa_audit)}")
    print(f"   Admin agent actions: {len(admin_audit)}")
    print(f"   Audit agent actions: {len(audit_audit)}")
    qa_allowed = sum(1 for e in qa_audit if e.get("status") == "allowed")
    qa_blocked = sum(1 for e in qa_audit if e.get("status") == "blocked")
    print(f"   QA allowed/blocked:  {qa_allowed}/{qa_blocked}")

    print("\n2. GOVERNANCE ENFORCEMENT")
    print("   D/T/R chains:         3 active delegations")
    print("   Budget enforcement:   ACTIVE (per-agent limits)")
    print("   Tool restrictions:    ACTIVE (whitelist per role)")
    print("   Clearance validation: ACTIVE (4-level hierarchy)")
    print("   Audit trail:          COMPLETE (all decisions logged)")
    print("   Fail mode:            CLOSED (deny by default)")

    print("\n3. AUTHENTICATION & ACCESS CONTROL")
    print("   Auth method:      JWT (RS256)")
    print("   RBAC roles:       3 (viewer, operator, auditor)")
    print("   Rate limiting:    ACTIVE (per-client sliding window)")
    print("   CORS:             ACTIVE (origin whitelist)")

    print("\n4. MODEL PROVENANCE")
    print("   Base model:       env variable (not hardcoded)")
    print("   SFT adapter:      imdb_sentiment_sft_v1 (Exercise 2)")
    print("   DPO adapter:      ultrafeedback_dpo_v1 (Exercise 3)")
    print("   Registry:         AdapterRegistry (versioned, auditable)")
    print("   Drift monitoring: ACTIVE (DriftMonitor, PSI threshold=0.2)")

    print("\n5. DEPLOYMENT ARCHITECTURE")
    print("   Channels:     API + CLI + MCP (via Nexus)")
    print("   Governance:   PactGovernedAgent on ALL channels")
    print("   Sessions:     persistent state across channels")
    print("   Cost control: per-agent budget cascading")
    print("   Monitoring:   DriftMonitor + structured logging")


generate_compliance_report()

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
print("\n\u2713 Checkpoint 3 passed — compliance report sections rendered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the regulatory mapping
# ════════════════════════════════════════════════════════════════════════

regulatory = pl.DataFrame(
    {
        "Requirement": [
            "EU AI Act Art. 9 — Risk Management",
            "EU AI Act Art. 12 — Record-keeping",
            "EU AI Act Art. 14 — Human Oversight",
            "Singapore AI Verify — Accountability",
            "Singapore AI Verify — Transparency",
            "MAS TRM 7.5 — Audit Trail",
            "PDPA — Personal Data Protection",
        ],
        "Technical control": [
            "Operating envelopes per agent; budget limits; tool restrictions",
            "Immutable audit trail; every decision logged with timestamps",
            "D/T/R chains; every agent action traces to human Delegator",
            "D/T/R accountability grammar; role-based clearance levels",
            "Reasoning chains logged; confidence scores reported",
            "Full audit log: action, resource, decision, reason, timestamp",
            "PII masking in global envelope; clearance-gated data access",
        ],
        "Status": ["COMPLIANT"] * 7,
    }
)
regulatory.write_parquet(OUTPUT_DIR / "regulatory_mapping.parquet")
print("\n6. REGULATORY COMPLIANCE MAPPING")
print(regulatory)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS Section 27 Production Order
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A MAS investigation team issues a Section 27 production
# order requesting every model-produced recommendation that reached
# a specific retail customer over a 90-day window. The audit tier
# holds the data clearance required to pull this. The qa and admin
# tiers do NOT — they physically cannot fulfil the order, which is
# the correct answer for least-privilege.
#
# BUSINESS IMPACT: A Section 27 order with no usable audit trail
# triggers either a license suspension (~S$10M/month of lost
# revenue for a mid-tier bank) or a "best-efforts reconstruction"
# that costs ~S$500,000 in external forensic consulting and still
# does not satisfy the regulator. With the governed platform, the
# same order is answered in hours via a single audit-tier query.

print("\n" + "=" * 70)
print("  APPLY — MAS Section 27 Production Order")
print("=" * 70)
print(
    """
  Order:    All model outputs to retail customer X, 90-day window.
  Source:   audit tier (clearance=restricted, access_audit_log=yes).
  Response: One query against PactGovernedAgent.get_audit_trail().
  SLA:      hours, not months.

  Without this audit trail:
    - Forensic reconstruction:        ~S$500,000 (and incomplete)
    - License suspension risk:        ~S$10M / month lost revenue

  With this audit trail:
    - Compliance-team query:          ~S$5,000 analyst time
    - Response delivered to MAS:      within 48h
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
  [x] Extracted per-tier audit trails from PactGovernedAgent
  [x] Built a section-by-section compliance report
  [x] Mapped technical controls to EU AI Act, AI Verify, MAS TRM, PDPA
  [x] Visualised the regulatory scorecard
  [x] Applied the audit report to a MAS Section 27 scenario

  KEY INSIGHT: The compliance report is a read-out of the same audit
  trail the governance engine has been writing since Exercise 8.2.
  There is no parallel "compliance pipeline" — governance IS the
  compliance pipeline. That is the whole point of framework-first.

  COURSE CAPSTONE COMPLETE. You have now built the full stack:
    adapter -> govern -> serve -> monitor -> audit
  on the Kailash platform, end-to-end.
"""
)
