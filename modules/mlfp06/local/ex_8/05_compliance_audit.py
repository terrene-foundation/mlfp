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
# THEORY — Compliance as a read-out
# ════════════════════════════════════════════════════════════════════════
# If governance is wired correctly, compliance is a SELECT against the
# audit trail — not a scramble. The audit tier (`audit`) has the
# clearance to pull the data; the qa and admin tiers structurally
# cannot, which is the correct answer for least-privilege.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild stack and drive sample traffic
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)
governance_engine = GovernanceEngine()


async def _init_engine() -> None:
    governance_engine.compile_org(write_org_yaml())


run_async(_init_engine())

base_qa = CapstoneQAAgent()

# TODO: rebuild the three tiers exactly as in 02_governance_pipeline.py
governed_qa = ____
governed_admin = ____
governed_audit = ____

agents_by_role = {"qa": governed_qa, "admin": governed_admin, "audit": governed_audit}


async def drive_sample_traffic() -> None:
    for q in eval_data["instruction"].to_list()[:3]:
        await handle_qa(q, role="qa", agents_by_role=agents_by_role)
    await handle_qa("Show metrics", role="admin", agents_by_role=agents_by_role)
    await handle_qa("Generate report", role="audit", agents_by_role=agents_by_role)


run_async(drive_sample_traffic())
assert len(agents_by_role) == 3
print("\u2713 Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract per-tier audit trails
# ════════════════════════════════════════════════════════════════════════

# TODO: call .get_audit_trail() on each governed agent
qa_audit = ____
admin_audit = ____
audit_audit = ____

print("Audit trail entry counts:")
print(f"  qa:    {len(qa_audit)}")
print(f"  admin: {len(admin_audit)}")
print(f"  audit: {len(audit_audit)}")
assert qa_audit is not None
print("\u2713 Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the section-by-section compliance report
# ════════════════════════════════════════════════════════════════════════


def generate_compliance_report() -> None:
    print("\n" + "=" * 60)
    print("  COMPLIANCE AUDIT REPORT")
    print("  System: MLFP Capstone Governed ML Platform")
    print("=" * 60)

    print("\n1. AGENT ACTIVITY SUMMARY")
    print(f"   QA agent actions:    {len(qa_audit)}")
    print(f"   Admin agent actions: {len(admin_audit)}")
    print(f"   Audit agent actions: {len(audit_audit)}")

    print("\n2. GOVERNANCE ENFORCEMENT")
    print("   D/T/R chains:         3 active delegations")
    print("   Budget enforcement:   ACTIVE")
    print("   Tool restrictions:    ACTIVE")
    print("   Clearance validation: ACTIVE")
    print("   Audit trail:          COMPLETE")
    print("   Fail mode:            CLOSED")


generate_compliance_report()
print("\n\u2713 Checkpoint 3 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise the regulatory mapping
# ════════════════════════════════════════════════════════════════════════

# TODO: build a polars DataFrame with three columns:
#       Requirement, Technical control, Status
#       Include at minimum these 7 Requirement rows:
#         EU AI Act Art. 9 — Risk Management
#         EU AI Act Art. 12 — Record-keeping
#         EU AI Act Art. 14 — Human Oversight
#         Singapore AI Verify — Accountability
#         Singapore AI Verify — Transparency
#         MAS TRM 7.5 — Audit Trail
#         PDPA — Personal Data Protection
regulatory = ____

regulatory.write_parquet(OUTPUT_DIR / "regulatory_mapping.parquet")
print(regulatory)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: MAS Section 27 Production Order
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A MAS investigation team issues a Section 27 order for
# every model-produced recommendation reaching a specific retail
# customer over a 90-day window. The audit tier can answer this in
# a single get_audit_trail() query. The qa and admin tiers cannot —
# which is the correct least-privilege answer.
#
# Without this audit trail: S$500,000 in forensic consulting or a
# license suspension worth ~S$10M/month of lost revenue.
# With it: ~S$5,000 analyst time, response delivered within 48h.

print("\n" + "=" * 70)
print("  APPLY — MAS Section 27 Production Order")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Extracted per-tier audit trails
  [x] Built a section-by-section compliance report
  [x] Mapped technical controls to EU AI Act / AI Verify / MAS TRM / PDPA
  [x] Applied the report to a MAS Section 27 scenario

  COURSE CAPSTONE COMPLETE. You have built the full stack:
    adapter -> govern -> serve -> monitor -> audit
  on the Kailash platform, end-to-end.
"""
)
