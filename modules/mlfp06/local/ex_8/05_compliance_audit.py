# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 8.5: Regulatory Compliance Audit Report
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Extract audit trails from a GovernedSupervisor for all three tiers
#   - Verify the hash-chained audit trail for tamper-evidence
#   - Generate every line of a regulator-facing report from live objects
#     (audit records, envelopes, registry, drift output) — and say
#     "not evidenced" where this run produced no evidence
#   - Map the evidence to EU AI Act, AI Verify, MAS TRM, PDPA
#   - Visualise which controls have evidence and which are gaps
#   - Apply the audit report to a regulator's production order
#
# PREREQUISITES: Exercises 8.1-8.4; Ollama running (`ollama serve`)
# ESTIMATED TIME: ~25 min
#
# TASKS:
#   1. Rebuild the governed stack and drive sample queries (allow + deny)
#   2. Extract the per-tier audit trails via ``supervisor.audit.to_list()``
#      and verify tamper-evidence via ``supervisor.audit.verify_chain()``
#   3. Build the section-by-section compliance report from live objects
#   4. Map the evidence to six regulatory requirements
#   5. Apply the report to a regulator's production order
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, preflight_ollama
from shared.mlfp06.ex_8 import (
    OUTPUT_DIR,
    build_capstone_stack,
    compile_capstone_governance,
    discover_trained_adapters,
    handle_qa,
    load_mmlu_eval,
    run_async,
)

# Task 1 makes real LLM calls; fail loudly now if Ollama is not running.
preflight_ollama(required_models=[DEFAULT_CHAT_MODEL])

# ════════════════════════════════════════════════════════════════════════
# THEORY — Compliance as a read-out, not a rewrite
# ════════════════════════════════════════════════════════════════════════
# If governance is wired correctly, a compliance audit is NOT a scramble
# — it is a query against evidence that already exists. Every capstone
# tier (qa, admin, audit) runs on a GovernedSupervisor, and each
# supervisor's ``audit`` attribute records every run into a hash-chained
# list:
#
#   supervisor.audit.to_list()        → the records themselves
#   supervisor.audit.verify_chain()   → structural tamper-evidence
#
# The discipline this exercise teaches: a report line is only as good as
# the object it was read from. Anything this run did not produce
# evidence for is printed as NOT EVIDENCED — never as ACTIVE or
# COMPLIANT. Evidence is not a legal ruling; a regulator decides that.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Rebuild stack + drive sample queries so audit trails exist
# ════════════════════════════════════════════════════════════════════════

eval_data = load_mmlu_eval(n_rows=100)

# TODO: Compile the capstone org (applies envelopes), then build the stack.
governance_engine, loaded = ____
agents_by_role, tiers = ____
tier_by_role = {t.role: t for t in tiers}

print("Governed stack rebuilt:")
for tier in tiers:
    print(
        f"  {tier.role:6s} -> budget=${tier.budget_usd:>5.1f}  "
        f"clearance={tier.clearance}"
    )

TRAFFIC = [
    *[(q, "qa", "generate_answer") for q in eval_data["instruction"].to_list()[:3]],
    ("Show model performance metrics", "admin", "view_metrics"),
    ("Generate quarterly compliance report", "audit", "generate_report"),
    ("Retrain the model on today's data", "qa", "update_model"),  # deny
    ("Export the full audit log", "qa", "access_audit_log"),  # deny
]


async def drive_sample_traffic() -> list[dict]:
    """Drive allow AND deny requests; log every refusal on the tier's trail."""
    outcomes = []
    for question, role, action in TRAFFIC:
        # TODO: Route through handle_qa with the engine and this action.
        r = ____
        if r["blocked"]:
            # A refusal that is not in the audit trail did not happen, as
            # far as an auditor can tell — record it.
            # TODO: record the refused action on this tier's audit trail.
            # Hint: agents_by_role[role].record_tool_use(..., blocked=True, reason=...)
            ____
        outcomes.append({"role": role, "action": action, "verdict": r["verdict"]})
    return outcomes


outcomes = pl.DataFrame(run_async(drive_sample_traffic()))
print(outcomes)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert len(agents_by_role) == 3, "Task 1: governed stack must be rebuilt"
assert (outcomes["verdict"] == "served").sum() == 5, "Task 1: 5 allowed requests served"
assert (outcomes["verdict"] == "blocked").sum() == 2, "Task 1: 2 deny cases blocked"
print("✓ Checkpoint 1 passed — stack rebuilt and sample traffic driven\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Extract per-tier audit trails + verify tamper-evidence
# ════════════════════════════════════════════════════════════════════════
#
# Each record references the prior record by hash — silently editing any
# record invalidates the chain. ``verify_chain()`` returns True only if
# the chain has not been tampered with.

# TODO: For every tier, read the audit records and verify the hash chain.
trails = ____
chains_valid = ____

print("Audit trail record counts + tamper-evidence:")
for role, records in trails.items():
    print(f"  {role:<6} {len(records):>3} records  chain_valid={chains_valid[role]}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert all(len(r) > 0 for r in trails.values()), "Task 2: every tier has records"
assert all(chains_valid.values()), "Task 2: every audit chain should verify"
print("✓ Checkpoint 2 passed — audit trails extracted and verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the section-by-section compliance report (live objects)
# ════════════════════════════════════════════════════════════════════════


def summarise_trail(records: list[dict]) -> dict[str, int]:
    """Count audit records by what they record (record_type / action)."""
    counts = Counter()
    for rec in records:
        action = rec["action"]
        if action.startswith("node_completed"):
            counts["completed"] += 1
        elif action.startswith("node_failed"):
            counts["failed"] += 1
        elif action.startswith("tool_blocked"):
            counts["refused"] += 1
        elif rec["record_type"] == "held":
            counts["held"] += 1
    return {k: counts.get(k, 0) for k in ("completed", "failed", "held", "refused")}


activity = {role: summarise_trail(records) for role, records in trails.items()}

# Deny-path probe per tier: an action the tier does not hold must be blocked.
all_tools = sorted({tool for t in tiers for tool in t.tools})
deny_probe = {}
for t in tiers:
    missing = next(a for a in all_tools + ["delete_all_records"] if a not in t.tools)
    # TODO: verify `missing` for this tier's address; keep the verdict level.
    deny_probe[t.role] = ____
# TODO: the same for the ML Director (D1-R1), who has no envelope.
director_level = ____

adapters_on_disk = discover_trained_adapters()
psi_path = OUTPUT_DIR / "psi_dashboard.parquet"
psi_df = pl.read_parquet(psi_path) if psi_path.exists() else None
generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
NOT_EVIDENCED = "NOT EVIDENCED in this run"


def generate_compliance_report() -> None:
    """Produce a regulator-facing read-out — every value read from an object."""
    print("\n" + "=" * 60)
    print("  COMPLIANCE AUDIT REPORT")
    print("  System: MLFP Capstone Governed ML Platform")
    print(f"  Generated: {generated_at}")
    print("=" * 60)

    print("\n1. AGENT ACTIVITY (from each tier's audit trail)")
    for role, c in activity.items():
        print(
            f"   {role:<6} completed={c['completed']}  failed={c['failed']}  "
            f"held={c['held']}  refused={c['refused']}"
        )

    print("\n2. GOVERNANCE ENFORCEMENT")
    print(f"   Envelopes applied from YAML: {len(loaded.envelopes)} "
          f"(then replaced by {len(tiers)} full tier envelopes)")
    for t in tiers:
        env = agents_by_role[t.role].envelope
        print(
            f"   {t.role:<6} clearance={env.confidentiality_clearance.value:<12} "
            f"budget=${env.financial.max_spend_usd:g}  deny-probe={deny_probe[t.role]}"
        )
    print(f"   ML Director (no envelope): {director_level}  <- open gap")
    print(f"   Audit chains verified: {chains_valid}")

    print("\n3. AUTHENTICATION & ACCESS CONTROL")
    print(f"   JWT / RBAC / rate limit / CORS: {NOT_EVIDENCED}")
    print("   (they are configured and exercised in 03_multichannel_serving.py)")

    print("\n4. MODEL PROVENANCE")
    print(f"   Served chat model (OLLAMA_CHAT_MODEL): {DEFAULT_CHAT_MODEL}")
    if adapters_on_disk:
        for a in adapters_on_disk:
            print(f"   Adapter on disk: {a['adapter_name']} ({a['method']}, base {a['base_model_id']})")
    else:
        print(f"   Fine-tuned adapters: {NOT_EVIDENCED} (none found on disk)")
    if psi_df is not None:
        alerts = psi_df.filter(pl.col("zone") == "Alert").height
        print(f"   Drift monitoring: {psi_df.height} windows checked, {alerts} in Alert (Ex 8.4)")
    else:
        print(f"   Drift monitoring: {NOT_EVIDENCED} (run 04_drift_monitoring.py)")

    print("\n5. DATA PROTECTION")
    print(f"   PII masking: {NOT_EVIDENCED} (no masking step exists in this platform)")
    print(f"   qa tier clearance: {tier_by_role['qa'].clearance}")


generate_compliance_report()

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert sum(c["completed"] for c in activity.values()) == 5, "Task 3: 5 served runs logged"
assert activity["qa"]["refused"] == 2, "Task 3: both qa refusals logged"
assert all(level == "blocked" for level in deny_probe.values())
print("\n✓ Checkpoint 3 passed — compliance report generated from live objects\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Map the evidence to regulatory requirements
# ════════════════════════════════════════════════════════════════════════

regulatory = pl.DataFrame(
    {
        "Requirement": [
            "EU AI Act Art. 9 — Risk management",
            "EU AI Act Art. 12 — Record-keeping",
            "EU AI Act Art. 14 — Human oversight",
            "Singapore AI Verify — Accountability",
            "MAS TRM Guidelines — Audit trail",
            "PDPA — Personal data protection",
        ],
        "Evidence in this run": [
            f"deny probes: {sorted(set(deny_probe.values()))}",
            f"{sum(len(r) for r in trails.values())} hash-chained records",
            f"{len(loaded.envelopes)} envelopes defined by the ML Director",
            f"envelope-less head: {director_level}",
            f"all chains verify: {all(chains_valid.values())}",
            "no PII masking implemented",
        ],
        "Evidence present": [
            all(level == "blocked" for level in deny_probe.values()),
            sum(len(r) for r in trails.values()) > 0,
            len(loaded.envelopes) == len(tiers),
            director_level == "blocked",  # False today: the head is unbounded
            all(chains_valid.values()),
            False,
        ],
    }
)
regulatory.write_parquet(OUTPUT_DIR / "regulatory_mapping.parquet")
print("\n6. REGULATORY EVIDENCE MAPPING (evidence, not a compliance ruling)")
print(regulatory)
gaps = regulatory.filter(~pl.col("Evidence present"))["Requirement"].to_list()
print(f"\n   Gaps to close before claiming these controls: {gaps}")

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert regulatory.height == 6, "Task 4: regulatory mapping must have 6 rows"
assert regulatory["Evidence present"][1] and regulatory["Evidence present"][4], (
    "Task 4: record-keeping and audit-trail evidence must be present"
)
print("\n✓ Checkpoint 4 passed — regulatory evidence mapped, gaps listed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: A Regulator's Production Order
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A financial regulator orders a Singapore bank to produce every
# model-produced recommendation that reached one retail customer over a
# 90-day window. The audit tier holds the clearance and the
# access_audit_log action needed to pull this; the qa tier does not — as
# the refused request in Task 1 shows — which is the correct answer for
# least-privilege.
#
# BUSINESS IMPACT (illustrative figures): without a usable audit trail
# the bank faces a "best-efforts reconstruction" that might cost
# ~S$500,000 in external forensic work and still not satisfy the
# regulator. With the governed platform the same order is a query over
# ``supervisor.audit.to_list()`` plus ``verify_chain()``.

print("\n" + "=" * 70)
print("  APPLY — Regulator's Production Order")
print("=" * 70)
audit_tier = tier_by_role["audit"]
print(
    f"""
  Order:    All model outputs to retail customer X, 90-day window.
  Source:   audit tier (clearance={audit_tier.clearance}, \
access_audit_log={'access_audit_log' in audit_tier.tools}).
  qa tier asking for the audit log in this run: \
{outcomes.filter(pl.col('action') == 'access_audit_log')['verdict'][0]}
  Response: one query against supervisor.audit.to_list() + verify_chain().

  Illustrative cost without a trail: ~S$500,000 forensic reconstruction.
"""
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
assert chains_valid["audit"], "Task 5: audit-tier chain must verify for evidence"
print("✓ Checkpoint 5 passed — audit-tier evidence ready for the regulator\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISATION — Evidence per requirement (from Task 4, not hand-coded)
# ════════════════════════════════════════════════════════════════════════

regs = [r.replace(" — ", "\n") for r in regulatory["Requirement"].to_list()]
statuses = ["evidence" if ok else "gap" for ok in regulatory["Evidence present"]]
color_map = {"evidence": "#4CAF50", "gap": "#F44336"}
colors = [color_map[s] for s in statuses]

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(regs, [1] * len(regs), color=colors, edgecolor="#333", linewidth=0.5)
for i, status in enumerate(statuses):
    ax.text(0.5, i, status.upper(), ha="center", va="center", fontweight="bold", color="white")
ax.set_xlim(0, 1)
ax.set_xticks([])
ax.invert_yaxis()
ax.set_title("Evidence Present in This Run, per Requirement")
ax.legend(
    handles=[mpatches.Patch(color=c, label=l.upper()) for l, c in color_map.items()],
    loc="lower right",
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_compliance_traffic_light.png", dpi=150)
plt.close(fig)
print(f"\nSaved: {OUTPUT_DIR / '05_compliance_traffic_light.png'}")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens over the qa tier's audit trail
# ══════════════════════════════════════════════════════════════════
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=agents_by_role["qa"].audit, run_id="ex_8_5_compliance")
print("\n── LLM Observatory: qa-tier audit snapshot ──")
print(obs.governance.audit_snapshot(last_n=20).select("action", "verdict", "reason"))
print(obs.governance.report())
print(f"  (Kaizen hash chain verified with audit.verify_chain(): {chains_valid['qa']})")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Extracted per-tier audit trails from GovernedSupervisor
  [x] Verified hash-chained tamper-evidence via audit.verify_chain()
  [x] Logged refusals on the audit trail, not just successes
  [x] Built every report line from a live object — or marked it NOT EVIDENCED
  [x] Mapped evidence to EU AI Act, AI Verify, MAS TRM and PDPA, gaps included
  [x] Applied the audit report to a regulator's production order

  KEY INSIGHT: The compliance report is a read-out of evidence the
  platform already produced. A line you cannot read from an object is a
  claim, not evidence — print it as a gap and go close it.

  COURSE CAPSTONE COMPLETE. You have now built the full stack:
    adapter -> govern -> serve -> monitor -> audit
  on the Kailash platform, end-to-end.
"""
)
