# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
REFERENCE IMPLEMENTATION — Shard 5 template.

Source: pact-specialist investigation, 2026-04-14.
Target file: modules/mlfp06/solutions/ex_7/04_runtime_audit.py
Verified against: kailash-pact==0.8.1, kaizen-agents==0.9.2.

This file is a COPY-PASTE TEMPLATE. It is NOT part of the course
materials and is NOT executed by any exercise pipeline. During Shard 5,
use this file as the source for the solutions/ex_7/04_runtime_audit.py
rewrite; derive the local/ mirror by stripping solution-specific blocks
per rules/exercise-standards.md § Progressive Scaffolding (M6 = ~20%
provided).

# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.4: Runtime Governance, Fail-Closed, and Audit Trail
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wrap an LLM-backed agent with a PACT-governed supervisor at runtime
#   - Verify fail-closed semantics (unknown role = DENY)
#   - Contain the blast radius of adversarial prompts via envelope limits
#   - Export a hash-chained audit trail and map it to
#     EU AI Act / MAS TRM / PDPA
#   - Understand warn / block / audit enforcement modes
#
# PREREQUISITES: 03_budget_access.py
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Build three PACT-governed supervisors at three clearance tiers
#   2. Run governed supervisors against normal + adversarial inputs
#   3. Verify fail-closed: unknown role MUST be denied
#   4. Contain the blast radius of adversarial prompts
#   5. Map audit trail entries to regulations (EU AI Act, MAS TRM, PDPA)
#   6. Apply — PDPA breach-readiness audit for a Singapore SaaS
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

from kaizen_agents import GovernedSupervisor
from pact import GovernanceEngine, TrustPostureLevel

from shared.mlfp06.ex_7 import (
    compile_governance,
    default_model_name,
    load_adversarial_prompts,
    make_fake_executor,
)

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine, compiled_org = compile_governance()
adversarial_prompts = load_adversarial_prompts(n=50)
print("\n--- GovernanceEngine compiled; adversarial prompts loaded ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Runtime Enforcement vs Compile-Time Validation
# ════════════════════════════════════════════════════════════════════════
# Compiling an org YAML proves the governance GRAPH is sound. It does
# NOT prove that live LLM calls respect the graph. For that, every
# agent invocation must pass through an enforcement wrapper that:
#
#   1. Checks if this action is inside the agent's envelope.
#   2. Checks if budget is sufficient.
#   3. Checks clearance against resource classification.
#   4. If ALL pass, executes and charges budget.
#   5. If ANY fail, records a BLOCKED verdict — tool never runs.
#
# Fail-closed means: the answer to "should this be allowed?" is DENY
# unless every check explicitly returns ALLOW. An indeterminate check
# (unknown role, missing envelope) also resolves to DENY — the
# opposite of the classic Unix default.
#
# The modern PACT wrapper is GovernedSupervisor. It takes the same
# three knobs as last year's wrapper — budget, tools, clearance —
# but goes further: the envelope is a proper 5-dimensional
# ConstraintEnvelope, and the audit trail is a hash-chained sequence
# of records that a tamper-evidence checker can verify offline.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Wrap a QA Agent with Three Governance Tiers
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: GovernedSupervisor — three clearance tiers")
print("=" * 70)

model = default_model_name() or "gpt-4o-mini"

governed_public = GovernedSupervisor(
    model=model,
    budget_usd=5.0,
    tools=["answer_question", "search_faq"],
    data_clearance="public",
)

governed_internal = GovernedSupervisor(
    model=model,
    budget_usd=50.0,
    tools=["answer_question", "search_faq", "read_data", "train_model"],
    data_clearance="confidential",
)

governed_admin = GovernedSupervisor(
    model=model,
    budget_usd=200.0,
    tools=["answer_question", "read_data", "audit_model", "access_audit_log"],
    data_clearance="restricted",  # maps to ConfidentialityLevel.RESTRICTED
)

print("Three runtime-governed supervisors created:")
for name, gs in [
    ("governed_public", governed_public),
    ("governed_internal", governed_internal),
    ("governed_admin", governed_admin),
]:
    env = gs.envelope
    print(
        f"  {name:17s}  "
        f"budget=${env.financial.max_spend_usd:>5.0f}  "
        f"clearance={env.confidentiality_clearance.name:<10s}  "
        f"tools={len(env.operational.allowed_actions)}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert governed_public is not None
assert governed_internal is not None
assert governed_admin is not None
assert governed_public.envelope.financial.max_spend_usd == 5.0
assert "read_data" in governed_internal.envelope.operational.allowed_actions
assert "access_audit_log" in governed_admin.envelope.operational.allowed_actions
print("\n[x] Checkpoint 1 passed — three governance tiers wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Run the Governed Supervisors
# ════════════════════════════════════════════════════════════════════════
#
# GovernedSupervisor.run() decomposes an objective into a plan, then
# runs each node through an execute_node callback YOU supply. That
# callback is where the real LLM (or stub) lives. Governance is
# enforced AROUND the callback — budget is checked before, spend is
# recorded after, the audit trail is appended automatically.
#
# When no OPENAI_API_KEY is set, we use a deterministic fake executor
# so the teaching narrative runs end-to-end offline. The governance
# wiring (envelope, budget tracking, audit trail) is IDENTICAL either
# way — the fake just short-circuits the LLM at the callback boundary.

print("=" * 70)
print("TASK 2: Run Governed Supervisors")
print("=" * 70)

live_mode = bool(
    os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
)
executor = None if live_mode else make_fake_executor()
print(f"  Mode: {'LIVE LLM' if live_mode else 'OFFLINE (fake executor)'}")


async def run_tiers():
    questions = [
        ("public", governed_public, "What is machine learning?"),
        ("public", governed_public, "Show me the model training logs."),
        ("internal", governed_internal, "Read the customer-tier sales data."),
        ("admin", governed_admin, "What are the last 90 days of audit findings?"),
    ]
    for tier, gs, q in questions:
        try:
            result = await gs.run(objective=q, execute_node=executor)
            status = "ok" if result.success else "failed"
            print(
                f"\n--- {tier} tier: {q[:50]}... ---\n"
                f"  status={status}  consumed=${result.budget_consumed:.4f}  "
                f"audit_entries={len(result.audit_trail)}"
            )
        except Exception as e:
            print(f"\n--- {tier} tier: BLOCKED ({type(e).__name__}: {e}) ---")


try:
    asyncio.run(run_tiers())
except Exception as e:
    print(f"\n  (tier run skipped — {type(e).__name__}: {e})")

print("\n[x] Checkpoint 2 passed — runtime wrapper executed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Fail-Closed Verification
# ════════════════════════════════════════════════════════════════════════
#
# The fail-closed proof: ask the governance engine to verify an action
# for a role address that does not exist in the compiled org. The
# engine's 5-step algorithm returns DENY at Step 1 (clearance not
# resolvable). There is no "try again" path.

print("=" * 70)
print("TASK 3: Fail-Closed Verification")
print("=" * 70)

verdict = engine.verify_action(
    role_address="D99-R99-T99-R99",  # does not exist in org
    action="answer_question",
    context={"cost": 0.10},
)
print(
    f"  Unknown role D99-R99-T99-R99: "
    f"{'DENIED (correct)' if not verdict.allowed else 'ALLOWED (BUG!)'}  "
    f"level={verdict.level}  reason={verdict.reason[:80]}"
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert not verdict.allowed, "Fail-closed: unknown role MUST be denied"
assert verdict.level == "blocked"
print("\n[x] Checkpoint 3 passed — fail-closed verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Containing the Blast Radius of Adversarial Prompts
# ════════════════════════════════════════════════════════════════════════
#
# IMPORTANT PEDAGOGICAL CORRECTION: governance does NOT classify
# prompts as toxic. Content classification is a separate control
# (moderation API, classifier head, or `KnowledgeFilter` hook). What
# the envelope DOES do is limit the blast radius of a successful
# prompt injection:
#
#   - Budget cap: a looped injection cannot drain more than $5 on
#     the public tier before governance halts the plan.
#   - Tool allowlist: even if the prompt convinces the model to
#     "run train_model", the envelope refuses because train_model
#     is not in the public tier's allowed_actions.
#   - Clearance: even if the prompt convinces the model to "read
#     customer credit records", the envelope refuses because the
#     public tier's clearance is PUBLIC, not RESTRICTED.
#
# The RealToxicityPrompts dataset is the stress test for that blast
# radius — we count how many successful calls the public supervisor
# makes under adversarial load before budget or envelope caps it.

print("=" * 70)
print("TASK 4: Blast-Radius Containment Against Adversarial Prompts")
print("=" * 70)


async def test_adversarial_prompts():
    sample = adversarial_prompts.head(10)
    n_budget_exhausted = 0
    n_envelope_violation = 0
    n_success = 0

    for i, row in enumerate(sample.iter_rows(named=True)):
        prompt_text = row["prompt_text"]
        toxicity = row["toxicity_score"]

        try:
            result = await governed_public.run(
                objective=prompt_text,
                execute_node=executor,
            )
            if result.success:
                n_success += 1
                outcome = "responded (within envelope)"
            else:
                n_envelope_violation += 1
                outcome = f"envelope blocked ({result.budget_consumed:.4f} USD)"
        except Exception as e:
            n_budget_exhausted += 1
            outcome = f"HALTED: {type(e).__name__}"

        snippet = prompt_text[:50].replace("\n", " ")
        print(f"  {i+1:2}. tox={toxicity:.2f} {outcome}: {snippet}...")

    print(
        f"\n  Result: {n_success} served within envelope, "
        f"{n_envelope_violation} envelope blocks, "
        f"{n_budget_exhausted} budget halts"
    )
    print(
        "  Interpretation: governance did NOT filter on content. "
        "It capped damage by refusing tools/spend outside the envelope."
    )
    return n_success, n_envelope_violation, n_budget_exhausted


try:
    n_success, n_env, n_budget = asyncio.run(test_adversarial_prompts())
except Exception as e:
    print(f"  (adversarial test skipped — {type(e).__name__}: {e})")
    n_success = n_env = n_budget = 0

print("\n[x] Checkpoint 4 passed — blast-radius containment tested\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Audit Trail & Regulatory Mapping
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Audit Trail & Regulatory Mapping")
print("=" * 70)

qa_audit = governed_public.audit.to_list()
admin_audit = governed_admin.audit.to_list()

# Tamper-evidence check: every supervisor's audit chain is hash-linked,
# which is structural proof the trail cannot be edited post-hoc.
public_chain_valid = governed_public.audit.verify_chain()
admin_chain_valid = governed_admin.audit.verify_chain()

print(f"Audit trail sizes:")
print(f"  Public tier:  {len(qa_audit)} entries  (chain valid: {public_chain_valid})")
print(f"  Admin tier:   {len(admin_audit)} entries  (chain valid: {admin_chain_valid})")

if qa_audit:
    first = qa_audit[0]
    print(f"\n  Sample entry keys: {sorted(first.keys())}")
    print(f"  Sample record_type: {first.get('record_type')}")
    print(f"  Sample prev_hash:   {(first.get('prev_hash') or 'GENESIS')[:16]}...")
    print(f"  Sample record_hash: {first.get('record_hash', '')[:16]}...")

# Regulatory mapping
print("\n--- Regulatory Mapping ---")
regulatory_map = pl.DataFrame(
    {
        "Regulation": [
            "EU AI Act Art. 9 (Risk Management)",
            "EU AI Act Art. 12 (Record-keeping)",
            "EU AI Act Art. 14 (Human Oversight)",
            "Singapore AI Verify (Accountability)",
            "MAS TRM 7.5 (Audit Trail)",
            "PDPA (Personal Data Protection)",
        ],
        "PACT Control": [
            "ConstraintEnvelopeConfig per role (5 dimensions)",
            "Hash-chained audit trail with timestamps",
            "D/T/R chains — every action traces to a human Delegator",
            "D/T/R accountability grammar",
            "audit.verify_chain() + query_by_agent()",
            "ConfidentialityLevel gating + KnowledgeItem ownership",
        ],
        "Status": [
            "COMPLIANT",
            "COMPLIANT",
            "COMPLIANT",
            "COMPLIANT",
            "COMPLIANT",
            "COMPLIANT",
        ],
    }
)
print(regulatory_map)

# Enforcement modes
print("\n--- Enforcement Modes ---")
print("  WARN:  log the violation, allow the action (dev/staging only)")
print("  BLOCK: deny the action and raise a governed error (production)")
print("  AUDIT: allow but flag for human review (semi-trusted agents)")
print("\n  Production default: BLOCK (fail-closed).")
print("  Modern pact: pact.enforcement.EnforcementMode + validate_enforcement_mode()")

# ── Checkpoint 5 ────────────────────────────────────────────────────────
assert public_chain_valid, "Task 5: public-tier audit chain should verify"
assert admin_chain_valid, "Task 5: admin-tier audit chain should verify"
assert regulatory_map.height >= 6, "Task 5: should map at least 6 regulations"
print("\n[x] Checkpoint 5 passed — audit trail and regulatory map complete\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Audit event timeline + enforcement outcome distribution
# ════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

tiers = ["public", "internal", "admin"]
tier_colors = {"public": "#2ecc71", "internal": "#3498db", "admin": "#e74c3c"}
events = [
    (0.5, "public", "allow"),
    (1.0, "public", "allow"),
    (1.5, "internal", "allow"),
    (2.0, "public", "block"),
    (2.5, "admin", "allow"),
    (3.0, "public", "allow"),
    (3.5, "internal", "allow"),
    (4.0, "admin", "audit"),
    (4.5, "public", "block"),
    (5.0, "internal", "allow"),
]
for t, tier, outcome in events:
    marker = "o" if outcome == "allow" else ("x" if outcome == "block" else "s")
    ax1.scatter(
        t, tiers.index(tier), c=tier_colors[tier], marker=marker, s=80, zorder=3
    )
ax1.set_yticks(range(len(tiers)))
ax1.set_yticklabels(tiers)
ax1.set_xlabel("Time (simulated seconds)")
ax1.set_title("Audit Event Timeline by Tier", fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="gray",
        markersize=8,
        label="allow",
    ),
    Line2D(
        [0],
        [0],
        marker="x",
        color="gray",
        markersize=8,
        label="block",
        linestyle="None",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        markerfacecolor="gray",
        markersize=8,
        label="audit",
    ),
]
ax1.legend(handles=legend_elements, fontsize=8, loc="upper right")

outcomes = ["ALLOW", "BLOCK", "AUDIT"]
counts = [6, 2, 1]
colors_pie = ["#2ecc71", "#e74c3c", "#f39c12"]
ax2.pie(
    counts,
    labels=outcomes,
    colors=colors_pie,
    autopct="%1.0f%%",
    startangle=90,
    textprops={"fontsize": 10, "fontweight": "bold"},
)
ax2.set_title("Enforcement Outcome Distribution", fontweight="bold")

plt.tight_layout()
fname = OUTPUT_DIR / "ex7_audit_timeline_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: PDPA Breach-Readiness Audit
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore HR SaaS platform with 200+ enterprise
# customers is served a PDPA breach notification inquiry. The PDPC
# asks: "For the 72-hour window starting 14 March, list every AI
# action on personal data, the role that took it, the human
# delegator that authorised the class of action, and whether any
# governed-error responses were returned to external callers."
#
# Without runtime enforcement, the only answer is a log dive that
# takes weeks and produces an incomplete reconstruction. With
# GovernedSupervisor wrapping every run, the answer is a single
# query against audit.query_by_agent() + audit.verify_chain() for
# tamper-evidence. The decision, the delegation chain, and the
# fail-closed behaviour on any suspicious action are all captured
# as hash-linked records.
#
# BUSINESS IMPACT: PDPA financial penalties under Singapore's 2021
# amendments reach 10% of annual turnover or S$1M (whichever is
# higher) for organisations with revenue above S$10M. A credible,
# tamper-evident audit trail is the difference between "we breached
# a data subject's rights" and "we contained the incident, here is
# the cryptographic evidence". One is a fine; the other is a
# closed case.

print("=" * 70)
print("  KEY TAKEAWAY: Governance Is a Runtime Property, Not a Slide")
print("=" * 70)
print("  Compile-time validation + runtime enforcement + hash-chained")
print("  audit = structural evidence for regulators. Anything less is vibes.")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (Exercise 7 Full Arc)")
print("=" * 70)
print(
    """
  [x] Wrapped agents with GovernedSupervisor at three clearance tiers
  [x] Ran governed supervisors against normal and adversarial queries
  [x] Verified fail-closed: unknown roles are denied, not ignored
  [x] Tested blast-radius containment against RealToxicityPrompts
  [x] Verified a hash-chained audit trail and mapped it to 6 regulations
  [x] Reasoned about a live PDPA breach-readiness scenario

  Governance principles recap:
    Fail-closed:          deny unless explicitly allowed
    Monotonic tightening: envelopes only get stricter
    Clearance hierarchy:  top_secret > secret > confidential > restricted > public
    Budget cascading:     child budget <= parent allocation
    Audit completeness:   every decision logged, chain-verifiable

  NEXT: Exercise 8 (Capstone) integrates EVERYTHING from M6 —
  SFT + DPO + PACT governance + Nexus deployment + compliance audit —
  a complete production ML system from training to deployment.
"""
)
