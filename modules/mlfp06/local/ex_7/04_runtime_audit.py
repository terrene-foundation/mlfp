# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.4: Runtime Governance, Deny Paths, and Audit Trail
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Run a real (local Ollama) LLM behind a GovernedSupervisor at runtime
#   - Prove a deny path: attach an envelope, then an out-of-envelope action
#     is BLOCKED — and see that the installed engine auto-approves a role
#     that has no envelope
#   - Contain the blast radius of adversarial prompts with a budget
#     envelope and an action allowlist
#   - Verify a hash-chained audit trail and map the evidence it provides
#     to the EU AI Act, MAS TRM and PDPA
#   - Know pact's real verdict levels (auto_approved / flagged / held /
#     blocked) and PactEngine's enforcement modes (enforce / shadow /
#     disabled)
#
# PREREQUISITES: 03_budget_access.py; Ollama running (`ollama serve`)
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Build three GovernedSupervisor tiers (public / confidential / secret)
#   2. Run the governed supervisors against normal inputs
#   3. Deny paths: attach an envelope, verify an out-of-envelope action
#   4. Contain the blast radius of adversarial prompts
#   5. Map audit-trail evidence to regulations (EU AI Act, MAS TRM, PDPA)
#   6. Apply — PDPA breach-readiness audit for a Singapore SaaS
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from kaizen_agents import GovernedSupervisor
from pact import (
    CommunicationConstraintConfig,
    ConfidentialityLevel,
    ConstraintEnvelopeConfig,
    DataAccessConstraintConfig,
    EnforcementMode,
    FinancialConstraintConfig,
    OperationalConstraintConfig,
    RoleEnvelope,
    TemporalConstraintConfig,
)

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, preflight_ollama
from shared.mlfp06.ex_7 import (
    clearance_chain_violations,
    compile_governance,
    default_model_name,
    load_adversarial_prompts,
    make_llm_executor,
)

OUTPUT_DIR = Path("outputs") / "ex7_governance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# This file makes real LLM calls. If Ollama is not running this raises
# OllamaUnreachableError telling you to start it (`ollama serve`).
preflight_ollama(required_models=[DEFAULT_CHAT_MODEL])

engine, org = compile_governance()
adversarial_prompts = load_adversarial_prompts(n=50)
print("\n--- GovernanceEngine compiled; adversarial prompts loaded ---\n")


# ════════════════════════════════════════════════════════════════════════
# THEORY — Runtime Enforcement vs Compile-Time Validation
# ════════════════════════════════════════════════════════════════════════
# Compiling an org YAML proves the governance GRAPH is sound. It does
# NOT prove that live LLM calls respect the graph. At runtime two
# different components do two different jobs:
#
#   GovernanceEngine.verify_action(role, action, context)
#       -> GovernanceVerdict with .level in
#          {auto_approved, flagged, held, blocked}
#          (.allowed is True for auto_approved and flagged)
#       It checks the role's ATTACHED envelope: allowed actions, cost
#       against the financial cap, and the other dimensions.
#
#   GovernedSupervisor.run(objective, execute_node=...)
#       Runs your executor (here: a real Ollama call), records the cost
#       the executor reports against its financial envelope, HOLDS further
#       work once the budget is used up, and appends a hash-chained audit
#       record for every step. It does not read prompt content and does
#       not decide which tools are allowed — that is verify_action's job.
#
# IMPORTANT — the installed default is NOT fail-closed. A role with no
# attached envelope, or an address that is not in the org, is
# auto-approved ("No envelope constraints -- action permitted"). A deny
# path exists only where you attached an envelope. So production
# governance needs two habits: attach an envelope to every agent role,
# and TEST the deny path (Task 3).


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Build Three GovernedSupervisor Tiers
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: GovernedSupervisor — three clearance tiers")
print("=" * 70)

model = default_model_name()

# TODO: Build three GovernedSupervisor tiers with model=model:
#   governed_public   — $5,   tools answer_question + search_faq,     "public"
#   governed_internal — $50,  + read_data + train_model,              "confidential"
#   governed_admin    — $200, answer_question, read_data, audit_model,
#                       access_audit_log,                             "secret"
# Hint: GovernedSupervisor(model=..., budget_usd=..., tools=[...],
#       data_clearance=...)  (pact: public < restricted < confidential < secret)
governed_public = ____
governed_internal = ____
governed_admin = ____

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
        f"clearance={env.confidentiality_clearance.name:<12s}  "
        f"tools={len(env.operational.allowed_actions)}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert governed_public.envelope.financial.max_spend_usd == 5.0
assert governed_internal.envelope.financial.max_spend_usd == 50.0
assert governed_admin.envelope.financial.max_spend_usd == 200.0
assert "read_data" in governed_internal.envelope.operational.allowed_actions
assert "access_audit_log" in governed_admin.envelope.operational.allowed_actions
assert "train_model" not in governed_public.envelope.operational.allowed_actions
assert governed_admin.envelope.confidentiality_clearance == ConfidentialityLevel.SECRET
print("\n[x] Checkpoint 1 passed — three governance tiers wired\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Run the Governed Supervisors (real LLM)
# ════════════════════════════════════════════════════════════════════════
#
# `make_llm_executor()` returns the execute_node callback: it sends the
# objective to the local Ollama model through make_delegate() and returns
# {result, cost, prompt_tokens, completion_tokens}. Ollama is free, so
# cost is $0 here. There is no offline stub — if the call fails, the
# supervisor marks the node FAILED and we stop with the real error.

print("=" * 70)
print("TASK 2: Run Governed Supervisors")
print("=" * 70)

# TODO: Build the real-LLM execute_node callback from the shared helper.
executor = ____


def node_errors(result) -> list[str]:
    """Collect the error message of every FAILED plan node."""
    return [n.error for n in result.plan.nodes.values() if n.error]


async def run_tiers() -> int:
    questions = [
        ("public", governed_public, "What is machine learning? Answer in two sentences."),
        ("public", governed_public, "Explain what a model training log contains."),
        ("internal", governed_internal, "List three checks before reading sales data."),
        ("admin", governed_admin, "What should an AI audit-findings review cover?"),
    ]
    successes = 0
    for tier, gs, q in questions:
        # TODO: Run the objective q through this tier's supervisor.
        result = ____
        if not result.success:
            raise RuntimeError(
                f"{tier} tier run failed: {node_errors(result)}. "
                "Is Ollama running? Start it with: ollama serve"
            )
        successes += 1
        answer = next(iter(result.results.values()))
        print(
            f"\n--- {tier} tier: {q[:50]} ---\n"
            f"  answer: {str(answer)[:120].replace(chr(10), ' ')}...\n"
            f"  consumed=${result.budget_consumed:.4f}  "
            f"audit_entries={len(result.audit_trail)}"
        )
    return successes


n_task2_success = asyncio.run(run_tiers())

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert n_task2_success == 4, "Task 2: every tier run should complete"
print("\n[x] Checkpoint 2 passed — runtime wrapper executed real LLM calls\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Deny Paths (Attach the Envelope First)
# ════════════════════════════════════════════════════════════════════════
#
# A deny path only exists where an envelope is attached. We:
#
#   1. Attach a public-tier envelope to customer_agent (D3-R1-T1-R1).
#   2. Ask the engine to verify `train_model` — not in that envelope.
#   3. Ask it to verify a $100 answer_question — over the $5 cap.
#   4. Ask the same train_model question for the department head
#      vp_customer (D3-R1), which has NO envelope.
#
# Steps 2-3 must be BLOCKED. Step 4 is AUTO-APPROVED by the installed
# default — that is the gap every org must close by attaching envelopes.

print("=" * 70)
print("TASK 3: Deny Paths (envelope attached vs. no envelope)")
print("=" * 70)

public_envelope = ConstraintEnvelopeConfig(
    id="customer_agent_envelope",
    description="customer_agent — bounded public tier",
    confidentiality_clearance=ConfidentialityLevel.PUBLIC,
    financial=FinancialConstraintConfig(max_spend_usd=5.0),
    operational=OperationalConstraintConfig(
        allowed_actions=["answer_question", "search_faq"],
        blocked_actions=[],
    ),
    temporal=TemporalConstraintConfig(blackout_periods=[]),
    data_access=DataAccessConstraintConfig(
        read_paths=["/public/*"],
        write_paths=[],
        blocked_data_types=[],
    ),
    communication=CommunicationConstraintConfig(allowed_channels=["internal"]),
    max_delegation_depth=3,
)
# TODO: Attach public_envelope to customer_agent (D3-R1-T1-R1), defined
#       by its head vp_customer (D3-R1). Do this BEFORE testing deny paths.
____

# TODO: Verify train_model (cost 0.10) for customer_agent.
out_of_envelope_verdict = ____
over_budget_verdict = engine.verify_action(
    role_address="D3-R1-T1-R1",
    action="answer_question",
    context={"cost": 100.0},
)
# TODO: Verify the same action for the department head D3-R1.
no_envelope_verdict = ____
for label, v in [
    ("customer_agent train_model ", out_of_envelope_verdict),
    ("customer_agent $100 answer ", over_budget_verdict),
    ("vp_customer train_model    ", no_envelope_verdict),
]:
    print(f"  {label} level={v.level:<13} allowed={v.allowed}")
    print(f"      reason: {v.reason[:110]}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert (
    not out_of_envelope_verdict.allowed
), "Task 3: an out-of-envelope action MUST be denied"
assert out_of_envelope_verdict.level == "blocked"
assert not over_budget_verdict.allowed, "Task 3: over-budget MUST be denied"
assert over_budget_verdict.level == "blocked"
assert no_envelope_verdict.level == "auto_approved", (
    "Task 3: the installed default auto-approves a role with no envelope"
)
print("\n[x] Checkpoint 3 passed — deny path verified where an envelope exists\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Containing the Blast Radius of Adversarial Prompts
# ════════════════════════════════════════════════════════════════════════
#
# Governance does NOT classify prompts as toxic. Content classification
# is a separate control (a moderation model or classifier). What the
# envelopes DO is cap the damage a successful injection can cause:
#
#   - Budget: once the financial envelope is used up, the supervisor
#     HOLDS every further node — a looped injection stops spending.
#   - Action allowlist: whatever the model is talked into, the most
#     damaging action it could request (`train_model`, `read_data`) is
#     blocked by verify_action for the customer-agent role.
#
# Ollama is free, so to make the budget observable this tier charges a
# NOTIONAL $0.01 per 1,000 tokens against a deliberately tiny $0.01
# budget. These are teaching numbers, not a bill.

print("=" * 70)
print("TASK 4: Blast-Radius Containment Against Adversarial Prompts")
print("=" * 70)

NOTIONAL_USD_PER_1K_TOKENS = 0.01
governed_blast = GovernedSupervisor(
    model=model,
    budget_usd=0.01,
    tools=["answer_question", "search_faq"],
    data_clearance="public",
)
blast_executor = make_llm_executor(
    notional_usd_per_1k_tokens=NOTIONAL_USD_PER_1K_TOKENS
)
INJECTED_ACTIONS = ["train_model", "read_data"]


async def test_adversarial_prompts() -> dict[str, int]:
    sample = adversarial_prompts.head(10)
    counts = {"served": 0, "held_budget": 0, "tool_blocked": 0, "tool_allowed": 0}

    for i, row in enumerate(sample.iter_rows(named=True)):
        prompt_text = row["prompt_text"]
        # TODO: Run the adversarial prompt through governed_blast.
        result = ____
        states = {n.state.name for n in result.plan.nodes.values()}
        if result.success:
            counts["served"] += 1
            outcome = f"served (notional ${result.budget_consumed:.4f})"
        elif "HELD" in states:
            counts["held_budget"] += 1
            outcome = "HELD — budget envelope exhausted"
        else:
            raise RuntimeError(
                f"prompt {i + 1}: LLM call failed: {node_errors(result)}. "
                "Start Ollama: ollama serve"
            )

        # Whatever the reply says, check the worst actions it could request.
        for action in INJECTED_ACTIONS:
            # TODO: Check the action for customer_agent (cost 0.01), then
            #       record the attempt on governed_blast's audit trail.
            # Hint: engine.verify_action(...); governed_blast.record_tool_use(
            #       action, blocked=..., reason=...)
            verdict = ____
            ____
            counts["tool_blocked" if not verdict.allowed else "tool_allowed"] += 1

        snippet = prompt_text[:45].replace("\n", " ")
        print(f"  {i + 1:2}. tox={row['toxicity_score']:.2f} {outcome}: {snippet}...")

    snap = governed_blast.budget.get_snapshot("root")
    print(
        f"\n  Result: {counts['served']} served, {counts['held_budget']} held by "
        f"the budget envelope; {counts['tool_blocked']} injected tool requests "
        f"blocked, {counts['tool_allowed']} allowed"
    )
    print(
        f"  Notional spend: ${snap.consumed:.4f} of ${snap.allocated:.2f} "
        "(the check runs BEFORE each call, so the last call can overshoot)"
    )
    return counts


blast_counts = asyncio.run(test_adversarial_prompts())

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert blast_counts["served"] + blast_counts["held_budget"] == 10, (
    "Task 4: every adversarial prompt must be either served or held"
)
assert blast_counts["tool_allowed"] == 0, (
    "Task 4: the customer-agent envelope must block every injected tool"
)
print("\n[x] Checkpoint 4 passed — blast radius bounded by budget + allowlist\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Audit Trail & Regulatory Evidence Mapping
# ════════════════════════════════════════════════════════════════════════
#
# The hash-chained audit trail is the structural evidence a regulator
# asks for: every supervisor records every step as a linked record, and
# `audit.verify_chain()` returns True only if no record was altered.

print("=" * 70)
print("TASK 5: Audit Trail & Regulatory Evidence Mapping")
print("=" * 70)

public_audit = governed_public.audit.to_list()
blast_audit = governed_blast.audit.to_list()
admin_audit = governed_admin.audit.to_list()
# TODO: Verify each supervisor's hash chain -> {"public": bool, ...}
chains_valid = ____
for tier, records in [("public", public_audit), ("blast", blast_audit), ("admin", admin_audit)]:
    by_type = Counter(r["record_type"] for r in records)
    print(f"  {tier:<7} {len(records):>3} records  chain valid: {chains_valid[tier]}")
    print(f"          by type: {dict(by_type)}")

if blast_audit:
    last = blast_audit[-1]
    print(f"\n  Sample record keys: {sorted(last.keys())}")
    print(f"  record_type={last['record_type']}  action={last['action']}")
    print(f"  prev_hash={last['prev_hash'][:16]}...  record_hash={last['record_hash'][:16]}...")

# Evidence produced by THIS run — each row is computed, not asserted.
n_blocked_records = sum(1 for r in blast_audit if r["action"].startswith("tool_blocked"))
regulatory_map = pl.DataFrame(
    {
        "Regulation": [
            "EU AI Act Art. 9 (risk management)",
            "EU AI Act Art. 12 (record-keeping)",
            "EU AI Act Art. 14 (human oversight)",
            "Singapore AI Verify (accountability)",
            "MAS TRM Guidelines (audit trail)",
            "PDPA (personal data protection)",
        ],
        "Evidence in this run": [
            f"deny path verified: {out_of_envelope_verdict.level}",
            f"{len(public_audit) + len(blast_audit) + len(admin_audit)} hash-chained records",
            f"{org.n_delegations} envelopes, each defined by a human head",
            f"clearance-chain violations: {clearance_chain_violations().height}",
            f"all chains verify: {all(chains_valid.values())}",
            f"public tier clearance: {governed_public.envelope.confidentiality_clearance.value}; "
            f"{n_blocked_records} blocked data/tool requests logged",
        ],
        "Evidence present": [
            out_of_envelope_verdict.level == "blocked",
            len(public_audit) > 0 and len(admin_audit) > 0,
            org.n_delegations == org.n_agents,
            clearance_chain_violations().height == 0,
            all(chains_valid.values()),
            governed_public.envelope.confidentiality_clearance
            == ConfidentialityLevel.PUBLIC
            and n_blocked_records > 0,
        ],
    }
)
print("\n--- Regulatory evidence map (evidence, not a compliance ruling) ---")
print(regulatory_map)

print("\n--- pact verdict levels (GovernanceEngine.verify_action) ---")
print("  auto_approved  allowed, no review")
print("  flagged        allowed, flagged for review")
print("  held           paused for human approval")
print("  blocked        denied")
print("\n--- PactEngine enforcement modes (pact.EnforcementMode) ---")
for mode in EnforcementMode:
    print(f"  {mode.value}")
print("  enforce = verdicts bind (default); shadow = log only, never block;")
print("  disabled = skip governance (needs PACT_ALLOW_DISABLED_MODE=true)")

# ── Checkpoint 5 ────────────────────────────────────────────────────────
assert all(chains_valid.values()), "Task 5: every audit chain should verify"
assert regulatory_map.height >= 6, "Task 5: should map at least 6 regulations"
assert regulatory_map["Evidence present"].all(), (
    "Task 5: every mapped control should have evidence from this run"
)
print("\n[x] Checkpoint 5 passed — audit trail verified, evidence mapped\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — Audit records by tier + outcome distribution (this run)
# ════════════════════════════════════════════════════════════════════════

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

tiers = ["public", "blast", "admin"]
markers = {"genesis": "D", "action": "o", "held": "s"}
for y, (tier, records) in enumerate(
    [("public", public_audit), ("blast", blast_audit), ("admin", admin_audit)]
):
    for idx, rec in enumerate(records):
        blocked = rec["action"].startswith("tool_blocked")
        ax1.scatter(
            idx,
            y,
            marker="x" if blocked else markers.get(rec["record_type"], "."),
            color="#e74c3c" if blocked or rec["record_type"] == "held" else "#2ecc71",
            s=50,
        )
ax1.set_yticks(range(len(tiers)))
ax1.set_yticklabels(tiers)
ax1.set_xlabel("Audit record index (chain order)")
ax1.set_title("Audit Records by Tier (x = blocked tool, ■ = held)", fontweight="bold")
ax1.grid(axis="x", alpha=0.3)

outcome_labels = ["served", "held (budget)", "tool blocked", "tool allowed"]
outcome_counts = [
    blast_counts["served"] + n_task2_success,
    blast_counts["held_budget"],
    blast_counts["tool_blocked"],
    blast_counts["tool_allowed"],
]
ax2.bar(outcome_labels, outcome_counts, color=["#2ecc71", "#f39c12", "#e74c3c", "#95a5a6"])
for i, c in enumerate(outcome_counts):
    ax2.text(i, c + 0.2, str(c), ha="center", fontsize=9)
ax2.set_title("Runtime Outcomes Measured in This Run", fontweight="bold")
ax2.set_ylabel("Count")

plt.tight_layout()
fname = OUTPUT_DIR / "ex7_audit_timeline_viz.png"
plt.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved: {fname}")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: PDPA Breach-Readiness Audit
# ════════════════════════════════════════════════════════════════════════
#
# SCENARIO: A Singapore HR SaaS platform with 200+ enterprise customers
# receives a data-breach inquiry from the regulator: "For the 72-hour
# window starting 14 March, list every AI action on personal data, the
# role that took it, the human head that authorised that class of action,
# and whether any request was refused."
#
# Without runtime governance, the only answer is a log dive that takes
# weeks and produces an incomplete reconstruction. With GovernedSupervisor
# around every run and verify_action in front of every tool, the answer is
# a query over `supervisor.audit.to_list()` (including tool_blocked
# records) plus `.verify_chain()` for tamper-evidence.
#
# BUSINESS IMPACT: under the 2020 PDPA amendments (in force from
# October 2022) the maximum financial penalty is 10% of annual Singapore
# turnover for organisations above S$10M turnover, or S$1M otherwise.
# A tamper-evident trail does not make a breach legal; it lets the
# organisation show quickly what happened and what was refused.

print("=" * 70)
print("  KEY TAKEAWAY: Governance Is a Runtime Property, Not a Slide")
print("=" * 70)
print("  Envelopes attached to every role + deny paths tested + a verified")
print("  audit chain = evidence. A role without an envelope is auto-approved.")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Governance lens over the supervisor audit
# ══════════════════════════════════════════════════════════════════
# The governance lens reads the blast-radius supervisor's audit records
# (supervisor.audit exposes .to_list()) and counts them by record type.
from shared.mlfp06.diagnostics import LLMObservatory

obs = LLMObservatory(governance=governed_blast.audit, run_id="ex_7_4_runtime")
# TODO: Take the governance lens's audit snapshot (last 200 records).
snapshot = ____
print("\n── LLM Observatory: governance audit snapshot ──")
print(snapshot.select("action", "verdict", "reason").tail(6))
print(obs.governance.report())
print(f"  (hash chain verified above with audit.verify_chain(): {chains_valid['blast']})")
# INTERPRETATION: "blocked" rows are the injected tool requests the
# envelope refused; "held" rows are prompts the budget envelope stopped.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (Exercise 7 Full Arc)")
print("=" * 70)
print(
    """
  [x] Ran a real local LLM behind GovernedSupervisor at three clearance tiers
  [x] Proved a deny path by attaching an envelope first
  [x] Saw the installed default auto-approve a role with no envelope
  [x] Bounded adversarial prompts with a budget envelope and an allowlist
  [x] Verified hash-chained audit trails and mapped the evidence
  [x] Reasoned about a live PDPA breach-readiness scenario

  Governance principles recap:
    Envelopes restrict:   no envelope = auto-approved (installed default)
    Test the deny path:   attach the envelope, then assert "blocked"
    Monotonic tightening: child envelopes never exceed their parent
    Clearance ladder:     public < restricted < confidential < secret < top_secret
    Budget cascading:     child budget <= parent allocation
    Audit completeness:   every step logged, chain-verifiable

  NEXT: Exercise 8 (Capstone) integrates EVERYTHING from M6 —
  SFT + DPO + PACT governance + Nexus deployment + compliance audit —
  a complete production ML system from training to deployment.
"""
)
