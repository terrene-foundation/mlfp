# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7.4: Runtime Governance, Fail-Closed, and Audit Trail
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Wrap a Kaizen BaseAgent with PactGovernedAgent at runtime
#   - Verify fail-closed semantics
#   - Block real adversarial prompts from RealToxicityPrompts
#   - Export an audit trail and map it to EU AI Act / MAS TRM / PDPA
#
# PREREQUISITES: 03_budget_access.py
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Wrap a Kaizen QA agent with three governance levels
#   2. Run the governed agents
#   3. Verify fail-closed for an unknown agent
#   4. Block real adversarial prompts
#   5. Map audit trail entries to regulations
#   6. Apply — PDPA breach-readiness audit
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kailash_pact import PactGovernedAgent

from shared.mlfp06.ex_7 import (
    compile_governance,
    default_model_name,
    load_adversarial_prompts,
)

engine, org = compile_governance()
adversarial_prompts = load_adversarial_prompts(n=50)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Runtime Enforcement vs Compile-Time Validation
# ════════════════════════════════════════════════════════════════════════
# Compile-time proves the graph is sound. Runtime proves live LLM calls
# respect the graph. You need both. Fail-closed means DENY unless every
# check returns ALLOW — unknown agents, missing envelopes = DENY.


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Wrap a QA agent with 3 governance levels
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 1: PactGovernedAgent Wrapping")
print("=" * 70)


class QASignature(Signature):
    """Answer questions within governance constraints."""

    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Governed response")
    confidence: float = OutputField(description="Answer confidence 0-1")


class QAAgent(BaseAgent):
    signature = QASignature
    model = default_model_name()
    max_llm_cost_usd = 5.0


base_qa = QAAgent()

# TODO: Wrap base_qa with PactGovernedAgent for three levels:
#       governed_public  — role="analyst", max_budget_usd=5,
#                          allowed_tools=["answer_question","search_faq"],
#                          clearance_level="public"
#       governed_internal — role="engineer", max_budget_usd=50,
#                          allowed_tools=["answer_question","search_faq","read_data","train_model"],
#                          clearance_level="confidential"
#       governed_admin   — role="auditor", max_budget_usd=200,
#                          allowed_tools=["answer_question","read_data","audit_model","access_audit_log"],
#                          clearance_level="restricted"
governed_public = ____
governed_internal = ____
governed_admin = ____

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert governed_public is not None
assert governed_internal is not None
assert governed_admin is not None
print("[x] Checkpoint 1 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Run the governed agents
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: Run Governed Agents")
print("=" * 70)


async def test_governed_agents():
    # TODO: Call governed_public.run(question="What is machine learning?")
    #       and print the first 200 chars of the answer.
    try:
        result = ____
        print(f"  Public answer: {str(result.answer)[:200]}...")
    except Exception as e:
        print(f"  Blocked: {type(e).__name__}: {e}")


try:
    asyncio.run(test_governed_agents())
except Exception as e:
    print(f"  (LLM call skipped — {type(e).__name__}: {e})")

print("\n[x] Checkpoint 2 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Fail-closed verification
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: Fail-Closed Verification")
print("=" * 70)

# TODO: Call engine.check_access with agent_id="unknown_agent",
#       resource="any_resource", action="any_action"
decision = ____

print(f"  Unknown agent: {'DENIED' if not decision.allowed else 'ALLOWED (BUG!)'}")

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert not decision.allowed
print("\n[x] Checkpoint 3 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Block real adversarial prompts
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Adversarial Prompt Blocking")
print("=" * 70)


async def test_adversarial_prompts():
    sample = adversarial_prompts.head(10)
    n_blocked = 0

    for i, row in enumerate(sample.iter_rows(named=True)):
        prompt_text = row["prompt_text"]
        toxicity = row["toxicity_score"]
        try:
            # TODO: check_access on customer_agent / external_input /
            #       answer_question, passing payload=prompt_text
            decision = ____
        except TypeError:
            decision = engine.check_access(
                agent_id="customer_agent",
                resource="external_input",
                action="answer_question",
            )
        allowed = getattr(decision, "allowed", False)
        if not allowed:
            n_blocked += 1
        status = "BLOCKED" if not allowed else "allowed"
        snippet = prompt_text[:50].replace("\n", " ")
        print(f"  {i+1:2}. tox={toxicity:.2f} {status}: {snippet}...")
    return n_blocked


try:
    n_blocked = asyncio.run(test_adversarial_prompts())
except Exception as e:
    print(f"  (skipped — {type(e).__name__}: {e})")
    n_blocked = 0

print("\n[x] Checkpoint 4 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Audit trail & regulatory mapping
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Audit Trail & Regulatory Mapping")
print("=" * 70)

try:
    qa_audit = governed_public.get_audit_trail()
    admin_audit = governed_admin.get_audit_trail()
except Exception:
    qa_audit, admin_audit = [], []

print(f"Audit entries: public={len(qa_audit)}  admin={len(admin_audit)}")

# TODO: Call engine.check_access for model_trainer / training_data / read
#       and bind to trace.
trace = ____

print(
    f"  model_trainer read training_data: "
    f"{'ALLOWED' if trace.allowed else 'DENIED'}"
)

# TODO: Build a polars DataFrame mapping at least 6 regulations to PACT
#       controls with a Status column. Columns: Regulation, PACT Control, Status
#       Regulations: EU AI Act Art. 9, Art. 12, Art. 14, Singapore AI Verify,
#                    MAS TRM 7.5, PDPA
regulatory_map = ____

print(regulatory_map)

# ── Checkpoint 5 ────────────────────────────────────────────────────────
assert trace.allowed
assert regulatory_map.height >= 6
print("\n[x] Checkpoint 5 passed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Apply: PDPA Breach-Readiness Audit
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore HR SaaS is served a PDPA inquiry asking, for a
# 72-hour window, every AI action on personal data, the responsible
# agent, the delegator, and every governed-error response. Without
# runtime enforcement, the log dive takes weeks; with PactGovernedAgent,
# it is a one-query answer.
#
# BUSINESS IMPACT: PDPA penalties reach 10% of annual turnover or
# S$1M (whichever is higher). A credible audit trail is the difference
# between a fine and a closed case.

print("=" * 70)
print("  KEY TAKEAWAY: Governance Is a Runtime Property, Not a Slide")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED (Exercise 7 Full Arc)")
print("=" * 70)
print(
    """
  [x] Wrapped Kaizen agents with PactGovernedAgent
  [x] Verified fail-closed semantics
  [x] Tested against RealToxicityPrompts
  [x] Exported audit trails, mapped to 6 regulations
  [x] Reasoned about a live PDPA breach-readiness scenario

  Governance principles recap:
    Fail-closed, monotonic tightening, cascading budgets, audit trail.

  NEXT: Exercise 8 (Capstone) integrates EVERYTHING from M6.
"""
)
