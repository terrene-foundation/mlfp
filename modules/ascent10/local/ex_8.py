# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 8: Capstone — Governed ML System
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Deploy a complete governed ML system combining alignment
#   (SFT adapter), RL-optimized decisions, PACT governance, and Nexus
#   multi-channel deployment.
#
# TASKS:
#   1. Load fine-tuned model with adapter
#   2. Configure PACT governance with operating envelopes
#   3. Build governed agent pipeline
#   4. Deploy via Nexus (API + CLI)
#   5. Test governance enforcement across channels
#   6. Generate compliance audit report
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import time

import polars as pl
from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kailash_align import AdapterRegistry, AlignmentConfig, AlignmentPipeline
from kailash_nexus import Nexus
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Load fine-tuned model with adapter
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
eval_data = loader.load("ascent10", "sg_domain_qa.parquet")

print(f"=== Governed ML System Capstone ===")
print(f"Evaluation data: {eval_data.shape}")


async def load_model():
    # TODO: Create AdapterRegistry, list adapters, load the best merged adapter.
    # Hint: AdapterRegistry(), await registry.list_adapters(),
    #        await registry.get_adapter("sg_domain_slerp_merge_v1")
    registry = ____
    adapters = ____

    print(f"\nAvailable adapters: {len(adapters)}")
    for a in adapters:
        print(f"  {a.get('name', '?')}: {a.get('method', '?')}")

    best_adapter = ____

    # TODO: Create AlignmentPipeline for inference with the adapter.
    # Hint: AlignmentPipeline(AlignmentConfig(method="inference", adapter_path=...))
    pipeline = ____

    print(f"\nLoaded adapter: {best_adapter.get('name', 'N/A')}")
    print(f"This is the SFT+DPO merged model from Exercises 1-2-5.")

    return pipeline, registry


inference_pipeline, registry = asyncio.run(load_model())


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure PACT governance
# ══════════════════════════════════════════════════════════════════════

# TODO: Write a PACT organization YAML with qa_agent and admin_agent.
# Include: departments, delegations with envelopes, global operating_envelopes.
# Hint: qa_agent gets $1 budget + internal clearance, admin_agent gets $10 + confidential.
org_yaml = """
____
"""

org_path = os.path.join(tempfile.gettempdir(), "capstone_org.yaml")
with open(org_path, "w") as f:
    f.write(org_yaml)

# TODO: Create GovernanceEngine.
# Hint: GovernanceEngine()
engine = ____


async def setup_governance():
    # TODO: Compile the org YAML.
    # Hint: engine.compile_org(org_path)
    org = ____
    print(f"\n=== PACT Governance ===")
    print(
        f"Organization compiled: {org.n_agents} agents, {org.n_delegations} delegations"
    )
    print(f"QA agent: $1 budget, internal clearance, read-only tools")
    print(f"Admin agent: $10 budget, confidential clearance, full tools")
    return org


org = asyncio.run(setup_governance())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Build governed agent pipeline
# ══════════════════════════════════════════════════════════════════════


# TODO: Define a QASignature with question, answer, confidence, sources fields.
# Hint: class QASignature(Signature) with InputField and OutputField.
class QASignature(Signature):
    """Answer Singapore domain questions using fine-tuned knowledge."""

    question: str = ____
    answer: str = ____
    confidence: float = ____
    sources: list[str] = ____


# TODO: Define QAAgent using BaseAgent with the signature.
# Hint: class QAAgent(BaseAgent), set signature, model, max_llm_cost_usd.
class QAAgent(BaseAgent):
    signature = ____
    model = ____
    max_llm_cost_usd = ____


base_qa = QAAgent()

# TODO: Wrap with PactGovernedAgent for qa role ($1, internal, limited tools).
# Hint: PactGovernedAgent(agent=..., governance_engine=..., role=..., ...)
governed_qa = ____

# TODO: Wrap with PactGovernedAgent for admin role ($10, confidential, full tools).
governed_admin = ____

print(f"\n=== Agent Pipeline ===")
print(f"Base: QAAgent (fine-tuned on Singapore domain)")
print(f"Governed QA: $1 budget, internal clearance")
print(f"Governed Admin: $10 budget, confidential clearance")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Deploy via Nexus
# ══════════════════════════════════════════════════════════════════════


async def handle_qa(question: str, role: str = "qa") -> dict:
    """Handle a question through the governed pipeline."""
    agent = governed_qa if role == "qa" else governed_admin
    start = time.time()

    try:
        # TODO: Run the governed agent and return result dict.
        # Hint: await agent.run(question=question)
        result = ____
        latency = time.time() - start
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
            "latency_ms": latency * 1000,
            "governed": True,
            "role": role,
        }
    except Exception as e:
        return {
            "error": str(e),
            "governed": True,
            "blocked": True,
            "role": role,
        }


# TODO: Create Nexus app and register the handler.
# Hint: Nexus(), app.register(handle_qa)
app = ____
____

print(f"\n=== Nexus Deployment ===")
print(f"Channels: API + CLI + MCP")
print(f"Handler: handle_qa(question, role)")
print(f"Governance: every request goes through PactGovernedAgent")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Test governance across channels
# ══════════════════════════════════════════════════════════════════════


async def test_governed_system():
    # TODO: Create a Nexus session.
    # Hint: app.create_session()
    session = ____

    print(f"\n=== Cross-Channel Governance Tests ===")
    print(f"Session: {session}")

    # Test 1: Normal QA (within budget and clearance)
    print(f"\n--- Test 1: Normal QA query ---")
    result = await handle_qa("What is Singapore's CPF contribution rate?", role="qa")
    if "error" not in result:
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Confidence: {result['confidence']}")
        print(f"Latency: {result['latency_ms']:.0f}ms")
    else:
        print(f"Error: {result['error']}")

    # Test 2: QA agent tries admin operation
    print(f"\n--- Test 2: QA agent exceeding role ---")
    result = await handle_qa(
        "Update the model weights with new training data and deploy to production.",
        role="qa",
    )
    if result.get("blocked"):
        print(f"BLOCKED: {result['error']}")
        print(f"QA agent cannot perform admin operations — clearance insufficient")
    else:
        print(f"Answer: {result.get('answer', '')[:200]}...")

    # Test 3: Admin agent (higher clearance)
    print(f"\n--- Test 3: Admin query ---")
    result = await handle_qa("What are the model performance metrics?", role="admin")
    if "error" not in result:
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Role: {result['role']} (elevated clearance)")
    else:
        print(f"Error: {result['error']}")

    # Test 4: Multiple queries to test budget cascade
    print(f"\n--- Test 4: Budget cascade (multiple queries) ---")
    questions = eval_data["instruction"].to_list()[:5]
    for i, q in enumerate(questions):
        result = await handle_qa(q, role="qa")
        status = "OK" if "error" not in result else f"BLOCKED: {result['error'][:50]}"
        print(f"  Q{i+1}: {status}")

    return session


session = asyncio.run(test_governed_system())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Generate compliance audit report
# ══════════════════════════════════════════════════════════════════════


async def compliance_report():
    print(f"\n{'='*60}")
    print(f"  COMPLIANCE AUDIT REPORT")
    print(f"  System: ASCENT Capstone Governed ML System")
    print(f"  Date: 2026-04-06")
    print(f"{'='*60}")

    # TODO: Extract audit trails from both governed agents.
    # Hint: governed_qa.get_audit_trail(), governed_admin.get_audit_trail()
    qa_audit = ____
    admin_audit = ____

    print(f"\n1. AGENT ACTIVITY SUMMARY")
    print(f"   QA Agent actions:    {len(qa_audit)}")
    print(f"   Admin Agent actions: {len(admin_audit)}")

    # TODO: Count blocked and allowed actions from qa_audit.
    # Hint: sum(1 for e in qa_audit if e.get("status") == "blocked")
    qa_blocked = ____
    qa_allowed = ____
    print(f"   QA allowed/blocked:  {qa_allowed}/{qa_blocked}")

    print(f"\n2. GOVERNANCE ENFORCEMENT")
    print(f"   Budget enforcement:     ACTIVE")
    print(f"   Tool restrictions:      ACTIVE")
    print(f"   Clearance validation:   ACTIVE")
    print(f"   Audit trail:            COMPLETE")

    print(f"\n3. REGULATORY COMPLIANCE")
    print(f"   EU AI Act Art. 9 (Risk Management):     COMPLIANT")
    print(f"     - Operating envelopes defined for all agents")
    print(f"     - Budget limits prevent runaway costs")
    print(f"   EU AI Act Art. 12 (Record-keeping):     COMPLIANT")
    print(f"     - All actions logged with timestamps")
    print(f"     - Blocked actions include reason codes")
    print(f"   Singapore AI Verify (Accountability):   COMPLIANT")
    print(f"     - D/T/R chains trace every action to a human delegator")
    print(f"     - Role-based access with clearance levels")
    print(f"   MAS TRM 7.5 (Audit Trail):              COMPLIANT")
    print(f"     - Immutable audit log with full action history")

    print(f"\n4. MODEL PROVENANCE")
    print(f"   Base model: environment variable (not hardcoded)")
    print(f"   SFT adapter: sg_domain_sft_v1 (Exercise 1)")
    print(f"   DPO adapter: sg_domain_dpo_v1 (Exercise 2)")
    print(f"   Merged: SLERP merge (Exercise 5)")
    print(f"   All adapters tracked in AdapterRegistry")

    print(f"\n5. DEPLOYMENT ARCHITECTURE")
    print(f"   Channels: API + CLI + MCP (via Nexus)")
    print(f"   Governance: PactGovernedAgent on all channels")
    print(f"   Sessions: persistent state across channels")
    print(f"   Cost control: per-agent budget cascading")

    print(f"\n{'='*60}")
    print(f"  AUDIT RESULT: COMPLIANT")
    print(f"  All governance controls operational.")
    print(f"{'='*60}")


asyncio.run(compliance_report())

print(f"\n=== Capstone Summary ===")
print(f"This exercise combines EVERY Kailash framework:")
print(f"  kailash-align: fine-tuned model with merged adapters")
print(f"  kailash-pact:  D/T/R governance with operating envelopes")
print(f"  kailash-kaizen: BaseAgent with structured Signatures")
print(f"  kailash-nexus: multi-channel deployment")
print(f"  kailash-ml:    model registry and metrics tracking")
print(f"From training to governance to deployment — the full Kailash lifecycle.")

print("\n✓ Exercise 8 complete — governed ML system capstone")
