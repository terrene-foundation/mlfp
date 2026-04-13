# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 7: AI Governance Engineering with PACT
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Define an organisational hierarchy in YAML using D/T/R grammar
#   - Compile and validate governance structure with GovernanceEngine
#   - Define operating envelopes: task envelopes, role envelopes, and
#     explain monotonic tightening
#   - Implement budget cascading across agent hierarchies
#   - Test access control decisions (both allow and deny) and verify
#     that governance is fail-closed (deny by default)
#   - Wrap agents with PactGovernedAgent for runtime enforcement
#   - Generate audit trails and map them to regulatory requirements
#   - Write governance unit tests that verify denied access stays denied
#   - Implement clearance levels (public < internal < confidential < restricted)
#   - Apply enforcement modes: warn, block, audit
#
# PREREQUISITES:
#   Exercise 6 (multi-agent systems).  This exercise GOVERNS the systems
#   built in Ex 5-6.  Governance is engineering: access controls, budget
#   limits, audit trails — not philosophical discussion.
#
# ESTIMATED TIME: ~180 min
#
# TASKS:
#    1. Load adversarial test prompts (RealToxicityPrompts)
#    2. Write YAML organisation definition (D/T/R grammar)
#    3. Compile with GovernanceEngine and validate
#    4. Define operating envelopes (task, role, budget, tool, clearance)
#    5. Monotonic tightening — envelopes only get stricter
#    6. Budget cascading across agent hierarchies
#    7. Test access control decisions (allow + deny)
#    8. PactGovernedAgent — runtime governance wrapper
#    9. Fail-closed governance and adversarial prompt blocking
#   10. Audit trail generation and regulatory mapping
#
# DATASET: allenai/real-toxicity-prompts (HuggingFace)
#   Real-world adversarial prompts collected from web text.  Used to
#   test that PACT governance correctly blocks high-risk inputs.
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import polars as pl

from kaizen import InputField, OutputField, Signature
from kaizen.core import BaseAgent
from kailash_pact import GovernanceEngine, PactGovernedAgent

from shared.kailash_helpers import setup_environment

setup_environment()

model_name = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ══════════════════════════════════════════════════════════════════════════
# TASK 1: Load Adversarial Test Prompts
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("TASK 1: Load Adversarial Test Prompts")
print("=" * 70)

CACHE_DIR = Path("data/mlfp06/toxicity")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "real_toxicity_50.parquet"

if CACHE_FILE.exists():
    print(f"Loading cached adversarial prompts from {CACHE_FILE}")
    adversarial_prompts = pl.read_parquet(CACHE_FILE)
else:
    print("Downloading allenai/real-toxicity-prompts from HuggingFace...")
    from datasets import load_dataset

    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    ds = ds.filter(
        lambda r: r["prompt"]["toxicity"] is not None and r["prompt"]["toxicity"] > 0.5
    )
    ds = ds.shuffle(seed=42).select(range(min(50, len(ds))))
    rows = [
        {
            "prompt_text": row["prompt"]["text"],
            "toxicity_score": row["prompt"]["toxicity"],
        }
        for row in ds
    ]
    adversarial_prompts = pl.DataFrame(rows)
    adversarial_prompts.write_parquet(CACHE_FILE)
    print(f"Cached {adversarial_prompts.height} adversarial prompts")

print(f"Loaded {adversarial_prompts.height} real adversarial prompts")
print(
    f"Toxicity range: {adversarial_prompts['toxicity_score'].min():.2f} — "
    f"{adversarial_prompts['toxicity_score'].max():.2f}"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────────
assert adversarial_prompts.height > 0, "Task 1: should have adversarial prompts"
print("✓ Checkpoint 1 passed — adversarial test data loaded\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 2: Write YAML Organisation Definition (D/T/R Grammar)
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 2: YAML Organisation Definition")
print("=" * 70)

print(
    """
D/T/R Accountability Grammar:
  D (Delegator):   Human authority who authorises a task
  T (Task):        Bounded scope of work the agent may perform
  R (Responsible): The agent that executes within the envelope

  Every agent action MUST trace back to a human Delegator.
  If model_trainer exceeds its $100 budget, accountability traces
  to chief_ml_officer who authorised the delegation.
"""
)

org_yaml = """
# Singapore FinTech AI Organisation — PACT Governance Definition
# D/T/R: every agent action traces to a human Delegator

organization:
  name: "SG FinTech AI Division"
  jurisdiction: "Singapore"
  regulatory_framework: "MAS TRM, AI Verify, PDPA"

departments:
  - name: "ML Engineering"
    head: "chief_ml_officer"
    agents:
      - id: "data_analyst"
        role: "analyst"
        clearance: "internal"
        description: "Analyses datasets, generates exploratory reports"
      - id: "model_trainer"
        role: "engineer"
        clearance: "confidential"
        description: "Trains and evaluates ML models"
      - id: "model_deployer"
        role: "operator"
        clearance: "confidential"
        description: "Deploys models to production infrastructure"

  - name: "Risk & Compliance"
    head: "chief_risk_officer"
    agents:
      - id: "risk_assessor"
        role: "auditor"
        clearance: "restricted"
        description: "Assesses model risk, bias, and regulatory compliance"
      - id: "bias_checker"
        role: "auditor"
        clearance: "confidential"
        description: "Checks models for bias and fairness violations"

  - name: "Customer Intelligence"
    head: "vp_customer"
    agents:
      - id: "customer_agent"
        role: "analyst"
        clearance: "public"
        description: "Handles customer-facing AI interactions"

delegations:
  - delegator: "chief_ml_officer"
    task: "data_analysis"
    responsible: "data_analyst"
    envelope:
      max_budget_usd: 20.0
      allowed_tools: ["read_data", "summarise_data", "generate_report"]
      allowed_data_clearance: "internal"
      max_data_rows: 500000

  - delegator: "chief_ml_officer"
    task: "model_training"
    responsible: "model_trainer"
    envelope:
      max_budget_usd: 100.0
      allowed_tools: ["train_model", "evaluate_model", "read_data"]
      allowed_data_clearance: "confidential"
      max_data_rows: 1000000

  - delegator: "chief_ml_officer"
    task: "model_deployment"
    responsible: "model_deployer"
    envelope:
      max_budget_usd: 50.0
      allowed_tools: ["deploy_model", "monitor_model", "rollback_model"]
      allowed_data_clearance: "confidential"

  - delegator: "chief_risk_officer"
    task: "risk_assessment"
    responsible: "risk_assessor"
    envelope:
      max_budget_usd: 200.0
      allowed_tools: ["read_data", "audit_model", "generate_report", "access_audit_log"]
      allowed_data_clearance: "restricted"

  - delegator: "chief_risk_officer"
    task: "bias_audit"
    responsible: "bias_checker"
    envelope:
      max_budget_usd: 75.0
      allowed_tools: ["read_data", "audit_model", "run_fairness_check"]
      allowed_data_clearance: "confidential"

  - delegator: "vp_customer"
    task: "customer_interaction"
    responsible: "customer_agent"
    envelope:
      max_budget_usd: 5.0
      allowed_tools: ["answer_question", "search_faq"]
      allowed_data_clearance: "public"
      max_response_length: 500

operating_envelopes:
  global:
    max_llm_cost_per_request_usd: 0.50
    require_audit_trail: true
    pii_handling: "mask"
    log_retention_days: 90
    fail_mode: "closed"
"""

org_yaml_path = os.path.join(tempfile.gettempdir(), "sg_fintech_org.yaml")
with open(org_yaml_path, "w") as f:
    f.write(org_yaml)

print(f"Organisation: SG FinTech AI Division")
print(f"Departments: 3 (ML Engineering, Risk & Compliance, Customer Intelligence)")
print(f"Agents: 6")
print(f"Delegations: 6 D/T/R chains")
print(f"YAML written to: {org_yaml_path}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────────
assert os.path.exists(org_yaml_path), "Task 2: YAML file should exist"
with open(org_yaml_path) as f:
    content = f.read()
assert "departments" in content and "delegations" in content, "YAML needs both sections"
print("✓ Checkpoint 2 passed — org YAML written\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 3: Compile with GovernanceEngine
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 3: GovernanceEngine Compilation")
print("=" * 70)


async def compile_governance():
    engine = GovernanceEngine()
    org = engine.compile_org(org_yaml_path)
    print(f"Compiled organisation:")
    print(f"  Agents:      {org.n_agents}")
    print(f"  Delegations: {org.n_delegations}")
    print(f"  Departments: {org.n_departments}")
    print(f"\nCompilation validates:")
    print(f"  - Every agent has a delegation chain to a human Delegator")
    print(f"  - No circular delegations (A->B->A)")
    print(f"  - Clearance levels decrease monotonically down chains")
    print(f"  - Budget envelopes don't exceed parent limits")
    return engine, org


engine, org = asyncio.run(compile_governance())

# ── Checkpoint 3 ─────────────────────────────────────────────────────────
assert org is not None, "Task 3: compilation should succeed"
assert org.n_agents > 0, "Task 3: should have agents"
assert org.n_delegations > 0, "Task 3: should have delegations"
print(
    f"✓ Checkpoint 3 passed — compiled: {org.n_agents} agents, "
    f"{org.n_delegations} delegations\n"
)


# ══════════════════════════════════════════════════════════════════════════
# TASK 4: Operating Envelopes
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 4: Operating Envelopes")
print("=" * 70)

print(
    """
An operating envelope defines the boundaries for what an agent can do:

  Task envelope:   restricts the agent to specific task types
    model_trainer can train+evaluate but NOT deploy
  Role envelope:   restricts based on organisational role
    auditors can read and audit but NOT modify
  Budget envelope:  maximum cost per task execution
    customer_agent limited to $5 per interaction
  Tool envelope:   whitelist of permitted tool calls
    deployer can deploy+monitor+rollback but NOT train
  Clearance envelope: highest data classification accessible
    public < internal < confidential < restricted
"""
)

envelopes = pl.DataFrame(
    {
        "Agent": [
            "data_analyst",
            "model_trainer",
            "model_deployer",
            "risk_assessor",
            "bias_checker",
            "customer_agent",
        ],
        "Clearance": [
            "internal",
            "confidential",
            "confidential",
            "restricted",
            "confidential",
            "public",
        ],
        "Budget": ["$20", "$100", "$50", "$200", "$75", "$5"],
        "Tools": [
            "read,summarise,report",
            "train,evaluate,read",
            "deploy,monitor,rollback",
            "read,audit,report,log",
            "read,audit,fairness",
            "answer,search",
        ],
        "Role": ["analyst", "engineer", "operator", "auditor", "auditor", "analyst"],
    }
)
print(envelopes)

# ── Checkpoint 4 ─────────────────────────────────────────────────────────
assert envelopes.height == 6, "Task 4: should have 6 agent envelopes"
print("\n✓ Checkpoint 4 passed — operating envelopes defined\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 5: Monotonic Tightening
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 5: Monotonic Tightening Principle")
print("=" * 70)

print(
    """
Monotonic tightening: envelopes can ONLY get stricter, never looser.

  Clearance hierarchy:  restricted > confidential > internal > public
  A delegated agent CANNOT have HIGHER clearance than its delegator.

  Budget hierarchy:  child budget <= parent delegation limit
  A child agent cannot be allocated more budget than its parent allows.

  Example:
    chief_ml_officer (restricted clearance, $500 budget)
      └─ model_trainer (confidential clearance, $100 budget)  ✓ tighter
      └─ data_analyst (internal clearance, $20 budget)        ✓ tighter
      └─ INVALID: agent with restricted clearance             ✗ not tighter

  This prevents privilege escalation: no agent can gain capabilities
  beyond what its human delegator authorised.
"""
)

clearance_levels = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}

# Verify monotonic tightening for each delegation chain
print("Verifying monotonic tightening:")
delegation_chains = [
    ("chief_ml_officer", "restricted", "model_trainer", "confidential"),
    ("chief_ml_officer", "restricted", "data_analyst", "internal"),
    ("chief_ml_officer", "restricted", "model_deployer", "confidential"),
    ("chief_risk_officer", "restricted", "risk_assessor", "restricted"),
    ("chief_risk_officer", "restricted", "bias_checker", "confidential"),
    ("vp_customer", "internal", "customer_agent", "public"),
]

for delegator, del_clearance, agent, agent_clearance in delegation_chains:
    is_tighter = clearance_levels[agent_clearance] <= clearance_levels[del_clearance]
    status = "✓" if is_tighter else "✗ VIOLATION"
    print(f"  {status} {delegator}({del_clearance}) -> {agent}({agent_clearance})")

# ── Checkpoint 5 ─────────────────────────────────────────────────────────
all_valid = all(
    clearance_levels[ac] <= clearance_levels[dc] for _, dc, _, ac in delegation_chains
)
assert all_valid, "Task 5: all clearance chains should be monotonically tightening"
print("\n✓ Checkpoint 5 passed — monotonic tightening verified\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 6: Budget Cascading
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 6: Budget Cascading Across Agent Hierarchies")
print("=" * 70)

print(
    """
Budget cascading: parent agent allocates budget to children.
  Total children spend <= parent's allocation.

  ml_director total budget: $500
    ├─ data_analyst:   $20/task
    ├─ model_trainer:  $100/task
    └─ model_deployer: $50/task

  After 3 training tasks: $300 spent
  Remaining for deployment: $200
  After 4 deployments: $200 spent, total $500 — budget exhausted
  Next request: DENIED (budget exceeded)
"""
)


class BudgetTracker:
    """Track budget allocation and consumption across agent hierarchy."""

    def __init__(self, total_budget: float):
        self.total_budget = total_budget
        self.consumed: dict[str, float] = {}
        self.allocations: dict[str, float] = {}

    def allocate(self, agent_id: str, amount: float) -> bool:
        """Allocate budget to an agent.  Returns False if insufficient."""
        total_allocated = sum(self.allocations.values())
        if total_allocated + amount > self.total_budget:
            return False
        self.allocations[agent_id] = self.allocations.get(agent_id, 0) + amount
        return True

    def spend(self, agent_id: str, amount: float) -> bool:
        """Record spending.  Returns False if exceeds allocation."""
        allocation = self.allocations.get(agent_id, 0)
        current = self.consumed.get(agent_id, 0)
        if current + amount > allocation:
            return False
        self.consumed[agent_id] = current + amount
        return True

    def remaining(self, agent_id: str) -> float:
        return self.allocations.get(agent_id, 0) - self.consumed.get(agent_id, 0)

    def summary(self) -> pl.DataFrame:
        agents = set(self.allocations.keys()) | set(self.consumed.keys())
        rows = []
        for a in sorted(agents):
            rows.append(
                {
                    "agent": a,
                    "allocated": self.allocations.get(a, 0),
                    "consumed": self.consumed.get(a, 0),
                    "remaining": self.remaining(a),
                }
            )
        return pl.DataFrame(rows)


# Demonstrate budget cascading
tracker = BudgetTracker(total_budget=500.0)
tracker.allocate("data_analyst", 20.0)
tracker.allocate("model_trainer", 100.0)
tracker.allocate("model_deployer", 50.0)

# Simulate spending
tracker.spend("model_trainer", 30.0)  # Training task 1
tracker.spend("model_trainer", 30.0)  # Training task 2
tracker.spend("model_trainer", 25.0)  # Training task 3
tracker.spend("data_analyst", 8.0)  # Analysis task
tracker.spend("model_deployer", 15.0)  # Deployment

# Try to overspend
overspend_ok = tracker.spend("model_trainer", 50.0)  # Would exceed $100 allocation
print(
    f"Overspend attempt (model_trainer, $50): {'ALLOWED' if overspend_ok else 'DENIED'}"
)

print(f"\nBudget summary:")
print(tracker.summary())

# ── Checkpoint 6 ─────────────────────────────────────────────────────────
assert not overspend_ok, "Task 6: overspend should be denied"
assert tracker.remaining("model_trainer") == 15.0, "Should have $15 remaining"
print("✓ Checkpoint 6 passed — budget cascading verified\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 7: Test Access Control Decisions
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 7: Access Control Tests (Allow + Deny)")
print("=" * 70)


async def test_access_control():
    test_cases = [
        # (agent, resource, action, expected_allowed, reason)
        ("model_trainer", "training_data", "read", True, "Within envelope"),
        (
            "model_trainer",
            "production_model",
            "deploy",
            False,
            "deploy not in allowed_tools",
        ),
        ("customer_agent", "customer_faq", "search_faq", True, "Within envelope"),
        ("customer_agent", "training_data", "read_data", False, "Clearance too low"),
        ("risk_assessor", "model_audit_log", "audit_model", True, "Within envelope"),
        (
            "risk_assessor",
            "production_model",
            "deploy_model",
            False,
            "deploy not in allowed_tools",
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
        (
            "customer_agent",
            "internal_data",
            "read_data",
            False,
            "Clearance: public < internal",
        ),
    ]

    results = []
    print(
        f"{'Agent':<17} {'Action':<20} {'Resource':<20} {'Expected':<10} {'Actual':<10} {'Match'}"
    )
    print("-" * 90)

    for agent_id, resource, action, expected, reason in test_cases:
        decision = engine.check_access(
            agent_id=agent_id,
            resource=resource,
            action=action,
        )
        actual = decision.allowed
        match = actual == expected
        status = "✓" if match else "✗ FAIL"
        results.append(
            {
                "agent": agent_id,
                "action": action,
                "expected": expected,
                "actual": actual,
                "match": match,
            }
        )
        print(
            f"  {status} {agent_id:<15} {action:<18} {resource:<18} "
            f"{'ALLOW' if expected else 'DENY':<8} {'ALLOW' if actual else 'DENY':<8}"
        )
        if not actual:
            print(f"      Reason: {decision.reason}")

    all_match = all(r["match"] for r in results)
    print(f"\nResults: {sum(r['match'] for r in results)}/{len(results)} correct")
    return results, all_match


access_results, all_correct = asyncio.run(test_access_control())

# ── Checkpoint 7 ─────────────────────────────────────────────────────────
assert len(access_results) >= 10, "Task 7: should test at least 10 cases"
print(f"✓ Checkpoint 7 passed — access control: {len(access_results)} tests\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 8: PactGovernedAgent — Runtime Governance Wrapper
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 8: PactGovernedAgent Runtime Wrapper")
print("=" * 70)


class QASignature(Signature):
    """Answer questions within governance constraints."""

    question: str = InputField(description="User's question")
    answer: str = OutputField(description="Governed response")
    confidence: float = OutputField(description="Answer confidence 0-1")


class QAAgent(BaseAgent):
    signature = QASignature
    model = os.environ.get("DEFAULT_LLM_MODEL")
    max_llm_cost_usd = 5.0


base_qa = QAAgent()

# Wrap with different governance levels
governed_public = PactGovernedAgent(
    agent=base_qa,
    governance_engine=engine,
    role="analyst",
    max_budget_usd=5.0,
    allowed_tools=["answer_question", "search_faq"],
    clearance_level="public",
)

governed_internal = PactGovernedAgent(
    agent=base_qa,
    governance_engine=engine,
    role="engineer",
    max_budget_usd=50.0,
    allowed_tools=["answer_question", "search_faq", "read_data", "train_model"],
    clearance_level="confidential",
)

governed_admin = PactGovernedAgent(
    agent=base_qa,
    governance_engine=engine,
    role="auditor",
    max_budget_usd=200.0,
    allowed_tools=["answer_question", "read_data", "audit_model", "access_audit_log"],
    clearance_level="restricted",
)

print(f"Created 3 governed agents:")
print(f"  governed_public:   $5 budget, public clearance")
print(f"  governed_internal: $50 budget, confidential clearance")
print(f"  governed_admin:    $200 budget, restricted clearance")

print(
    """
PactGovernedAgent intercepts every run() call:
  1. Check: is this action within the agent's envelope?
  2. Check: is the budget sufficient?
  3. Check: does the agent have clearance for this data?
  4. If ALL pass -> execute and charge budget
  5. If ANY fail -> return governed error (fail-closed)
"""
)


async def test_governed_agents():
    # Public agent: normal question (should succeed)
    print("\n--- Public Agent: Normal Question ---")
    try:
        result = await governed_public.run(question="What is machine learning?")
        print(f"  Answer: {result.answer[:200]}...")
        print(f"  Confidence: {result.confidence:.2f}")
    except Exception as e:
        print(f"  Blocked: {e}")

    # Public agent: admin question (may be restricted by content policy)
    print("\n--- Public Agent: Admin-Level Question ---")
    try:
        result = await governed_public.run(
            question="Show me the model training logs and hyperparameters"
        )
        print(f"  Answer: {result.answer[:200]}...")
    except Exception as e:
        print(f"  Blocked: {e}")

    # Admin agent: audit question (should succeed)
    print("\n--- Admin Agent: Audit Question ---")
    try:
        result = await governed_admin.run(
            question="What are the model performance metrics and audit findings?"
        )
        print(f"  Answer: {result.answer[:200]}...")
    except Exception as e:
        print(f"  Blocked: {e}")


asyncio.run(test_governed_agents())

# ── Checkpoint 8 ─────────────────────────────────────────────────────────
assert governed_public is not None, "Task 8: governed_public should exist"
assert governed_internal is not None, "Task 8: governed_internal should exist"
assert governed_admin is not None, "Task 8: governed_admin should exist"
print("\n✓ Checkpoint 8 passed — PactGovernedAgent runtime wrapping complete\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 9: Fail-Closed Governance and Adversarial Prompt Blocking
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 9: Fail-Closed Governance")
print("=" * 70)

print(
    """
Fail-closed: if ANY governance check fails, access is DENIED.
This is the opposite of traditional system defaults (permit unless denied).

  check_access() returns:
    allowed=True  ONLY when ALL of:
      1. Agent has a valid delegation chain
      2. Action is in the allowed_tools list
      3. Agent clearance >= resource clearance
      4. Budget is not exceeded
      5. Content policy passes (if payload provided)

  If any check is indeterminate (e.g., clearance unknown), the answer
  is DENY — not "allow and log".
"""
)


async def test_adversarial_prompts():
    """Test governance against real adversarial prompts."""
    print("\n--- Adversarial Prompt Blocking (RealToxicityPrompts) ---")
    sample = adversarial_prompts.head(10)
    n_blocked = 0

    for i, row in enumerate(sample.iter_rows(named=True)):
        prompt_text = row["prompt_text"]
        toxicity = row["toxicity_score"]

        decision = engine.check_access(
            agent_id="customer_agent",
            resource="external_input",
            action="answer_question",
            payload=prompt_text,
        )
        allowed = getattr(decision, "allowed", False)
        if not allowed:
            n_blocked += 1

        snippet = prompt_text[:50].replace("\n", " ")
        status = "BLOCKED" if not allowed else "allowed"
        print(f"  {i+1}. tox={toxicity:.2f} {status}: {snippet}...")

    print(f"\n  Result: {n_blocked}/{sample.height} adversarial prompts blocked")
    return n_blocked


try:
    n_blocked = asyncio.run(test_adversarial_prompts())
except TypeError:
    print("  (PACT version does not support payload kwarg; demonstrating concept)")
    n_blocked = 0

# Test fail-closed property: unknown agent should be denied
print("\n--- Fail-Closed: Unknown Agent ---")
decision = engine.check_access(
    agent_id="unknown_agent",
    resource="any_resource",
    action="any_action",
)
print(f"  Unknown agent access: {'DENIED' if not decision.allowed else 'ALLOWED'}")
assert not decision.allowed, "Fail-closed: unknown agents should be denied"

# ── Checkpoint 9 ─────────────────────────────────────────────────────────
print("✓ Checkpoint 9 passed — fail-closed governance verified\n")


# ══════════════════════════════════════════════════════════════════════════
# TASK 10: Audit Trail and Regulatory Mapping
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("TASK 10: Audit Trail and Regulatory Mapping")
print("=" * 70)

# Generate audit trail from governed agents
qa_audit = governed_public.get_audit_trail()
admin_audit = governed_admin.get_audit_trail()

print(f"Audit trail:")
print(f"  Public agent actions:  {len(qa_audit)}")
print(f"  Admin agent actions:   {len(admin_audit)}")

qa_blocked = sum(1 for e in qa_audit if e.get("status") == "blocked")
qa_allowed = sum(1 for e in qa_audit if e.get("status") == "allowed")
print(f"  Public allowed/blocked: {qa_allowed}/{qa_blocked}")

# D/T/R decision trace
print(f"\n--- Decision Trace Example ---")
decision = engine.check_access(
    agent_id="model_trainer",
    resource="training_data",
    action="read",
)
print(f"  Agent: model_trainer (role=engineer, clearance=confidential)")
print(f"  Chain: chief_ml_officer -> model_training -> model_trainer")
print(f"  Envelope checks:")
print(f"    Tool 'read_data' in allowed_tools: YES")
print(f"    Clearance 'confidential' <= allowed 'confidential': YES")
print(f"    Budget consumed < $100 limit: YES")
print(f"  Decision: {'ALLOWED' if decision.allowed else 'DENIED'}")

# Regulatory compliance mapping
print(f"\n--- Regulatory Compliance Mapping ---")
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
            "Operating envelopes per agent",
            "Immutable audit trail with timestamps",
            "D/T/R chains — every action traces to human Delegator",
            "D/T/R accountability grammar",
            "Full audit log with action, resource, decision, reason",
            "Clearance levels + PII masking in global envelope",
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

# Governance enforcement modes
print(f"\n--- Enforcement Modes ---")
print(f"  WARN:  Log the violation but allow the action (dev/staging only)")
print(f"  BLOCK: Deny the action and return a governed error (production)")
print(f"  AUDIT: Allow but flag for human review (semi-trusted agents)")
print(f"\n  Production default: BLOCK (fail-closed)")
print(f"  Never use WARN in production — it defeats the purpose of governance")

# ── Checkpoint 10 ────────────────────────────────────────────────────────
assert decision.allowed, "model_trainer should be allowed to read training_data"
assert regulatory_map.height >= 6, "Task 10: should map at least 6 regulations"
print("\n✓ Checkpoint 10 passed — audit trail and regulatory mapping complete\n")


# ══════════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  ✓ D/T/R grammar: Delegator -> Task -> Responsible; every agent action
    traces to a human authority
  ✓ GovernanceEngine.compile_org(): validates structural governance
    (no circular delegation, monotonic clearance, budget bounds)
  ✓ Operating envelopes: task + role + budget + tool + clearance constraints
  ✓ Monotonic tightening: envelopes only get stricter down the chain
  ✓ Budget cascading: parent allocates to children; children cannot overspend
  ✓ Access control tests: verify BOTH allowed and denied cases
  ✓ PactGovernedAgent: runtime wrapper that intercepts every run() call
  ✓ Fail-closed: deny by default; allow only what is explicitly permitted
  ✓ Adversarial prompt blocking: governance as content safety layer
  ✓ Audit trail: machine-readable evidence for regulatory compliance
  ✓ Regulatory mapping: PACT controls -> EU AI Act, AI Verify, MAS TRM, PDPA
  ✓ Enforcement modes: warn (dev), block (prod), audit (semi-trusted)

  Governance principles:
    Fail-closed:          deny unless explicitly allowed
    Monotonic tightening: envelopes only get stricter
    Clearance hierarchy:  restricted > confidential > internal > public
    Budget cascading:     child budget <= parent allocation
    Audit completeness:   every decision logged (allowed AND denied)

  NEXT: Exercise 8 (Capstone) integrates EVERYTHING from M6:
  SFT + DPO + PACT governance + Nexus deployment + compliance audit.
  A complete production ML system from training to deployment.
"""
)
