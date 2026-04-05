# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 7: Agent Governance at Scale
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Govern a multi-agent ML pipeline using PACT clearance levels,
#   budget cascading, and dereliction handling. Demonstrate how governance
#   properties propagate from an org-level policy down through a supervisor
#   to worker agents — and what happens when agents violate constraints.
#
# TASKS:
#   1. Define a multi-tier organisation with clearance levels
#   2. Build a multi-agent pipeline (supervisor + 3 workers)
#   3. Demonstrate budget cascading (child <= parent at every level)
#   4. Test dereliction scenarios: what happens when agents fail or overspend
#   5. AuditChain: verify tamper-evident logging of every governance event
#   6. Scale governance: apply policy to 10 agents in a single org
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import polars as pl

from kailash_pact import GovernanceEngine, compile_org, Address
from kailash_pact import GovernanceContext, RoleEnvelope, TaskEnvelope
from kailash_pact import PactGovernedAgent, AuditChain
from kailash_pact.governance import (
    ClearanceLevel,
    BudgetCascade,
    DerelictionHandler,
    DerelictionPolicy,
    GovernanceViolation,
)
from kaizen_agents import Delegate, SupervisorWorkerPattern

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()

model = os.environ.get("DEFAULT_LLM_MODEL", os.environ.get("OPENAI_PROD_MODEL"))


# ── Data Context ──────────────────────────────────────────────────────

loader = ASCENTDataLoader()
credit = loader.load("ascent03", "sg_credit_scoring.parquet")

data_context = (
    f"Singapore Credit Scoring: {credit.height:,} rows, "
    f"{len(credit.columns)} columns, default rate "
    f"{credit['default'].mean():.1%}"
)

print(f"=== Agent Governance at Scale ===")
print(f"Dataset: {data_context}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Multi-tier organisation with clearance levels
# ══════════════════════════════════════════════════════════════════════
#
# Clearance levels add a security layer on top of RBAC permissions.
# A clearance level is a numeric tier (0 = public, 4 = top-secret).
# Resources are tagged with a minimum required clearance.
# An agent can only access a resource if its clearance >= resource level.
#
# Clearance hierarchy used here:
#   0 — PUBLIC     (anyone)
#   1 — INTERNAL   (employees)
#   2 — SENSITIVE  (approved analysts)
#   3 — RESTRICTED (senior staff + ML engineers)
#   4 — CRITICAL   (executives + model risk only)

# TODO: Write the org YAML string defining:
#   - 5 clearance levels (0=PUBLIC ... 4=CRITICAL)
#   - 6 data_classifications (credit_applications=3, model_predictions=2,
#       audit_logs=4, feature_store=2, production_logs=1, public_reports=0)
#   - department "ml_pipeline" with teams "orchestration" and "workers":
#       pipeline_supervisor: clearance=3, tools=[profile_data, describe_column,
#           target_analysis, train_model, register_model], max_cost_usd=10.0
#       data_quality_worker: clearance=2, tools=[profile_data, describe_column], max_cost_usd=2.0
#       feature_worker:      clearance=2, tools=[describe_column, target_analysis], max_cost_usd=2.0
#       model_worker:        clearance=3, tools=[train_model, register_model], max_cost_usd=3.0
#   - department "risk_governance" / team "oversight":
#       risk_officer: clearance=4, tools=[read_audit_chain], max_cost_usd=1.0,
#                     can_approve_production=true
org_yaml_content = ____  # Hint: multiline YAML string — see solution for full structure

import tempfile
from pathlib import Path

org_file = Path(tempfile.mktemp(suffix=".yaml"))
org_file.write_text(org_yaml_content)

from kailash_pact import load_org_yaml

# TODO: Load and compile the org. load_org_yaml returns an org object;
#   compile_org(org) returns a compiled result.
org = ____  # Hint: load_org_yaml(str(org_file))
compiled = ____  # Hint: compile_org(org)

print(f"\n=== Organisation Compiled ===")
print(f"Valid: {compiled.valid}")
if compiled.errors:
    print(f"Errors: {compiled.errors}")
print(f"Departments: {[d.name for d in compiled.departments]}")
print(f"Total roles: {compiled.total_roles}")
print(f"Clearance levels: 0 (PUBLIC) -> 4 (CRITICAL)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build governed multi-agent pipeline
# ══════════════════════════════════════════════════════════════════════


async def build_governed_pipeline():
    """Create GovernanceEngine and wrap agents with governance contexts."""
    # TODO: Instantiate GovernanceEngine from the compiled org.
    engine = ____  # Hint: GovernanceEngine(compiled)

    # TODO: Define Address objects for each role.
    #   Address(department, team, role)
    supervisor_addr = (
        ____  # Hint: Address("ml_pipeline", "orchestration", "pipeline_supervisor")
    )
    dq_worker_addr = (
        ____  # Hint: Address("ml_pipeline", "workers", "data_quality_worker")
    )
    feat_worker_addr = ____  # Hint: Address("ml_pipeline", "workers", "feature_worker")
    model_worker_addr = ____  # Hint: Address("ml_pipeline", "workers", "model_worker")

    # TODO: Create governance contexts for each address using engine.create_context.
    supervisor_ctx = ____  # Hint: await engine.create_context(supervisor_addr)
    dq_ctx = ____  # Hint: await engine.create_context(dq_worker_addr)
    feat_ctx = ____  # Hint: await engine.create_context(feat_worker_addr)
    model_ctx = ____  # Hint: await engine.create_context(model_worker_addr)

    print(f"\n=== Governance Contexts ===")
    for name, ctx in [
        ("supervisor", supervisor_ctx),
        ("dq_worker", dq_ctx),
        ("feat_worker", feat_ctx),
        ("model_worker", model_ctx),
    ]:
        print(
            f"  {name:<18} clearance={ctx.clearance_level}  "
            f"budget=${ctx.max_cost_usd:.2f}  "
            f"tools={ctx.tools_allowed}"
        )

    # TODO: Wrap each base Delegate agent with PactGovernedAgent.
    #   PactGovernedAgent(agent=Delegate(model=model), governance_context=ctx)
    supervisor_agent = ____  # Hint: PactGovernedAgent(agent=Delegate(model=model), governance_context=supervisor_ctx)
    dq_agent = ____  # Hint: PactGovernedAgent(agent=Delegate(model=model), governance_context=dq_ctx)
    feat_agent = ____  # Hint: PactGovernedAgent(agent=Delegate(model=model), governance_context=feat_ctx)
    model_agent = ____  # Hint: PactGovernedAgent(agent=Delegate(model=model), governance_context=model_ctx)

    return engine, {
        "supervisor": (supervisor_agent, supervisor_ctx),
        "dq_worker": (dq_agent, dq_ctx),
        "feat_worker": (feat_agent, feat_ctx),
        "model_worker": (model_agent, model_ctx),
    }


engine, governed_agents = asyncio.run(build_governed_pipeline())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Budget cascading
# ══════════════════════════════════════════════════════════════════════
#
# Budget cascading enforces: child_budget <= parent_budget at every level.
#
# Cascade hierarchy:
#   Org budget ($20) -> Supervisor ($10) -> Workers ($2-3 each)
#
# Rules:
#   1. Supervisor cannot be allocated more than the org allows
#   2. Workers cannot be allocated more than the supervisor's remaining budget
#   3. Supervisor's remaining budget = supervisor_budget - sum(worker_budgets)
#   4. If supervisor is over-budget, PACT raises BudgetViolation (not silently clamps)


async def demonstrate_budget_cascade():
    print(f"\n=== Budget Cascading ===")

    # TODO: Instantiate BudgetCascade with org_total_budget=20.0 and governance_engine.
    cascade = (
        ____  # Hint: BudgetCascade(org_total_budget=20.0, governance_engine=engine)
    )

    supervisor_ctx = governed_agents["supervisor"][1]
    dq_ctx = governed_agents["dq_worker"][1]
    feat_ctx = governed_agents["feat_worker"][1]
    model_ctx = governed_agents["model_worker"][1]

    # TODO: Validate the cascade from supervisor down to workers using cascade.validate.
    #   parent_ctx=supervisor_ctx, child_contexts=[dq_ctx, feat_ctx, model_ctx]
    cascade_report = await cascade.validate(
        parent_ctx=____,  # Hint: supervisor_ctx
        child_contexts=____,  # Hint: [dq_ctx, feat_ctx, model_ctx]
    )

    print(f"Org budget:         ${cascade_report.org_budget:.2f}")
    print(f"Supervisor budget:  ${cascade_report.parent_budget:.2f}")
    print(f"Workers allocated:  ${cascade_report.children_total:.2f}")
    print(f"  dq_worker:        ${dq_ctx.max_cost_usd:.2f}")
    print(f"  feat_worker:      ${feat_ctx.max_cost_usd:.2f}")
    print(f"  model_worker:     ${model_ctx.max_cost_usd:.2f}")
    print(f"Supervisor reserve: ${cascade_report.parent_reserve:.2f}")
    print(f"Cascade valid:      {cascade_report.is_valid}")

    # Attempt to create a child task that exceeds supervisor budget
    print(f"\n--- Attempt: child requests $8 (supervisor only has $10) ---")
    try:
        # TODO: Create a TaskEnvelope with max_cost_usd=8.0 under the supervisor.
        #   TaskEnvelope(parent=RoleEnvelope(address=..., max_cost_usd=supervisor_ctx.max_cost_usd,
        #                                    data_access=list(supervisor_ctx.data_access)),
        #                max_cost_usd=8.0, data_access=["credit_applications"])
        excessive_task = TaskEnvelope(
            parent=RoleEnvelope(
                address=____,  # Hint: Address("ml_pipeline", "orchestration", "pipeline_supervisor")
                max_cost_usd=____,  # Hint: supervisor_ctx.max_cost_usd
                data_access=____,  # Hint: list(supervisor_ctx.data_access)
            ),
            max_cost_usd=____,  # Hint: 8.0
            data_access=____,  # Hint: ["credit_applications"]
        )
        print(
            f"  Granted: ${excessive_task.effective_max_cost_usd:.2f} (within parent)"
        )
    except GovernanceViolation as e:
        print(f"  BLOCKED: {e}")

    # Attempt to create a task exceeding parent by a large margin
    print(f"\n--- Attempt: child requests $50 (supervisor budget is $10) ---")
    try:
        # TODO: Create a TaskEnvelope with max_cost_usd=50.0 under the supervisor.
        #   Expect PACT to clamp or raise GovernanceViolation.
        over_task = TaskEnvelope(
            parent=RoleEnvelope(
                address=____,  # Hint: Address("ml_pipeline", "orchestration", "pipeline_supervisor")
                max_cost_usd=____,  # Hint: supervisor_ctx.max_cost_usd
                data_access=____,  # Hint: list(supervisor_ctx.data_access)
            ),
            max_cost_usd=____,  # Hint: 50.0
            data_access=____,  # Hint: ["credit_applications"]
        )
        print(
            f"  Tightened to: ${over_task.effective_max_cost_usd:.2f} "
            f"(clamped to parent limit)"
        )
    except GovernanceViolation as e:
        print(f"  BLOCKED: {e}")


asyncio.run(demonstrate_budget_cascade())


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Dereliction scenarios
# ══════════════════════════════════════════════════════════════════════
#
# Dereliction = agent fails to fulfil its governance obligation.
# Three dereliction types:
#   OVERSPEND     — agent exceeds its cost budget
#   SCOPE_BREACH  — agent accesses data/tools outside its context
#   TIMEOUT       — agent does not complete within allowed time
#   SILENT_FAIL   — agent returns no output (drops the task)
#
# DerelictionPolicy defines what happens on each type:
#   HALT          — immediately stop this agent, propagate to supervisor
#   QUARANTINE    — mark agent as untrusted, route tasks to backup
#   ALERT_ONLY    — log the violation, continue execution
#   ROLLBACK      — undo any state changes the agent made


async def test_dereliction():
    print(f"\n=== Dereliction Handling ===")

    dq_agent, dq_ctx = governed_agents["dq_worker"]

    # TODO: Define a DerelictionPolicy with:
    #   overspend=HALT, scope_breach=HALT,
    #   timeout=ALERT_ONLY, silent_fail=ROLLBACK
    policy = DerelictionPolicy(
        overspend=____,  # Hint: DerelictionPolicy.Action.HALT
        scope_breach=____,  # Hint: DerelictionPolicy.Action.HALT
        timeout=____,  # Hint: DerelictionPolicy.Action.ALERT_ONLY
        silent_fail=____,  # Hint: DerelictionPolicy.Action.ROLLBACK
    )

    # TODO: Create a DerelictionHandler with governance_engine, policy,
    #   notify_supervisor=True, log_to_audit_chain=True.
    handler = DerelictionHandler(
        governance_engine=____,  # Hint: engine
        policy=____,  # Hint: policy
        notify_supervisor=____,  # Hint: True
        log_to_audit_chain=____,  # Hint: True
    )

    # Scenario A: OVERSPEND
    print(f"\nScenario A: OVERSPEND (budget $2.00, request $5.00)")
    # TODO: Create a GovernanceViolation for OVERSPEND and handle it.
    overspend_violation = GovernanceViolation(
        agent_address=____,  # Hint: Address("ml_pipeline", "workers", "data_quality_worker")
        violation_type=____,  # Hint: "OVERSPEND"
        details=____,  # Hint: {"budget": 2.0, "requested": 5.0}
    )
    action_a = await handler.handle(____)  # Hint: overspend_violation
    print(f"  Dereliction type: OVERSPEND")
    print(f"  Policy action:    {action_a.action}")
    print(f"  Agent halted:     {action_a.agent_halted}")
    print(f"  Supervisor notified: {action_a.supervisor_notified}")
    print(f"  Audit event ID:   {action_a.audit_event_id}")

    # Scenario B: SCOPE_BREACH
    print(
        f"\nScenario B: SCOPE_BREACH (tries to access audit_logs, clearance 4 required)"
    )
    # TODO: Create a GovernanceViolation for SCOPE_BREACH and handle it.
    scope_violation = GovernanceViolation(
        agent_address=____,  # Hint: Address("ml_pipeline", "workers", "data_quality_worker")
        violation_type=____,  # Hint: "SCOPE_BREACH"
        details=____,  # Hint: {"resource": "audit_logs", "required_clearance": 4, "agent_clearance": 2}
    )
    action_b = await handler.handle(____)  # Hint: scope_violation
    print(f"  Dereliction type: SCOPE_BREACH")
    print(f"  Policy action:    {action_b.action}")
    print(f"  Agent halted:     {action_b.agent_halted}")
    print(f"  Audit event ID:   {action_b.audit_event_id}")

    # Scenario C: TIMEOUT
    print(f"\nScenario C: TIMEOUT (non-critical worker, ALERT_ONLY policy)")
    # TODO: Create a GovernanceViolation for TIMEOUT and handle it.
    timeout_violation = GovernanceViolation(
        agent_address=____,  # Hint: Address("ml_pipeline", "workers", "feature_worker")
        violation_type=____,  # Hint: "TIMEOUT"
        details=____,  # Hint: {"timeout_seconds": 30, "elapsed_seconds": 47}
    )
    action_c = await handler.handle(____)  # Hint: timeout_violation
    print(f"  Dereliction type: TIMEOUT")
    print(f"  Policy action:    {action_c.action}")
    print(f"  Agent halted:     {action_c.agent_halted}")
    print(f"  Execution continues: {not action_c.agent_halted}")

    # Scenario D: SILENT_FAIL
    print(f"\nScenario D: SILENT_FAIL (model_worker returns nothing -> rollback)")
    # TODO: Create a GovernanceViolation for SILENT_FAIL and handle it.
    silent_violation = GovernanceViolation(
        agent_address=____,  # Hint: Address("ml_pipeline", "workers", "model_worker")
        violation_type=____,  # Hint: "SILENT_FAIL"
        details=____,  # Hint: {"expected_output": "model_metrics", "actual_output": None}
    )
    action_d = await handler.handle(____)  # Hint: silent_violation
    print(f"  Dereliction type: SILENT_FAIL")
    print(f"  Policy action:    {action_d.action}")
    print(f"  State rolled back: {action_d.state_rolled_back}")

    return handler


handler = asyncio.run(test_dereliction())


# ══════════════════════════════════════════════════════════════════════
# TASK 5: AuditChain — tamper-evident governance log
# ══════════════════════════════════════════════════════════════════════
#
# AuditChain records every governance event in a cryptographic chain.
# Each entry: event_type, agent_address, timestamp, details,
# and a SHA-256 hash of (previous_hash || event_data).
#
# Properties:
#   Tamper-evident: modifying any entry breaks the hash chain
#   Sequential:     events have a total ordering (by sequence number)
#   Durable:        persisted to storage, survives process restart
#   Queryable:      filter by agent, event type, time range


async def inspect_audit_chain():
    print(f"\n=== AuditChain ===")

    # TODO: Instantiate AuditChain with governance_engine=engine.
    audit = ____  # Hint: AuditChain(governance_engine=engine)

    # TODO: Retrieve the last 20 events using audit.get_events(limit=20).
    events = ____  # Hint: await audit.get_events(limit=20)
    print(f"Events logged: {len(events)}")
    print(f"\nEvent log:")
    for event in events:
        print(
            f"  [{event.sequence:04d}] {event.timestamp:%H:%M:%S}  "
            f"{event.event_type:<20} {str(event.agent_address):<45} "
            f"hash={event.hash[:12]}..."
        )

    # TODO: Verify chain integrity using audit.verify_chain().
    integrity = ____  # Hint: await audit.verify_chain()
    print(f"\nChain integrity: {'VALID' if integrity.is_valid else 'BROKEN'}")
    if not integrity.is_valid:
        print(f"  Broken at sequence: {integrity.broken_at}")
        print(f"  Reason: {integrity.reason}")

    # TODO: Filter for DERELICTION events using audit.get_events(event_type=..., limit=10).
    dereliction_events = (
        ____  # Hint: await audit.get_events(event_type="DERELICTION", limit=10)
    )
    print(f"\nDereliction events: {len(dereliction_events)}")
    for e in dereliction_events:
        print(
            f"  {e.event_type} — {e.agent_address} — {e.details.get('violation_type')}"
        )

    return audit


audit = asyncio.run(inspect_audit_chain())


# ══════════════════════════════════════════════════════════════════════
# TASK 6: Scale governance to 10 agents
# ══════════════════════════════════════════════════════════════════════
#
# Production ML pipelines may have 10-50 agents running concurrently.
# GovernanceEngine is thread-safe and designed for high concurrency.
# This task demonstrates applying governance to 10 agents in parallel
# and verifying that budget constraints hold across the fleet.


async def governance_at_scale():
    print(f"\n=== Governance at Scale (10 agents) ===")

    fleet_budget_total = 20.0
    n_workers = 8
    per_worker_budget = fleet_budget_total / (n_workers + 2)

    fleet_contexts = []
    for i in range(n_workers):
        role = "data_quality_worker" if i % 2 == 0 else "feature_worker"
        addr = Address("ml_pipeline", "workers", role)

        # TODO: Create a governance context for the address, then wrap it
        #   in a TaskEnvelope under a supervisor RoleEnvelope.
        #   TaskEnvelope(
        #     parent=RoleEnvelope(address=supervisor_addr, max_cost_usd=per_worker_budget,
        #                         data_access=list(ctx.data_access)),
        #     max_cost_usd=per_worker_budget, data_access=list(ctx.data_access))
        ctx = await engine.create_context(addr)
        task = TaskEnvelope(
            parent=RoleEnvelope(
                address=____,  # Hint: Address("ml_pipeline", "orchestration", "pipeline_supervisor")
                max_cost_usd=____,  # Hint: per_worker_budget
                data_access=____,  # Hint: list(ctx.data_access)
            ),
            max_cost_usd=____,  # Hint: per_worker_budget
            data_access=____,  # Hint: list(ctx.data_access)
        )

        fleet_contexts.append((f"worker_{i:02d}", role, task.effective_max_cost_usd))

    total_allocated = sum(b for _, _, b in fleet_contexts)
    print(f"Fleet size:         {n_workers} workers + 1 supervisor")
    print(f"Org budget:         ${fleet_budget_total:.2f}")
    print(f"Per-worker budget:  ${per_worker_budget:.2f}")
    print(f"Total allocated:    ${total_allocated:.2f}")
    print(f"Budget utilisation: {total_allocated / fleet_budget_total:.0%}")
    print(f"\nFleet:")
    for name, role, budget in fleet_contexts:
        print(f"  {name} ({role:<22}): ${budget:.2f}")

    violations = [
        (name, budget)
        for name, _, budget in fleet_contexts
        if budget > per_worker_budget
    ]
    print(f"\nBudget violations: {len(violations)} (expected 0)")
    assert len(violations) == 0, f"Budget cascade violated: {violations}"
    print(f"All allocations satisfy cascade constraint (child <= parent)")

    # TODO: Run parallel governance access checks for all fleet agents using
    #   asyncio.gather and engine.can_access.
    async def check_agent(name: str, ctx_budget: float) -> bool:
        addr = Address("ml_pipeline", "workers", "data_quality_worker")
        decision = await engine.can_access(
            ____, ____
        )  # Hint: addr, "credit_applications"
        return decision

    checks = await asyncio.gather(
        *[check_agent(name, budget) for name, _, budget in fleet_contexts]
    )

    allowed = sum(checks)
    print(f"\nParallel governance checks: {n_workers} agents")
    print(f"  Allowed:  {allowed}/{n_workers}")
    print(f"  Denied:   {n_workers - allowed}/{n_workers}")


asyncio.run(governance_at_scale())

# Cleanup
org_file.unlink()


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"   GOVERNANCE AT SCALE — KEY PROPERTIES")
print(f"{'=' * 70}")
print(
    """
Clearance Levels:
  -> Resource classification independent of role permissions
  -> Agent clearance must be >= resource classification
  -> Prevents privilege escalation even with correct role

Budget Cascading:
  -> Org -> Supervisor -> Workers: strict monotonic tightening
  -> Supervisor cannot grant more than it has
  -> Workers cannot exceed their parent supervisor's remaining budget
  -> Prevents "budget laundering" through delegation chains

Dereliction Handling:
  -> HALT: safety-critical violations (overspend, scope breach)
  -> ALERT_ONLY: non-critical failures (timeout on low-priority task)
  -> ROLLBACK: partial state consistency (silent fail)
  -> Every dereliction is logged to AuditChain

AuditChain:
  -> Cryptographic hash chain — tamper-evident
  -> Every governance event: access check, dereliction, budget spend
  -> Queryable by agent, event type, time range
  -> The regulatory artefact: proves governance was enforced

Scale:
  -> GovernanceEngine is thread-safe, concurrency-safe
  -> Budget cascade applies across N agents in O(N) time
  -> AuditChain writes are serialised (correctness) but buffered (speed)
"""
)

print(
    "Exercise 7 complete — agent governance at scale with clearance, cascade, dereliction"
)
