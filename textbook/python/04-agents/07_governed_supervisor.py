# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Governed Supervisor
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use GovernedSupervisor for budget-governed multi-agent execution
# LEVEL: Advanced
# PARITY: Equivalent — Rust has orchestration runtime with budget envelopes
# VALIDATES: GovernedSupervisor, SupervisorResult, PACT governance,
#            budget tracking, audit trail, clearance enforcement
#
# Run: uv run python textbook/python/04-agents/07_governed_supervisor.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import os

from kaizen_agents import GovernedSupervisor, SupervisorResult
from kaizen_agents.supervisor import HoldRecord
from kailash.trust import ConfidentialityLevel

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. GovernedSupervisor — progressive disclosure ────────────────────
# GovernedSupervisor hides the 20-concept L3 surface area behind
# three layers:
#
# Layer 1 (simple):   model + budget_usd -> run()
# Layer 2 (configured): tools, data_clearance, warning_threshold
# Layer 3 (advanced):  direct access to governance subsystems
#
# Defaults follow PACT default-deny: empty tools, $1 budget, PUBLIC.

# ── 2. Layer 1: minimal instantiation ─────────────────────────────────
# Just model and budget. Everything else gets sensible defaults.

supervisor = GovernedSupervisor(
    model=model,
    budget_usd=10.0,
)

assert isinstance(supervisor, GovernedSupervisor)
assert supervisor.model == model
assert supervisor.tools == [], "Default-deny: no tools allowed"
assert supervisor.clearance_level == ConfidentialityLevel.PUBLIC

# ── 3. Layer 2: configured supervisor ─────────────────────────────────
# Add tools, clearance, and threshold configuration.

configured = GovernedSupervisor(
    model=model,
    budget_usd=25.0,
    tools=["read_file", "grep", "write_report"],
    data_clearance="restricted",
    warning_threshold=0.70,
    timeout_seconds=600.0,
    max_children=15,
    max_depth=3,
    policy_source="jack.hong@example.com",
)

assert configured.tools == ["read_file", "grep", "write_report"]
assert configured.clearance_level == ConfidentialityLevel.RESTRICTED
assert len(configured.tools) == 3

# ── 4. Budget validation ──────────────────────────────────────────────
# Budget must be finite and non-negative. PACT governance enforces this.

try:
    GovernedSupervisor(model=model, budget_usd=-1.0)
    assert False, "Negative budget should raise ValueError"
except ValueError:
    pass

try:
    GovernedSupervisor(model=model, budget_usd=float("inf"))
    assert False, "Infinite budget should raise ValueError"
except ValueError:
    pass

try:
    GovernedSupervisor(model=model, budget_usd=float("nan"))
    assert False, "NaN budget should raise ValueError"
except ValueError:
    pass

# ── 5. Timeout validation ─────────────────────────────────────────────

try:
    GovernedSupervisor(model=model, timeout_seconds=0)
    assert False, "Zero timeout should raise ValueError"
except ValueError:
    pass

try:
    GovernedSupervisor(model=model, timeout_seconds=-10)
    assert False, "Negative timeout should raise ValueError"
except ValueError:
    pass

# ── 6. Data clearance levels ──────────────────────────────────────────
# PACT defines confidentiality levels. The supervisor maps
# user-friendly strings to ConfidentialityLevel enum values.

clearance_mapping = {
    "public": ConfidentialityLevel.PUBLIC,
    "internal": ConfidentialityLevel.RESTRICTED,
    "restricted": ConfidentialityLevel.RESTRICTED,
    "confidential": ConfidentialityLevel.CONFIDENTIAL,
    "secret": ConfidentialityLevel.SECRET,
    "top_secret": ConfidentialityLevel.TOP_SECRET,
}

for name, expected_level in clearance_mapping.items():
    s = GovernedSupervisor(model=model, data_clearance=name)
    assert s.clearance_level == expected_level, f"{name} -> {expected_level}"

# Invalid clearance raises ValueError
try:
    GovernedSupervisor(model=model, data_clearance="invalid")
    assert False, "Invalid clearance should raise ValueError"
except ValueError:
    pass

# ── 7. Constraint envelope (Layer 2 access) ───────────────────────────
# The envelope encapsulates financial, operational, temporal, and
# data access constraints. It is returned as a deep copy (immutable).

envelope = supervisor.envelope
assert envelope is not None
assert envelope.financial is not None
assert envelope.financial.max_spend_usd == 10.0

# Deep copy: mutations do not affect the supervisor
envelope_copy = supervisor.envelope
assert envelope_copy is not envelope, "Deep copy each time"

# ── 8. Layer 3: governance subsystems (read-only views) ────────────────
# Each subsystem is exposed via a read-only proxy that only allows
# query methods. Mutation methods are blocked.

# Audit trail
audit = supervisor.audit
assert audit is not None

# Budget tracker
budget = supervisor.budget
snapshot = budget.get_snapshot("root")
assert snapshot is not None, "Root budget is always allocated"

# Accountability tracker
accountability = supervisor.accountability
assert accountability is not None

# Cascade manager
cascade = supervisor.cascade
assert cascade is not None

# Clearance enforcer
clearance = supervisor.clearance
assert clearance is not None

# Dereliction detector
dereliction = supervisor.dereliction
assert dereliction is not None

# Bypass manager
bypass = supervisor.bypass_manager
assert bypass is not None

# Vacancy manager
vacancy = supervisor.vacancy
assert vacancy is not None

# ── 9. SupervisorResult — execution outcome ───────────────────────────
# GovernedSupervisor.run() returns a frozen SupervisorResult dataclass.

result = SupervisorResult(
    success=True,
    results={"task-0": "Analysis complete"},
    budget_consumed=2.50,
    budget_allocated=10.0,
)

assert result.success is True
assert result.results["task-0"] == "Analysis complete"
assert result.budget_consumed == 2.50
assert result.budget_allocated == 10.0
assert result.audit_trail == [], "Empty by default"
assert result.events == [], "Empty by default"
assert result.modifications == [], "Empty by default"

# SupervisorResult is frozen (immutable after construction)
try:
    result.success = False  # type: ignore[misc]
    assert False, "Frozen dataclass should not allow mutation"
except AttributeError:
    pass

# ── 10. Dry run with default executor ─────────────────────────────────
# When no execute_node callback is provided, GovernedSupervisor
# uses a default no-op executor useful for plan validation.


async def _dry_run():
    s = GovernedSupervisor(model=model, budget_usd=5.0)
    result = await s.run("Analyze the codebase for security issues")
    assert isinstance(result, SupervisorResult)
    assert result.success is True, "Dry run always succeeds"
    assert result.budget_consumed == 0.0, "No-op executor has zero cost"
    assert len(result.audit_trail) > 0, "Audit trail recorded"
    assert result.plan is not None, "Plan was built"
    return result


dry_result = asyncio.run(_dry_run())
assert dry_result.success is True

# ── 11. Custom executor callback ──────────────────────────────────────
# For real execution, provide an async callback that runs each node.
#
# async def my_executor(spec, inputs):
#     # spec.description contains the node's task description
#     # inputs contains resolved upstream outputs + context
#     result = await call_llm(spec.description, inputs)
#     return {"result": result, "cost": 0.05}
#
# result = await supervisor.run(
#     "Analyze and report",
#     execute_node=my_executor,
# )

# ── 12. Tool use recording ────────────────────────────────────────────
# CLI/entrypoint code can record tool invocations in the audit trail.

supervisor.record_tool_use("read_file", {"path": "/src/main.py"})
supervisor.record_tool_use(
    "grep", {"pattern": "TODO"}, blocked=True, reason="not in allowed tools"
)

audit_entries = supervisor.audit.to_list()
assert len(audit_entries) > 0, "Tool use recorded in audit trail"

# ── 13. Cost recording ────────────────────────────────────────────────
# External code can record costs against the session budget.

supervisor.record_cost(1.50, source="llm_tokens")
budget_snap = supervisor.budget.get_snapshot("root")
assert budget_snap.consumed >= 1.50

# Invalid costs are silently ignored (safety guard)
supervisor.record_cost(-1.0, source="invalid")
supervisor.record_cost(float("inf"), source="invalid")
supervisor.record_cost(float("nan"), source="invalid")

print("PASS: 04-agents/07_governed_supervisor")
