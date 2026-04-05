# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — ML / ML Agents & AgentGuardrailMixin
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Demonstrate the 6 ML Kaizen agents, the AgentInfusionProtocol
#            double opt-in pattern, and the 5 mandatory guardrails
#            (confidence, cost budget, approval gate, baseline comparison,
#            audit trail).
# LEVEL: Advanced
# PARITY: Python-only (kailash-ml has no Rust counterpart)
# VALIDATES: DataScientistAgent, FeatureEngineerAgent, ModelSelectorAgent,
#            ExperimentInterpreterAgent, DriftAnalystAgent,
#            RetrainingDecisionAgent, AgentGuardrailMixin, GuardrailConfig,
#            CostTracker, AuditEntry, ApprovalRequest, ApprovalResult,
#            AgentInfusionProtocol
#
# NOTE: Running the agents requires `pip install kailash-ml[agents]`
#       (kailash-kaizen).  This tutorial validates the types and
#       guardrail machinery WITHOUT calling live LLMs.
#
# Run: uv run python textbook/python/05-ml/15_ml_agents.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

from kailash_ml.engines._guardrails import (
    AgentGuardrailMixin,
    ApprovalRequest,
    ApprovalResult,
    AuditEntry,
    CostTracker,
    GuardrailBudgetExceededError,
    GuardrailConfig,
)
from kailash_ml.types import AgentInfusionProtocol

# ════════════════════════════════════════════════════════════════════════
# Part A: Agent Discovery (lazy imports)
# ════════════════════════════════════════════════════════════════════════

# ── 1. The 6 ML agents are lazy-loaded via __getattr__ ──────────────
# They require `pip install kailash-ml[agents]`.  We verify the lazy
# loading mechanism works by importing the module itself.

from kailash_ml import agents  # noqa: F401

# The __all__ exposes all 6 agents
expected_agents = {
    "DataScientistAgent",
    "FeatureEngineerAgent",
    "ModelSelectorAgent",
    "ExperimentInterpreterAgent",
    "DriftAnalystAgent",
    "RetrainingDecisionAgent",
}

assert set(agents.__all__) == expected_agents

# Attempting to import a non-existent agent raises AttributeError
try:
    agents.__getattr__("NonExistentAgent")
    assert False, "Should raise AttributeError"
except AttributeError:
    pass  # Expected

# ════════════════════════════════════════════════════════════════════════
# Part B: AgentInfusionProtocol (double opt-in)
# ════════════════════════════════════════════════════════════════════════

# ── 2. AgentInfusionProtocol is a runtime-checkable Protocol ─────────

assert hasattr(AgentInfusionProtocol, "suggest_model")
assert hasattr(AgentInfusionProtocol, "suggest_features")
assert hasattr(AgentInfusionProtocol, "interpret_results")
assert hasattr(AgentInfusionProtocol, "interpret_drift")

# Double opt-in means:
#   1. pip install kailash-ml[agents]   (install the kaizen dependency)
#   2. agent=True in config             (opt-in at call site)
# Without BOTH, agent features are not used.

# ════════════════════════════════════════════════════════════════════════
# Part C: Guardrail Config & Validation
# ════════════════════════════════════════════════════════════════════════

# ── 3. GuardrailConfig — defaults ────────────────────────────────────

config = GuardrailConfig()
assert config.max_llm_cost_usd == 1.0
assert config.auto_approve is False
assert config.require_baseline is True
assert config.audit_trail is True
assert config.min_confidence == 0.5

# ── 4. GuardrailConfig — NaN/Inf rejection ──────────────────────────

try:
    GuardrailConfig(max_llm_cost_usd=float("nan"))
    assert False, "Should reject NaN"
except ValueError:
    pass  # Expected

try:
    GuardrailConfig(max_llm_cost_usd=float("inf"))
    assert False, "Should reject Inf"
except ValueError:
    pass  # Expected

try:
    GuardrailConfig(max_llm_cost_usd=-5.0)
    assert False, "Should reject negative"
except ValueError:
    pass  # Expected

try:
    GuardrailConfig(min_confidence=float("nan"))
    assert False, "Should reject NaN confidence"
except ValueError:
    pass  # Expected

# ════════════════════════════════════════════════════════════════════════
# Part D: CostTracker (Guardrail 2 — Cost Budget)
# ════════════════════════════════════════════════════════════════════════

# ── 5. CostTracker — tracks cumulative LLM costs ────────────────────

tracker = CostTracker(max_budget_usd=0.10)
assert tracker.total_spent == 0.0
assert tracker.remaining == 0.10
assert len(tracker.calls) == 0

# Record a call
cost = tracker.record("test-model", input_tokens=100, output_tokens=50)
assert cost > 0
assert tracker.total_spent == cost
assert tracker.remaining == 0.10 - cost
assert len(tracker.calls) == 1

call = tracker.calls[0]
assert call["model"] == "test-model"
assert call["input_tokens"] == 100
assert call["output_tokens"] == 50

# Record many calls to exceed budget
try:
    tracker.record("test-model", input_tokens=100000, output_tokens=100000)
    assert False, "Should raise GuardrailBudgetExceededError"
except GuardrailBudgetExceededError:
    pass  # Expected

# Reset tracker
tracker.reset()
assert tracker.total_spent == 0.0
assert len(tracker.calls) == 0

# ── 6. CostTracker — validation ──────────────────────────────────────

try:
    CostTracker(max_budget_usd=-1.0)
    assert False, "Should reject negative budget"
except ValueError:
    pass

try:
    CostTracker(max_budget_usd=float("nan"))
    assert False, "Should reject NaN budget"
except ValueError:
    pass

# ════════════════════════════════════════════════════════════════════════
# Part E: AgentGuardrailMixin (5 Mandatory Guardrails)
# ════════════════════════════════════════════════════════════════════════


class MockEngine(AgentGuardrailMixin):
    """Engine that uses the guardrail mixin."""

    def __init__(self, config: GuardrailConfig | None = None) -> None:
        self._init_guardrails(config)


# ── 7. Guardrail 1: Confidence scores ────────────────────────────────

engine = MockEngine(GuardrailConfig(min_confidence=0.6))

# High confidence passes
assert engine._check_confidence(0.8, "TestAgent") is True

# Low confidence fails (engine should fall back to algorithmic mode)
assert engine._check_confidence(0.3, "TestAgent") is False

# Boundary
assert engine._check_confidence(0.6, "TestAgent") is True
assert engine._check_confidence(0.59, "TestAgent") is False

# ── 8. Guardrail 2: Cost budget (via mixin) ──────────────────────────

budget_engine = MockEngine(GuardrailConfig(max_llm_cost_usd=0.05))
assert budget_engine._budget_remaining == 0.05

budget_engine._record_cost("model", 100, 50)
assert budget_engine._budget_remaining < 0.05

try:
    budget_engine._record_cost("model", 100000, 100000)
    assert False, "Should exceed budget"
except GuardrailBudgetExceededError:
    pass

# ── 9. Guardrail 3: Human approval gate ──────────────────────────────

# auto_approve=False (default) — creates approval requests
approval_engine = MockEngine(GuardrailConfig(auto_approve=False))
request = approval_engine._request_approval(
    agent_name="ModelSelector",
    recommendation_summary="Use GradientBoosting (confidence 0.85)",
    confidence=0.85,
    baseline_comparison="RF baseline: 0.88 accuracy",
)

assert isinstance(request, ApprovalRequest)
assert request.agent_name == "ModelSelector"
assert request.confidence == 0.85

# Approve it
result = approval_engine.approve(request.id, approved_by="human_reviewer")
assert isinstance(result, ApprovalResult)
assert result.approved is True
assert result.approved_by == "human_reviewer"

# Create another and reject
request2 = approval_engine._request_approval("DriftAnalyst", "Retrain immediately", 0.7)
result2 = approval_engine.reject(request2.id, "reviewer", reason="Not urgent")
assert result2.approved is False

# auto_approve=True — skips approval
auto_engine = MockEngine(GuardrailConfig(auto_approve=True))
no_request = auto_engine._request_approval("Agent", "Summary", 0.9)
assert no_request is None  # No approval needed

# ── 10. Guardrail 4: Baseline comparison ─────────────────────────────
# The require_baseline flag means the engine must run a pure algorithmic
# baseline alongside the agent.  This is checked in engine logic, not
# the mixin.  The mixin records it in audit entries.

assert engine._guardrail_config.require_baseline is True

# ── 11. Guardrail 5: Audit trail ─────────────────────────────────────

audit_engine = MockEngine(GuardrailConfig(audit_trail=True))

entry = audit_engine._log_audit(
    agent_name="DataScientist",
    engine_name="AutoMLEngine",
    input_summary="200 rows, 5 features, classification",
    output_summary="Recommended GradientBoosting, confidence 0.85",
    confidence=0.85,
    llm_cost_usd=0.003,
    approved_by="human",
    baseline_result="RF accuracy: 0.88",
)

assert isinstance(entry, AuditEntry)
assert entry.agent_name == "DataScientist"
assert entry.engine_name == "AutoMLEngine"
assert entry.confidence == 0.85
assert entry.llm_cost_usd == 0.003
assert entry.approved_by == "human"
assert entry.baseline_result == "RF accuracy: 0.88"

# Buffered entries
assert len(audit_engine.audit_entries) == 1
assert audit_engine.audit_entries[0].id == entry.id

# ── 12. AuditEntry serialization round-trip ──────────────────────────

ae_dict = entry.to_dict()
ae_restored = AuditEntry.from_dict(ae_dict)
assert ae_restored.id == entry.id
assert ae_restored.agent_name == entry.agent_name
assert ae_restored.confidence == entry.confidence
assert ae_restored.llm_cost_usd == entry.llm_cost_usd

# ── 13. Edge case: approve non-existent request ─────────────────────

try:
    approval_engine.approve("nonexistent-id", "reviewer")
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected

try:
    approval_engine.reject("nonexistent-id", "reviewer")
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected

print("PASS: 05-ml/15_ml_agents")
