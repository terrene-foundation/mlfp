# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Tree-of-Thoughts Agent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use TreeOfThoughtsAgent for exploration-based reasoning
# LEVEL: Advanced
# PARITY: Python-only — no Rust equivalent
# VALIDATES: ToTAgent, ToTAgentConfig, ToTSignature, ToTPath,
#            ToTEvaluation, path generation, evaluation, selection
#
# Run: uv run python textbook/python/04-agents/09_tree_of_thoughts.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

from kaizen_agents.agents.specialized.tree_of_thoughts import (
    ToTAgent,
    ToTAgentConfig,
    ToTSignature,
    ToTPath,
    ToTEvaluation,
)
from kaizen.core.base_agent import BaseAgent

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. Tree-of-Thoughts pattern ───────────────────────────────────────
# ToT explores multiple reasoning paths in parallel, then evaluates
# and selects the best one. This is fundamentally different from:
#
#   CoT:   Single linear reasoning path (one shot)
#   ReAct: Single iterative path with tool observations
#   ToT:   N parallel paths, evaluated, best selected
#
# The four phases:
#   1. GENERATE: Create N diverse reasoning paths
#   2. EVALUATE: Score each path independently
#   3. SELECT:   Choose the highest-scoring path
#   4. EXECUTE:  Extract the result from the best path

# ── 2. ToTPath and ToTEvaluation — typed structures ───────────────────
# TypedDict definitions provide type safety for paths and evaluations.

# ToTPath has required and optional fields
sample_path: ToTPath = {
    "path_id": 1,
    "reasoning": "Approach via cost-benefit analysis",
}
assert sample_path["path_id"] == 1
assert sample_path["reasoning"] == "Approach via cost-benefit analysis"

# ToTEvaluation scores a path
sample_eval: ToTEvaluation = {
    "path": {"path_id": 1, "reasoning": "Cost-benefit analysis"},
    "score": 0.85,
    "reasoning": "Well-structured approach with clear trade-offs",
}
assert sample_eval["score"] == 0.85

# ── 3. ToTSignature — multi-path I/O structure ────────────────────────

assert "task" in ToTSignature._signature_inputs

assert "paths" in ToTSignature._signature_outputs
assert "evaluations" in ToTSignature._signature_outputs
assert "best_path" in ToTSignature._signature_outputs
assert "final_result" in ToTSignature._signature_outputs

# ── 4. ToTAgentConfig — exploration-specific settings ─────────────────

config = ToTAgentConfig()

assert config.num_paths == 5, "Generate 5 reasoning paths by default"
assert config.max_paths == 20, "Safety limit: max 20 paths"
assert config.evaluation_criteria == "quality", "Default: quality-based evaluation"
assert config.parallel_execution is True, "Parallel generation by default"
assert config.temperature == 0.9, "High temperature for diverse paths"

# num_paths cannot exceed max_paths
try:
    ToTAgentConfig(num_paths=25, max_paths=20)
except ValueError:
    pass  # Config itself does not validate, but ToTAgent does

# Custom configuration
custom_config = ToTAgentConfig(
    num_paths=10,
    max_paths=20,
    evaluation_criteria="creativity",
    parallel_execution=True,
)
assert custom_config.num_paths == 10
assert custom_config.evaluation_criteria == "creativity"

# ── 5. ToTAgent instantiation ─────────────────────────────────────────
# ToT uses AsyncSingleShotStrategy (default) because the multi-path
# generation is handled internally, not via MultiCycleStrategy.
# Each path generation is a separate BaseAgent.run() call.

agent = ToTAgent(
    llm_provider="mock",
    model=model,
    num_paths=5,
    evaluation_criteria="quality",
)

assert isinstance(agent, ToTAgent)
assert isinstance(agent, BaseAgent)
assert agent.tot_config.num_paths == 5

# ── 6. Path validation (num_paths vs max_paths) ──────────────────────
# ToTAgent validates that num_paths does not exceed max_paths.

try:
    ToTAgent(
        llm_provider="mock",
        model=model,
        num_paths=25,
        max_paths=20,
    )
    assert False, "num_paths > max_paths should raise ValueError"
except ValueError as e:
    assert "exceeds" in str(e).lower()

# ── 7. Path evaluation logic ──────────────────────────────────────────
# The agent evaluates paths based on four criteria:
#   - Completeness: reasoning length > 20 chars (0.3)
#   - No errors: no "error" key in path (0.3)
#   - Structured steps: has non-empty steps list (0.2)
#   - Reasoning quality: reasoning > 100 chars (0.2)

# Good path: meets all criteria
good_path = {
    "path_id": 1,
    "reasoning": "A" * 150,  # Long reasoning
    "steps": ["step1", "step2", "step3"],
}
good_eval = agent._evaluate_path(good_path, "test task")
assert good_eval["score"] == 1.0, "Perfect path gets 1.0"

# Path with error: loses 0.3
error_path = {
    "path_id": 2,
    "reasoning": "A" * 150,
    "steps": ["step1"],
    "error": "generation failed",
}
error_eval = agent._evaluate_path(error_path, "test task")
assert error_eval["score"] == 0.7, "Error costs 0.3"

# Empty path: scores 0.0
empty_path = {"path_id": 3, "reasoning": ""}
empty_eval = agent._evaluate_path(empty_path, "test task")
assert empty_eval["score"] == 0.0, "Empty path scores 0.0"

# Short path: partial score
short_path = {"path_id": 4, "reasoning": "Brief reasoning about the problem"}
short_eval = agent._evaluate_path(short_path, "test task")
assert 0.0 < short_eval["score"] < 1.0, "Short path gets partial score"

# ── 8. Best path selection ────────────────────────────────────────────
# _select_best_path picks the evaluation with the highest score.

evaluations = [
    {"path": {"path_id": 1}, "score": 0.6, "reasoning": "OK"},
    {"path": {"path_id": 2}, "score": 0.9, "reasoning": "Excellent"},
    {"path": {"path_id": 3}, "score": 0.3, "reasoning": "Weak"},
]

best = agent._select_best_path(evaluations)
assert best["score"] == 0.9
assert best["path"]["path_id"] == 2

# Empty evaluations: graceful fallback
empty_best = agent._select_best_path([])
assert empty_best["score"] == 0.0

# ── 9. Input validation ───────────────────────────────────────────────

empty_result = agent.run(task="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["paths"] == []
assert empty_result["evaluations"] == []
assert empty_result["final_result"] == ""

# ── 10. Return structure ──────────────────────────────────────────────
# ToTAgent.run() returns:
# {
#     "paths": [
#         {"path_id": 0, "reasoning": "...", "steps": [...]},
#         {"path_id": 1, "reasoning": "...", "steps": [...]},
#         ...
#     ],
#     "evaluations": [
#         {"path": {...}, "score": 0.85, "reasoning": "..."},
#         ...
#     ],
#     "best_path": {"path": {...}, "score": 0.92, "reasoning": "..."},
#     "final_result": "The recommended approach is..."
# }
#
# Error codes: INVALID_INPUT, PATH_GENERATION_FAILED,
#              EVALUATION_FAILED, SELECTION_FAILED, EXECUTION_FAILED

# ── 11. Node metadata ─────────────────────────────────────────────────

assert ToTAgent.metadata.name == "ToTAgent"
assert "tot" in ToTAgent.metadata.tags
assert "multi-path" in ToTAgent.metadata.tags
assert "parallel" in ToTAgent.metadata.tags

print("PASS: 04-agents/09_tree_of_thoughts")
