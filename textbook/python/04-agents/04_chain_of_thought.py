# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / Chain-of-Thought Agent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use ChainOfThoughtAgent for step-by-step reasoning
# LEVEL: Intermediate
# PARITY: Python-only — no Rust equivalent
# VALIDATES: ChainOfThoughtAgent, ChainOfThoughtConfig,
#            ChainOfThoughtSignature, verification, text extraction
#
# Run: uv run python textbook/python/04-agents/04_chain_of_thought.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kaizen_agents.agents.specialized.chain_of_thought import (
    ChainOfThoughtAgent,
    ChainOfThoughtConfig,
    ChainOfThoughtSignature,
)
from kaizen.core.base_agent import BaseAgent

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. Chain-of-Thought pattern ────────────────────────────────────────
# CoT decomposes complex problems into explicit reasoning steps.
# Unlike ReAct (iterative cycles with tools), CoT is a single-shot
# structured reasoning pass with transparent intermediate steps.
#
# The LLM fills in five structured steps:
#   step1: Problem understanding
#   step2: Data identification and organization
#   step3: Systematic calculation or analysis
#   step4: Solution verification
#   step5: Final answer formulation
#
# This makes the reasoning auditable — every step is visible.

# ── 2. ChainOfThoughtSignature — five-step structure ──────────────────

assert "problem" in ChainOfThoughtSignature._signature_inputs
assert "context" in ChainOfThoughtSignature._signature_inputs

# Five reasoning steps + final answer + confidence
assert "step1" in ChainOfThoughtSignature._signature_outputs
assert "step2" in ChainOfThoughtSignature._signature_outputs
assert "step3" in ChainOfThoughtSignature._signature_outputs
assert "step4" in ChainOfThoughtSignature._signature_outputs
assert "step5" in ChainOfThoughtSignature._signature_outputs
assert "final_answer" in ChainOfThoughtSignature._signature_outputs
assert "confidence" in ChainOfThoughtSignature._signature_outputs

# context is optional (has default)
context_field = ChainOfThoughtSignature._signature_inputs["context"]
assert context_field.required is False

# ── 3. ChainOfThoughtConfig — reasoning-specific settings ─────────────

config = ChainOfThoughtConfig()

assert config.temperature == 0.1, "Low temperature for precise reasoning"
assert config.max_tokens == 1500, "Larger context for step-by-step output"
assert config.timeout == 45, "Longer timeout for complex reasoning"
assert config.reasoning_steps == 5, "Five reasoning steps"
assert config.confidence_threshold == 0.7
assert config.enable_verification is True, "Verification enabled by default"

# Custom configuration for stricter reasoning
strict_config = ChainOfThoughtConfig(
    confidence_threshold=0.9,
    reasoning_steps=5,
    enable_verification=True,
)
assert strict_config.confidence_threshold == 0.9

# ── 4. ChainOfThoughtAgent instantiation ──────────────────────────────
# CoT uses AsyncSingleShotStrategy (the default). Unlike ReAct,
# it does NOT iterate — it produces all five steps in one LLM call.

agent = ChainOfThoughtAgent(
    llm_provider="mock",
    model=model,
    confidence_threshold=0.7,
    enable_verification=True,
)

assert isinstance(agent, ChainOfThoughtAgent)
assert isinstance(agent, BaseAgent)
assert agent.cot_config.enable_verification is True

# ── 5. Input validation ───────────────────────────────────────────────

empty_result = agent.run(problem="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["confidence"] == 0.0
assert empty_result["final_answer"] == "Please provide a clear problem to solve."

# All five steps present even on error
for i in range(1, 6):
    assert f"step{i}" in empty_result

# ── 6. Text response extraction (fallback parser) ─────────────────────
# When the LLM returns plain text instead of structured JSON (e.g.,
# with mock provider), CoT extracts structure from the text.
# This is a safety net, not the primary path.

text_response = """Step 1: Understand the problem - we need to multiply 15 by 23.
Step 2: Break down the numbers - 15 = 10 + 5, 23 = 20 + 3.
Step 3: Apply distributive property - (10+5)(20+3) = 200+30+100+15.
Step 4: Sum the parts - 200+30+100+15 = 345.
Step 5: Verify by estimation - 15*20=300, 15*3=45, total=345. Correct.
Final answer: 345"""

extracted = agent._extract_from_text_response(text_response)

assert "step1" in extracted
assert "final_answer" in extracted
assert extracted["final_answer"] == "345"
assert extracted["confidence"] == 0.5, "Default confidence for text-parsed responses"

# Missing steps get empty strings
sparse_text = "Final answer: 42"
sparse_result = agent._extract_from_text_response(sparse_text)
assert sparse_result["final_answer"] == "42"
for i in range(1, 6):
    assert f"step{i}" in sparse_result  # All step keys present

# ── 7. Verification flag ──────────────────────────────────────────────
# When enable_verification=True, CoT adds a "verified" boolean
# to the result: True if confidence >= threshold.
#
# result = agent.run(problem="Calculate 15 * 23")
# {
#     "step1": "Understanding the problem...",
#     "step2": "Identifying the numbers...",
#     "step3": "Systematic calculation...",
#     "step4": "Verifying the result...",
#     "step5": "Formulating final answer...",
#     "final_answer": "345",
#     "confidence": 0.95,
#     "verified": True,   # confidence >= threshold
# }
#
# When confidence < threshold:
# {
#     "final_answer": "...",
#     "confidence": 0.4,
#     "verified": False,
#     "warning": "Low confidence (0.40 < 0.7)",
# }

# Verify config controls verification behavior
no_verify_agent = ChainOfThoughtAgent(
    llm_provider="mock",
    model=model,
    enable_verification=False,
)
assert no_verify_agent.cot_config.enable_verification is False

# ── 8. CoT vs ReAct: when to use which ────────────────────────────────
# CoT: Single-shot structured reasoning. Best for:
#   - Math problems, logical deduction
#   - Step-by-step analysis that doesn't need external data
#   - Audit-trail requirements (every step recorded)
#
# ReAct: Multi-cycle iterative reasoning with tools. Best for:
#   - Tasks requiring real-time data (search, file reading)
#   - Multi-step workflows where each step depends on observations
#   - Open-ended exploration where the path isn't known upfront

# ── 9. Node metadata ──────────────────────────────────────────────────

assert ChainOfThoughtAgent.metadata.name == "ChainOfThoughtAgent"
assert "chain-of-thought" in ChainOfThoughtAgent.metadata.tags
assert "reasoning" in ChainOfThoughtAgent.metadata.tags
assert "verification" in ChainOfThoughtAgent.metadata.tags

print("PASS: 04-agents/04_chain_of_thought")
