# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Agents / SimpleQA Agent
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a SimpleQA agent for question answering
# LEVEL: Basic
# PARITY: Python-only — no Rust equivalent
# VALIDATES: SimpleQAAgent, SimpleQAConfig, QASignature, zero-config pattern
#
# Run: uv run python textbook/python/04-agents/02_simple_qa.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os

from kaizen_agents.agents.specialized.simple_qa import (
    SimpleQAAgent,
    SimpleQAConfig,
    QASignature,
)
from kaizen.core.base_agent import BaseAgent
from kaizen.signatures import InputField, OutputField, Signature

model = os.environ.get("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")

# ── 1. QASignature — structured I/O for question answering ────────────
# SimpleQA uses a Signature with typed inputs and outputs.
# The LLM receives the signature fields and returns structured data.

sig = QASignature()

assert "question" in QASignature._signature_inputs
assert "context" in QASignature._signature_inputs
assert "answer" in QASignature._signature_outputs
assert "confidence" in QASignature._signature_outputs
assert "reasoning" in QASignature._signature_outputs

# context has a default (optional), question does not (required)
context_field = QASignature._signature_inputs["context"]
assert context_field.required is False, "Context is optional"

question_field = QASignature._signature_inputs["question"]
assert question_field.required is True, "Question is required"

# ── 2. SimpleQAConfig — progressive configuration ─────────────────────
# SimpleQAConfig is a dataclass with sensible defaults.
# Parameters can be overridden via constructor, env vars, or config object.

config = SimpleQAConfig()

assert config.temperature == 0.1, "Low temperature for factual answers"
assert config.max_tokens == 300, "Short answers by default"
assert config.timeout == 30
assert config.retry_attempts == 3
assert config.min_confidence_threshold == 0.5
assert config.max_turns is None, "Memory disabled by default (opt-in)"

# Override specific parameters
custom_config = SimpleQAConfig(
    temperature=0.3,
    max_tokens=500,
    min_confidence_threshold=0.7,
)
assert custom_config.temperature == 0.3
assert custom_config.min_confidence_threshold == 0.7

# ── 3. Zero-config instantiation ──────────────────────────────────────
# SimpleQAAgent works out of the box with no arguments.
# It reads KAIZEN_* environment variables or uses defaults.

agent = SimpleQAAgent(llm_provider="mock")

assert isinstance(agent, SimpleQAAgent)
assert isinstance(agent, BaseAgent), "SimpleQAAgent extends BaseAgent"
assert agent.qa_config is not None

# ── 4. Progressive configuration via constructor ──────────────────────
# Override individual parameters without creating a config object.

configured_agent = SimpleQAAgent(
    llm_provider="mock",
    model=model,
    temperature=0.2,
    max_tokens=400,
    min_confidence_threshold=0.8,
)

assert configured_agent.qa_config.temperature == 0.2
assert configured_agent.qa_config.max_tokens == 400
assert configured_agent.qa_config.min_confidence_threshold == 0.8

# ── 5. Configuration via config object ────────────────────────────────
# For full control, pass a SimpleQAConfig directly.

full_config = SimpleQAConfig(
    llm_provider="mock",
    model=model,
    temperature=0.1,
    max_tokens=300,
    timeout=60,
    retry_attempts=5,
    min_confidence_threshold=0.6,
)

config_agent = SimpleQAAgent(config=full_config)
assert config_agent.qa_config.timeout == 60
assert config_agent.qa_config.retry_attempts == 5

# ── 6. Memory opt-in with max_turns ───────────────────────────────────
# Memory is disabled by default. Set max_turns to enable BufferMemory
# for multi-turn conversation continuity.

memory_agent = SimpleQAAgent(
    llm_provider="mock",
    max_turns=10,
)

assert memory_agent.qa_config.max_turns == 10

# ── 7. Input validation ───────────────────────────────────────────────
# SimpleQA validates input before sending to the LLM.
# Empty or whitespace-only questions return an error immediately.

empty_result = agent.run(question="")
assert empty_result["error"] == "INVALID_INPUT"
assert empty_result["confidence"] == 0.0

whitespace_result = agent.run(question="   ")
assert whitespace_result["error"] == "INVALID_INPUT"

# ── 8. Return structure ───────────────────────────────────────────────
# SimpleQA.run() returns a dict with answer, confidence, reasoning,
# and optional warning/error fields.
#
# result = agent.run(question="What is the capital of France?")
# {
#     "answer": "Paris",
#     "confidence": 0.95,
#     "reasoning": "France's capital is a well-known fact",
# }
#
# When confidence < min_confidence_threshold, a warning is added:
# {
#     "answer": "...",
#     "confidence": 0.3,
#     "reasoning": "...",
#     "warning": "Low confidence (0.30 < 0.5)"
# }
#
# NOTE: We do not call run() with real questions here because
# it requires an LLM API key. The above shows the return format.

# ── 9. Node metadata for Studio discovery ─────────────────────────────
# SimpleQAAgent exposes NodeMetadata for Kailash Studio auto-discovery.

assert SimpleQAAgent.metadata.name == "SimpleQAAgent"
assert "qa" in SimpleQAAgent.metadata.tags
assert "question-answering" in SimpleQAAgent.metadata.tags

# ── 10. Strategy: AsyncSingleShotStrategy (default) ───────────────────
# SimpleQA is an interactive agent (not autonomous). It uses
# AsyncSingleShotStrategy: one LLM call per run() invocation.
# This is the default when no strategy= is passed to BaseAgent.

# Autonomous agents (ReAct, RAG) use MultiCycleStrategy instead.

print("PASS: 04-agents/02_simple_qa")
