# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Kaizen / Signatures
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define AI agent interfaces using Signature, InputField, OutputField
# LEVEL: Basic
# PARITY: Equivalent — Rust uses AgentConfig struct; Python uses class-based
#         Signature with metaclass (DSPy-inspired declarative pattern)
# VALIDATES: Signature, InputField, OutputField, class-based and programmatic
#
# Run: uv run python textbook/python/03-kaizen/01_signatures.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from kaizen import Signature, InputField, OutputField

# ── 1. Class-based Signature (recommended) ──────────────────────────
# Signatures declare WHAT an agent does — its inputs and outputs.
# The docstring becomes the agent's system instructions.
# InputField/OutputField use `description=` (preferred) or `desc=`.


class QASignature(Signature):
    """You are a helpful question-answering assistant."""

    question: str = InputField(description="The question to answer")
    context: str = InputField(description="Additional context", default="")
    answer: str = OutputField(description="Clear, accurate answer")
    confidence: float = OutputField(description="Confidence score 0.0 to 1.0")


# Verify the signature was processed by SignatureMeta
assert hasattr(QASignature, "_signature_inputs"), "Metaclass extracts inputs"
assert hasattr(QASignature, "_signature_outputs"), "Metaclass extracts outputs"

assert "question" in QASignature._signature_inputs
assert "context" in QASignature._signature_inputs
assert "answer" in QASignature._signature_outputs
assert "confidence" in QASignature._signature_outputs

# Check input field properties
question_field = QASignature._signature_inputs["question"]
assert isinstance(question_field, InputField)
assert question_field.required is True, "No default → required"

context_field = QASignature._signature_inputs["context"]
assert context_field.required is False, "Has default → optional"
assert context_field.default == ""

print(f"QA inputs: {list(QASignature._signature_inputs.keys())}")
print(f"QA outputs: {list(QASignature._signature_outputs.keys())}")

# ── 2. Signature with intent and guidelines ─────────────────────────
# __intent__ defines WHY the agent exists (high-level purpose).
# __guidelines__ define HOW the agent should behave (constraints).


class CustomerSupportSignature(Signature):
    """You are a customer support agent for an e-commerce platform."""

    __intent__ = "Resolve customer issues efficiently and empathetically"
    __guidelines__ = [
        "Acknowledge concerns before proposing solutions",
        "Use empathetic language",
        "Escalate if unresolved in 3 turns",
    ]

    customer_message: str = InputField(description="Customer's message")
    order_history: str = InputField(description="Recent order history", default="")
    response: str = OutputField(description="Support response")
    action: str = OutputField(description="Action to take: resolve, escalate, transfer")


assert CustomerSupportSignature._signature_intent == (
    "Resolve customer issues efficiently and empathetically"
)
assert len(CustomerSupportSignature._signature_guidelines) == 3

# ── 3. Programmatic Signature (backward-compatible) ─────────────────
# For dynamic use cases where inputs/outputs aren't known at class time.

sig = Signature(
    inputs=["query", "context"],
    outputs=["result"],
    name="search",
    description="Search for relevant information",
)

assert sig is not None, "Programmatic signature creation works"

# ── 4. Signature inheritance ────────────────────────────────────────
# Signatures can be extended via inheritance. Child fields merge with
# parent fields (child overrides on name collision).


class DetailedQA(QASignature):
    """Extended QA with source citations."""

    sources: str = OutputField(description="List of sources used")


# Inherits parent fields + adds new ones
assert "question" in DetailedQA._signature_inputs, "Inherits parent input"
assert "answer" in DetailedQA._signature_outputs, "Inherits parent output"
assert "sources" in DetailedQA._signature_outputs, "Adds new output"

# ── 5. Edge case: empty signature ───────────────────────────────────


class EmptySignature(Signature):
    """A signature with no fields."""

    pass


assert len(EmptySignature._signature_inputs) == 0
assert len(EmptySignature._signature_outputs) == 0

# ── 6. Field metadata ──────────────────────────────────────────────
# Additional kwargs on InputField/OutputField are stored as metadata.


class RichSignature(Signature):
    """Signature with rich field metadata."""

    data: str = InputField(
        description="Input data",
        format="json",
        max_length=10000,
    )
    result: str = OutputField(description="Processed result")


data_field = RichSignature._signature_inputs["data"]
assert data_field.metadata.get("format") == "json"
assert data_field.metadata.get("max_length") == 10000

print("PASS: 03-kaizen/01_signatures")
