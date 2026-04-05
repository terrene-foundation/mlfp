# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — Kaizen / Structured Output
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Define structured output schemas for agent responses
# LEVEL: Advanced
# PARITY: Python-only — Rust uses response_schema module differently
# VALIDATES: Signature with typed OutputField, structured response patterns
#
# Run: uv run python textbook/python/03-kaizen/06_structured_output.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from typing import List

from kaizen import Signature, InputField, OutputField

# ── 1. Typed output fields ──────────────────────────────────────────
# OutputField types tell the LLM the expected format. The agent's
# response is parsed to match these types.


class SentimentSignature(Signature):
    """Analyze the sentiment of the given text."""

    text: str = InputField(description="Text to analyze")
    sentiment: str = OutputField(description="One of: positive, negative, neutral")
    confidence: float = OutputField(description="Confidence score 0.0 to 1.0")
    keywords: str = OutputField(description="Comma-separated key phrases")


assert "sentiment" in SentimentSignature._signature_outputs
assert "confidence" in SentimentSignature._signature_outputs
assert "keywords" in SentimentSignature._signature_outputs

# ── 2. Multi-field structured output ────────────────────────────────
# Signatures can request complex structured responses.


class ExtractEntitiesSignature(Signature):
    """Extract named entities from text."""

    text: str = InputField(description="Text to process")
    context: str = InputField(description="Domain context", default="general")
    people: str = OutputField(description="JSON array of person names")
    organizations: str = OutputField(description="JSON array of org names")
    locations: str = OutputField(description="JSON array of locations")
    dates: str = OutputField(description="JSON array of dates mentioned")
    summary: str = OutputField(description="One-sentence summary")


assert len(ExtractEntitiesSignature._signature_outputs) == 5
assert len(ExtractEntitiesSignature._signature_inputs) == 2

# ── 3. Boolean/decision output ──────────────────────────────────────


class ModerationSignature(Signature):
    """Check if content violates guidelines."""

    content: str = InputField(description="Content to moderate")
    guidelines: str = InputField(description="Moderation guidelines")
    is_safe: bool = OutputField(description="True if content is safe")
    reason: str = OutputField(description="Explanation of decision")
    category: str = OutputField(description="Violation category if unsafe, else 'none'")


assert "is_safe" in ModerationSignature._signature_outputs

# ── 4. Signature for structured data extraction ─────────────────────


class InvoiceExtractionSignature(Signature):
    """Extract structured data from an invoice image or text."""

    invoice_text: str = InputField(description="Raw invoice text")
    vendor_name: str = OutputField(description="Name of the vendor")
    invoice_number: str = OutputField(description="Invoice number")
    total_amount: str = OutputField(
        description="Total amount as string (e.g., '1234.56')"
    )
    currency: str = OutputField(description="Currency code (e.g., 'SGD', 'USD')")
    line_items: str = OutputField(
        description="JSON array of {description, quantity, price}"
    )


assert len(InvoiceExtractionSignature._signature_outputs) == 5

# ── 5. Pattern: Signature → Agent → Structured response ────────────
# In practice, a Signature is attached to an agent, and the agent's
# LLM call is constrained to produce the declared output fields:
#
#   from kaizen_agents import Agent
#   agent = Agent(signature=SentimentSignature, model=model)
#   result = await agent.run(text="Kailash SDK is amazing!")
#   print(result.sentiment)    # "positive"
#   print(result.confidence)   # 0.95
#   print(result.keywords)     # "Kailash SDK, amazing"

print("PASS: 03-kaizen/06_structured_output")
