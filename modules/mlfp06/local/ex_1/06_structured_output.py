# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP06 — Exercise 1.6: Structured Output with Kaizen Signature
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Define a typed Kaizen Signature with InputField and OutputField
#   - Drive an LLM with a type-safe schema instead of free-form text
#   - Access validated results by Signature field name (result["sentiment"])
#   - Understand why Signatures are the production standard
#
# PREREQUISITES: 01_zero_shot.py .. 05_self_consistency.py
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — why types beat free-form JSON
#   2. Build — the ReviewExtraction Signature
#   3. Train — run the signature-backed agent across SST-2 eval docs
#   4. Visualise — typed field access
#   5. Apply — driver incident-report extraction at a ride-hailing platform
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from kaizen import InputField, OutputField, Signature
from kaizen.core.base_agent import BaseAgent

from shared.mlfp06._ollama_bootstrap import DEFAULT_CHAT_MODEL, OLLAMA_BASE_URL
from shared.mlfp06.ex_1 import ensure_ollama, get_eval_docs, plot_extraction_accuracy

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Typed Signatures Beat Free-Form JSON
# ════════════════════════════════════════════════════════════════════════
# Free-form JSON fails: format drift, schema drift, silent data loss.
# Kaizen Signatures declare types; Kaizen renders them into a schema
# prompt and unpacks the reply into a dict keyed by the declared output
# field names. This is the production standard.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the Signature
# ════════════════════════════════════════════════════════════════════════


class ReviewExtraction(Signature):
    """Extract structured information from a movie review snippet."""

    # TODO: Declare input field review_text: str = InputField(description=...)
    ____

    # TODO: Declare output fields:
    #   sentiment: str — "positive" or "negative"
    #   confidence: float — 0.0 to 1.0
    #   key_phrases: list[str] — up to 5 phrases
    #   targets: list[str] — aspects evaluated (acting, plot, visuals)
    #   tone: str — enthusiastic, measured, disappointed, angry, sarcastic
    ____


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN (run the Signature-backed agent)
# ════════════════════════════════════════════════════════════════════════


# BaseAgent wiring for a local Ollama model (kaizen 2.28):
#   - llm_provider/model/base_url route the call to the Ollama daemon; the
#     model comes from OLLAMA_CHAT_MODEL via the course bootstrap
#   - use_async_llm=True is required for `await agent.run_async(...)`
#   - response_format + structured_output_mode="explicit" make the agent
#     request JSON that it can unpack into the Signature's output fields
# No dollar budget: a local model is free, so a USD cap would be meaningless.
OLLAMA_AGENT_CONFIG = {
    "llm_provider": "ollama",
    "model": DEFAULT_CHAT_MODEL,
    "base_url": OLLAMA_BASE_URL,
    "use_async_llm": True,
    "response_format": {"type": "json_object"},
    "structured_output_mode": "explicit",
}
OUTPUT_FIELDS = ["sentiment", "confidence", "key_phrases", "targets", "tone"]


async def run_signature_extraction() -> list[dict]:
    ensure_ollama()  # fails loudly with "ollama serve" if the daemon is down
    # TODO: Construct a BaseAgent with config=OLLAMA_AGENT_CONFIG and an
    # INSTANCE of your ReviewExtraction Signature.
    agent = ____

    docs = get_eval_docs().head(10)
    results: list[dict] = []
    # TODO: For each text, `await agent.run_async(review_text=text[:800])`.
    # The result is a dict keyed by OutputField names. If any name in
    # OUTPUT_FIELDS is missing, raise RuntimeError (no placeholder values).
    # Then validate VALUES: confidence must be float()-coercible — the model
    # sometimes returns explicit nulls; LLM output is untrusted input.
    # Append each result; print the first 3 using result["sentiment"],
    # result["confidence"], result["key_phrases"], result["targets"],
    # result["tone"].
    ____
    return results


print("\n" + "=" * 70)
print("  Structured Output via Kaizen Signature")
print("=" * 70)
signature_results = asyncio.run(run_signature_extraction())


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(signature_results) > 0, "Task 3: Signature extraction should produce results"
sample = signature_results[0]
assert "sentiment" in sample, "Result should have 'sentiment' field"
assert "confidence" in sample, "Result should have 'confidence' field"
assert 0.0 <= float(sample["confidence"]) <= 1.0, "Confidence should be in [0, 1]"
assert "key_phrases" in sample, "Result should have 'key_phrases' field"
assert isinstance(sample["key_phrases"], list), "key_phrases should be a list"
assert "tone" in sample, "Result should have 'tone' field"
print(
    f"\n[ok] Checkpoint passed — Signature extraction: "
    f"sentiment='{sample['sentiment']}', confidence={float(sample.get('confidence', 0)):.2f}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE — dict field access + extraction accuracy chart
# ════════════════════════════════════════════════════════════════════════
# run_async returns a dict keyed by the Signature's OutputField names —
# no string matching, no hand-written JSON parsing, no normalise_label().
avg_conf = sum(float(r["confidence"]) for r in signature_results) / len(
    signature_results
)
tones = [r["tone"] for r in signature_results]
print(f"\n  Avg confidence across {len(signature_results)} reviews: {avg_conf:.2f}")
print(f"  Tone distribution: {tones}")

# R9A: visual proof — extraction accuracy per output field type
plot_extraction_accuracy(
    signature_results,
    field_names=["sentiment", "confidence", "key_phrases", "targets", "tone"],
    title="Structured Output — Extraction Rate per Field",
    filename="ex1_06_extraction_accuracy.png",
)

# INTERPRETATION: The Signature output is directly usable by downstream
# code. No parsing layer, no format drift. When the LLM's reply cannot be
# unpacked into the schema, this script raises instead of inventing a
# value — the failure is LOUD and FIXABLE, not silent and corrupting.
# The bar chart counts non-empty values per field. Every field is present
# (we raise otherwise), so a bar below 100% means the model returned an
# EMPTY value — typically for list fields like key_phrases/targets.
# Fields below ~90% need tighter OutputField descriptions.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Driver Incident-Report Extraction at a Ride-Hailing Platform
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Southeast Asian ride-hailing platform
# receives over a thousand driver-submitted incident reports per day.
# Each free-text report must be decomposed into a structured record for
# the risk + insurance pipeline:
#   - incident_type (collision, theft, passenger_dispute, mechanical)
#   - severity (minor, moderate, severe)
#   - parties_involved (list of strings: "driver", "passenger", "other_vehicle")
#   - location_landmark (free text)
#   - claim_required (bool)
#   - urgency (immediate, 24h, 72h)
#
# Why Kaizen Signatures fit here:
#   - The downstream pipeline is STRONGLY TYPED — DataFlow models expect
#     specific fields and types. A missing field means a row insert fails.
#   - An insurance partner's API requires strict JSON schema compliance.
#   - Silent misclassification is unrecoverable downstream — once a
#     "severe" report is tagged "minor", the claim is routed to the wrong
#     queue and may miss a notification deadline.
#
# Free-form JSON prompting fails this use case: when the LLM returns
# "sevrity" instead of "severity", a hand-written parser silently drops
# the field. With a Signature the missing field is detected and the
# record is rejected loudly, exactly as Task 3 does above.
#
# BUSINESS IMPACT (illustrative figures): suppose each pipeline error
# costs ~S$180 in rework, customer contact and claim re-routing. At
# 1,200 reports/day, cutting the parse-error rate from 8% to 0.5% avoids
# ~90 errors/day ≈ S$16K/day. Your extraction-rate chart above is the
# measured starting point for that estimate on your own model.
#
# DEPLOYMENT NOTE: Keep the Signature next to the DataFlow model
# definition, so a schema change updates both the LLM output and the
# database column in one place.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Defined a Kaizen Signature with typed InputField + OutputField
  [x] Built a BaseAgent subclass backed by the Signature schema
  [x] Accessed results via dict keys validated by the Signature
  [x] Understood why Signatures solve free-form JSON's failure modes
  [x] Sized the approach against a ride-hailing incident pipeline

  KEY INSIGHT: Every other technique in this exercise treats LLM output
  as strings to parse. Signatures treat it as typed data to validate.
  In production, the difference is the gap between "silent corruption"
  and "loud, fixable error".

  Where this goes next:
    - Exercise 2: fine-tune the base model with LoRA adapters
    - Exercise 3: DPO (Direct Preference Optimisation) — skip the
      reward model from RLHF entirely
    - Exercises 7-8: wire all of this into PACT governance and Nexus
      multi-channel deployment
"""
)
