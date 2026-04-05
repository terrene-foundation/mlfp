# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Data Provenance
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Track data provenance and lineage metadata with typed
#            wrappers for field values
# LEVEL: Advanced
# PARITY: Python-only
# VALIDATES: Provenance[T], ProvenanceMetadata, SourceType enum,
#            serialization (to_dict/from_dict), confidence validation
#
# Run: uv run python textbook/python/01-dataflow/09_provenance.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from datetime import UTC, datetime

from dataflow import Provenance, ProvenanceMetadata, SourceType

# ── 1. SourceType enum — where a field value came from ──────────────
# Classification of data origins. All values are str-backed.

assert SourceType.EXCEL_CELL.value == "excel_cell"
assert SourceType.API_QUERY.value == "api_query"
assert SourceType.CALCULATED.value == "calculated"
assert SourceType.AGENT_DERIVED.value == "agent_derived"
assert SourceType.MANUAL.value == "manual"
assert SourceType.DATABASE.value == "database"
assert SourceType.FILE.value == "file"

# Construct from string
assert SourceType("manual") == SourceType.MANUAL

print("SourceType: 7 origin types defined")

# ── 2. ProvenanceMetadata — origin and confidence ───────────────────
# Describes where a value came from, how confident we are, and
# optionally tracks previous values and change reasons.

meta = ProvenanceMetadata(
    source_type=SourceType.API_QUERY,
    source_detail="Singapore Open Data API — HDB resale prices Q1 2026",
    confidence=0.95,
)

assert meta.source_type == SourceType.API_QUERY
assert meta.source_detail == "Singapore Open Data API — HDB resale prices Q1 2026"
assert meta.confidence == 0.95
assert meta.previous_value is None
assert meta.change_reason == ""
assert meta.extracted_at is not None, "Defaults to now (UTC)"
assert isinstance(meta.extracted_at, datetime)
print(
    f"ProvenanceMetadata: source={meta.source_type.value}, confidence={meta.confidence}"
)

# ── 3. ProvenanceMetadata — full fields ─────────────────────────────
# Track previous values and change reasons for audit trails.

meta_updated = ProvenanceMetadata(
    source_type=SourceType.AGENT_DERIVED,
    source_detail="PriceAdjustmentAgent v2.1",
    confidence=0.88,
    previous_value=450000.0,
    change_reason="Agent adjusted for inflation index 2026-Q1",
    extracted_at=datetime(2026, 3, 15, 10, 30, 0, tzinfo=UTC),
)

assert meta_updated.previous_value == 450000.0
assert meta_updated.change_reason == "Agent adjusted for inflation index 2026-Q1"
assert meta_updated.extracted_at.year == 2026
print(f"Updated metadata: previous_value={meta_updated.previous_value}")

# ── 4. Confidence validation ────────────────────────────────────────
# Confidence must be a finite number between 0.0 and 1.0.

# Valid boundaries
ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=0.0)
ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=1.0)

# Invalid: out of range
try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=1.5)
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "between 0.0 and 1.0" in str(e)

try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=-0.1)
    assert False, "Should raise ValueError"
except ValueError as e:
    assert "between 0.0 and 1.0" in str(e)

# Invalid: NaN and infinity
import math

try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=float("nan"))
    assert False, "Should raise ValueError"
except ValueError:
    pass

try:
    ProvenanceMetadata(source_type=SourceType.MANUAL, confidence=float("inf"))
    assert False, "Should raise ValueError"
except ValueError:
    pass

print("Confidence validation: boundary checks passed")

# ── 5. SourceType coercion ──────────────────────────────────────────
# source_type accepts both enum members and string values.

meta_from_str = ProvenanceMetadata(source_type="database", confidence=1.0)
assert meta_from_str.source_type == SourceType.DATABASE
assert isinstance(meta_from_str.source_type, SourceType)

# ── 6. ProvenanceMetadata serialization ─────────────────────────────
# to_dict() and from_dict() for JSON-compatible storage.

meta_dict = meta.to_dict()
assert meta_dict["source_type"] == "api_query"
assert meta_dict["confidence"] == 0.95
assert (
    meta_dict["source_detail"] == "Singapore Open Data API — HDB resale prices Q1 2026"
)
assert isinstance(meta_dict["extracted_at"], str), "datetime serialized to ISO string"

# Round-trip
meta_restored = ProvenanceMetadata.from_dict(meta_dict)
assert meta_restored.source_type == meta.source_type
assert meta_restored.confidence == meta.confidence
assert meta_restored.source_detail == meta.source_detail
print("ProvenanceMetadata serialization: round-trip verified")

# ── 7. Provenance[T] — typed value wrapper ──────────────────────────
# Provenance wraps a value with its metadata. It is generic over the
# value type: Provenance[float], Provenance[str], Provenance[int], etc.

price = Provenance(
    value=485000.0,
    metadata=ProvenanceMetadata(
        source_type=SourceType.API_QUERY,
        source_detail="HDB resale API — Ang Mo Kio 4-room",
        confidence=0.99,
    ),
)

assert price.value == 485000.0
assert isinstance(price.value, float)
assert price.metadata.source_type == SourceType.API_QUERY
assert price.metadata.confidence == 0.99
print(
    f"Provenance[float]: value={price.value}, source={price.metadata.source_type.value}"
)

# String provenance
address = Provenance(
    value="Block 123 Ang Mo Kio Ave 3",
    metadata=ProvenanceMetadata(
        source_type=SourceType.MANUAL,
        source_detail="User input via form",
        confidence=1.0,
    ),
)

assert address.value == "Block 123 Ang Mo Kio Ave 3"
assert address.metadata.source_type == SourceType.MANUAL

# Integer provenance
floor_area = Provenance(
    value=92,
    metadata=ProvenanceMetadata(
        source_type=SourceType.EXCEL_CELL,
        source_detail="HDB_data_2026.xlsx!Sheet1!C42",
        confidence=0.9,
    ),
)

assert floor_area.value == 92
assert floor_area.metadata.source_detail == "HDB_data_2026.xlsx!Sheet1!C42"

# ── 8. Provenance serialization ─────────────────────────────────────
# to_dict() flattens value + metadata into a single dict.

price_dict = price.to_dict()
assert price_dict["value"] == 485000.0
assert price_dict["source_type"] == "api_query"
assert price_dict["confidence"] == 0.99
assert "extracted_at" in price_dict
print(f"Provenance serialized: {list(price_dict.keys())}")

# Round-trip with from_dict()
price_restored = Provenance.from_dict(price_dict)
assert price_restored.value == price.value
assert price_restored.metadata.source_type == price.metadata.source_type
assert price_restored.metadata.confidence == price.metadata.confidence
print("Provenance serialization: round-trip verified")

# ── 9. Tracking value changes ──────────────────────────────────────
# Use previous_value and change_reason to build an audit trail.

revised_price = Provenance(
    value=492000.0,
    metadata=ProvenanceMetadata(
        source_type=SourceType.CALCULATED,
        source_detail="Inflation adjustment model v3",
        confidence=0.85,
        previous_value=485000.0,
        change_reason="Q1 2026 inflation adjustment (+1.4%)",
    ),
)

assert revised_price.value == 492000.0
assert revised_price.metadata.previous_value == 485000.0
assert revised_price.metadata.change_reason == "Q1 2026 inflation adjustment (+1.4%)"
print(f"Value change: {revised_price.metadata.previous_value} -> {revised_price.value}")

# Serialize and check previous_value survives round-trip
revised_dict = revised_price.to_dict()
assert revised_dict["previous_value"] == 485000.0
assert revised_dict["change_reason"] == "Q1 2026 inflation adjustment (+1.4%)"

restored = Provenance.from_dict(revised_dict)
assert restored.metadata.previous_value == 485000.0

# ── 10. Agent-derived provenance ────────────────────────────────────
# When an AI agent produces or transforms a value, track it as
# AGENT_DERIVED with appropriate confidence.

prediction = Provenance(
    value=510000.0,
    metadata=ProvenanceMetadata(
        source_type=SourceType.AGENT_DERIVED,
        source_detail="HDBPricePredictionAgent — XGBoost ensemble",
        confidence=0.72,
        change_reason="12-month forward prediction based on 2025-2026 trends",
    ),
)

assert prediction.metadata.source_type == SourceType.AGENT_DERIVED
assert prediction.metadata.confidence == 0.72
assert prediction.metadata.confidence < 1.0, "Predictions carry uncertainty"
print(
    f"Agent prediction: {prediction.value} (confidence={prediction.metadata.confidence})"
)

# ── 11. Edge case: None value ───────────────────────────────────────
# Provenance can wrap None to indicate a missing-but-tracked field.

missing = Provenance(
    value=None,
    metadata=ProvenanceMetadata(
        source_type=SourceType.DATABASE,
        source_detail="Legacy system — field not migrated",
        confidence=0.0,
    ),
)

assert missing.value is None
assert missing.metadata.confidence == 0.0
missing_dict = missing.to_dict()
assert missing_dict["value"] is None

restored_missing = Provenance.from_dict(missing_dict)
assert restored_missing.value is None
print("None value: provenance tracks missing data with zero confidence")

print("PASS: 01-dataflow/09_provenance")
