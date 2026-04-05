# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Data Classification
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Apply data classification levels, retention policies, and
#            masking strategies to model fields
# LEVEL: Intermediate
# PARITY: Python-only
# VALIDATES: DataClassification, RetentionPolicy, MaskingStrategy,
#            FieldClassification, ClassificationPolicy, @classify
#            decorator, get_field_classification()
#
# Run: uv run python textbook/python/01-dataflow/06_classification.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.classification import (
    ClassificationPolicy,
    DataClassification,
    FieldClassification,
    MaskingStrategy,
    RetentionPolicy,
    classify,
    get_field_classification,
)

# ── 1. DataClassification enum — sensitivity levels ─────────────────
# Ordered from least to most sensitive.
# All enums are str-backed for JSON-friendly serialization.

assert DataClassification.PUBLIC.value == "public"
assert DataClassification.INTERNAL.value == "internal"
assert DataClassification.SENSITIVE.value == "sensitive"
assert DataClassification.PII.value == "pii"
assert DataClassification.GDPR.value == "gdpr"
assert DataClassification.HIGHLY_CONFIDENTIAL.value == "highly_confidential"

# String-backed: can construct from string
assert DataClassification("pii") == DataClassification.PII

print("DataClassification levels: PUBLIC -> HIGHLY_CONFIDENTIAL")

# ── 2. RetentionPolicy enum — how long data is kept ────────────────

assert RetentionPolicy.INDEFINITE.value == "indefinite"
assert RetentionPolicy.DAYS_30.value == "days_30"
assert RetentionPolicy.DAYS_90.value == "days_90"
assert RetentionPolicy.YEARS_1.value == "years_1"
assert RetentionPolicy.YEARS_7.value == "years_7"
assert RetentionPolicy.UNTIL_CONSENT_REVOKED.value == "until_consent_revoked"

print(
    "RetentionPolicy: INDEFINITE, DAYS_30, DAYS_90, YEARS_1, YEARS_7, UNTIL_CONSENT_REVOKED"
)

# ── 3. MaskingStrategy enum — how field values are obscured ────────

assert MaskingStrategy.NONE.value == "none"
assert MaskingStrategy.HASH.value == "hash"
assert MaskingStrategy.REDACT.value == "redact"
assert MaskingStrategy.LAST_FOUR.value == "last_four"
assert MaskingStrategy.ENCRYPT.value == "encrypt"

print("MaskingStrategy: NONE, HASH, REDACT, LAST_FOUR, ENCRYPT")

# ── 4. @classify decorator — attach classification to a class ──────
# Stack multiple @classify decorators on a class.
# Each decorator classifies one field with sensitivity, retention,
# and masking metadata.


@classify(
    "email",
    DataClassification.PII,
    RetentionPolicy.UNTIL_CONSENT_REVOKED,
    MaskingStrategy.REDACT,
)
@classify(
    "name",
    DataClassification.PII,
    RetentionPolicy.YEARS_1,
    MaskingStrategy.REDACT,
)
@classify(
    "phone",
    DataClassification.PII,
    RetentionPolicy.YEARS_1,
    MaskingStrategy.LAST_FOUR,
)
@classify(
    "notes",
    DataClassification.INTERNAL,
    RetentionPolicy.INDEFINITE,
    MaskingStrategy.NONE,
)
@dataclass
class Customer:
    name: str = ""
    email: str = ""
    phone: str = ""
    notes: str = ""


# Verify classification metadata is stored on the class
assert hasattr(Customer, "__field_classifications__")
assert len(Customer.__field_classifications__) == 4

# ── 5. get_field_classification() — look up a field's metadata ──────
# Returns a FieldClassification dataclass or None for unclassified fields.

email_fc = get_field_classification(Customer, "email")
assert email_fc is not None
assert isinstance(email_fc, FieldClassification)
assert email_fc.classification == DataClassification.PII
assert email_fc.retention == RetentionPolicy.UNTIL_CONSENT_REVOKED
assert email_fc.masking == MaskingStrategy.REDACT
print(
    f"email: classification={email_fc.classification.value}, masking={email_fc.masking.value}"
)

phone_fc = get_field_classification(Customer, "phone")
assert phone_fc is not None
assert phone_fc.masking == MaskingStrategy.LAST_FOUR

notes_fc = get_field_classification(Customer, "notes")
assert notes_fc is not None
assert notes_fc.classification == DataClassification.INTERNAL
assert notes_fc.masking == MaskingStrategy.NONE

# Unclassified field returns None
unknown_fc = get_field_classification(Customer, "nonexistent_field")
assert unknown_fc is None, "Unclassified field returns None"

# ── 6. FieldClassification — serialization ──────────────────────────

fc_dict = email_fc.to_dict()
assert fc_dict == {
    "classification": "pii",
    "retention": "until_consent_revoked",
    "masking": "redact",
}

# Round-trip: dict -> FieldClassification
fc_restored = FieldClassification.from_dict(fc_dict)
assert fc_restored == email_fc, "Round-trip serialization preserves data"
print("FieldClassification serialization: round-trip verified")

# ── 7. ClassificationPolicy — runtime policy registry ──────────────
# ClassificationPolicy maps model fields to classification metadata.
# Can register models decorated with @classify or set fields manually.

policy = ClassificationPolicy()

# Register a decorated model
policy.register_model(Customer)

# Look up classification level as a string
level = policy.classify("Customer", "email")
assert level == "pii", f"Expected 'pii', got '{level}'"

level = policy.classify("Customer", "notes")
assert level == "internal"

# Unclassified field defaults to "public"
level = policy.classify("Customer", "nonexistent")
assert level == "public", "Unclassified defaults to public"

# Unregistered model also defaults to "public"
level = policy.classify("UnknownModel", "field")
assert level == "public"

print("ClassificationPolicy: classify() lookups verified")

# ── 8. ClassificationPolicy — programmatic field setting ────────────
# Use set_field() to classify fields without the @classify decorator.

policy.set_field(
    "Order",
    "credit_card",
    DataClassification.HIGHLY_CONFIDENTIAL,
    RetentionPolicy.DAYS_90,
    MaskingStrategy.ENCRYPT,
)

cc_level = policy.classify("Order", "credit_card")
assert cc_level == "highly_confidential"

cc_fc = policy.get_field("Order", "credit_card")
assert cc_fc is not None
assert cc_fc.retention == RetentionPolicy.DAYS_90
assert cc_fc.masking == MaskingStrategy.ENCRYPT

# ── 9. ClassificationPolicy — get all fields for a model ───────────

customer_fields = policy.get_model_fields("Customer")
assert len(customer_fields) == 4, "Four classified fields"
assert "email" in customer_fields
assert "name" in customer_fields
assert "phone" in customer_fields
assert "notes" in customer_fields

print(f"Customer classified fields: {list(customer_fields.keys())}")

# ── 10. Retention days lookup ───────────────────────────────────────
# Convert RetentionPolicy enum to concrete days.

days = policy.retention_days_for_policy(RetentionPolicy.DAYS_30)
assert days == 30

days = policy.retention_days_for_policy(RetentionPolicy.YEARS_1)
assert days == 365

days = policy.retention_days_for_policy(RetentionPolicy.YEARS_7)
assert days == 2555

# Indefinite and consent-based return None (no fixed expiry)
days = policy.retention_days_for_policy(RetentionPolicy.INDEFINITE)
assert days is None

days = policy.retention_days_for_policy(RetentionPolicy.UNTIL_CONSENT_REVOKED)
assert days is None

# Default retention days per classification level
days = policy.get_retention_days("sensitive")
assert days == 365, "Sensitive default: 1 year"

days = policy.get_retention_days("highly_confidential")
assert days == 2555, "Highly confidential default: ~7 years"

days = policy.get_retention_days("public")
assert days is None, "Public: no retention limit"

print("Retention days: lookups verified")

# ── 11. Edge case: @classify with defaults ──────────────────────────
# retention defaults to INDEFINITE, masking defaults to NONE


@classify("status", DataClassification.INTERNAL)
@dataclass
class MinimalModel:
    status: str = ""


fc = get_field_classification(MinimalModel, "status")
assert fc is not None
assert fc.classification == DataClassification.INTERNAL
assert fc.retention == RetentionPolicy.INDEFINITE, "Default retention"
assert fc.masking == MaskingStrategy.NONE, "Default masking"

print("PASS: 01-dataflow/06_classification")
