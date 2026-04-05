# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# Kailash SDK Textbook — DataFlow / Field-Level Validators
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use field-level validators (email, range, pattern, length, etc.)
#            to validate model instance data at runtime
# LEVEL: Intermediate
# PARITY: Python-only
# VALIDATES: field_validator, email_validator, range_validator,
#            pattern_validator, length_validator, phone_validator,
#            url_validator, uuid_validator, one_of_validator,
#            validate_model, ValidationResult, FieldValidationError
#
# Run: uv run python textbook/python/01-dataflow/05_validators.py
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass

from dataflow.validation import (
    FieldValidationError,
    ValidationResult,
    email_validator,
    field_validator,
    length_validator,
    one_of_validator,
    pattern_validator,
    phone_validator,
    range_validator,
    url_validator,
    uuid_validator,
    validate_model,
)

# ── 1. Simple validators — value -> bool ────────────────────────────
# These validators take a single value and return True/False.
# They are used directly, not as factories.

# Email validator: RFC 5322 simplified pattern
assert email_validator("alice@example.com") is True
assert email_validator("bob@terrene.org") is True
assert email_validator("not-an-email") is False
assert email_validator("") is False
assert email_validator(42) is False, "Non-string returns False"

# URL validator: requires http/https scheme
assert url_validator("https://terrene.org") is True
assert url_validator("http://example.com/path?q=1") is True
assert url_validator("ftp://files.example.com") is False, "Only http/https"
assert url_validator("example.com") is False, "Scheme required"

# UUID validator: any UUID version
assert uuid_validator("550e8400-e29b-41d4-a716-446655440000") is True
assert uuid_validator("not-a-uuid") is False

# Phone validator: E.164 and common formats
assert phone_validator("+65 6234 5678") is True
assert phone_validator("+1-555-123-4567") is True
assert phone_validator("12345") is False, "Too short"

print("Simple validators: all checks passed")

# ── 2. Factory validators — return a Callable ──────────────────────
# These take configuration parameters and return a validator function.

# length_validator: checks string/sequence length bounds
check_name = length_validator(min_len=1, max_len=50)
assert check_name("Alice") is True
assert check_name("") is False, "Below min_len"
assert check_name("A" * 51) is False, "Above max_len"

# range_validator: checks numeric value bounds
check_age = range_validator(min_val=0, max_val=150)
assert check_age(25) is True
assert check_age(-1) is False, "Below min_val"
assert check_age(200) is False, "Above max_val"
assert check_age("twenty") is False, "Non-numeric returns False"

# Unbounded: only min or only max
check_positive = range_validator(min_val=0)
assert check_positive(0) is True
assert check_positive(999999) is True
assert check_positive(-1) is False

# pattern_validator: checks string against a regex (fullmatch)
check_code = pattern_validator(r"[A-Z]{2}-\d{4}")
assert check_code("SG-1234") is True
assert check_code("sg-1234") is False, "Case-sensitive fullmatch"
assert check_code("SG-12345") is False, "Extra digit fails fullmatch"

# one_of_validator: checks membership in an allowed set
check_tier = one_of_validator(["free", "pro", "enterprise"])
assert check_tier("pro") is True
assert check_tier("premium") is False, "Not in allowed set"

print("Factory validators: all checks passed")

# ── 3. Decorating a model class with @field_validator ───────────────
# Stack multiple @field_validator decorators on a class.
# Validators are stored in __field_validators__ and run by validate_model().


@field_validator("email", email_validator)
@field_validator("name", length_validator(min_len=1, max_len=100))
@field_validator("age", range_validator(min_val=0, max_val=150))
@field_validator("tier", one_of_validator(["free", "pro", "enterprise"]))
@dataclass
class UserProfile:
    name: str = ""
    email: str = ""
    age: int = 0
    tier: str = "free"


# Verify validators were registered on the class
assert hasattr(UserProfile, "__field_validators__"), "Validators attached to class"
assert len(UserProfile.__field_validators__) == 4, "Four validators registered"

# ── 4. validate_model() — run all validators on an instance ─────────
# Returns a ValidationResult that collects ALL errors (not fail-fast).

# Valid instance
good_user = UserProfile(name="Alice", email="alice@example.com", age=30, tier="pro")
result = validate_model(good_user)

assert isinstance(result, ValidationResult)
assert result.valid is True, "All validators pass"
assert len(result.errors) == 0
print(f"Valid user: valid={result.valid}, errors={len(result.errors)}")

# Invalid instance — multiple errors collected
bad_user = UserProfile(name="", email="not-an-email", age=-5, tier="premium")
result = validate_model(bad_user)

assert result.valid is False, "Validation fails"
assert len(result.errors) == 4, "All four fields fail validation"
print(f"Invalid user: valid={result.valid}, errors={len(result.errors)}")

# ── 5. Inspecting ValidationResult and FieldValidationError ─────────
# Each error is a FieldValidationError with field, message, validator, value.

for err in result.errors:
    assert isinstance(err, FieldValidationError)
    assert isinstance(err.field, str)
    assert isinstance(err.message, str)
    assert isinstance(err.validator, str)

# Check specific errors
error_fields = {e.field for e in result.errors}
assert "name" in error_fields, "name validation failed"
assert "email" in error_fields, "email validation failed"
assert "age" in error_fields, "age validation failed"
assert "tier" in error_fields, "tier validation failed"

# Serialize to dict
result_dict = result.to_dict()
assert result_dict["valid"] is False
assert len(result_dict["errors"]) == 4
print(f"Serialized result: valid={result_dict['valid']}")

# ── 6. Partial validation — only some fields fail ──────────────────

partial_bad = UserProfile(name="Bob", email="bad", age=25, tier="free")
result = validate_model(partial_bad)

assert result.valid is False
assert len(result.errors) == 1, "Only email fails"
assert result.errors[0].field == "email"
print(f"Partial invalid: {result.errors[0].field} failed")

# ── 7. Merging results ─────────────────────────────────────────────
# ValidationResult.merge() combines errors from multiple validations.

result_a = ValidationResult()
result_a.add_error(
    field_name="field_a",
    message="Field A invalid",
    validator="custom",
    value="bad",
)

result_b = ValidationResult()
result_b.add_error(
    field_name="field_b",
    message="Field B invalid",
    validator="custom",
    value="worse",
)

result_a.merge(result_b)
assert len(result_a.errors) == 2, "Merged errors from both results"
assert result_a.valid is False

# ── 8. Edge cases ──────────────────────────────────────────────────

# range_validator rejects NaN and infinity
import math

check_finite = range_validator(min_val=0, max_val=100)
assert check_finite(float("nan")) is False, "NaN rejected"
assert check_finite(float("inf")) is False, "Infinity rejected"

# length_validator works on lists too
check_items = length_validator(min_len=1, max_len=5)
assert check_items([1, 2, 3]) is True
assert check_items([]) is False, "Empty list below min_len"

# Invalid range_validator construction
try:
    range_validator(min_val=100, max_val=0)
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected: min_val > max_val

# Invalid length_validator construction
try:
    length_validator(min_len=10, max_len=5)
    assert False, "Should raise ValueError"
except ValueError:
    pass  # Expected: min_len > max_len

print("Edge cases: all checks passed")

print("PASS: 01-dataflow/05_validators")
