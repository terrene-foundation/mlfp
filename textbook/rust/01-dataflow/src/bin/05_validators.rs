// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Validators
//!
//! OBJECTIVE: Apply field-level and cross-field validators to enforce data integrity.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs provide EmailValidator, LengthValidator, PatternValidator,
//!         RangeValidator, etc. Rust uses trait objects (Box<dyn FieldValidator>).
//! VALIDATES: FieldValidator, EmailValidator, LengthValidator, PatternValidator,
//!            RangeValidator, ValidationLayer, CrossFieldRule
//!
//! Run: cargo run -p tutorial-dataflow --bin 05_validators

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    let mut df = DataFlow::new("sqlite::memory:")
        .await
        .expect("connect");

    // ── 1. Built-in Validators ──
    // DataFlow provides validators that run before database writes.
    // Invalid data is rejected with a structured ValidationError.

    // EmailValidator: checks RFC 5322 format.
    let email_v = EmailValidator::new();
    assert!(email_v.validate(&Value::from("alice@example.com")).is_ok());
    assert!(email_v.validate(&Value::from("not-an-email")).is_err());

    // LengthValidator: enforces string length bounds.
    let len_v = LengthValidator::new(2, 50);
    assert!(len_v.validate(&Value::from("ok")).is_ok());
    assert!(len_v.validate(&Value::from("x")).is_err()); // too short
    assert!(len_v
        .validate(&Value::from("x".repeat(51).as_str()))
        .is_err()); // too long

    // RangeValidator: enforces numeric bounds.
    let range_v = RangeValidator::new(0.0, 200.0);
    assert!(range_v.validate(&Value::Integer(100)).is_ok());
    assert!(range_v.validate(&Value::Integer(-1)).is_err());
    assert!(range_v.validate(&Value::Integer(201)).is_err());

    // PatternValidator: enforces regex patterns.
    let pattern_v = PatternValidator::new(r"^[A-Z]{2}\d{4}$").expect("valid regex");
    assert!(pattern_v.validate(&Value::from("AB1234")).is_ok());
    assert!(pattern_v.validate(&Value::from("abc")).is_err());

    // ── 2. Model with Validators ──
    // Attach validators to fields during model definition.

    let model = ModelDefinition::new("Contact", "contacts")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("name", FieldType::Text, |f| {
            f.required()
                .validator(Box::new(LengthValidator::new(1, 100)))
        })
        .field("email", FieldType::Text, |f| {
            f.required().validator(Box::new(EmailValidator::new()))
        })
        .field("phone", FieldType::Text, |f| {
            f.validator(Box::new(PhoneValidator::new()))
        })
        .field("age", FieldType::Integer, |f| {
            f.validator(Box::new(RangeValidator::new(0.0, 150.0)))
        })
        .auto_timestamps();

    df.register_model(model).expect("register Contact");
    let express = df.express();

    // ── 3. Valid Data Passes ──
    let mut valid = ValueMap::new();
    valid.insert(Arc::from("name"), Value::from("Alice Smith"));
    valid.insert(Arc::from("email"), Value::from("alice@example.com"));
    valid.insert(Arc::from("age"), Value::Integer(30));

    let result = express.create("Contact", valid).await;
    assert!(result.is_ok());

    // ── 4. Invalid Data Rejected ──
    // Invalid email is caught by the validator, not by the database.

    let mut invalid = ValueMap::new();
    invalid.insert(Arc::from("name"), Value::from("Bob"));
    invalid.insert(Arc::from("email"), Value::from("not-an-email"));

    let result = express.create("Contact", invalid).await;
    assert!(result.is_err());

    // Invalid range.
    let mut bad_age = ValueMap::new();
    bad_age.insert(Arc::from("name"), Value::from("Charlie"));
    bad_age.insert(Arc::from("email"), Value::from("charlie@test.com"));
    bad_age.insert(Arc::from("age"), Value::Integer(-5));

    let result = express.create("Contact", bad_age).await;
    assert!(result.is_err());

    // ── 5. Validation Layer ──
    // ValidationLayer aggregates multiple validators for a model
    // and provides batch validation with detailed error reporting.

    let mut layer = ValidationLayer::new();
    layer.add_rule(ValidationRule::required("name"));
    layer.add_rule(ValidationRule::required("email"));

    // Cross-field rules validate relationships between fields.
    layer.add_cross_field_rule(CrossFieldRule::new(
        CrossFieldComparison::LessThan {
            field_a: "start_date".into(),
            field_b: "end_date".into(),
        },
        "start_date must be before end_date",
    ));

    // Validate a record.
    let mut record = ValueMap::new();
    record.insert(Arc::from("name"), Value::from("Test"));
    record.insert(Arc::from("email"), Value::from("test@test.com"));

    let errors = layer.validate(&record);
    assert!(errors.is_empty(), "valid record should pass");

    // Missing required field.
    let empty = ValueMap::new();
    let errors = layer.validate(&empty);
    assert!(!errors.is_empty(), "missing fields should fail");

    println!("PASS: 01-dataflow/05_validators");
}
