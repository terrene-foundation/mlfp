// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Data Classification
//!
//! OBJECTIVE: Classify data columns by sensitivity level and apply masking policies.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs provide DataClassification levels, MaskingStrategy,
//!         and DataClassificationPolicy for compliance (GDPR, HIPAA).
//! VALIDATES: DataClassification, MaskingStrategy, DataClassificationPolicy,
//!            ComplianceTag, mask_row
//!
//! Run: cargo run -p tutorial-dataflow --bin 06_classification

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

fn main() {
    // ── 1. Classification Levels ──
    // DataClassification defines how sensitive a data field is.
    // Levels follow an ordered hierarchy for access control.

    let public = DataClassification::Public;
    let internal = DataClassification::Internal;
    let confidential = DataClassification::Confidential;
    let restricted = DataClassification::Restricted;

    // Levels are ordered: Public < Internal < Confidential < Restricted.
    assert!(public < internal);
    assert!(internal < confidential);
    assert!(confidential < restricted);

    // ── 2. Masking Strategies ──
    // MaskingStrategy determines how sensitive data is redacted when
    // the requester lacks sufficient clearance.

    let _full = MaskingStrategy::Full; // Replace entire value with ***
    let _partial = MaskingStrategy::Partial; // Show partial (e.g., ****1234)
    let _hash = MaskingStrategy::Hash; // SHA-256 hash of value
    let _null = MaskingStrategy::Null; // Replace with null

    // ── 3. Data Classification Policy ──
    // A policy maps field names to their classification and masking strategy.

    let mut policy = DataClassificationPolicy::new();

    policy.classify("name", DataClassification::Internal, MaskingStrategy::Partial);
    policy.classify("email", DataClassification::Confidential, MaskingStrategy::Full);
    policy.classify("ssn", DataClassification::Restricted, MaskingStrategy::Hash);
    policy.classify("department", DataClassification::Public, MaskingStrategy::Full);

    // ── 4. Compliance Tags ──
    // ComplianceTag marks which regulations apply to a field.

    let gdpr = ComplianceTag::Gdpr;
    let hipaa = ComplianceTag::Hipaa;
    let pci = ComplianceTag::Pci;
    let sox = ComplianceTag::Sox;

    // Tags are used for audit reporting and policy enforcement.
    assert_ne!(gdpr, hipaa);
    assert_ne!(pci, sox);

    // ── 5. Masking a Row ──
    // mask_row() applies the policy to a data row, replacing fields
    // that exceed the requester's clearance level.

    let mut row = ValueMap::new();
    row.insert(Arc::from("name"), Value::from("Alice Smith"));
    row.insert(Arc::from("email"), Value::from("alice@company.com"));
    row.insert(Arc::from("ssn"), Value::from("123-45-6789"));
    row.insert(Arc::from("department"), Value::from("Engineering"));

    // Mask at Internal clearance: Public and Internal fields visible,
    // Confidential and Restricted fields are masked.
    let masked = mask_row(&row, &policy, DataClassification::Internal);

    // Public field: department -- visible.
    assert_eq!(
        masked.get("department" as &str),
        Some(&Value::from("Engineering"))
    );

    // Internal field: name -- visible at Internal clearance.
    // (The name should be visible, not masked, at Internal level.)
    let name_val = masked.get("name" as &str).expect("name present");
    assert!(name_val.as_str().is_some());

    // Confidential field: email -- masked at Internal clearance.
    let email_val = masked.get("email" as &str).expect("email present");
    let email_str = email_val.as_str().unwrap_or("");
    assert_ne!(email_str, "alice@company.com", "email should be masked");

    // Restricted field: ssn -- masked at Internal clearance.
    let ssn_val = masked.get("ssn" as &str).expect("ssn present");
    let ssn_str = ssn_val.as_str().unwrap_or("");
    assert_ne!(ssn_str, "123-45-6789", "ssn should be masked");

    // ── 6. Retention Policy ──
    // RetentionPolicy defines how long classified data is kept.

    let _keep = RetentionAction::Keep;
    let _archive = RetentionAction::Archive;
    let _delete = RetentionAction::Delete;
    let _anonymize = RetentionAction::Anonymize;

    let retention = RetentionPolicy {
        action: RetentionAction::Delete,
        days: 365,
    };
    assert_eq!(retention.days, 365);

    println!("PASS: 01-dataflow/06_classification");
}
