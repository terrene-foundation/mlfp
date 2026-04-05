// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Provenance
//!
//! OBJECTIVE: Track data lineage with ProvenanceData for audit and compliance.
//! LEVEL: Advanced
//! PARITY: Full -- Both SDKs provide FieldProvenance, ProvenanceData, and
//!         ProvenanceOptions for tracking who changed what and when.
//! VALIDATES: ProvenanceData, FieldProvenance, SourceType, ProvenanceOptions,
//!            provenance_column_name, extract_provenance_from_row
//!
//! Run: cargo run -p tutorial-dataflow --bin 09_provenance

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

fn main() {
    // ── 1. What Is Provenance? ──
    // Data provenance tracks the origin, transformations, and lineage
    // of every piece of data. Required for compliance (GDPR Article 30,
    // SOX, HIPAA) and ML model explainability.

    // ── 2. SourceType ──
    // Identifies where data came from.

    let _user_input = SourceType::UserInput;
    let _api = SourceType::ApiImport;
    let _system = SourceType::SystemGenerated;
    let _derived = SourceType::Derived;

    // ── 3. FieldProvenance ──
    // Tracks provenance per field: who provided it, when, and from where.

    let field_prov = FieldProvenance {
        field_name: "email".to_string(),
        source_type: SourceType::UserInput,
        source_id: Some("user_form_123".to_string()),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    assert_eq!(field_prov.field_name, "email");
    assert!(matches!(field_prov.source_type, SourceType::UserInput));

    // ── 4. ProvenanceData ──
    // Aggregates provenance across all fields of a record.

    let prov = ProvenanceData {
        record_id: "record_001".to_string(),
        model_name: "Contact".to_string(),
        fields: vec![
            FieldProvenance {
                field_name: "name".to_string(),
                source_type: SourceType::UserInput,
                source_id: Some("signup_form".to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
            FieldProvenance {
                field_name: "score".to_string(),
                source_type: SourceType::Derived,
                source_id: Some("ml_pipeline_v2".to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
        ],
    };

    assert_eq!(prov.model_name, "Contact");
    assert_eq!(prov.fields.len(), 2);

    // ── 5. ProvenanceOptions ──
    // Controls what provenance metadata is captured.

    let opts = ProvenanceOptions {
        enabled: true,
        track_source: true,
        track_timestamp: true,
    };

    assert!(opts.enabled);
    assert!(opts.track_source);

    // ── 6. Provenance Column Naming ──
    // provenance_column_name() generates a standard column name for
    // storing provenance JSON alongside the data field.

    let col = provenance_column_name("email");
    assert!(col.contains("email"));
    assert!(col.contains("provenance") || col.contains("prov"));

    // ── 7. Extract Provenance from Row ──
    // extract_provenance_from_row() reads provenance metadata from a
    // fetched database row that includes provenance columns.

    let mut row = ValueMap::new();
    row.insert(Arc::from("name"), Value::from("Alice"));
    row.insert(Arc::from("email"), Value::from("alice@example.com"));

    // Provenance is stored as JSON in a companion column.
    let prov_json = serde_json::json!({
        "source_type": "user_input",
        "source_id": "form_456",
        "timestamp": "2026-01-15T10:30:00Z"
    });
    let prov_col = provenance_column_name("email");
    row.insert(Arc::from(prov_col.as_str()), Value::from(prov_json.to_string().as_str()));

    let extracted = extract_provenance_from_row(&row, "email");
    assert!(
        extracted.is_some(),
        "provenance should be extractable from row"
    );

    if let Some(field_prov) = extracted {
        assert_eq!(field_prov.field_name, "email");
    }

    println!("PASS: 01-dataflow/09_provenance");
}
