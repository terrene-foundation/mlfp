// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Sync Express
//!
//! OBJECTIVE: Use DataFlowExpressSync for blocking CRUD in non-async contexts.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python has synchronous wrappers via run_sync();
//!         Rust provides DataFlowExpressSync with identical API surface.
//! VALIDATES: DataFlowExpressSync, create, read, update, delete, list
//!
//! Run: cargo run -p tutorial-dataflow --bin 04_sync_express

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

// DataFlowExpressSync wraps the async express API with blocking calls.
// Useful for CLI tools, scripts, and contexts where async is not available.

#[tokio::main]
async fn main() {
    // ── Setup ──
    let mut df = DataFlow::new("sqlite::memory:")
        .await
        .expect("connect");

    let model = ModelDefinition::new("Note", "notes")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("title", FieldType::Text, |f| f.required())
        .field("body", FieldType::Text, |f| f)
        .auto_timestamps();

    df.register_model(model).expect("register");

    // ── Get Sync Express Handle ──
    // DataFlowExpressSync provides the same API as DataFlowExpress
    // but blocks the calling thread instead of returning futures.

    let sync_express = df.express_sync();

    // ── Create (Sync) ──
    let mut data = ValueMap::new();
    data.insert(Arc::from("title"), Value::from("Meeting notes"));
    data.insert(Arc::from("body"), Value::from("Discussed roadmap"));

    let created = sync_express.create("Note", data).expect("sync create");
    let note_id = created
        .get("id" as &str)
        .expect("id")
        .as_i64()
        .expect("integer id");
    assert!(note_id > 0);

    // ── Read (Sync) ──
    let note = sync_express
        .read("Note", &note_id.to_string())
        .expect("sync read");
    assert_eq!(note.get("title" as &str), Some(&Value::from("Meeting notes")));

    // ── Update (Sync) ──
    let mut updates = ValueMap::new();
    updates.insert(Arc::from("body"), Value::from("Discussed roadmap and milestones"));

    let updated = sync_express
        .update("Note", &note_id.to_string(), updates)
        .expect("sync update");
    assert_eq!(
        updated.get("body" as &str),
        Some(&Value::from("Discussed roadmap and milestones"))
    );

    // ── List (Sync) ──
    // Create a few more notes first.
    for title in ["Design doc", "Sprint review", "Retrospective"] {
        let mut data = ValueMap::new();
        data.insert(Arc::from("title"), Value::from(title));
        sync_express.create("Note", data).expect("create");
    }

    let all = sync_express
        .list("Note", ListOptions::default())
        .expect("sync list");
    assert_eq!(all.len(), 4);

    // ── Delete (Sync) ──
    sync_express
        .delete("Note", &note_id.to_string())
        .expect("sync delete");

    let remaining = sync_express
        .list("Note", ListOptions::default())
        .expect("sync list after delete");
    assert_eq!(remaining.len(), 3);

    println!("PASS: 01-dataflow/04_sync_express");
}
