// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Express CRUD
//!
//! OBJECTIVE: Use DataFlowExpress for single-record CRUD operations (create, read, update, delete, list).
//! LEVEL: Basic
//! PARITY: Full -- Python uses await db.express.create(); Rust uses df.express().create().await.
//!         Express is the default for simple CRUD (~23x faster than workflow-based CRUD).
//! VALIDATES: DataFlowExpress, create, read, update, delete, list, count
//!
//! Run: cargo run -p tutorial-dataflow --bin 02_express_crud

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // ── Setup ──
    let mut df = DataFlow::new("sqlite::memory:")
        .await
        .expect("connect");

    let model = ModelDefinition::new("Task", "tasks")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("title", FieldType::Text, |f| f.required())
        .field("done", FieldType::Boolean, |f| f.default_value("false"))
        .field("priority", FieldType::Integer, |f| f.default_value("0"))
        .auto_timestamps();

    df.register_model(model).expect("register Task");

    let express = df.express();

    // ── Create ──
    // express.create(model_name, data) inserts a single record.
    // Data is a ValueMap (BTreeMap<Arc<str>, Value>).

    let mut data = ValueMap::new();
    data.insert(Arc::from("title"), Value::from("Learn Kailash DataFlow"));
    data.insert(Arc::from("priority"), Value::Integer(1));

    let created = express.create("Task", data).await.expect("create task");
    assert!(created.contains_key("id" as &str));
    let task_id = created
        .get("id" as &str)
        .expect("id field")
        .as_i64()
        .expect("id is integer");
    assert!(task_id > 0);

    // ── Read ──
    // express.read(model_name, id) fetches a single record by primary key.

    let task = express
        .read("Task", &task_id.to_string())
        .await
        .expect("read task");
    assert_eq!(
        task.get("title" as &str),
        Some(&Value::from("Learn Kailash DataFlow"))
    );

    // ── Update ──
    // express.update(model_name, id, fields) updates specific fields.

    let mut updates = ValueMap::new();
    updates.insert(Arc::from("done"), Value::Bool(true));
    updates.insert(Arc::from("priority"), Value::Integer(5));

    let updated = express
        .update("Task", &task_id.to_string(), updates)
        .await
        .expect("update task");
    assert_eq!(updated.get("done" as &str), Some(&Value::Bool(true)));
    assert_eq!(updated.get("priority" as &str), Some(&Value::Integer(5)));

    // ── Create More Records ──

    for title in ["Build a workflow", "Write tests", "Deploy to production"] {
        let mut data = ValueMap::new();
        data.insert(Arc::from("title"), Value::from(title));
        express.create("Task", data).await.expect("create");
    }

    // ── Count ──
    // express.count(model_name) returns the total record count.

    let count = express.count("Task").await.expect("count");
    assert_eq!(count, 4);

    // ── List ──
    // express.list(model_name, options) returns paginated results.

    let all_tasks = express
        .list("Task", ListOptions::default())
        .await
        .expect("list all");
    assert_eq!(all_tasks.len(), 4);

    // List with limit.
    let limited = express
        .list(
            "Task",
            ListOptions {
                limit: Some(2),
                ..Default::default()
            },
        )
        .await
        .expect("list limited");
    assert_eq!(limited.len(), 2);

    // ── Delete ──
    // express.delete(model_name, id) removes a record.

    express
        .delete("Task", &task_id.to_string())
        .await
        .expect("delete task");

    let count = express.count("Task").await.expect("count after delete");
    assert_eq!(count, 3);

    // Verify the deleted record is gone.
    let result = express.read("Task", &task_id.to_string()).await;
    assert!(result.is_err());

    println!("PASS: 01-dataflow/02_express_crud");
}
