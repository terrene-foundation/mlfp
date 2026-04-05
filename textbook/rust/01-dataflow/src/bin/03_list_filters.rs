// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / List Filters
//!
//! OBJECTIVE: Use FilterCondition and FilterOp for querying records with conditions.
//! LEVEL: Basic
//! PARITY: Full -- Python uses dict-based filters; Rust uses FilterCondition::new()
//!         with FilterOp enum for type-safe query construction.
//! VALIDATES: FilterCondition, FilterOp, ListOptions with filters
//!
//! Run: cargo run -p tutorial-dataflow --bin 03_list_filters

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // ── Setup: Create and Populate ──
    let mut df = DataFlow::new("sqlite::memory:")
        .await
        .expect("connect");

    let model = ModelDefinition::new("Employee", "employees")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("name", FieldType::Text, |f| f.required())
        .field("department", FieldType::Text, |f| f.required())
        .field("salary", FieldType::Integer, |f| f.required())
        .field("active", FieldType::Boolean, |f| f.default_value("true"))
        .auto_timestamps();

    df.register_model(model).expect("register");

    let express = df.express();

    // Insert sample employees.
    let employees = vec![
        ("Alice", "Engineering", 90_000, true),
        ("Bob", "Engineering", 85_000, true),
        ("Charlie", "Marketing", 75_000, true),
        ("Diana", "Marketing", 80_000, false),
        ("Eve", "Sales", 70_000, true),
    ];

    for (name, dept, salary, active) in &employees {
        let mut data = ValueMap::new();
        data.insert(Arc::from("name"), Value::from(*name));
        data.insert(Arc::from("department"), Value::from(*dept));
        data.insert(Arc::from("salary"), Value::Integer(*salary));
        data.insert(Arc::from("active"), Value::Bool(*active));
        express.create("Employee", data).await.expect("insert");
    }

    // ── FilterOp::Eq -- Equality ──
    // Filter by exact match on a field value.

    let eng_filter = FilterCondition::new("department", FilterOp::Eq, Value::from("Engineering"));

    let eng = express
        .list(
            "Employee",
            ListOptions {
                filters: vec![eng_filter],
                ..Default::default()
            },
        )
        .await
        .expect("list engineering");
    assert_eq!(eng.len(), 2);

    // ── FilterOp::Gt -- Greater Than ──
    // Salary > 80,000

    let high_salary = FilterCondition::new("salary", FilterOp::Gt, Value::Integer(80_000));

    let rich = express
        .list(
            "Employee",
            ListOptions {
                filters: vec![high_salary],
                ..Default::default()
            },
        )
        .await
        .expect("list high salary");
    assert_eq!(rich.len(), 2); // Alice (90k), Bob (85k)

    // ── FilterOp::Lt -- Less Than ──

    let low_salary = FilterCondition::new("salary", FilterOp::Lt, Value::Integer(80_000));

    let budget = express
        .list(
            "Employee",
            ListOptions {
                filters: vec![low_salary],
                ..Default::default()
            },
        )
        .await
        .expect("list low salary");
    assert_eq!(budget.len(), 2); // Charlie (75k), Eve (70k)

    // ── Combined Filters (AND) ──
    // Multiple filters are combined with AND.

    let active_eng = express
        .list(
            "Employee",
            ListOptions {
                filters: vec![
                    FilterCondition::new("department", FilterOp::Eq, Value::from("Marketing")),
                    FilterCondition::new("active", FilterOp::Eq, Value::Bool(true)),
                ],
                ..Default::default()
            },
        )
        .await
        .expect("list active marketing");
    assert_eq!(active_eng.len(), 1); // Only Charlie (Diana is inactive)

    // ── FilterOp::Gte / FilterOp::Lte -- Range ──
    // Salary >= 75,000 AND salary <= 85,000

    let mid_range = express
        .list(
            "Employee",
            ListOptions {
                filters: vec![
                    FilterCondition::new("salary", FilterOp::Gte, Value::Integer(75_000)),
                    FilterCondition::new("salary", FilterOp::Lte, Value::Integer(85_000)),
                ],
                ..Default::default()
            },
        )
        .await
        .expect("list mid range");
    assert_eq!(mid_range.len(), 3); // Bob (85k), Charlie (75k), Diana (80k)

    // ── FilterOp::Ne -- Not Equal ──

    let not_sales = FilterCondition::new("department", FilterOp::Ne, Value::from("Sales"));
    let non_sales = express
        .list(
            "Employee",
            ListOptions {
                filters: vec![not_sales],
                ..Default::default()
            },
        )
        .await
        .expect("list not sales");
    assert_eq!(non_sales.len(), 4);

    // ── Pagination: limit + offset ──

    let page1 = express
        .list(
            "Employee",
            ListOptions {
                limit: Some(2),
                offset: Some(0),
                ..Default::default()
            },
        )
        .await
        .expect("page 1");
    assert_eq!(page1.len(), 2);

    let page2 = express
        .list(
            "Employee",
            ListOptions {
                limit: Some(2),
                offset: Some(2),
                ..Default::default()
            },
        )
        .await
        .expect("page 2");
    assert_eq!(page2.len(), 2);

    println!("PASS: 01-dataflow/03_list_filters");
}
