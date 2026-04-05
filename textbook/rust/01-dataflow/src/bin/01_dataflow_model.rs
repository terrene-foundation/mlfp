// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Model Definition
//!
//! OBJECTIVE: Define database models using ModelDefinition builder and FieldType enum.
//! LEVEL: Basic
//! PARITY: Full -- Python uses @db.model decorator with type annotations;
//!         Rust uses ModelDefinition::new() with builder pattern and FieldType enum.
//! VALIDATES: DataFlow, ModelDefinition, FieldType, FieldDef, register_model
//!
//! Run: cargo run -p tutorial-dataflow --bin 01_dataflow_model

use kailash_dataflow::prelude::*;

#[tokio::main]
async fn main() {
    // ── 1. Create a DataFlow Instance ──
    // DataFlow connects to a database. SQLite in-memory is simplest for
    // tutorials. In production, use PostgreSQL via a connection string.

    let mut df = DataFlow::new("sqlite::memory:")
        .await
        .expect("connect to sqlite");

    // ── 2. Define a Model ──
    // ModelDefinition describes a table: name, table name, and fields.
    // Each field has a name, type, and optional configuration via closure.
    //
    // Python equivalent:
    //   @db.model
    //   class User:
    //       id: int = field(primary_key=True)
    //       name: str
    //       email: str

    let user_model = ModelDefinition::new("User", "users")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("name", FieldType::Text, |f| f.required())
        .field("email", FieldType::Text, |f| f.required().unique())
        .field("age", FieldType::Integer, |f| f)
        .auto_timestamps();

    // auto_timestamps() adds created_at and updated_at fields automatically.
    // These are managed by the framework -- never set them manually.

    // ── 3. Inspect Model Metadata ──
    // The model carries metadata about its fields before registration.

    assert_eq!(user_model.name(), "User");
    assert_eq!(user_model.table_name(), "users");

    let fields = user_model.fields();
    assert!(fields.len() >= 4); // id, name, email, age (+ timestamps)

    // ── 4. FieldType Enum ──
    // FieldType maps to SQL column types via the dialect layer.
    //
    //   Integer  -> INTEGER / INT
    //   Text     -> TEXT / VARCHAR
    //   Boolean  -> BOOLEAN / INT (SQLite)
    //   Float    -> REAL / DOUBLE
    //   DateTime -> DATETIME / TIMESTAMP
    //   Json     -> JSON / TEXT (SQLite)
    //   Blob     -> BLOB / BYTEA

    let product_model = ModelDefinition::new("Product", "products")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("name", FieldType::Text, |f| f.required())
        .field("price", FieldType::Float, |f| f.required())
        .field("active", FieldType::Boolean, |f| f.default_value("true"))
        .field("metadata", FieldType::Json, |f| f)
        .field("image", FieldType::Blob, |f| f)
        .auto_timestamps();

    assert_eq!(product_model.name(), "Product");

    // ── 5. Register Models ──
    // register_model() validates the definition and creates the table.
    // After registration, 11 CRUD/bulk workflow nodes are available:
    //   CreateUser, ReadUser, UpdateUser, DeleteUser, ListUser,
    //   UpsertUser, CountUser, BulkCreateUser, BulkUpdateUser,
    //   BulkDeleteUser, BulkUpsertUser

    df.register_model(user_model).expect("register User");
    df.register_model(product_model).expect("register Product");

    // ── 6. Model with Foreign Key ──
    // ForeignKeyRef links a field to another model's field.

    let order_model = ModelDefinition::new("Order", "orders")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("user_id", FieldType::Integer, |f| {
            f.required().foreign_key(ForeignKeyRef::new("users", "id"))
        })
        .field("total", FieldType::Float, |f| f.required())
        .field("status", FieldType::Text, |f| {
            f.required().default_value("pending")
        })
        .auto_timestamps();

    df.register_model(order_model).expect("register Order");

    // ── 7. Verify Registration ──
    // After registration, models are accessible via the DataFlow instance.

    let registered = df.registered_models();
    assert!(registered.contains(&"User".to_string()));
    assert!(registered.contains(&"Product".to_string()));
    assert!(registered.contains(&"Order".to_string()));
    assert_eq!(registered.len(), 3);

    println!("PASS: 01-dataflow/01_dataflow_model");
}
