// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- DataFlow / Transactions
//!
//! OBJECTIVE: Use DataFlowTransaction for atomic multi-operation database changes.
//! LEVEL: Intermediate
//! PARITY: Full -- Python uses `async with db.transaction() as tx:`;
//!         Rust uses `df.begin_transaction().await` with explicit commit/rollback.
//!         RAII rollback ensures no partial writes on drop.
//! VALIDATES: DataFlowTransaction, begin_transaction, commit, rollback, RAII safety
//!
//! Run: cargo run -p tutorial-dataflow --bin 07_transactions

use kailash_dataflow::prelude::*;
use kailash_value::{Value, ValueMap};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // ── Setup ──
    let mut df = DataFlow::new("sqlite::memory:")
        .await
        .expect("connect");

    let model = ModelDefinition::new("Account", "accounts")
        .field("id", FieldType::Integer, |f| f.primary_key())
        .field("name", FieldType::Text, |f| f.required())
        .field("balance", FieldType::Integer, |f| f.required())
        .auto_timestamps();

    df.register_model(model).expect("register");

    let express = df.express();

    // Create two accounts.
    let mut alice_data = ValueMap::new();
    alice_data.insert(Arc::from("name"), Value::from("Alice"));
    alice_data.insert(Arc::from("balance"), Value::Integer(1000));
    let alice = express
        .create("Account", alice_data)
        .await
        .expect("create alice");
    let alice_id = alice.get("id" as &str).unwrap().as_i64().unwrap();

    let mut bob_data = ValueMap::new();
    bob_data.insert(Arc::from("name"), Value::from("Bob"));
    bob_data.insert(Arc::from("balance"), Value::Integer(500));
    let bob = express
        .create("Account", bob_data)
        .await
        .expect("create bob");
    let bob_id = bob.get("id" as &str).unwrap().as_i64().unwrap();

    // ── 1. Successful Transaction: Transfer ──
    // Transfer 200 from Alice to Bob atomically.

    {
        let mut tx = df.begin_transaction().await.expect("begin tx");

        // Debit Alice.
        let mut debit = ValueMap::new();
        debit.insert(Arc::from("balance"), Value::Integer(800)); // 1000 - 200
        tx.update("Account", &alice_id.to_string(), debit)
            .await
            .expect("debit alice");

        // Credit Bob.
        let mut credit = ValueMap::new();
        credit.insert(Arc::from("balance"), Value::Integer(700)); // 500 + 200
        tx.update("Account", &bob_id.to_string(), credit)
            .await
            .expect("credit bob");

        // Commit makes both changes permanent.
        tx.commit().await.expect("commit transfer");
    }

    // Verify balances after successful transfer.
    let alice_after = express
        .read("Account", &alice_id.to_string())
        .await
        .expect("read alice");
    assert_eq!(
        alice_after.get("balance" as &str),
        Some(&Value::Integer(800))
    );

    let bob_after = express
        .read("Account", &bob_id.to_string())
        .await
        .expect("read bob");
    assert_eq!(
        bob_after.get("balance" as &str),
        Some(&Value::Integer(700))
    );

    // ── 2. RAII Rollback: Transaction Dropped Without Commit ──
    // If a transaction is dropped without calling commit(),
    // all changes are automatically rolled back.

    {
        let mut tx = df.begin_transaction().await.expect("begin tx");

        // Attempt to zero out Alice's balance.
        let mut zero = ValueMap::new();
        zero.insert(Arc::from("balance"), Value::Integer(0));
        tx.update("Account", &alice_id.to_string(), zero)
            .await
            .expect("zero alice");

        // tx is dropped here without commit() -- automatic rollback.
    }

    // Alice's balance is unchanged (still 800).
    let alice_check = express
        .read("Account", &alice_id.to_string())
        .await
        .expect("read alice after rollback");
    assert_eq!(
        alice_check.get("balance" as &str),
        Some(&Value::Integer(800))
    );

    // ── 3. Transaction with Create and Delete ──
    // Transactions can mix create, update, and delete operations.

    {
        let mut tx = df.begin_transaction().await.expect("begin tx");

        let mut new_account = ValueMap::new();
        new_account.insert(Arc::from("name"), Value::from("Charlie"));
        new_account.insert(Arc::from("balance"), Value::Integer(300));
        tx.create("Account", new_account)
            .await
            .expect("create charlie");

        tx.commit().await.expect("commit create");
    }

    let count = express.count("Account").await.expect("count");
    assert_eq!(count, 3); // Alice, Bob, Charlie

    println!("PASS: 01-dataflow/07_transactions");
}
