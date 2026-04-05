// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / Hello Nexus
//!
//! OBJECTIVE: Create a Nexus instance, register handlers, and inspect the registration.
//! LEVEL: Basic
//! PARITY: Full -- Python has Nexus() with same constructor; Rust has Nexus::new().
//!         Both share handler registration and handler_count().
//! VALIDATES: Nexus, ClosureHandler, handler(), handler_count
//!
//! Run: cargo run -p tutorial-nexus --bin 01_hello_nexus

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. Create a Nexus Instance ──
    // Nexus is the multi-channel deployment platform. Register a handler
    // once and it becomes available as HTTP API, CLI, and MCP tool.

    let mut nexus = Nexus::new();
    assert_eq!(nexus.handler_count(), 0);

    // ── 2. Register a Handler ──
    // ClosureHandler wraps an async closure as a handler.
    // with_params() declares the handler's input parameters for
    // OpenAPI generation and CLI argument parsing.

    nexus.handler(
        "greet",
        ClosureHandler::with_params(
            |inputs: ValueMap| async move {
                let name = inputs
                    .get("name" as &str)
                    .and_then(|v| v.as_str())
                    .unwrap_or("World");
                Ok(Value::from(format!("Hello, {name}!")))
            },
            vec![HandlerParam::new("name", HandlerParamType::String)],
        ),
    );

    assert_eq!(nexus.handler_count(), 1);

    // ── 3. Register Multiple Handlers ──

    nexus.handler(
        "add",
        ClosureHandler::with_params(
            |inputs: ValueMap| async move {
                let a = inputs
                    .get("a" as &str)
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let b = inputs
                    .get("b" as &str)
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                Ok(Value::Integer(a + b))
            },
            vec![
                HandlerParam::new("a", HandlerParamType::Integer),
                HandlerParam::new("b", HandlerParamType::Integer),
            ],
        ),
    );

    nexus.handler(
        "echo",
        ClosureHandler::with_params(
            |inputs: ValueMap| async move {
                let message = inputs
                    .get("message" as &str)
                    .cloned()
                    .unwrap_or(Value::Null);
                Ok(message)
            },
            vec![HandlerParam::new("message", HandlerParamType::String)],
        ),
    );

    assert_eq!(nexus.handler_count(), 3);

    // ── 4. Handler Discovery ──
    // Registered handlers are available for all three channels:
    //   HTTP: POST /api/greet, POST /api/add, POST /api/echo
    //   CLI:  nexus greet --name "Alice"
    //   MCP:  tool_call("greet", {"name": "Alice"})

    // The handler registry tracks all registered handlers by name.
    // In a real app, you'd call nexus.start() to begin serving.

    println!("PASS: 02-nexus/01_hello_nexus");
}
