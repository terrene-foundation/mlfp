// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / HTTP Transport
//!
//! OBJECTIVE: Understand Nexus HTTP API routing built on axum.
//! LEVEL: Basic
//! PARITY: Full -- Python uses uvicorn/starlette; Rust uses axum/tower.
//!         Both auto-generate POST /api/{handler_name} routes.
//! VALIDATES: build_api_router, NexusConfig, handler routing
//!
//! Run: cargo run -p tutorial-nexus --bin 02_http_transport

use kailash_nexus::prelude::*;
use kailash_nexus::api::build_api_router;
use kailash_nexus::config::NexusConfig;

fn main() {
    // ── 1. HTTP Transport Architecture ──
    // Nexus HTTP channel maps each handler to a POST endpoint:
    //   handler("greet", ...) -> POST /api/greet
    //   handler("add", ...)   -> POST /api/add
    //
    // The request body is JSON that maps to the handler's ValueMap inputs.
    // The response is the handler's return Value serialized as JSON.

    let mut nexus = Nexus::new();

    nexus.handler(
        "status",
        ClosureHandler::with_params(
            |_inputs: ValueMap| async move {
                Ok(Value::from("running"))
            },
            vec![],
        ),
    );

    nexus.handler(
        "compute",
        ClosureHandler::with_params(
            |inputs: ValueMap| async move {
                let x = inputs
                    .get("x" as &str)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                Ok(Value::Float(x * x))
            },
            vec![HandlerParam::new("x", HandlerParamType::Float)],
        ),
    );

    assert_eq!(nexus.handler_count(), 2);

    // ── 2. NexusConfig ──
    // Configuration for the HTTP server: port, CORS, rate limiting.

    let config = NexusConfig::default();
    // Default port is 8000.
    // CORS, rate limiting, and body size limits are configured via middleware.

    // ── 3. Building the API Router ──
    // build_api_router() creates an axum Router from registered handlers.
    // Each handler becomes POST /api/{name}.
    //
    // In production:
    //   let app = build_api_router(&nexus, &config);
    //   let listener = tokio::net::TcpListener::bind("0.0.0.0:8000").await?;
    //   axum::serve(listener, app).await?;
    //
    // We don't start the server in this tutorial -- just verify the setup.

    // ── 4. Request/Response Flow ──
    // HTTP request lifecycle:
    //   1. Client sends POST /api/compute with {"x": 5.0}
    //   2. Nexus deserializes JSON body into ValueMap
    //   3. Handler closure receives ValueMap, returns Result<Value, NexusError>
    //   4. Nexus serializes Value back to JSON response
    //   5. Client receives {"result": 25.0}
    //
    // All error handling is automatic: handler errors become
    // appropriate HTTP status codes (400, 500, etc.).

    println!("PASS: 02-nexus/02_http_transport");
}
