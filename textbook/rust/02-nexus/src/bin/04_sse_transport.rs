// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / SSE Transport
//!
//! OBJECTIVE: Understand Server-Sent Events (SSE) for real-time streaming in Nexus.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs support SSE streaming via the EventBus.
//!         Rust uses axum's SSE support; Python uses starlette's EventSourceResponse.
//! VALIDATES: EventBus, NexusEvent, WsBroadcaster concept
//!
//! Run: cargo run -p tutorial-nexus --bin 04_sse_transport

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. Server-Sent Events (SSE) ──
    // SSE is a one-way server-to-client streaming protocol over HTTP.
    // Nexus uses SSE for:
    //   - MCP tool responses (streaming token-by-token)
    //   - Real-time event notifications
    //   - Progress updates for long-running handlers

    // ── 2. EventBus ──
    // The EventBus is a publish-subscribe system for real-time events.
    // Handlers can emit events, and SSE clients receive them.

    let event_bus = EventBus::new();

    // Publish an event.
    event_bus.publish(NexusEvent::new("task_started", "Processing request #42"));
    event_bus.publish(NexusEvent::new("task_progress", "50% complete"));
    event_bus.publish(NexusEvent::new("task_completed", "Done"));

    // ── 3. NexusEvent ──
    // Events have a type (for filtering) and a payload (the data).
    // Clients can subscribe to specific event types.

    let event = NexusEvent::new("deployment", "Service v2.1 deployed");
    assert_eq!(event.event_type(), "deployment");

    // ── 4. WebSocket Broadcaster ──
    // For bidirectional communication, Nexus also supports WebSocket.
    // WsBroadcaster manages multiple connected clients.

    let _broadcaster = WsBroadcaster::new();

    // In production:
    //   broadcaster.broadcast("Hello, all clients!").await;
    //
    // WebSocket endpoints are registered alongside HTTP handlers:
    //   nexus.ws_handler("chat", ws_handler_fn);

    // ── 5. Streaming Patterns ──
    // Pattern 1: Event-driven (fire and forget)
    //   event_bus.publish(NexusEvent::new("log", "message"));
    //
    // Pattern 2: Progress tracking
    //   for i in 0..100 {
    //       event_bus.publish(NexusEvent::new("progress", format!("{i}%")));
    //   }
    //
    // Pattern 3: WebSocket chat
    //   broadcaster.broadcast(serde_json::to_string(&msg)?).await;

    println!("PASS: 02-nexus/04_sse_transport");
}
