// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / Event Bus
//!
//! OBJECTIVE: Use EventBus for publish-subscribe messaging between handlers and clients.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs provide EventBus with publish/subscribe.
//!         Rust uses tokio broadcast channels; Python uses asyncio queues.
//! VALIDATES: EventBus, NexusEvent, publish, subscribe pattern
//!
//! Run: cargo run -p tutorial-nexus --bin 07_event_bus

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. EventBus ──
    // The EventBus enables real-time communication between:
    //   - Handlers emitting progress updates
    //   - SSE clients receiving live notifications
    //   - WebSocket clients in bidirectional channels
    //   - Background services reporting status

    let bus = EventBus::new();

    // ── 2. Publishing Events ──
    // NexusEvent carries a type string (for filtering) and a data payload.

    bus.publish(NexusEvent::new("user.created", "Alice signed up"));
    bus.publish(NexusEvent::new("user.created", "Bob signed up"));
    bus.publish(NexusEvent::new("order.placed", "Order #1234 placed"));

    // ── 3. Event Types as Namespaces ──
    // Use dot-separated event types for hierarchical filtering:
    //   "user.created"   -- specific event
    //   "user.*"         -- all user events (conceptual -- filter in subscriber)
    //   "system.health"  -- system heartbeat

    let events = vec![
        NexusEvent::new("deploy.started", "v2.1 deploying"),
        NexusEvent::new("deploy.progress", "50%"),
        NexusEvent::new("deploy.completed", "v2.1 live"),
    ];

    for event in &events {
        bus.publish(event.clone());
    }

    // ── 4. Integration with Handlers ──
    // Handlers can access the EventBus to emit progress events:
    //
    //   nexus.handler("long_task", ClosureHandler::with_params(
    //       |inputs: ValueMap| async move {
    //           let bus = /* injected via handler context */;
    //           bus.publish(NexusEvent::new("task.started", "..."));
    //           // ... do work ...
    //           bus.publish(NexusEvent::new("task.completed", "..."));
    //           Ok(Value::from("done"))
    //       },
    //       vec![],
    //   ));
    //
    // SSE clients see these events in real-time.

    // ── 5. EventBus Patterns ──
    // Pattern: Progress Tracking
    //   Emit periodic events during long-running operations.
    //   Client displays a progress bar based on event data.
    //
    // Pattern: Live Dashboard
    //   Emit metric events (CPU, memory, request count).
    //   Dashboard subscribes and updates in real-time.
    //
    // Pattern: Notification Feed
    //   Emit user-facing events (order updates, messages).
    //   Client renders notification toast on each event.

    println!("PASS: 02-nexus/07_event_bus");
}
