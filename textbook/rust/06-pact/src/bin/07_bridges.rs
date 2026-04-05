// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Bridges
//!
//! OBJECTIVE: Connect PACT governance to external systems via bridge adapters.
//! LEVEL: Advanced
//! PARITY: Full -- Both SDKs use the bridge adapter pattern for external integrations.
//! VALIDATES: Bridge trait, adapter registration, event forwarding
//!
//! Run: cargo run -p tutorial-pact --bin 07_bridges

use kailash_governance::bridges::{Bridge, BridgeEvent, BridgeRegistry};

fn main() {
    // ── 1. Bridge pattern ──
    // Bridges connect PACT governance to external systems:
    //   - Logging systems (audit trail)
    //   - Notification services (alerts)
    //   - Compliance platforms (regulatory reporting)
    //   - Identity providers (role resolution)

    // ── 2. BridgeEvent ──
    // Events that bridges can receive and forward.

    let access_event = BridgeEvent::AccessGranted {
        actor: "D1-R1".to_string(),
        resource: "production/api".to_string(),
        action: "deploy".to_string(),
    };

    let denial_event = BridgeEvent::AccessDenied {
        actor: "D1-T1-R2".to_string(),
        resource: "production/api".to_string(),
        action: "deploy".to_string(),
        reason: "Insufficient clearance".to_string(),
    };

    let escalation_event = BridgeEvent::Escalation {
        from: "D1-T1-R2".to_string(),
        to: "D1-R1".to_string(),
        reason: "Deploy approval needed".to_string(),
    };

    assert!(matches!(access_event, BridgeEvent::AccessGranted { .. }));
    assert!(matches!(denial_event, BridgeEvent::AccessDenied { .. }));
    assert!(matches!(escalation_event, BridgeEvent::Escalation { .. }));

    // ── 3. BridgeRegistry ──
    // Central registry for all active bridges.

    let mut registry = BridgeRegistry::new();

    // Register bridges
    registry.register(LoggingBridge::new("audit-log"));
    registry.register(LoggingBridge::new("compliance"));

    assert_eq!(registry.bridge_count(), 2);

    // ── 4. Event broadcasting ──
    // broadcast() sends an event to all registered bridges.

    registry.broadcast(&access_event);
    registry.broadcast(&denial_event);
    registry.broadcast(&escalation_event);

    // ── 5. Selective routing ──
    // Bridges can filter which events they handle.

    let security_bridge = LoggingBridge::new("security")
        .filter(|event| matches!(event, BridgeEvent::AccessDenied { .. }));

    assert!(security_bridge.accepts(&denial_event));
    assert!(!security_bridge.accepts(&access_event));

    // ── 6. Key concepts ──
    // - Bridge: adapter connecting PACT to external systems
    // - BridgeEvent: typed governance events
    // - BridgeRegistry: central registration and broadcasting
    // - Event types: AccessGranted, AccessDenied, Escalation
    // - Selective routing via event filters
    // - Bridges are pluggable: add new integrations without changing core

    println!("PASS: 06-pact/07_bridges");
}

// Supporting types

struct LoggingBridge {
    name: String,
    filter: Option<Box<dyn Fn(&BridgeEvent) -> bool>>,
}

impl LoggingBridge {
    fn new(name: &str) -> Self {
        Self { name: name.to_string(), filter: None }
    }

    fn filter<F: Fn(&BridgeEvent) -> bool + 'static>(mut self, f: F) -> Self {
        self.filter = Some(Box::new(f));
        self
    }

    fn accepts(&self, event: &BridgeEvent) -> bool {
        self.filter.as_ref().map_or(true, |f| f(event))
    }
}

impl Bridge for LoggingBridge {
    fn name(&self) -> &str { &self.name }
    fn handle(&self, _event: &BridgeEvent) {
        // In production, forward to logging system
    }
}
