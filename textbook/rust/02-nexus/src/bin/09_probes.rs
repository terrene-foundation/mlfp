// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Nexus / Health & Readiness Probes
//!
//! OBJECTIVE: Configure Kubernetes health, readiness, and startup probes.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python uses ProbeManager with state transitions;
//!         Rust uses the same ProbeManager with Arc<Mutex<>> for thread safety.
//! VALIDATES: ProbeManager, ProbeState, ProbeResponse, state transitions,
//!            liveness/readiness/startup checks, readiness callbacks
//!
//! Run: cargo run -p tutorial-nexus --bin 09_probes

use kailash_nexus::prelude::*;

fn main() {
    // ── 1. ProbeState enum ──
    // ProbeState models the Kubernetes probe lifecycle. Transitions are
    // monotonic: Starting -> Ready -> Draining. Failed is terminal
    // (only reset() can recover from it).

    assert_eq!(ProbeState::Starting.as_str(), "starting");
    assert_eq!(ProbeState::Ready.as_str(), "ready");
    assert_eq!(ProbeState::Draining.as_str(), "draining");
    assert_eq!(ProbeState::Failed.as_str(), "failed");

    // ── 2. Create a ProbeManager ──
    // ProbeManager is thread-safe (Arc<Mutex<>> internally).

    let probes = ProbeManager::new();

    assert_eq!(probes.state(), ProbeState::Starting);
    assert!(probes.is_alive());      // Not FAILED, so alive
    assert!(!probes.is_ready());     // Not READY yet
    assert!(!probes.is_started());   // Not past STARTING yet

    // ── 3. Liveness probe (/healthz) ──
    // check_liveness() returns 200 for all states except Failed.
    // Tells Kubernetes the process is alive and should not be restarted.

    let liveness = probes.check_liveness();
    assert_eq!(liveness.status(), "ok");
    assert_eq!(liveness.http_status(), 200);

    // ── 4. Startup probe (/startup) ──
    // check_startup() returns 200 once past STARTING state.

    let startup = probes.check_startup();
    assert_eq!(startup.status(), "starting");
    assert_eq!(startup.http_status(), 503); // Not started yet

    // ── 5. Readiness probe (/readyz) ──
    // check_readiness() returns 200 only in READY state and when all
    // readiness callbacks pass.

    let readiness = probes.check_readiness();
    assert_eq!(readiness.status(), "not_ready");
    assert_eq!(readiness.http_status(), 503);

    // ── 6. State transition: Starting -> Ready ──
    // mark_ready() transitions to Ready. Returns true on success.

    let success = probes.mark_ready();
    assert!(success);
    assert_eq!(probes.state(), ProbeState::Ready);
    assert!(probes.is_alive());
    assert!(probes.is_ready());
    assert!(probes.is_started());

    // Now all three probes return 200
    assert_eq!(probes.check_liveness().http_status(), 200);
    assert_eq!(probes.check_readiness().http_status(), 200);
    assert_eq!(probes.check_startup().http_status(), 200);

    // ── 7. Workflow count tracking ──
    // ProbeManager tracks registered workflow count in probe details.

    probes.set_workflow_count(5);
    let readiness = probes.check_readiness();
    assert_eq!(readiness.http_status(), 200);

    // ── 8. Readiness callbacks ──
    // Additional readiness checks registered as closures.
    // Each must return true for readiness to pass.

    probes.add_readiness_check("db_check", || true);
    probes.add_readiness_check("model_check", || true);

    assert_eq!(probes.check_readiness().http_status(), 200);

    // ── 9. Failing readiness callback ──
    // If any callback returns false, readiness fails.

    let probes2 = ProbeManager::new();
    probes2.mark_ready();
    probes2.add_readiness_check("always_fails", || false);

    let readiness = probes2.check_readiness();
    assert_eq!(readiness.http_status(), 503);
    assert_eq!(readiness.status(), "not_ready");

    // ── 10. State transition: Ready -> Draining ──
    // mark_draining() signals graceful shutdown. Kubernetes stops
    // sending new traffic but lets existing requests complete.

    let success = probes.mark_draining();
    assert!(success);
    assert_eq!(probes.state(), ProbeState::Draining);
    assert!(probes.is_alive());      // Still alive (don't restart)
    assert!(!probes.is_ready());     // No longer accepting traffic
    assert!(probes.is_started());    // Past startup

    assert_eq!(probes.check_liveness().http_status(), 200);
    assert_eq!(probes.check_readiness().http_status(), 503);

    // ── 11. State transition: -> Failed ──
    // mark_failed() is terminal. The process should be restarted.

    let probes3 = ProbeManager::new();
    probes3.mark_ready();
    let success = probes3.mark_failed("Out of memory");
    assert!(success);
    assert_eq!(probes3.state(), ProbeState::Failed);
    assert!(!probes3.is_alive());

    let liveness = probes3.check_liveness();
    assert_eq!(liveness.http_status(), 503);
    assert_eq!(liveness.status(), "failed");

    // ── 12. Invalid transitions are rejected ──
    // The state machine enforces valid transitions.

    let probes4 = ProbeManager::new();

    // Can't go Starting -> Draining (must be Ready first)
    assert!(!probes4.mark_draining());
    assert_eq!(probes4.state(), ProbeState::Starting);

    // Can't go Failed -> Ready (Failed is terminal)
    probes4.mark_failed("crash");
    assert!(!probes4.mark_ready());
    assert_eq!(probes4.state(), ProbeState::Failed);

    // ── 13. Reset for recovery/testing ──
    // reset() is the only way to recover from Failed.

    probes4.reset();
    assert_eq!(probes4.state(), ProbeState::Starting);
    assert!(probes4.is_alive());
    assert!(probes4.mark_ready());

    // ── 14. ProbeResponse serialization ──
    // to_json() produces a JSON-serializable representation.

    let resp = probes4.check_liveness();
    let json = resp.to_json();
    assert!(json.contains("ok") || json.contains("ready"));

    // ── 15. Key concepts ──
    // - ProbeState: Starting -> Ready -> Draining (Failed is terminal)
    // - ProbeManager: thread-safe, atomic state transitions
    // - check_liveness() -> /healthz: 200 unless Failed
    // - check_readiness() -> /readyz: 200 only in Ready + callbacks pass
    // - check_startup() -> /startup: 200 once past Starting
    // - mark_ready(), mark_draining(), mark_failed(): state transitions
    // - add_readiness_check(name, callback): custom readiness checks
    // - set_workflow_count(n): track registered workflows
    // - reset(): recover from Failed (for testing/recovery)

    println!("PASS: 02-nexus/09_probes");
}
