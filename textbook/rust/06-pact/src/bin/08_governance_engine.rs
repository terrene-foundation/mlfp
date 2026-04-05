// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Governance Engine
//!
//! OBJECTIVE: Use GovernanceEngine for unified governance enforcement.
//! LEVEL: Advanced
//! PARITY: Full -- Both SDKs use GovernanceEngine with identical fail-closed semantics.
//! VALIDATES: GovernanceEngine, policy evaluation, envelope enforcement, audit
//!
//! Run: cargo run -p tutorial-pact --bin 08_governance_engine

use kailash_governance::engine::GovernanceEngine;

fn main() {
    // ── 1. GovernanceEngine ──
    // The engine is the unified entry point for all governance operations.
    // It composes: access control, envelopes, clearance, and bridges.
    //
    // Thread-safe (Arc<Mutex<>>) for concurrent agent access.
    // Fail-closed: any governance check failure blocks the operation.

    let engine = GovernanceEngine::builder()
        .organization_yaml(r#"
organization:
  name: "TestOrg"
  departments:
    - name: "Engineering"
      roles:
        - name: "Lead"
          clearance: 4
        - name: "Developer"
          clearance: 2
"#)
        .default_deny()
        .build()
        .expect("valid config");

    assert_eq!(engine.organization_name(), "TestOrg");

    // ── 2. Unified check ──
    // check() evaluates all governance constraints in one call:
    //   - Access policy (is the action permitted?)
    //   - Clearance level (does the actor have sufficient clearance?)
    //   - Envelope constraints (within budget/action limits?)

    let result = engine.check("D1-R1", "read", "code/*");
    assert!(result.is_allowed());

    // ── 3. Fail-closed semantics ──
    // When any check fails, the entire operation is denied.
    // No partial access -- all constraints must pass.

    // Unknown actor -> denied
    let unknown = engine.check("D99-R1", "deploy", "production/api");
    assert!(!unknown.is_allowed());

    // ── 4. Decision explanation ──
    // Every decision includes an explanation for audit purposes.

    let decision = engine.check("D1-R2", "deploy", "production/api");
    assert!(decision.explanation().len() > 0);

    // ── 5. Audit trail ──
    // All checks are recorded for compliance and debugging.

    let audit = engine.audit_trail();
    assert!(audit.len() >= 3, "All checks are logged");

    // ── 6. Engine configuration ──
    // The engine supports runtime reconfiguration.

    let engine2 = GovernanceEngine::builder()
        .organization_yaml(r#"
organization:
  name: "Updated"
  departments:
    - name: "Security"
      roles:
        - name: "CISO"
          clearance: 5
"#)
        .default_deny()
        .build()
        .expect("valid config");

    assert_eq!(engine2.organization_name(), "Updated");

    // ── 7. Key concepts ──
    // - GovernanceEngine: unified governance enforcement
    // - Composes access control + clearance + envelopes + bridges
    // - Fail-closed: all constraints must pass
    // - Thread-safe for concurrent agent access
    // - Decision explanations for audit
    // - Audit trail for compliance
    // - Builder pattern for configuration

    println!("PASS: 06-pact/08_governance_engine");
}
