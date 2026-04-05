// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Clearance Levels
//!
//! OBJECTIVE: Define and enforce clearance-based access controls.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs use 5-level clearance with identical semantics.
//! VALIDATES: ClearanceLevel, clearance checks, escalation, verification gradient
//!
//! Run: cargo run -p tutorial-pact --bin 03_clearance

use kailash_governance::clearance::{ClearanceLevel, check_clearance};

fn main() {
    // ── 1. Clearance levels ──
    // PACT defines 5 clearance levels, from lowest to highest.

    assert_eq!(ClearanceLevel::Public.level(), 1);
    assert_eq!(ClearanceLevel::Internal.level(), 2);
    assert_eq!(ClearanceLevel::Confidential.level(), 3);
    assert_eq!(ClearanceLevel::Secret.level(), 4);
    assert_eq!(ClearanceLevel::TopSecret.level(), 5);

    // ── 2. Clearance comparison ──
    // Higher level grants access to lower-level resources.

    assert!(ClearanceLevel::Secret > ClearanceLevel::Confidential);
    assert!(ClearanceLevel::Public < ClearanceLevel::Internal);
    assert!(ClearanceLevel::TopSecret >= ClearanceLevel::TopSecret);

    // ── 3. Access checks ──
    // check_clearance(actor, resource) returns true if actor's clearance
    // meets or exceeds the resource's required clearance.

    assert!(check_clearance(ClearanceLevel::Secret, ClearanceLevel::Confidential));
    assert!(check_clearance(ClearanceLevel::TopSecret, ClearanceLevel::TopSecret));
    assert!(!check_clearance(ClearanceLevel::Internal, ClearanceLevel::Secret));
    assert!(!check_clearance(ClearanceLevel::Public, ClearanceLevel::Confidential));

    // ── 4. Clearance from integer ──

    assert_eq!(ClearanceLevel::from_level(1), ClearanceLevel::Public);
    assert_eq!(ClearanceLevel::from_level(3), ClearanceLevel::Confidential);
    assert_eq!(ClearanceLevel::from_level(5), ClearanceLevel::TopSecret);

    // ── 5. Verification gradient ──
    // Higher clearance operations require more verification:
    //   Level 1 (Public): No verification
    //   Level 2 (Internal): Basic validation
    //   Level 3 (Confidential): Standard review
    //   Level 4 (Secret): Enhanced review + approval
    //   Level 5 (TopSecret): Full audit trail + multi-party approval

    let verification_requirements = |level: ClearanceLevel| -> &str {
        match level {
            ClearanceLevel::Public => "none",
            ClearanceLevel::Internal => "basic",
            ClearanceLevel::Confidential => "standard",
            ClearanceLevel::Secret => "enhanced",
            ClearanceLevel::TopSecret => "full_audit",
        }
    };

    assert_eq!(verification_requirements(ClearanceLevel::Public), "none");
    assert_eq!(verification_requirements(ClearanceLevel::TopSecret), "full_audit");

    // ── 6. Escalation ──
    // When an actor lacks clearance, they must escalate to a higher authority.
    // The escalation target must have sufficient clearance.

    let actor_clearance = ClearanceLevel::Internal;
    let required = ClearanceLevel::Secret;
    let escalation_target = ClearanceLevel::TopSecret;

    assert!(!check_clearance(actor_clearance, required));
    assert!(check_clearance(escalation_target, required));

    // ── 7. Key concepts ──
    // - 5 clearance levels: Public < Internal < Confidential < Secret < TopSecret
    // - check_clearance(actor, resource): boolean access check
    // - Higher clearance grants access to lower-level resources
    // - Verification gradient: more verification for higher clearance
    // - Escalation: route to higher authority when clearance insufficient
    // - Clearance is attached to D/T/R roles in the organization

    println!("PASS: 06-pact/03_clearance");
}
