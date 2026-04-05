// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Operating Envelopes
//!
//! OBJECTIVE: Define and enforce operating envelopes for AI agents.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs use Envelope with identical constraint semantics.
//! VALIDATES: Envelope, EnvelopeConstraint, budget limits, action restrictions
//!
//! Run: cargo run -p tutorial-pact --bin 04_envelopes

use kailash_governance::envelopes::{Envelope, EnvelopeConstraint};

fn main() {
    // ── 1. Operating Envelope ──
    // An envelope defines the boundaries within which an AI agent operates.
    // It constrains: budget, actions, data access, time, and escalation.

    let envelope = Envelope::new("code-reviewer")
        .budget_usd(5.0)
        .max_actions(100)
        .allowed_action("read_file")
        .allowed_action("search_code")
        .allowed_action("write_review")
        .denied_action("delete_file")
        .denied_action("push_to_main")
        .clearance_required(3);

    assert_eq!(envelope.name(), "code-reviewer");
    assert_eq!(envelope.budget_usd(), 5.0);
    assert_eq!(envelope.max_actions(), 100);

    // ── 2. Action enforcement ──
    // Check whether an action is permitted within the envelope.

    assert!(envelope.is_action_allowed("read_file"));
    assert!(envelope.is_action_allowed("search_code"));
    assert!(!envelope.is_action_allowed("delete_file"));
    assert!(!envelope.is_action_allowed("push_to_main"));

    // ── 3. Budget tracking ──
    // Envelopes track consumed budget.

    let mut tracked = envelope.clone();
    tracked.record_cost(2.0);
    assert_eq!(tracked.consumed_usd(), 2.0);
    assert_eq!(tracked.remaining_usd(), 3.0);
    assert!(!tracked.is_budget_exhausted());

    tracked.record_cost(3.5);
    assert!(tracked.is_budget_exhausted());

    // ── 4. Action counting ──
    // Track actions taken against the max_actions limit.

    let mut counted = envelope.clone();
    for _ in 0..50 {
        counted.record_action();
    }
    assert_eq!(counted.actions_taken(), 50);
    assert!(!counted.is_action_limit_reached());

    for _ in 0..50 {
        counted.record_action();
    }
    assert!(counted.is_action_limit_reached());

    // ── 5. EnvelopeConstraint ──
    // Individual constraints can be checked independently.

    let budget_constraint = EnvelopeConstraint::Budget { max_usd: 10.0 };
    let action_constraint = EnvelopeConstraint::MaxActions { limit: 50 };
    let clearance_constraint = EnvelopeConstraint::Clearance { min_level: 3 };

    assert!(matches!(budget_constraint, EnvelopeConstraint::Budget { .. }));
    assert!(matches!(action_constraint, EnvelopeConstraint::MaxActions { .. }));
    assert!(matches!(clearance_constraint, EnvelopeConstraint::Clearance { .. }));

    // ── 6. Nested envelopes ──
    // Sub-agents inherit parent envelope constraints.
    // The sub-agent's envelope must be within the parent's limits.

    let parent_envelope = Envelope::new("supervisor")
        .budget_usd(20.0)
        .max_actions(500);

    let child_envelope = Envelope::new("worker")
        .budget_usd(5.0) // Must be <= parent's 20.0
        .max_actions(100); // Must be <= parent's 500

    assert!(child_envelope.budget_usd() <= parent_envelope.budget_usd());
    assert!(child_envelope.max_actions() <= parent_envelope.max_actions());

    // ── 7. Key concepts ──
    // - Envelope: operating boundaries for AI agents
    // - Budget constraint: maximum USD spend
    // - Action constraint: maximum number of actions
    // - Action allow/deny lists: permitted operations
    // - Clearance requirement: minimum clearance level
    // - Budget and action tracking during execution
    // - Nested envelopes: child must be within parent limits
    // - Fail-closed: denied by default, explicitly allowed

    println!("PASS: 06-pact/04_envelopes");
}
