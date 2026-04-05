// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Access Control
//!
//! OBJECTIVE: Implement role-based access control using PACT governance.
//! LEVEL: Intermediate
//! PARITY: Full -- Both SDKs use AccessPolicy with identical evaluation logic.
//! VALIDATES: AccessPolicy, AccessRule, AccessDecision, policy evaluation
//!
//! Run: cargo run -p tutorial-pact --bin 05_access

use kailash_governance::access::{
    AccessDecision, AccessPolicy, AccessRequest, AccessRule,
};

fn main() {
    // ── 1. AccessRule ──
    // Rules define who can do what on which resource.

    let rule1 = AccessRule::allow("D1-R1")  // CTO
        .action("deploy")
        .resource("production/*");

    let rule2 = AccessRule::allow("D1-T1-R1")  // Backend Lead
        .action("deploy")
        .resource("staging/*");

    let rule3 = AccessRule::deny("D1-T1-R2")  // Developer
        .action("deploy")
        .resource("production/*");

    assert!(rule1.is_allow());
    assert!(rule2.is_allow());
    assert!(rule3.is_deny());

    // ── 2. AccessPolicy ──
    // A policy collects rules and evaluates access requests.

    let policy = AccessPolicy::new("deployment-policy")
        .rule(rule1)
        .rule(rule2)
        .rule(rule3)
        .default_deny(); // Deny if no rule matches

    assert_eq!(policy.name(), "deployment-policy");
    assert_eq!(policy.rule_count(), 3);

    // ── 3. Access requests ──
    // An AccessRequest pairs an actor address with an action and resource.

    let cto_deploy = AccessRequest::new("D1-R1", "deploy", "production/api");
    let lead_staging = AccessRequest::new("D1-T1-R1", "deploy", "staging/api");
    let dev_production = AccessRequest::new("D1-T1-R2", "deploy", "production/api");
    let unknown = AccessRequest::new("D2-R1", "deploy", "production/api");

    // ── 4. Policy evaluation ──

    let decision1 = policy.evaluate(&cto_deploy);
    assert!(matches!(decision1, AccessDecision::Allow));

    let decision2 = policy.evaluate(&lead_staging);
    assert!(matches!(decision2, AccessDecision::Allow));

    let decision3 = policy.evaluate(&dev_production);
    assert!(matches!(decision3, AccessDecision::Deny));

    // No matching rule -> default deny
    let decision4 = policy.evaluate(&unknown);
    assert!(matches!(decision4, AccessDecision::Deny));

    // ── 5. Wildcard matching ──
    // Resources use glob-style wildcards.
    // "production/*" matches "production/api", "production/web", etc.

    let api_request = AccessRequest::new("D1-R1", "deploy", "production/api");
    let web_request = AccessRequest::new("D1-R1", "deploy", "production/web");

    assert!(matches!(policy.evaluate(&api_request), AccessDecision::Allow));
    assert!(matches!(policy.evaluate(&web_request), AccessDecision::Allow));

    // ── 6. Multiple policies ──
    // Different resources can have separate policies.

    let data_policy = AccessPolicy::new("data-policy")
        .rule(AccessRule::allow("D1-R1").action("read").resource("data/*"))
        .rule(AccessRule::allow("D1-R1").action("write").resource("data/*"))
        .rule(AccessRule::allow("D1-T1-R1").action("read").resource("data/*"))
        .rule(AccessRule::deny("D1-T1-R2").action("write").resource("data/production/*"))
        .default_deny();

    let dev_read = AccessRequest::new("D1-T1-R2", "read", "data/staging/users");
    // No explicit allow for D1-T1-R2 read -> default deny
    assert!(matches!(data_policy.evaluate(&dev_read), AccessDecision::Deny));

    // ── 7. Key concepts ──
    // - AccessRule: allow/deny for actor + action + resource
    // - AccessPolicy: collection of rules with default behavior
    // - AccessRequest: actor address + action + resource
    // - Policy evaluation: first matching rule wins
    // - Default deny: reject if no rule matches (fail-closed)
    // - Wildcard resources: "resource/*" for glob matching
    // - Actors identified by D/T/R addresses

    println!("PASS: 06-pact/05_access");
}
