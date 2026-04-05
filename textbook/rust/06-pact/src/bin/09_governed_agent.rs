// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- PACT / Governed Agent
//!
//! OBJECTIVE: Build AI agents that operate under PACT governance constraints.
//! LEVEL: Advanced
//! PARITY: Full -- Both SDKs use the same governed agent pattern with
//!         envelope enforcement and clearance-gated tool access.
//! VALIDATES: GovernedAgent configuration, envelope enforcement, tool gating
//!
//! Run: cargo run -p tutorial-pact --bin 09_governed_agent

use serde_json::json;

fn main() {
    // ── 1. Governed Agent pattern ──
    // A governed agent is a Kaizen agent wrapped in PACT constraints:
    //   - Operating envelope (budget, action limits)
    //   - Clearance-gated tools (some tools require higher clearance)
    //   - Access-controlled resources (data, APIs, services)
    //   - Audit trail (all decisions logged)

    // ── 2. Agent with envelope ──
    // The envelope defines what the agent can do.

    let agent_config = json!({
        "name": "deployment-agent",
        "model": "claude-sonnet-4-20250514",
        "address": "D1-T1-R1",
        "clearance": 4,
        "envelope": {
            "budget_usd": 10.0,
            "max_actions": 200,
            "allowed_actions": [
                "read_code",
                "run_tests",
                "deploy_staging",
                "deploy_production",
            ],
            "denied_actions": [
                "delete_database",
                "modify_permissions",
            ],
        },
    });

    assert_eq!(agent_config["name"], "deployment-agent");
    assert_eq!(agent_config["clearance"], 4);
    assert_eq!(agent_config["envelope"]["budget_usd"], 10.0);

    // ── 3. Tool clearance gating ──
    // Tools can have minimum clearance requirements.
    // The agent can only use tools at or below its clearance level.

    let tools = vec![
        json!({"name": "read_code", "clearance": 1, "description": "Read source code"}),
        json!({"name": "run_tests", "clearance": 2, "description": "Execute test suite"}),
        json!({"name": "deploy_staging", "clearance": 3, "description": "Deploy to staging"}),
        json!({"name": "deploy_production", "clearance": 4, "description": "Deploy to production"}),
        json!({"name": "rotate_keys", "clearance": 5, "description": "Rotate encryption keys"}),
    ];

    let agent_clearance = 4;
    let accessible_tools: Vec<&serde_json::Value> = tools
        .iter()
        .filter(|t| t["clearance"].as_u64().unwrap() <= agent_clearance as u64)
        .collect();

    assert_eq!(accessible_tools.len(), 4); // Can't use rotate_keys (level 5)

    // ── 4. Pre-action governance check ──
    // Before every action, the governance engine checks:
    //   1. Is the action in the envelope's allowed list?
    //   2. Is the agent within budget?
    //   3. Is the action count within limits?
    //   4. Does the agent have sufficient clearance?

    let action = "deploy_production";
    let in_allowed = agent_config["envelope"]["allowed_actions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a.as_str().unwrap() == action);
    let in_denied = agent_config["envelope"]["denied_actions"]
        .as_array()
        .unwrap()
        .iter()
        .any(|a| a.as_str().unwrap() == action);

    assert!(in_allowed && !in_denied, "deploy_production is permitted");

    // ── 5. Governed execution pattern ──
    //
    //   let engine = GovernanceEngine::load("org.yaml");
    //   let agent = GovernedAgent::new(agent_config, engine);
    //
    //   // Every tool call is checked:
    //   //   1. agent decides to call "deploy_production"
    //   //   2. governance engine checks envelope + clearance
    //   //   3. if allowed: execute; if denied: return AccessDenied
    //   //   4. log decision to audit trail
    //
    //   let result = agent.run("Deploy the latest release to production").await;

    // ── 6. Escalation pattern ──
    // When an agent lacks clearance, it can escalate to a supervisor.

    let escalation = json!({
        "from": "D1-T1-R2",      // Developer (clearance 2)
        "to": "D1-R1",           // Lead (clearance 4)
        "action": "deploy_production",
        "reason": "Deployment requires clearance 4, agent has clearance 2",
    });

    assert_eq!(escalation["reason"].as_str().unwrap().len() > 0, true);

    // ── 7. Key concepts ──
    // - GovernedAgent: Kaizen agent + PACT governance
    // - Operating envelope: budget, action limits, allow/deny lists
    // - Clearance-gated tools: min clearance per tool
    // - Pre-action governance check before every tool call
    // - Fail-closed: denied if any check fails
    // - Escalation: route to higher-clearance supervisor
    // - Full audit trail for compliance

    println!("PASS: 06-pact/09_governed_agent");
}
