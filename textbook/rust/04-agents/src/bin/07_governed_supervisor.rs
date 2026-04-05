// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Governed Supervisor
//!
//! OBJECTIVE: Build a governed supervisor agent that manages sub-agents under PACT.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python has GovernedSupervisor;
//!         Rust has SupervisorConfig with PACT integration for governance.
//! VALIDATES: SupervisorConfig, delegation patterns, budget allocation,
//!            PACT envelope enforcement, agent lifecycle
//!
//! Run: cargo run -p tutorial-agents --bin 07_governed_supervisor

use kaizen_agents::governance::{
    GovernancePolicy, SupervisorConfig, SubAgentSpec,
};

fn main() {
    // ── 1. Governed supervisor pattern ──
    // A supervisor orchestrates multiple sub-agents under governance:
    //   - Routes tasks to the most appropriate sub-agent
    //   - Enforces budget limits per sub-agent
    //   - Applies PACT operating envelopes
    //   - Monitors and escalates failures

    // ── 2. SubAgentSpec -- define sub-agents ──
    // Each sub-agent has a name, description, and capability card
    // that the supervisor uses for routing decisions.

    let code_reviewer = SubAgentSpec::new("code-reviewer")
        .description("Reviews code for bugs, style, and best practices")
        .capability("code_review")
        .capability("security_audit")
        .max_budget_usd(2.0);

    assert_eq!(code_reviewer.name(), "code-reviewer");
    assert_eq!(code_reviewer.capabilities().len(), 2);

    let test_writer = SubAgentSpec::new("test-writer")
        .description("Writes comprehensive test suites")
        .capability("test_generation")
        .capability("coverage_analysis")
        .max_budget_usd(1.5);

    let doc_writer = SubAgentSpec::new("doc-writer")
        .description("Writes technical documentation")
        .capability("documentation")
        .max_budget_usd(1.0);

    // ── 3. GovernancePolicy ──
    // Governance policies constrain supervisor behavior.

    let policy = GovernancePolicy::new()
        .total_budget_usd(10.0)
        .max_delegation_depth(2) // Supervisor -> sub-agent -> sub-sub-agent
        .require_justification(true) // Supervisor must justify routing decisions
        .escalation_threshold(3); // Escalate after 3 failed attempts

    assert_eq!(policy.total_budget_usd(), 10.0);
    assert_eq!(policy.max_delegation_depth(), 2);
    assert!(policy.require_justification());

    // ── 4. SupervisorConfig ──
    // Combines sub-agents with governance policy.

    let config = SupervisorConfig::new("dev-supervisor", "claude-sonnet-4-20250514")
        .sub_agent(code_reviewer)
        .sub_agent(test_writer)
        .sub_agent(doc_writer)
        .governance(policy);

    assert_eq!(config.name(), "dev-supervisor");
    assert_eq!(config.sub_agent_count(), 3);
    assert!(config.has_sub_agent("code-reviewer"));
    assert!(config.has_sub_agent("test-writer"));
    assert!(config.has_sub_agent("doc-writer"));

    // ── 5. Routing pattern ──
    // The supervisor uses LLM reasoning to route tasks.
    // It examines each sub-agent's capability card and description,
    // then decides which agent should handle the task.
    //
    // CRITICAL: Routing is LLM-based, NOT keyword matching.
    // The supervisor's signature includes agent descriptions as context.

    // ── 6. Budget allocation ──
    // The supervisor tracks budget across all sub-agents.
    // Each sub-agent has an individual cap within the total budget.

    let total = config.total_budget_usd();
    assert_eq!(total, 10.0);

    // ── 7. Delegation chain ──
    // Governed supervisors track the full delegation chain:
    //   User -> Supervisor -> SubAgent -> (optional sub-sub-agent)
    //
    // max_delegation_depth prevents infinite delegation loops.

    // ── 8. Execution pattern ──
    //
    //   let supervisor = GovernedSupervisor::new(config).await;
    //   let result = supervisor.run("Review and test the auth module").await;
    //   // Supervisor routes to code-reviewer, then test-writer
    //   // Each sub-agent operates within its budget cap
    //
    // NOTE: We do not call run() here as it requires an LLM API key.

    // ── 9. Key concepts ──
    // - GovernedSupervisor: orchestrates sub-agents under governance
    // - SubAgentSpec: name, description, capabilities, budget cap
    // - GovernancePolicy: total budget, delegation depth, escalation
    // - LLM-based routing using capability cards (no keyword matching)
    // - Budget tracked per sub-agent and in aggregate
    // - Delegation chain tracked for auditability
    // - PACT integration for operating envelope enforcement

    println!("PASS: 04-agents/07_governed_supervisor");
}
