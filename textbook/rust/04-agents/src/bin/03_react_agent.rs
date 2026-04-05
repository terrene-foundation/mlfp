// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / ReAct Agent
//!
//! OBJECTIVE: Build a ReAct agent that reasons, acts, and observes iteratively.
//! LEVEL: Intermediate
//! PARITY: Equivalent -- Python has ReActAgent with MultiCycleStrategy;
//!         Rust uses ReActConfig with the same iterative execution model.
//! VALIDATES: ReActConfig, ActionType, convergence detection,
//!            multi-cycle execution, tool_calls-based convergence
//!
//! Run: cargo run -p tutorial-agents --bin 03_react_agent

use kaizen_agents::agents::react::{ActionType, ReActConfig};

fn main() {
    // ── 1. ReAct pattern: Reason + Act + Observe ──
    // ReAct agents operate in iterative cycles:
    //   1. REASON: Think about the current state and what to do next
    //   2. ACT: Execute an action (tool call, clarification, or finish)
    //   3. OBSERVE: Examine the result of the action
    //   4. REPEAT: Until converged (finish action or high confidence)
    //
    // This is fundamentally different from single-shot agents (SimpleQA).
    // ReAct is autonomous: it decides its own execution path.

    // ── 2. ActionType -- structured action vocabulary ──
    // The ReAct agent uses a fixed vocabulary of action types.

    assert_eq!(ActionType::ToolUse.as_str(), "tool_use");
    assert_eq!(ActionType::Finish.as_str(), "finish");
    assert_eq!(ActionType::Clarify.as_str(), "clarify");

    // ── 3. ReActConfig -- configuration ──
    // ReAct-specific settings control iteration limits and convergence.

    let config = ReActConfig::default();

    assert_eq!(config.max_cycles(), 10);
    assert!((config.confidence_threshold() - 0.7).abs() < 0.001);
    assert!(config.mcp_discovery_enabled());
    assert!(!config.parallel_tools());

    // Custom configuration
    let custom = ReActConfig::builder()
        .max_cycles(15)
        .confidence_threshold(0.8)
        .parallel_tools(true)
        .build();

    assert_eq!(custom.max_cycles(), 15);
    assert!((custom.confidence_threshold() - 0.8).abs() < 0.001);

    // ── 4. Convergence detection -- objective (tool_calls) ──
    // Objective convergence uses the tool_calls field.
    // This implements Claude Code's while(tool_call_exists) pattern.
    //
    //   tool_calls present and non-empty -> NOT converged (continue)
    //   tool_calls present but empty     -> CONVERGED (stop)

    // Simulate: tool calls present -> continue
    let has_tools = serde_json::json!({
        "tool_calls": [{"name": "search", "params": {"query": "flights"}}],
        "action": "tool_use",
        "confidence": 0.5,
    });
    assert!(!check_convergence(&config, &has_tools));

    // Simulate: empty tool calls -> converged
    let empty_tools = serde_json::json!({
        "tool_calls": [],
        "action": "finish",
        "confidence": 0.9,
    });
    assert!(check_convergence(&config, &empty_tools));

    // ── 5. Convergence detection -- subjective fallback ──
    // When tool_calls is absent, fall back to action/confidence checks.

    // action == "finish" -> converged
    let finish = serde_json::json!({"action": "finish", "confidence": 0.6});
    assert!(check_convergence(&config, &finish));

    // confidence >= threshold -> converged
    let high_conf = serde_json::json!({"action": "tool_use", "confidence": 0.9});
    assert!(check_convergence(&config, &high_conf));

    // action == "tool_use" with low confidence -> NOT converged
    let low_conf = serde_json::json!({"action": "tool_use", "confidence": 0.3});
    assert!(!check_convergence(&config, &low_conf));

    // ── 6. ReAct return structure ──
    // ReActAgent returns:
    //   {
    //       "thought": "I need to search for flights...",
    //       "action": "tool_use",
    //       "action_input": {"tool": "search", "query": "flights to Paris"},
    //       "confidence": 0.85,
    //       "need_tool": true,
    //       "cycles_used": 3,
    //       "total_cycles": 10,
    //   }

    // ── 7. Key concepts ──
    // - ReAct: Reason-Act-Observe iterative loop
    // - ActionType: ToolUse, Finish, Clarify
    // - max_cycles: hard limit on iterations
    // - confidence_threshold: convergence threshold
    // - Objective convergence: tool_calls presence
    // - Subjective fallback: action + confidence
    // - Multi-cycle strategy vs single-shot
    // - Rust uses ReActConfig; Python uses ReActConfig with same semantics

    println!("PASS: 04-agents/03_react_agent");
}

/// Simulate convergence detection logic.
fn check_convergence(config: &ReActConfig, result: &serde_json::Value) -> bool {
    // Objective: check tool_calls field
    if let Some(tool_calls) = result.get("tool_calls") {
        if let Some(arr) = tool_calls.as_array() {
            return arr.is_empty();
        }
    }
    // Subjective fallback
    if let Some(action) = result.get("action").and_then(|v| v.as_str()) {
        if action == "finish" {
            return true;
        }
    }
    if let Some(conf) = result.get("confidence").and_then(|v| v.as_f64()) {
        if conf >= config.confidence_threshold() {
            return true;
        }
        return false;
    }
    true // Default safe fallback
}
