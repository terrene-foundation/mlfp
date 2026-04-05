// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Delegate with Tools & Events
//!
//! OBJECTIVE: Use DelegateEngine with tool registration and event handling patterns.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python has Delegate with ToolRegistry and DelegateEvent;
//!         Rust has DelegateEngine with ToolRegistry and DelegateEvent enum.
//! VALIDATES: ToolRegistry, ToolDef, DelegateEvent variants, tool-delegate integration
//!
//! Run: cargo run -p tutorial-agents --bin 01_delegate

use kailash_kaizen::agent::DelegateConfig;
use kailash_kaizen::cost::BudgetTracker;
use kaizen_agents::delegate_engine::{
    DelegateEvent, ToolDef, ToolRegistry,
};

fn main() {
    // ── 1. ToolRegistry -- create and register tools ──
    // ToolRegistry holds tools the DelegateEngine can call. Each tool has:
    //   - name: unique identifier used in function calling
    //   - description: human-readable description for the model
    //   - parameters: JSON Schema for the tool's arguments

    let mut registry = ToolRegistry::new();

    assert_eq!(registry.tool_count(), 0);

    // ── 2. Register tools ──
    // Each tool is defined with a name, description, and parameter schema.

    registry.register(ToolDef::new(
        "read_file",
        "Read a file from the filesystem",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }),
    ));

    assert!(registry.has_tool("read_file"));
    assert!(!registry.has_tool("nonexistent"));
    assert_eq!(registry.tool_count(), 1);

    // ── 3. Multiple tool registration ──
    // Real agents typically have many tools.

    registry.register(ToolDef::new(
        "search",
        "Search for information in the codebase",
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }),
    ));

    assert_eq!(registry.tool_count(), 2);
    assert!(registry.has_tool("search"));

    // ── 4. OpenAI function-calling format ──
    // to_openai_tools() converts tools to the wire format the model sees.

    let openai_tools = registry.to_openai_tools();
    assert_eq!(openai_tools.len(), 2);
    assert_eq!(openai_tools[0]["type"], "function");
    assert_eq!(openai_tools[0]["function"]["name"], "read_file");

    // ── 5. ToolDef inspection ──
    // Individual tool definitions carry metadata.

    let write_tool = ToolDef::new(
        "write_file",
        "Write content to a file",
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }),
    );

    assert_eq!(write_tool.name(), "write_file");
    let openai_fmt = write_tool.to_openai_format();
    assert_eq!(openai_fmt["type"], "function");
    assert_eq!(openai_fmt["function"]["name"], "write_file");

    // ── 6. DelegateEvent variants ──
    // DelegateEngine yields typed events during execution.
    // Rust uses enum variants for exhaustive pattern matching.
    //
    //   DelegateEvent::TextDelta { text }
    //   DelegateEvent::ToolCallStart { call_id, name }
    //   DelegateEvent::ToolCallEnd { call_id, name, result, error }
    //   DelegateEvent::TurnComplete { text, usage }
    //   DelegateEvent::BudgetExhausted { budget, consumed }
    //   DelegateEvent::Error { message, details }

    let td = DelegateEvent::TextDelta {
        text: "Hello".to_string(),
    };
    assert!(matches!(td, DelegateEvent::TextDelta { .. }));

    let tcs = DelegateEvent::ToolCallStart {
        call_id: "call_001".to_string(),
        name: "read_file".to_string(),
    };
    assert!(matches!(tcs, DelegateEvent::ToolCallStart { .. }));

    let tce = DelegateEvent::ToolCallEnd {
        call_id: "call_001".to_string(),
        name: "read_file".to_string(),
        result: "file contents".to_string(),
        error: None,
    };
    assert!(matches!(tce, DelegateEvent::ToolCallEnd { .. }));

    let tc = DelegateEvent::TurnComplete {
        text: "Analysis complete".to_string(),
        prompt_tokens: 100,
        completion_tokens: 50,
    };
    assert!(matches!(tc, DelegateEvent::TurnComplete { .. }));

    let be = DelegateEvent::BudgetExhausted {
        budget_usd: 5.0,
        consumed_usd: 5.01,
    };
    assert!(matches!(be, DelegateEvent::BudgetExhausted { .. }));

    let err = DelegateEvent::Error {
        message: "Connection timeout".to_string(),
    };
    assert!(matches!(err, DelegateEvent::Error { .. }));

    // ── 7. Event stream consumption pattern ──
    // The standard Rust pattern for consuming events:
    //
    //   while let Some(event) = stream.next().await {
    //       match event {
    //           DelegateEvent::TextDelta { text } => print!("{text}"),
    //           DelegateEvent::ToolCallStart { name, .. } => show_spinner(&name),
    //           DelegateEvent::ToolCallEnd { name, result, .. } => {
    //               hide_spinner(&name);
    //           }
    //           DelegateEvent::TurnComplete { prompt_tokens, .. } => {
    //               println!("\nTokens: {prompt_tokens}");
    //           }
    //           DelegateEvent::BudgetExhausted { .. } => break,
    //           DelegateEvent::Error { message } => eprintln!("Error: {message}"),
    //       }
    //   }

    // ── 8. Budget tracking with tools ──
    // Budget tracking applies to the full session including tool calls.

    let mut tracker = BudgetTracker::new(5.0);
    assert_eq!(tracker.budget_usd(), 5.0);
    assert_eq!(tracker.consumed_usd(), 0.0);
    assert_eq!(tracker.remaining_usd(), 5.0);

    // ── 9. Config with tools ──
    // DelegateConfig references the ToolRegistry.

    let _config = DelegateConfig::new("claude-sonnet-4-20250514")
        .system_prompt("You are a code reviewer. Use the available tools.")
        .budget_usd(5.0);

    println!("PASS: 04-agents/01_delegate");
}
