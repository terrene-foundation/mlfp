// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Kaizen / Agent Configuration
//!
//! OBJECTIVE: Configure agents with KaizenConfig and AgentManager.
//! LEVEL: Intermediate
//! PARITY: Full -- Python has KaizenConfig and AgentManager with same progressive disclosure;
//!         Rust uses typed config structs and a thread-safe AgentManager.
//! VALIDATES: KaizenConfig, AgentManager, configure(), framework setup
//!
//! Run: cargo run -p tutorial-kaizen --bin 03_agent_config

use kailash_kaizen::agent::{AgentConfig, AgentManager, KaizenConfig};

fn main() {
    // ── 1. KaizenConfig -- framework-wide settings ──
    // KaizenConfig stores global settings for the Kaizen framework.
    // These apply to all new Kaizen instances unless overridden.

    let config = KaizenConfig::default();

    assert!(config.signature_programming_enabled());
    assert!(config.multi_agent_coordination());
    assert!(config.transparency_enabled());

    // ── 2. Custom configuration ──
    // Builder pattern for setting framework options.

    let custom = KaizenConfig::builder()
        .signature_programming(true)
        .mcp_integration(false)
        .multi_agent_coordination(true)
        .transparency(true)
        .build();

    assert!(custom.signature_programming_enabled());
    assert!(!custom.mcp_integration_enabled());

    // ── 3. AgentManager -- lifecycle management ──
    // AgentManager tracks agent instances, their state, and provides
    // discovery/shutdown capabilities. Thread-safe via Arc<Mutex<>>.

    let manager = AgentManager::new();

    assert_eq!(manager.agent_count(), 0);

    // ── 4. AgentConfig -- per-agent settings ──
    // Each agent has its own config with model, tools, and limits.

    let agent_config = AgentConfig::new("my-agent", "claude-sonnet-4-20250514")
        .max_turns(10)
        .temperature(0.7);

    assert_eq!(agent_config.name(), "my-agent");
    assert_eq!(agent_config.model(), "claude-sonnet-4-20250514");
    assert_eq!(agent_config.max_turns(), 10);

    // ── 5. Register agents with manager ──
    // The manager tracks agents by name for discovery and lifecycle.

    manager.register("my-agent", agent_config);
    assert_eq!(manager.agent_count(), 1);
    assert!(manager.has_agent("my-agent"));
    assert!(!manager.has_agent("nonexistent"));

    // ── 6. Environment-based configuration ──
    // In production, model names and API keys come from environment:
    //
    //   let model = std::env::var("DEFAULT_LLM_MODEL")
    //       .unwrap_or_else(|_| "claude-sonnet-4-20250514".to_string());
    //   let config = AgentConfig::new("reviewer", &model);
    //
    // NEVER hardcode model names or API keys.

    // ── 7. Key concepts ──
    // - KaizenConfig: framework-wide settings (signature, MCP, multi-agent)
    // - AgentConfig: per-agent settings (model, turns, temperature)
    // - AgentManager: thread-safe lifecycle management and discovery
    // - Configuration is always from environment in production
    // - Python: kaizen.configure() sets globals; Rust: KaizenConfig::builder()

    println!("PASS: 03-kaizen/03_agent_config");
}
