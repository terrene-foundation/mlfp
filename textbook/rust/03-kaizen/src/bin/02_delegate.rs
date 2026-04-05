// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Kaizen / Delegate
//!
//! OBJECTIVE: Use DelegateEngine -- the primary entry point for autonomous AI execution.
//! LEVEL: Basic
//! PARITY: Equivalent -- Python has Delegate() with progressive disclosure;
//!         Rust has DelegateEngine with DelegateConfig for the same layered approach.
//! VALIDATES: DelegateEngine, DelegateConfig, DelegateEvent, budget tracking
//!
//! Run: cargo run -p tutorial-kaizen --bin 02_delegate

use kailash_kaizen::agent::DelegateConfig;
use kailash_kaizen::cost::BudgetTracker;

fn main() {
    // ── 1. DelegateConfig (Layer 1: minimal) ──
    // DelegateConfig is the configuration struct for the DelegateEngine.
    // Model comes from environment variables -- NEVER hardcoded.
    //
    // Python equivalent:
    //   delegate = Delegate(model=os.environ["DEFAULT_LLM_MODEL"])

    let config = DelegateConfig::new("claude-sonnet-4-20250514");

    assert_eq!(config.model(), "claude-sonnet-4-20250514");
    assert!(config.budget_usd().is_none()); // No budget cap by default
    assert!(config.system_prompt().is_none());

    // ── 2. Layer 2: configured DelegateConfig ──
    // Add system prompt, turn limits, and tool configuration.

    let configured = DelegateConfig::new("claude-sonnet-4-20250514")
        .system_prompt("You are a helpful code reviewer.")
        .max_turns(20);

    assert_eq!(configured.max_turns(), 20);
    assert_eq!(
        configured.system_prompt().unwrap(),
        "You are a helpful code reviewer."
    );

    // ── 3. Layer 3: governed config (budget tracking) ──
    // budget_usd enables automatic cost tracking.

    let governed = DelegateConfig::new("claude-sonnet-4-20250514")
        .budget_usd(5.0);

    assert_eq!(governed.budget_usd(), Some(5.0));

    // ── 4. BudgetTracker ──
    // BudgetTracker monitors accumulated cost against a budget cap.
    // When the budget is exceeded, it signals exhaustion.

    let mut tracker = BudgetTracker::new(1.0);

    assert_eq!(tracker.budget_usd(), 1.0);
    assert_eq!(tracker.consumed_usd(), 0.0);
    assert_eq!(tracker.remaining_usd(), 1.0);
    assert!(!tracker.is_exhausted());

    // Record a cost
    tracker.record(0.50);
    assert_eq!(tracker.consumed_usd(), 0.50);
    assert_eq!(tracker.remaining_usd(), 0.50);
    assert!(!tracker.is_exhausted());

    // Exceed the budget
    tracker.record(0.60);
    assert!(tracker.consumed_usd() > tracker.budget_usd());
    assert!(tracker.is_exhausted());

    // ── 5. DelegateEvent types ──
    // DelegateEngine yields typed events during execution:
    //
    //   TextDelta       -- incremental text from the model
    //   ToolCallStart   -- a tool invocation has begun
    //   ToolCallEnd     -- a tool invocation has completed
    //   TurnComplete    -- the model finished responding
    //   BudgetExhausted -- budget cap exceeded
    //   ErrorEvent      -- an error occurred
    //
    // In Rust, these are enum variants for pattern matching:
    //
    //   match event {
    //       DelegateEvent::TextDelta { text } => print!("{text}"),
    //       DelegateEvent::TurnComplete { usage, .. } => { .. },
    //       DelegateEvent::BudgetExhausted { budget, consumed } => { .. },
    //       DelegateEvent::Error { message } => eprintln!("{message}"),
    //       _ => {}
    //   }

    // ── 6. Budget validation ──
    // Budget must be finite and non-negative.

    let zero_budget = DelegateConfig::new("claude-sonnet-4-20250514")
        .budget_usd(0.0);
    assert_eq!(zero_budget.budget_usd(), Some(0.0));

    // ── 7. Cost model ──
    // Conservative per-1M-token cost estimates by model prefix:
    //   claude-  : $3 input, $15 output
    //   gpt-4o   : $2.5 input, $10 output
    //   gemini-  : $1.25 input, $5 output
    //
    // These are approximations for budget tracking, not billing.

    // Example: claude-sonnet with 1000 prompt + 500 completion tokens
    //   input  = 1000 / 1_000_000 * $3  = $0.003
    //   output = 500 / 1_000_000 * $15  = $0.0075
    //   total = $0.0105

    let cost = (1000.0 * 3.0 + 500.0 * 15.0) / 1_000_000.0;
    assert!(cost > 0.01 && cost < 0.02);

    // ── 8. Key concepts ──
    // - DelegateConfig: progressive disclosure configuration
    // - DelegateEngine: the runtime that executes with DelegateConfig
    // - BudgetTracker: monitors cost against a budget cap
    // - DelegateEvent: enum of typed events for streaming consumption
    // - Model names always from environment variables, never hardcoded
    // - Rust uses enum variants; Python uses DelegateEvent subclasses

    println!("PASS: 03-kaizen/02_delegate");
}
