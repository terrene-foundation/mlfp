// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Kaizen / Cost Tracking and Budgets
//!
//! OBJECTIVE: Track LLM costs and enforce budget limits on agents.
//! LEVEL: Advanced
//! PARITY: Full -- Python has Delegate.budget_usd with BudgetExhausted event;
//!         Rust has BudgetTracker with the same semantics and CostEstimator.
//! VALIDATES: BudgetTracker, CostEstimator, budget enforcement, cost calculation
//!
//! Run: cargo run -p tutorial-kaizen --bin 05_cost_tracking

use kailash_kaizen::cost::{BudgetTracker, CostEstimator};

fn main() {
    // ── 1. Budget-tracked agent ──
    // BudgetTracker sets a hard cap on estimated cost. When accumulated
    // cost exceeds the budget, is_exhausted() returns true.

    let mut tracker = BudgetTracker::new(1.0);

    assert_eq!(tracker.budget_usd(), 1.0);
    assert_eq!(tracker.consumed_usd(), 0.0);
    assert_eq!(tracker.remaining_usd(), 1.0);
    assert!(!tracker.is_exhausted());

    // ── 2. Cost estimation internals ──
    // CostEstimator uses model prefix heuristics for approximate cost.
    // This is NOT billing data -- it's for budget enforcement.
    //
    // Example: claude-sonnet with 1000 prompt + 500 completion tokens
    //   input cost  = 1000 / 1_000_000 * $3  = $0.003
    //   output cost = 500 / 1_000_000 * $15  = $0.0075
    //   total = $0.0105 per turn

    let estimator = CostEstimator::for_model("claude-sonnet-4-20250514");

    let cost = estimator.estimate(1000, 500);
    assert!(cost > 0.01 && cost < 0.02, "Expected ~$0.0105, got {cost}");

    // ── 3. Record cost against budget ──
    // record() adds to the consumed total and checks budget.

    tracker.record(cost);
    assert!(tracker.consumed_usd() > 0.01);
    assert!(!tracker.is_exhausted());

    // Multiple turns accumulate
    for _ in 0..90 {
        tracker.record(cost);
    }
    assert!(tracker.consumed_usd() > 0.9);

    // Eventually exceeds budget
    for _ in 0..20 {
        tracker.record(cost);
    }
    assert!(tracker.is_exhausted());

    // ── 4. No budget = unlimited ──
    // When no budget is set, tracking still works but never exhausts.

    let unlimited = BudgetTracker::unlimited();
    assert!(!unlimited.is_exhausted());
    assert!(unlimited.budget_usd().is_infinite());

    // ── 5. Zero budget ──
    // Valid: immediately exhausted on first turn.

    let mut zero = BudgetTracker::new(0.0);
    assert!(!zero.is_exhausted()); // Not exhausted until a cost is recorded
    zero.record(0.001);
    assert!(zero.is_exhausted());

    // ── 6. CostEstimator for different providers ──

    let gpt4o_est = CostEstimator::for_model("gpt-4o");
    let gpt4o_cost = gpt4o_est.estimate(1000, 500);
    // input: $2.5/1M, output: $10/1M
    assert!(gpt4o_cost > 0.007 && gpt4o_cost < 0.008);

    let gemini_est = CostEstimator::for_model("gemini-pro");
    let gemini_cost = gemini_est.estimate(1000, 500);
    // input: $1.25/1M, output: $5/1M
    assert!(gemini_cost > 0.003 && gemini_cost < 0.004);

    // Unknown model uses conservative defaults (claude rates)
    let unknown_est = CostEstimator::for_model("unknown-model");
    let unknown_cost = unknown_est.estimate(1000, 500);
    assert!(unknown_cost > 0.0);

    // ── 7. Production pattern: budget per request ──
    // In production, create a new BudgetTracker per request:
    //
    //   let mut tracker = BudgetTracker::new(0.50);
    //   let config = DelegateConfig::new(&model).budget_usd(0.50);
    //   // ... stream events, check tracker.is_exhausted() after each turn

    // ── 8. Key concepts ──
    // - BudgetTracker: monitors cost against a hard cap
    // - CostEstimator::for_model(): model-specific cost calculator
    // - estimate(prompt_tokens, completion_tokens) -> f64
    // - record(cost): add cost and check budget
    // - is_exhausted(): true when consumed > budget
    // - unlimited(): no cap, tracking-only mode
    // - Cost estimates are approximations, not billing data

    println!("PASS: 03-kaizen/05_cost_tracking");
}
