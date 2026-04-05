// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- Agents / Tree of Thoughts
//!
//! OBJECTIVE: Build agents that explore multiple reasoning paths simultaneously.
//! LEVEL: Advanced
//! PARITY: Equivalent -- Python has TreeOfThoughts agent;
//!         Rust uses ToTConfig with parallel branch exploration.
//! VALIDATES: ToTConfig, BranchConfig, evaluation and pruning strategies,
//!            thought tree structure
//!
//! Run: cargo run -p tutorial-agents --bin 09_tree_of_thoughts

use kaizen_agents::agents::tree_of_thoughts::{
    BranchConfig, PruningStrategy, ToTConfig,
};

fn main() {
    // ── 1. Tree of Thoughts pattern ──
    // ToT extends Chain of Thought by exploring MULTIPLE reasoning paths:
    //   1. BRANCH: Generate N different approaches to the problem
    //   2. EVALUATE: Score each branch's quality and promise
    //   3. PRUNE: Discard low-scoring branches
    //   4. EXPAND: Continue the most promising branches
    //   5. SELECT: Choose the best final answer across all branches
    //
    // This is more expensive than CoT but excels at:
    //   - Creative problem-solving
    //   - Planning tasks
    //   - Complex puzzles
    //   - Tasks with multiple valid approaches

    // ── 2. ToTConfig ──
    // Configuration controls the tree exploration strategy.

    let config = ToTConfig::default();

    assert_eq!(config.num_branches(), 3); // Explore 3 paths
    assert_eq!(config.max_depth(), 3);    // Up to 3 levels deep
    assert!(matches!(
        config.pruning_strategy(),
        PruningStrategy::TopK
    ));

    // ── 3. Custom configuration ──

    let custom = ToTConfig::builder()
        .num_branches(5)
        .max_depth(4)
        .pruning_strategy(PruningStrategy::Threshold(0.6))
        .evaluation_prompt("Rate this approach from 0 to 1 for correctness and feasibility.")
        .build();

    assert_eq!(custom.num_branches(), 5);
    assert_eq!(custom.max_depth(), 4);

    // ── 4. BranchConfig ──
    // Each branch tracks its reasoning path and evaluation score.

    let branch = BranchConfig::new("approach_1")
        .thought("Start by breaking the problem into sub-problems")
        .score(0.8);

    assert_eq!(branch.name(), "approach_1");
    assert!((branch.score() - 0.8).abs() < 0.001);

    let branch2 = BranchConfig::new("approach_2")
        .thought("Apply a greedy algorithm to find a quick solution")
        .score(0.5);

    let branch3 = BranchConfig::new("approach_3")
        .thought("Use dynamic programming for optimal solution")
        .score(0.9);

    // ── 5. Pruning strategies ──

    // TopK: keep the top N branches by score
    assert!(matches!(
        ToTConfig::default().pruning_strategy(),
        PruningStrategy::TopK
    ));

    // Threshold: keep branches above a minimum score
    let threshold_config = ToTConfig::builder()
        .pruning_strategy(PruningStrategy::Threshold(0.6))
        .build();
    assert!(matches!(
        threshold_config.pruning_strategy(),
        PruningStrategy::Threshold(_)
    ));

    // ── 6. Tree structure ──
    // The tree is built level by level:
    //
    //   Level 0: Problem statement
    //   Level 1: [Branch A (0.8), Branch B (0.5), Branch C (0.9)]
    //            Branch B pruned (score < threshold)
    //   Level 2: [Branch A expanded (0.85), Branch C expanded (0.95)]
    //   Level 3: [Branch A final, Branch C final]
    //   Selection: Branch C chosen (highest cumulative score)

    // Simulate scoring and selection
    let branches = vec![branch, branch2, branch3];
    let best = branches
        .iter()
        .max_by(|a, b| a.score().partial_cmp(&b.score()).unwrap())
        .unwrap();
    assert_eq!(best.name(), "approach_3");

    // ── 7. Execution pattern ──
    //
    //   let agent = TreeOfThoughts::new(config, model).await;
    //   let result = agent.run("Plan a migration from monolith to microservices").await;
    //   println!("Best approach: {}", result["selected_branch"]);
    //   println!("Reasoning: {}", result["reasoning"]);
    //   println!("Branches explored: {}", result["branches_explored"]);
    //
    // NOTE: We do not call run() here as it requires an LLM API key.

    // ── 8. Key concepts ──
    // - ToT: explores multiple reasoning paths in parallel
    // - BranchConfig: individual path with thought and score
    // - PruningStrategy: TopK or Threshold for eliminating weak branches
    // - More expensive than CoT but better for complex/creative tasks
    // - Tree depth controls reasoning complexity
    // - LLM evaluates each branch (self-evaluation pattern)
    // - Final selection picks the highest-scoring complete path

    println!("PASS: 04-agents/09_tree_of_thoughts");
}
