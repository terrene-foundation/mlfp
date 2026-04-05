// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- RL / Reward Shaping
//!
//! OBJECTIVE: Design reward functions — sparse vs dense rewards, potential-based shaping,
//!            curriculum learning, and common pitfalls.
//! LEVEL: Advanced
//! PARITY: Partial -- kailash-rl provides environments with configurable rewards
//!         (GridWorld step_reward, goal/trap rewards). Formal reward shaping utilities
//!         are pending.
//! VALIDATES: Reward design patterns, potential-based shaping, curriculum, reward pitfalls
//!
//! Run: cargo run -p tutorial-rl --bin 05_reward_shaping

use kailash_rl::prelude::*;
use kailash_rl::env::grid_world::GridWorld;
use kailash_rl::env::frozen_lake::FrozenLake;
use kailash_rl::algo::tabular::q_learning::QLearning;
use kailash_rl::training::train_tabular;

fn main() {
    // ── 1. Sparse vs dense rewards ──
    // Sparse: reward only at episode end (e.g., FrozenLake: +1 at goal, 0 elsewhere).
    // Dense: reward at every step (e.g., distance to goal, progress signals).
    //
    // Sparse is harder to learn from (long credit assignment chain).
    // Dense provides faster learning but risks reward hacking.

    // Sparse: FrozenLake only rewards reaching the goal
    let mut sparse_env = FrozenLake::new(false);
    let eps = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut sparse_agent = QLearning::new(0.8, 0.95, eps);
    let sparse_stats = train_tabular(&mut sparse_env, &mut sparse_agent, 3000, Some(42)).unwrap();

    // Dense: GridWorld with per-step reward
    // Small negative per-step reward encourages shortest path
    let mut dense_env = GridWorld::new(4, 4)
        .with_goal(3, 3, 1.0);
    let eps2 = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut dense_agent = QLearning::new(0.8, 0.95, eps2);
    let dense_stats = train_tabular(&mut dense_env, &mut dense_agent, 3000, Some(42)).unwrap();

    // Both should learn, but dense reward typically converges faster
    // (on simple problems, sparse can also work well)
    let sparse_sr = sparse_stats.episodes.iter().rev().take(500)
        .filter(|e| e.terminated && e.total_reward > 0.0).count() as f64 / 500.0;
    let dense_sr = dense_stats.episodes.iter().rev().take(500)
        .filter(|e| e.terminated && e.total_reward > 0.0).count() as f64 / 500.0;

    assert!(sparse_sr > 0.0 || dense_sr > 0.0, "At least one should learn");

    // ── 2. Potential-based reward shaping ──
    // Adding a shaping reward F(s, s') = gamma * Phi(s') - Phi(s) preserves
    // the optimal policy while accelerating learning.
    //
    // Phi(s) is a potential function (e.g., negative distance to goal).
    // This is the ONLY form of shaping guaranteed not to change the optimal policy.

    // 4x4 grid, goal at (3,3)
    let goal_row = 3_i32;
    let goal_col = 3_i32;

    // Potential function: negative Manhattan distance to goal
    let potential = |state: u64| -> f64 {
        let row = (state / 4) as i32;
        let col = (state % 4) as i32;
        let distance = (goal_row - row).abs() + (goal_col - col).abs();
        -(distance as f64)
    };

    // Verify potential: at goal it is maximum (0), far away it is negative
    assert_eq!(potential(15), 0.0);  // At goal (3,3)
    assert_eq!(potential(0), -6.0);  // At start (0,0): distance = 6
    assert!(potential(15) > potential(0));

    // Shaping reward: F = gamma * Phi(s') - Phi(s)
    let gamma = 0.99;
    let state = 0_u64;      // (0,0), Phi = -6
    let next_state = 1_u64;  // (0,1), Phi = -5

    let shaping_reward = gamma * potential(next_state) - potential(state);
    assert!(shaping_reward > 0.0, "Moving toward goal gets positive shaping");

    let away_state = 4_u64;  // (1,0), Phi = -5
    let shaping_away = gamma * potential(away_state) - potential(state);
    // Moving right vs moving down from (0,0) -- both reduce distance by 1
    assert!(shaping_away > 0.0);

    // ── 3. Reward components ──
    // Complex tasks often decompose reward into weighted components.

    let components = vec![
        RewardComponent { name: "goal_reached".into(), weight: 1.0, value: 0.0 },
        RewardComponent { name: "step_penalty".into(), weight: 0.01, value: -1.0 },
        RewardComponent { name: "progress".into(), weight: 0.1, value: 0.5 },
        RewardComponent { name: "collision_penalty".into(), weight: 0.5, value: 0.0 },
    ];

    let total_reward: f64 = components.iter().map(|c| c.weight * c.value).sum();
    assert!((total_reward - (0.0 + -0.01 + 0.05 + 0.0)).abs() < 1e-10);

    // ── 4. Reward scaling ──
    // Large reward magnitudes cause learning instability.
    // Normalize rewards to keep gradients stable.

    let raw_rewards = vec![-100.0, -50.0, 0.0, 50.0, 1000.0];
    let mean: f64 = raw_rewards.iter().sum::<f64>() / raw_rewards.len() as f64;
    let variance: f64 = raw_rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
        / raw_rewards.len() as f64;
    let std_dev = variance.sqrt();

    let normalized: Vec<f64> = raw_rewards.iter()
        .map(|&r| (r - mean) / (std_dev + 1e-8))
        .collect();

    // Normalized rewards should have approximately zero mean
    let norm_mean: f64 = normalized.iter().sum::<f64>() / normalized.len() as f64;
    assert!(norm_mean.abs() < 1e-10, "Normalized rewards should be zero-mean");

    // ── 5. Reward clipping ──
    // Clip extreme rewards to prevent gradient explosions.
    // Common in DQN: clip all rewards to [-1, 1].

    let clipped: Vec<f64> = raw_rewards.iter()
        .map(|&r| r.clamp(-1.0, 1.0))
        .collect();

    for &r in &clipped {
        assert!(r >= -1.0 && r <= 1.0);
    }

    // ── 6. Common reward pitfalls ──

    // Pitfall 1: Reward hacking
    // If reward = -distance_to_goal, agent might oscillate near goal
    // to avoid the terminal state (collecting zero penalty indefinitely).
    //
    // Fix: add terminal bonus that dominates step rewards.
    let step_penalty = -0.01_f64;
    let terminal_bonus = 10.0_f64;
    assert!(terminal_bonus > step_penalty.abs() * 100.0,
            "Terminal bonus should dominate step penalties");

    // Pitfall 2: Reward shaping changes optimal policy
    // Bad: r_shaped = r + distance_improvement (not potential-based)
    // Good: r_shaped = r + gamma * Phi(s') - Phi(s) (potential-based)
    //
    // Only potential-based shaping is guaranteed to preserve optimality.

    // Pitfall 3: Reward delay
    // Very delayed rewards make credit assignment hard.
    // Use intermediate rewards or temporal difference learning.
    let episode_length = 1000;
    let discount_at_end = gamma.powi(episode_length);
    assert!(discount_at_end < 0.001,
            "Reward 1000 steps away is heavily discounted");

    // ── 7. Curriculum learning ──
    // Start with easy tasks and progressively increase difficulty.
    // This provides dense reward signals early, then sparse as the agent improves.

    let curriculum = vec![
        CurriculumStage {
            name: "2x2 grid".into(),
            grid_size: 2,
            has_obstacles: false,
            episodes: 500,
        },
        CurriculumStage {
            name: "3x3 grid".into(),
            grid_size: 3,
            has_obstacles: false,
            episodes: 1000,
        },
        CurriculumStage {
            name: "4x4 with walls".into(),
            grid_size: 4,
            has_obstacles: true,
            episodes: 2000,
        },
        CurriculumStage {
            name: "4x4 with walls and traps".into(),
            grid_size: 4,
            has_obstacles: true,
            episodes: 3000,
        },
    ];

    assert_eq!(curriculum.len(), 4);
    // Each stage has more episodes (harder tasks need more training)
    for i in 1..curriculum.len() {
        assert!(curriculum[i].episodes >= curriculum[i - 1].episodes);
    }

    // ── 8. Intrinsic motivation ──
    // When extrinsic rewards are sparse, add intrinsic rewards for:
    //   - Curiosity: reward for visiting novel states
    //   - Count-based: reward inversely proportional to visit count
    //   - Random Network Distillation (RND): prediction error as novelty
    //
    // r_total = r_extrinsic + beta * r_intrinsic

    let visit_counts: std::collections::HashMap<u64, u64> = vec![
        (0, 100), (1, 50), (2, 5), (3, 1), (4, 0),
    ].into_iter().collect();

    let intrinsic_rewards: Vec<f64> = (0..5).map(|s| {
        let count = *visit_counts.get(&s).unwrap_or(&0);
        1.0 / ((count as f64) + 1.0).sqrt() // Count-based bonus
    }).collect();

    // Rarely-visited states get higher intrinsic reward
    assert!(intrinsic_rewards[4] > intrinsic_rewards[0]);
    assert!(intrinsic_rewards[3] > intrinsic_rewards[1]);

    // ── 9. Key concepts ──
    // - Sparse reward: only at episode end (harder to learn)
    // - Dense reward: every step (faster learning, risk of hacking)
    // - Potential-based shaping: F = gamma*Phi(s') - Phi(s) preserves optimal policy
    // - Reward components: weighted decomposition for complex tasks
    // - Normalization: zero-mean, unit-variance for stable gradients
    // - Clipping: bound rewards to [-1, 1] to prevent gradient explosions
    // - Curriculum: easy tasks first, progressively harder
    // - Intrinsic motivation: curiosity/count-based bonuses for exploration
    // - Pitfalls: reward hacking, non-potential shaping, extreme delays

    println!("PASS: 08-rl/05_reward_shaping");
}

struct RewardComponent {
    name: String,
    weight: f64,
    value: f64,
}

struct CurriculumStage {
    name: String,
    grid_size: usize,
    has_obstacles: bool,
    episodes: u64,
}
