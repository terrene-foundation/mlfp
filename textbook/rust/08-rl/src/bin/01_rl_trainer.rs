// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- RL / RL Trainer Setup
//!
//! OBJECTIVE: Set up RL training — environment registration, algorithm selection,
//!            training configuration, and the train_tabular entry point.
//! LEVEL: Basic
//! PARITY: Full -- kailash-rl provides Environment trait, tabular algorithms,
//!         EpsilonSchedule, train_tabular, and built-in environments.
//! VALIDATES: Environment trait, train_tabular, EpsilonSchedule, TrainingStats, EpisodeStats
//!
//! Run: cargo run -p tutorial-rl --bin 01_rl_trainer

use kailash_rl::prelude::*;
use kailash_rl::env::frozen_lake::FrozenLake;
use kailash_rl::algo::tabular::q_learning::QLearning;
use kailash_rl::training::{train_tabular, run_tabular_episode};

fn main() {
    // ── 1. Environment creation ──
    // Environments define the RL task: observation space, action space,
    // reset, and step. FrozenLake is a 4x4 grid navigation task.

    let mut env = FrozenLake::new(false); // false = deterministic (non-slippery)

    // Inspect spaces
    let obs_space = env.observation_space();
    let act_space = env.action_space();

    // FrozenLake has 16 states (4x4 grid) and 4 actions (L/D/R/U)
    assert_eq!(obs_space.n().unwrap(), 16);
    assert_eq!(act_space.n().unwrap(), 4);

    // ── 2. Epsilon schedule ──
    // Controls exploration vs exploitation during training.
    // Linear decay: start at 1.0 (full exploration) -> 0.01 (mostly greedy).

    let mut epsilon = EpsilonSchedule::linear(1.0, 0.01, 5000);

    // Starts fully exploratory
    let first = epsilon.current();
    assert!((first - 1.0).abs() < 1e-10);

    // Decays as steps are taken
    for _ in 0..2500 {
        let _ = epsilon.get_and_step();
    }
    let midpoint = epsilon.current();
    assert!(midpoint < 1.0 && midpoint > 0.01, "Should be between start and end");

    // Can reset schedule
    epsilon.reset();
    assert!((epsilon.current() - 1.0).abs() < 1e-10);

    // ── 3. Algorithm setup ──
    // Q-Learning: off-policy TD control.
    // Parameters:
    //   alpha: learning rate (how fast Q-values update)
    //   gamma: discount factor (how much future rewards matter)
    //   epsilon: exploration schedule

    let eps = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut agent = QLearning::new(
        0.8,   // alpha: aggressive learning (good for tabular)
        0.95,  // gamma: value future rewards highly
        eps,
    );

    // ── 4. Single episode ──
    // Run one episode: reset env, agent acts, env steps, until done.

    let stats = run_tabular_episode(&mut env, &mut agent, Some(42))
        .expect("episode should complete");

    assert!(stats.steps > 0, "Episode took at least one step");
    // total_reward is 1.0 if goal reached, 0.0 if fell in hole

    // ── 5. Full training run ──
    // Train for multiple episodes and track statistics.

    let training_stats = train_tabular(&mut env, &mut agent, 1000, Some(123))
        .expect("training should complete");

    assert_eq!(training_stats.episodes.len(), 1000);

    // Mean reward and length
    let mean_reward = training_stats.mean_reward();
    let mean_length = training_stats.mean_length();
    assert!(mean_length > 0.0, "Episodes have nonzero length");

    // Success rate: fraction of episodes reaching the goal
    let success_rate = training_stats.success_rate();
    assert!(success_rate >= 0.0 && success_rate <= 1.0);

    // ── 6. Learning progress ──
    // Compare early vs late performance to verify learning.

    let early: Vec<&EpisodeStats> = training_stats.episodes.iter().take(100).collect();
    let late: Vec<&EpisodeStats> = training_stats.episodes.iter().rev().take(100).collect();

    let early_reward: f64 = early.iter().map(|e| e.total_reward).sum::<f64>() / 100.0;
    let late_reward: f64 = late.iter().map(|e| e.total_reward).sum::<f64>() / 100.0;

    // On deterministic FrozenLake, Q-Learning should improve
    assert!(late_reward >= early_reward, "Agent should improve over training");

    // ── 7. Extended training for convergence ──
    // With enough episodes on deterministic FrozenLake, Q-Learning converges.

    let eps2 = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut agent2 = QLearning::new(0.8, 0.95, eps2);
    let mut env2 = FrozenLake::new(false);

    let long_stats = train_tabular(&mut env2, &mut agent2, 10_000, Some(42))
        .expect("training should complete");

    // Check convergence in last 1000 episodes
    let last_1000: Vec<&EpisodeStats> = long_stats.episodes.iter().rev().take(1000).collect();
    let converged_rate = last_1000.iter()
        .filter(|e| e.terminated && e.total_reward > 0.0)
        .count() as f64 / 1000.0;

    assert!(converged_rate > 0.7, "Q-Learning should converge on deterministic FrozenLake");

    // ── 8. EpisodeStats fields ──

    let sample = &training_stats.episodes[0];
    let _reward = sample.total_reward;    // Cumulative reward
    let _steps = sample.steps;            // Number of steps taken
    let _terminated = sample.terminated;  // Reached terminal state?

    // ── 9. Key concepts ──
    // - Environment: observation space, action space, reset(), step()
    // - EpsilonSchedule: linear decay from exploration to exploitation
    // - Q-Learning: off-policy TD control (alpha, gamma, epsilon)
    // - run_tabular_episode: single episode execution
    // - train_tabular: multi-episode training with seed management
    // - TrainingStats: episodes, mean_reward, mean_length, success_rate
    // - Learning progress: compare early vs late episode performance
    // - Convergence: stable high success rate in late episodes

    println!("PASS: 08-rl/01_rl_trainer");
}
