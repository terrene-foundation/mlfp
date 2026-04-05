// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- RL / Training Loops
//!
//! OBJECTIVE: Understand RL training loops — the PPO clipped objective, replay buffers,
//!            rollout collection, and convergence monitoring.
//! LEVEL: Advanced
//! PARITY: Partial -- kailash-rl provides ReplayBuffer, RolloutBuffer, TrajectoryBuffer,
//!         and tabular training. PPO neural-net training is conceptual (pending deep RL).
//! VALIDATES: ReplayBuffer, RolloutBuffer, Transition, training loop pattern, convergence
//!
//! Run: cargo run -p tutorial-rl --bin 04_training_loops

use kailash_rl::prelude::*;
use kailash_rl::buffer::replay::{ReplayBuffer, Transition};
use kailash_rl::buffer::rollout::{RolloutBuffer, RolloutStep};
use kailash_rl::env::frozen_lake::FrozenLake;
use kailash_rl::algo::tabular::q_learning::QLearning;
use kailash_rl::training::train_tabular;

fn main() {
    // ── 1. Replay buffer ──
    // Off-policy algorithms (DQN, SAC) store transitions in a replay buffer
    // and sample random mini-batches for learning.
    //
    // This breaks temporal correlation and improves stability.

    let mut buffer = ReplayBuffer::new(1000).expect("valid capacity");

    // Store transitions
    for i in 0..50 {
        let transition = Transition {
            state: SpaceValue::Discrete(i % 16),
            action: SpaceValue::Discrete(i % 4),
            reward: if i % 10 == 0 { 1.0 } else { 0.0 },
            next_state: SpaceValue::Discrete((i + 1) % 16),
            done: i % 10 == 0,
        };
        buffer.push(transition);
    }

    assert_eq!(buffer.len(), 50);

    // Sample a mini-batch
    let mut rng = rand::thread_rng();
    let batch = buffer.sample(8, &mut rng).expect("enough transitions");
    assert_eq!(batch.len(), 8);

    // Verify batch contents are valid
    for transition in &batch {
        // Rewards should be 0.0 or 1.0
        assert!(transition.reward == 0.0 || transition.reward == 1.0);
    }

    // ── 2. Buffer capacity ──
    // When full, oldest transitions are evicted (circular buffer).

    let mut small_buf = ReplayBuffer::new(5).expect("valid");
    for i in 0..10 {
        small_buf.push(Transition {
            state: SpaceValue::Discrete(i),
            action: SpaceValue::Discrete(0),
            reward: i as f64,
            next_state: SpaceValue::Discrete(i + 1),
            done: false,
        });
    }

    // Only keeps last 5 transitions
    assert_eq!(small_buf.len(), 5);

    // Cannot create buffer with zero capacity
    assert!(ReplayBuffer::new(0).is_err());

    // Cannot sample more than available
    assert!(small_buf.sample(10, &mut rng).is_err());

    // ── 3. Rollout buffer ──
    // On-policy algorithms (PPO, A2C) collect complete rollouts before updating.
    // Unlike replay buffers, rollout data is discarded after each update.

    let mut rollout = RolloutBuffer::new(0.99, 0.95).expect("valid params");

    // Collect a trajectory
    for i in 0..10 {
        rollout.push(RolloutStep {
            observation: SpaceValue::Discrete(i),
            action: SpaceValue::Discrete(i % 4),
            reward: if i == 9 { 1.0 } else { -0.01 },
            value: 0.5 - (i as f64 * 0.05),
            log_prob: -1.5,
            terminated: i == 9,
            truncated: false,
        });
    }

    assert_eq!(rollout.len(), 10);

    // Compute returns and advantages via GAE
    let rollout_data = rollout.compute_returns(0.0).expect("non-empty buffer");
    assert_eq!(rollout_data.rewards.len(), 10);
    assert!((rollout_data.rewards[9] - 1.0).abs() < 1e-10); // Terminal reward

    // Advantages should be computed
    assert_eq!(rollout_data.advantages.len(), 10);

    // ── 4. PPO clipped objective (conceptual) ──
    // PPO maximizes:
    //   L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)
    //
    // where:
    //   r_t = pi(a|s) / pi_old(a|s)  (probability ratio)
    //   A_t = advantage estimate
    //   eps = clip range (typically 0.2)
    //
    // The clip prevents the policy from changing too much in one update.

    let clip_range = 0.2_f64;

    // Case 1: Positive advantage, ratio within clip range
    let ratio1 = 1.1_f64;
    let advantage1 = 2.0_f64;
    let unclipped1 = ratio1 * advantage1;
    let clipped_ratio1 = ratio1.clamp(1.0 - clip_range, 1.0 + clip_range);
    let clipped1 = clipped_ratio1 * advantage1;
    let ppo_loss1 = unclipped1.min(clipped1);
    assert!((ppo_loss1 - unclipped1).abs() < 1e-10, "Within range: use unclipped");

    // Case 2: Positive advantage, ratio above clip range
    let ratio2 = 1.5_f64;
    let advantage2 = 2.0_f64;
    let unclipped2 = ratio2 * advantage2;
    let clipped_ratio2 = ratio2.clamp(1.0 - clip_range, 1.0 + clip_range);
    let clipped2 = clipped_ratio2 * advantage2;
    let ppo_loss2 = unclipped2.min(clipped2);
    assert!(ppo_loss2 < unclipped2, "Above range: clipped is smaller");
    assert!((clipped_ratio2 - 1.2).abs() < 1e-10, "Clamped to 1+eps");

    // Case 3: Negative advantage (bad action), ratio increases
    let ratio3 = 1.5_f64;
    let advantage3 = -1.0_f64;
    let unclipped3 = ratio3 * advantage3;  // -1.5
    let clipped_ratio3 = ratio3.clamp(1.0 - clip_range, 1.0 + clip_range);
    let clipped3 = clipped_ratio3 * advantage3;  // -1.2
    let ppo_loss3 = unclipped3.min(clipped3);
    assert!((ppo_loss3 - unclipped3).abs() < 1e-10, "Negative advantage: use larger penalty");

    // ── 5. Generalized Advantage Estimation (GAE) ──
    // GAE computes advantage estimates that trade off bias and variance:
    //   A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}
    //   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    //
    // lambda=0: TD(0) estimate (low variance, high bias)
    // lambda=1: Monte Carlo estimate (high variance, low bias)
    // lambda=0.95: typical default (good balance)

    let gamma = 0.99;
    let lambda = 0.95;

    // Simple example: 3-step trajectory
    let rewards = vec![0.0, 0.0, 1.0];
    let values = vec![0.5, 0.7, 0.9, 0.0]; // V(s0), V(s1), V(s2), V(terminal)

    // TD errors
    let deltas: Vec<f64> = (0..3)
        .map(|t| rewards[t] + gamma * values[t + 1] - values[t])
        .collect();

    assert!(deltas[2] > 0.0, "Getting reward should have positive TD error");

    // GAE advantages (backward pass)
    let mut advantages = vec![0.0; 3];
    let mut gae = 0.0;
    for t in (0..3).rev() {
        gae = deltas[t] + gamma * lambda * gae;
        advantages[t] = gae;
    }

    // All advantages should be positive (reward propagates backward)
    assert!(advantages[0] > 0.0);
    assert!(advantages[2] > 0.0);

    // ── 6. Convergence monitoring ──
    // Track metrics across training to detect convergence or instability.

    let mut env = FrozenLake::new(false);
    let eps = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut agent = QLearning::new(0.8, 0.95, eps);

    let stats = train_tabular(&mut env, &mut agent, 5000, Some(42)).unwrap();

    // Window-based success rate
    let window_size = 500;
    let mut windowed_rates = Vec::new();
    for start in (0..stats.episodes.len()).step_by(window_size) {
        let end = (start + window_size).min(stats.episodes.len());
        let window = &stats.episodes[start..end];
        let rate = window.iter()
            .filter(|e| e.terminated && e.total_reward > 0.0)
            .count() as f64 / window.len() as f64;
        windowed_rates.push(rate);
    }

    // Success rate should generally increase over windows
    assert!(!windowed_rates.is_empty());
    if windowed_rates.len() >= 2 {
        let last = *windowed_rates.last().unwrap();
        let first = windowed_rates[0];
        assert!(last >= first, "Performance should improve or stay stable");
    }

    // ── 7. Training loop pattern ──
    // The general deep RL training loop:
    //
    //   for epoch in 0..num_epochs {
    //       // 1. Collect rollouts
    //       let rollout = collect_rollout(&env, &policy, rollout_length);
    //
    //       // 2. Compute advantages (GAE)
    //       let advantages = compute_gae(&rollout, gamma, lambda);
    //
    //       // 3. Update policy (multiple mini-batch passes)
    //       for _ in 0..num_minibatch_passes {
    //           let batch = sample_minibatch(&rollout, &advantages);
    //           let loss = compute_ppo_loss(&batch, clip_range);
    //           optimizer.step(loss);
    //       }
    //
    //       // 4. Log metrics
    //       log_metrics(epoch, &rollout);
    //   }

    // ── 8. Key concepts ──
    // - ReplayBuffer: circular buffer for off-policy sampling (DQN, SAC)
    // - RolloutBuffer: on-policy trajectory storage (PPO, A2C), used once
    // - PPO clipped objective: prevents large policy updates
    // - clip_range (eps=0.2): limits probability ratio to [0.8, 1.2]
    // - GAE: advantage estimation trading off bias/variance (lambda=0.95)
    // - TD error: delta = r + gamma * V(s') - V(s)
    // - Convergence monitoring: windowed success rate should increase
    // - On-policy: collect rollout -> update -> discard (PPO, A2C)
    // - Off-policy: store in buffer -> sample randomly -> update (DQN, SAC)

    println!("PASS: 08-rl/04_training_loops");
}
