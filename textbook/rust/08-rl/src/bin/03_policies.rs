// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- RL / Policies and Value Functions
//!
//! OBJECTIVE: Understand policy representations — Q-tables (tabular), epsilon-greedy
//!            action selection, value functions, and the TabularAgent trait.
//! LEVEL: Intermediate
//! PARITY: Full -- kailash-rl provides TabularAgent trait, Q-Learning, SARSA,
//!         Expected SARSA, Monte Carlo, and EpsilonSchedule.
//! VALIDATES: TabularAgent, Q-Learning, SARSA, EpsilonSchedule, Q-table, best_action
//!
//! Run: cargo run -p tutorial-rl --bin 03_policies

use kailash_rl::prelude::*;
use kailash_rl::algo::tabular::{
    q_learning::QLearning,
    sarsa::Sarsa,
    expected_sarsa::ExpectedSarsa,
    monte_carlo::{MonteCarlo, McVariant},
};
use kailash_rl::env::frozen_lake::FrozenLake;
use kailash_rl::training::train_tabular;

fn main() {
    // ── 1. TabularAgent trait ──
    // The interface all tabular RL agents implement:
    //   act(state, num_actions) -> action
    //   update(experience) -> ()
    //   q_table() -> &HashMap<(state, action), value>
    //   q_value(state, action) -> f64
    //   best_action(state, num_actions) -> action (greedy)

    let eps = EpsilonSchedule::linear(1.0, 0.01, 5000);
    let mut agent = QLearning::new(0.5, 0.99, eps);

    // Before training, Q-table is empty -> all values are 0.0
    assert_eq!(agent.q_value(0, 0), 0.0);
    assert_eq!(agent.q_value(0, 1), 0.0);

    // Q-table starts empty
    assert!(agent.q_table().is_empty());

    // ── 2. Q-Learning update ──
    // Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))
    //
    // Off-policy: uses max over next actions (not the action actually taken).
    // This makes Q-Learning converge to the optimal policy regardless of
    // the exploration strategy used.

    let experience = TabularExperience {
        state: 0,
        action: 2,    // Right
        reward: 0.0,
        next_state: 1,
        done: false,
        num_actions: 4,
        next_action: None, // Q-Learning doesn't use this
    };

    agent.update(&experience).expect("update should succeed");

    // Q(0, 2) should now be nonzero (or zero if next state had zero values)
    // It starts as: 0 + 0.5 * (0 + 0.99 * max(Q(1,a)) - 0) = 0
    // (all Q-values still zero for state 1)
    assert_eq!(agent.q_value(0, 2), 0.0);

    // After reaching a reward, values propagate backward
    let reward_exp = TabularExperience {
        state: 14,
        action: 2,
        reward: 1.0,
        next_state: 15, // Goal
        done: true,
        num_actions: 4,
        next_action: None,
    };

    agent.update(&reward_exp).expect("update");
    assert!(agent.q_value(14, 2) > 0.0, "Q-value should increase after reward");

    // ── 3. SARSA ──
    // Q(s,a) += alpha * (r + gamma * Q(s',a') - Q(s,a))
    //
    // On-policy: uses the actual next action a' (not max).
    // Learns the value of the policy being followed (including exploration).
    // Tends to find safer paths (avoids high-penalty states).

    let eps2 = EpsilonSchedule::linear(1.0, 0.01, 5000);
    let mut sarsa_agent = Sarsa::new(0.5, 0.99, eps2);

    // SARSA uses next_action in its update
    let sarsa_exp = TabularExperience {
        state: 0,
        action: 2,
        reward: 0.0,
        next_state: 1,
        done: false,
        num_actions: 4,
        next_action: Some(1), // The action actually taken next
    };

    sarsa_agent.update(&sarsa_exp).expect("update");

    // ── 4. Expected SARSA ──
    // Q(s,a) += alpha * (r + gamma * E[Q(s',a')] - Q(s,a))
    //
    // Uses expected value over all next actions (weighted by policy probability).
    // Smoother than SARSA, less optimistic than Q-Learning.

    let eps3 = EpsilonSchedule::linear(1.0, 0.01, 5000);
    let _expected_sarsa = ExpectedSarsa::new(0.5, 0.99, eps3);

    // ── 5. Monte Carlo ──
    // Updates Q-values using complete episode returns (no bootstrapping).
    // Unbiased but high variance.
    //
    // First-visit MC: update Q(s,a) only on the first visit to (s,a)
    // in each episode.

    let eps4 = EpsilonSchedule::linear(1.0, 0.01, 5000);
    let _mc_agent = MonteCarlo::new(0.99, eps4, McVariant::FirstVisit);

    // ── 6. Algorithm comparison on FrozenLake ──
    // Train each algorithm and compare convergence.

    let num_episodes = 5000;
    let seed = Some(42_u64);

    // Q-Learning
    let eps_ql = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut ql = QLearning::new(0.8, 0.95, eps_ql);
    let mut env_ql = FrozenLake::new(false);
    let stats_ql = train_tabular(&mut env_ql, &mut ql, num_episodes, seed).unwrap();

    // SARSA
    let eps_sa = EpsilonSchedule::linear(1.0, 0.01, 50_000);
    let mut sa = Sarsa::new(0.8, 0.95, eps_sa);
    let mut env_sa = FrozenLake::new(false);
    let stats_sa = train_tabular(&mut env_sa, &mut sa, num_episodes, seed).unwrap();

    // Both should learn something on deterministic FrozenLake
    let last_500_ql: Vec<_> = stats_ql.episodes.iter().rev().take(500).collect();
    let last_500_sa: Vec<_> = stats_sa.episodes.iter().rev().take(500).collect();

    let sr_ql = last_500_ql.iter().filter(|e| e.terminated && e.total_reward > 0.0).count() as f64 / 500.0;
    let sr_sa = last_500_sa.iter().filter(|e| e.terminated && e.total_reward > 0.0).count() as f64 / 500.0;

    assert!(sr_ql > 0.5, "Q-Learning should learn on deterministic FrozenLake");
    assert!(sr_sa > 0.5, "SARSA should learn on deterministic FrozenLake");

    // ── 7. Q-table inspection ──
    // After training, the Q-table reveals the learned policy.

    let q_table = ql.q_table();
    assert!(!q_table.is_empty(), "Q-table should have entries after training");

    // Best action for the start state
    let best = ql.best_action(0, 4);
    assert!(best < 4, "Best action must be valid");

    // Q-value for best action should be positive (goal is reachable)
    let best_q = ql.q_value(0, best);
    assert!(best_q > 0.0, "Start state should have positive Q-value");

    // ── 8. Epsilon-greedy exploration ──
    // During training, agents select actions epsilon-greedily:
    //   With probability epsilon: random action (explore)
    //   With probability 1-epsilon: best action (exploit)
    //
    // As epsilon decays, the agent exploits more and explores less.

    let mut schedule = EpsilonSchedule::linear(1.0, 0.01, 100);

    // Track epsilon decay
    let mut epsilon_values = Vec::new();
    for _ in 0..120 {
        epsilon_values.push(schedule.get_and_step());
    }

    // First value should be high (exploratory)
    assert!(epsilon_values[0] > 0.9);
    // Last values should be at minimum (exploitative)
    assert!(epsilon_values.last().unwrap() < &0.02);
    // Monotonically non-increasing
    for i in 1..epsilon_values.len() {
        assert!(epsilon_values[i] <= epsilon_values[i - 1] + 1e-10);
    }

    // ── 9. Key concepts ──
    // - TabularAgent: Q-table-based agent interface (act, update, q_table)
    // - Q-Learning: off-policy, uses max Q(s',a') — finds optimal policy
    // - SARSA: on-policy, uses actual next action — finds safe policy
    // - Expected SARSA: uses expected value — smoother than SARSA
    // - Monte Carlo: uses complete episode returns — unbiased, high variance
    // - Q-table: HashMap<(state, action), value> — the learned knowledge
    // - best_action: greedy action selection from Q-table
    // - Epsilon-greedy: explore (random) with probability epsilon, else exploit
    // - EpsilonSchedule: linearly decays epsilon from high to low

    println!("PASS: 08-rl/03_policies");
}
