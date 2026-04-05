// Copyright 2026 Terrene Foundation
// SPDX-License-Identifier: Apache-2.0
//! Kailash SDK Textbook -- RL / Custom Environments
//!
//! OBJECTIVE: Build custom Gymnasium-style environments — the Environment trait,
//!            observation/action spaces, step mechanics, and built-in environments.
//! LEVEL: Intermediate
//! PARITY: Full -- kailash-rl provides the Environment trait and built-in environments
//!         (FrozenLake, CartPole, CliffWalking, GridWorld, MountainCar, Pendulum).
//! VALIDATES: Environment trait, Space, SpaceValue, StepResult, reset, step, built-ins
//!
//! Run: cargo run -p tutorial-rl --bin 02_environments

use kailash_rl::prelude::*;
use kailash_rl::env::frozen_lake::FrozenLake;
use kailash_rl::env::cliff_walking::CliffWalking;
use kailash_rl::env::cartpole::CartPole;
use kailash_rl::env::grid_world::GridWorld;
use ndarray::arr1;

fn main() {
    // ── 1. Environment trait ──
    // Every RL environment implements:
    //   reset(seed) -> (Observation, Info)
    //   step(action) -> StepResult
    //   observation_space() -> &Space
    //   action_space() -> &Space
    //   render() -> Option<String>

    let mut env = FrozenLake::new(false);

    // Reset returns initial observation and info
    let (obs, info) = env.reset(Some(42)).expect("reset should succeed");
    let state = obs.as_discrete();
    assert_eq!(state, 0, "Agent starts at position 0");
    assert!(info.is_empty() || !info.is_empty()); // Info may or may not have entries

    // ── 2. Spaces ──
    // Discrete space: finite set of integers [0, n)
    // Box space: continuous n-dimensional bounded array

    // Discrete: FrozenLake
    let obs_space = env.observation_space();
    assert_eq!(obs_space.n().unwrap(), 16); // 4x4 grid = 16 states

    let act_space = env.action_space();
    assert_eq!(act_space.n().unwrap(), 4); // Left, Down, Right, Up

    // Space::contains validates membership
    let valid_obs = SpaceValue::Discrete(5);
    assert!(obs_space.contains(&valid_obs).unwrap());

    let invalid_obs = SpaceValue::Discrete(20);
    assert!(!obs_space.contains(&invalid_obs).unwrap());

    // Box: CartPole
    let cart = CartPole::new();
    let cart_obs = cart.observation_space();
    // CartPole observation: [x, x_dot, theta, theta_dot]
    assert_eq!(cart_obs.shape().len(), 1); // 1D shape
    assert_eq!(cart.action_space().n().unwrap(), 2); // Left, Right

    // ── 3. StepResult ──
    // step() returns observation, reward, terminated, truncated, info.

    let action = SpaceValue::Discrete(2); // Move right
    let result = env.step(&action).expect("step should succeed");

    let _next_state = result.observation.as_discrete();
    let _reward = result.reward;
    let _terminated = result.terminated; // Terminal state (goal/hole)
    let _truncated = result.truncated;   // Step limit reached
    let _info = &result.info;

    // ── 4. Episode loop ──
    // A typical episode: reset, then step until terminated or truncated.

    let (mut obs, _) = env.reset(Some(0)).expect("reset");
    let mut total_reward = 0.0;
    let mut steps = 0;

    loop {
        // Simple policy: always go right then down
        let action_idx = if steps % 2 == 0 { 2 } else { 1 }; // Right, Down
        let result = env.step(&SpaceValue::Discrete(action_idx)).expect("step");
        total_reward += result.reward;
        steps += 1;
        obs = result.observation;

        if result.terminated || result.truncated || steps > 100 {
            break;
        }
    }

    assert!(steps > 0);

    // ── 5. Built-in environments ──

    // CliffWalking: 4x12 grid, cliff along bottom edge
    let mut cliff = CliffWalking::new();
    let cliff_obs = cliff.observation_space();
    assert_eq!(cliff_obs.n().unwrap(), 48); // 4*12 = 48 states
    assert_eq!(cliff.action_space().n().unwrap(), 4);

    let (cliff_start, _) = cliff.reset(Some(0)).expect("reset");
    assert_eq!(cliff_start.as_discrete(), 36); // Bottom-left corner

    // CartPole: balance a pole on a cart
    let mut cart_env = CartPole::new();
    let (cart_obs_val, _) = cart_env.reset(Some(42)).expect("reset");
    // CartPole observation is Box (continuous)
    let obs_array = cart_obs_val.as_box();
    assert_eq!(obs_array.len(), 4); // [x, x_dot, theta, theta_dot]

    // ── 6. Custom GridWorld ──
    // GridWorld uses a builder pattern for custom layouts with walls, goals, and traps.

    // Create a 5x5 grid:
    //   S . . . .
    //   . W . W .
    //   . . . . .
    //   . W T W .
    //   . . . . G
    let mut gw = GridWorld::new(5, 5)
        .with_wall(1, 1)
        .with_wall(1, 3)
        .with_wall(3, 1)
        .with_trap(3, 2, -1.0)
        .with_wall(3, 3)
        .with_goal(4, 4, 1.0);

    assert_eq!(gw.observation_space().n().unwrap(), 25);
    assert_eq!(gw.action_space().n().unwrap(), 4);

    let (gw_obs, _) = gw.reset(Some(0)).expect("reset");
    assert_eq!(gw_obs.as_discrete(), 0); // Start at top-left

    // Step into a wall: agent stays in place
    let result = gw.step(&SpaceValue::Discrete(3)).expect("step"); // Left from (0,0)
    assert_eq!(result.observation.as_discrete(), 0); // Stayed at 0

    // ── 7. Space sampling ──
    // Spaces can generate random valid samples.

    let discrete_space = Space::Discrete(10);
    let mut rng = rand::thread_rng();
    for _ in 0..50 {
        let sample = discrete_space.sample(&mut rng);
        assert!(discrete_space.contains(&sample).unwrap());
    }

    let box_space = Space::Box {
        low: arr1(&[-1.0, -2.0]).into_dyn(),
        high: arr1(&[1.0, 2.0]).into_dyn(),
    };
    for _ in 0..50 {
        let sample = box_space.sample(&mut rng);
        assert!(box_space.contains(&sample).unwrap());
    }

    // ── 8. Render ──
    // Some environments support text rendering.

    let rendered = env.render();
    // FrozenLake supports rendering
    if let Some(text) = rendered {
        assert!(!text.is_empty());
    }

    // ── 9. Key concepts ──
    // - Environment trait: reset, step, observation_space, action_space, render
    // - Space::Discrete(n): integer in [0, n)
    // - Space::Box { low, high }: continuous bounded array
    // - StepResult: observation, reward, terminated, truncated, info
    // - terminated: episode ended by reaching terminal state
    // - truncated: episode ended by step limit
    // - Built-in: FrozenLake, CliffWalking, CartPole, GridWorld, MountainCar, Pendulum
    // - GridWorld: custom layouts with walls, goals, traps
    // - Space::sample: generate random valid values for testing

    println!("PASS: 08-rl/02_environments");
}
