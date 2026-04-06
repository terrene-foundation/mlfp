# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 3: RL Fundamentals with RLTrainer
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Understand MDPs, value functions, and Q-learning by
#   building a simple RL agent, then use RLTrainer for a real
#   environment.
#
# TASKS:
#   1. Define MDP for grid-world (states, actions, transitions, rewards)
#   2. Implement value iteration
#   3. Implement Q-learning from scratch
#   4. Compare with RLTrainer(algorithm="dqn")
#   5. Visualize learning curves
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import random

import polars as pl

from kailash_ml import ModelVisualizer, RLTrainer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define MDP for grid-world
# ══════════════════════════════════════════════════════════════════════

# 4x4 grid-world: agent navigates from (0,0) to (3,3)
# Actions: 0=up, 1=right, 2=down, 3=left
GRID_SIZE = 4
GOAL = (3, 3)
OBSTACLES = [(1, 1), (2, 2)]
ACTIONS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
ACTION_NAMES = {0: "up", 1: "right", 2: "down", 3: "left"}


def get_next_state(state: tuple[int, int], action: int) -> tuple[int, int]:
    """Transition function: T(s, a) -> s'."""
    # TODO: Compute next state given action, stay in bounds, avoid obstacles.
    # Hint: Use ACTIONS[action] for (dr, dc), check bounds and OBSTACLES.
    dr, dc = ACTIONS[action]
    nr, nc = ____
    if ____:
        return (nr, nc)
    return state  # Invalid move — stay


def get_reward(state: tuple[int, int], next_state: tuple[int, int]) -> float:
    """Reward function: R(s, a, s')."""
    # TODO: Return +10 for reaching GOAL, -1 for hitting wall/obstacle, -0.1 for step cost.
    # Hint: Check if next_state == GOAL, next_state == state, else step cost.
    if ____:
        return 10.0
    if ____:
        return -1.0
    return -0.1


def get_all_states() -> list[tuple[int, int]]:
    """All non-obstacle states in the grid."""
    # TODO: Return list of (r, c) for all cells not in OBSTACLES.
    # Hint: List comprehension over range(GRID_SIZE) x range(GRID_SIZE).
    return ____


states = get_all_states()
print("=== Grid-World MDP ===")
print(f"States: {len(states)}, Actions: {len(ACTIONS)}")
print(f"Goal: {GOAL}, Obstacles: {OBSTACLES}")
print(f"Reward: +10 (goal), -1 (wall/obstacle), -0.1 (step)")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Implement value iteration
# ══════════════════════════════════════════════════════════════════════

gamma = 0.99
theta = 1e-6  # Convergence threshold

# Initialize value function
V = {s: 0.0 for s in states}
V[GOAL] = 0.0  # Terminal state

iteration = 0
while True:
    delta = 0.0
    for s in states:
        if s == GOAL:
            continue
        old_v = V[s]
        # TODO: Bellman optimality update: V(s) = max_a [R(s,a,s') + gamma * V(s')]
        # Hint: Loop over ACTIONS, compute get_next_state + get_reward for each.
        action_values = []
        for a in ACTIONS:
            s_next = ____
            r = ____
            action_values.append(____)
        V[s] = ____
        delta = max(delta, abs(old_v - V[s]))
    iteration += 1
    if delta < theta:
        break

# TODO: Extract policy from value function.
# Hint: For each state, pick the action that maximizes R + gamma * V(s').
policy = {}
for s in states:
    if s == GOAL:
        policy[s] = -1  # Terminal
        continue
    best_a = ____
    policy[s] = best_a

print(f"\n=== Value Iteration ===")
print(f"Converged in {iteration} iterations")
print("\nValue function:")
for r in range(GRID_SIZE):
    row = []
    for c in range(GRID_SIZE):
        if (r, c) in OBSTACLES:
            row.append("  XXX ")
        else:
            row.append(f"{V.get((r, c), 0):6.2f}")
    print("  ".join(row))

print("\nOptimal policy:")
for r in range(GRID_SIZE):
    row = []
    for c in range(GRID_SIZE):
        if (r, c) in OBSTACLES:
            row.append(" X ")
        elif (r, c) == GOAL:
            row.append(" G ")
        else:
            row.append(f" {ACTION_NAMES[policy[(r, c)]][0].upper()} ")
    print("".join(row))


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement Q-learning from scratch
# ══════════════════════════════════════════════════════════════════════

Q = {(s, a): 0.0 for s in states for a in ACTIONS}
alpha = 0.1  # Learning rate
epsilon = 0.3  # Exploration rate
n_episodes = 500
episode_rewards = []

for ep in range(n_episodes):
    state = (0, 0)
    total_reward = 0.0
    steps = 0

    while state != GOAL and steps < 100:
        # TODO: Epsilon-greedy action selection.
        # Hint: random.random() < epsilon -> random action, else argmax Q.
        if ____:
            action = ____
        else:
            action = ____

        next_state = get_next_state(state, action)
        reward = get_reward(state, next_state)

        # TODO: Q-learning update rule.
        # Hint: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        best_next = ____
        Q[(state, action)] += ____

        total_reward += reward
        state = next_state
        steps += 1

    episode_rewards.append(total_reward)

    # TODO: Decay epsilon.
    # Hint: epsilon = max(0.01, epsilon * 0.995)
    epsilon = ____

# TODO: Extract Q-learning policy from Q-table.
# Hint: For each state, pick the action with highest Q-value.
q_policy = {}
for s in states:
    q_policy[s] = ____

print("\n=== Q-Learning ===")
print(f"Episodes: {n_episodes}")
print(f"Final avg reward (last 50): {sum(episode_rewards[-50:]) / 50:.2f}")
print(
    f"Q-policy matches value iteration: "
    f"{sum(1 for s in states if s != GOAL and q_policy[s] == policy[s])}"
    f"/{len(states) - 1} states"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Compare with RLTrainer(algorithm="dqn")
# ══════════════════════════════════════════════════════════════════════

# TODO: Create RLTrainer with DQN algorithm.
# Hint: RLTrainer(algorithm="dqn", env_name=..., learning_rate=..., gamma=...,
#        epsilon_start=..., epsilon_end=..., epsilon_decay=..., buffer_size=...,
#        batch_size=..., n_epochs=...)
trainer = ____

# TODO: Train the DQN agent.
# Hint: trainer.train()
rl_result = ____

print("\n=== RLTrainer DQN ===")
print(f"Total reward: {rl_result.total_reward:.2f}")
print(f"Episodes: {len(rl_result.episode_lengths)}")
print(
    f"Avg episode length: {sum(rl_result.episode_lengths) / len(rl_result.episode_lengths):.1f}"
)
print(f"Policy learned: {rl_result.policy is not None}")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Visualize learning curves
# ══════════════════════════════════════════════════════════════════════

# TODO: Build polars DataFrame for Q-learning curve.
# Hint: pl.DataFrame with episode, reward, method columns.
q_learning_curve = ____

# TODO: Compute rolling average for smoothed curve.
# Hint: .with_columns(pl.col("reward").rolling_mean(window_size=20).alias(...))
window = 20
smoothed = ____

# TODO: Plot learning curve with ModelVisualizer.
# Hint: ModelVisualizer(), visualizer.plot_learning_curve(data=..., x_col=..., y_col=..., ...)
visualizer = ____
____

# TODO: Create summary comparison DataFrame.
# Hint: pl.DataFrame with method, type, requires_model, scalable, final_performance columns.
print("\n=== Method Comparison ===")
comparison = ____
print(comparison)

print(
    "\n✓ Exercise 3 complete — RL fundamentals: MDP, value iteration, Q-learning, DQN"
)
