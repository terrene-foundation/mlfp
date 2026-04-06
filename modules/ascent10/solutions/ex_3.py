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
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    # Stay in bounds and avoid obstacles
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and (nr, nc) not in OBSTACLES:
        return (nr, nc)
    return state  # Invalid move — stay


def get_reward(state: tuple[int, int], next_state: tuple[int, int]) -> float:
    """Reward function: R(s, a, s')."""
    if next_state == GOAL:
        return 10.0
    if next_state == state:  # Hit wall/obstacle
        return -1.0
    return -0.1  # Step cost


def get_all_states() -> list[tuple[int, int]]:
    """All non-obstacle states in the grid."""
    return [
        (r, c)
        for r in range(GRID_SIZE)
        for c in range(GRID_SIZE)
        if (r, c) not in OBSTACLES
    ]


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
        # Bellman optimality: V(s) = max_a [R(s,a,s') + gamma * V(s')]
        action_values = []
        for a in ACTIONS:
            s_next = get_next_state(s, a)
            r = get_reward(s, s_next)
            action_values.append(r + gamma * V[s_next])
        V[s] = max(action_values)
        delta = max(delta, abs(old_v - V[s]))
    iteration += 1
    if delta < theta:
        break

# Extract policy from value function
policy = {}
for s in states:
    if s == GOAL:
        policy[s] = -1  # Terminal
        continue
    best_a = max(
        ACTIONS.keys(),
        key=lambda a: get_reward(s, get_next_state(s, a))
        + gamma * V[get_next_state(s, a)],
    )
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
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(list(ACTIONS.keys()))
        else:
            action = max(ACTIONS.keys(), key=lambda a: Q[(state, a)])

        next_state = get_next_state(state, action)
        reward = get_reward(state, next_state)

        # Q-learning update: Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        best_next = max(Q[(next_state, a)] for a in ACTIONS)
        Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])

        total_reward += reward
        state = next_state
        steps += 1

    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(0.01, epsilon * 0.995)

# Extract Q-learning policy
q_policy = {}
for s in states:
    q_policy[s] = max(ACTIONS.keys(), key=lambda a: Q[(s, a)])

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

trainer = RLTrainer(
    algorithm="dqn",
    env_name="grid_world",
    learning_rate=3e-4,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=32,
    n_epochs=200,
)

rl_result = trainer.train()

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

# Build polars DataFrame for visualization
q_learning_curve = pl.DataFrame(
    {
        "episode": list(range(n_episodes)),
        "reward": episode_rewards,
        "method": ["Q-learning"] * n_episodes,
    }
)

# Rolling average for smoothed curve
window = 20
smoothed = q_learning_curve.with_columns(
    pl.col("reward").rolling_mean(window_size=window).alias("smoothed_reward")
)

visualizer = ModelVisualizer()
visualizer.plot_learning_curve(
    data=smoothed,
    x_col="episode",
    y_col="smoothed_reward",
    title="Q-Learning Convergence (Grid-World)",
    output_path="./rl_learning_curve.html",
)

# Summary comparison
print("\n=== Method Comparison ===")
comparison = pl.DataFrame(
    {
        "method": ["Value Iteration", "Q-Learning", "RLTrainer DQN"],
        "type": ["Model-based", "Model-free (tabular)", "Model-free (neural)"],
        "requires_model": [True, False, False],
        "scalable": [False, False, True],
        "final_performance": [
            f"{V[(0,0)]:.2f}",
            f"{sum(episode_rewards[-50:]) / 50:.2f}",
            f"{rl_result.total_reward:.2f}",
        ],
    }
)
print(comparison)

print(
    "\n✓ Exercise 3 complete — RL fundamentals: MDP, value iteration, Q-learning, DQN"
)
