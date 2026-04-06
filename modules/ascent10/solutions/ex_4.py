# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT10 — Exercise 4: PPO Training for Inventory Management
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Train a PPO agent for inventory management — clipped
#   objective, GAE, reward shaping — using RLTrainer.
#
# TASKS:
#   1. Define inventory environment (state: stock, demand; action: order qty)
#   2. Configure RLTrainer with PPO
#   3. Implement reward function with penalties
#   4. Train and compare vs heuristic baseline
#   5. Analyze policy behavior across demand scenarios
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import random

import polars as pl

from kailash_ml import ModelVisualizer, RLTrainer

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Define inventory environment
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
demand_data = loader.load("ascent10", "inventory_demand.parquet")

print(f"=== Inventory Management Environment ===")
print(f"Demand data: {demand_data.shape}")


class InventoryEnv:
    """Simple inventory management environment.

    State: (current_stock, day_of_week, avg_recent_demand)
    Action: order_quantity (0 to max_order)
    Reward: revenue from sales - holding cost - stockout penalty - order cost
    """

    def __init__(
        self,
        demand_pattern: list[float],
        max_stock: int = 100,
        max_order: int = 50,
        holding_cost: float = 0.5,
        stockout_penalty: float = 5.0,
        order_cost: float = 1.0,
        unit_price: float = 10.0,
    ):
        self.demand_pattern = demand_pattern
        self.max_stock = max_stock
        self.max_order = max_order
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.order_cost = order_cost
        self.unit_price = unit_price
        self.stock = max_stock // 2
        self.day = 0
        self.total_reward = 0.0

    def reset(self) -> list[float]:
        self.stock = self.max_stock // 2
        self.day = 0
        self.total_reward = 0.0
        return self._get_state()

    def _get_state(self) -> list[float]:
        avg_demand = sum(self.demand_pattern[max(0, self.day - 7) : self.day + 1]) / 7
        return [
            self.stock / self.max_stock,
            (self.day % 7) / 6.0,
            avg_demand / 50.0,
        ]

    def step(self, action: int) -> tuple[list[float], float, bool]:
        # Apply order
        order_qty = min(action, self.max_order)
        self.stock = min(self.stock + order_qty, self.max_stock)

        # Demand arrives
        demand_idx = self.day % len(self.demand_pattern)
        demand = int(self.demand_pattern[demand_idx] + random.gauss(0, 3))
        demand = max(0, demand)

        # Calculate reward
        sold = min(self.stock, demand)
        revenue = sold * self.unit_price
        holding = self.stock * self.holding_cost
        stockout = max(0, demand - self.stock) * self.stockout_penalty
        order_expense = order_qty * self.order_cost

        reward = revenue - holding - stockout - order_expense
        self.stock = max(0, self.stock - demand)
        self.day += 1
        self.total_reward += reward

        done = self.day >= len(self.demand_pattern)
        return self._get_state(), reward, done


# Create environment with demand patterns from data
demand_values = demand_data["demand"].to_list()[:90]  # 90 days
env = InventoryEnv(demand_pattern=demand_values)

state = env.reset()
print(f"State space: [stock_ratio, day_of_week, avg_demand]")
print(f"Action space: order quantity (0 to {env.max_order})")
print(f"Initial state: {state}")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure RLTrainer with PPO
# ══════════════════════════════════════════════════════════════════════

trainer = RLTrainer(
    algorithm="ppo",
    env_name="inventory_management",
    learning_rate=3e-4,
    gamma=0.99,  # Discount factor — value future rewards highly
    clip_epsilon=0.2,  # PPO clipping — prevent large policy updates
    n_epochs=100,
    batch_size=64,
    gae_lambda=0.95,  # GAE — balance bias vs variance in advantage estimation
)

print(f"\n=== PPO Configuration ===")
print(f"Algorithm: Proximal Policy Optimization")
print(f"Key hyperparameters:")
print(f"  gamma={trainer.gamma}: discount factor (0.99 = long-term planning)")
print(f"  clip_epsilon={trainer.clip_epsilon}: limits policy change per update")
print(f"  gae_lambda={trainer.gae_lambda}: GAE bias-variance trade-off")
print(f"  learning_rate={trainer.learning_rate}")
print(f"\nPPO objective: maximize reward while staying close to the old policy")
print(f"L_CLIP = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Implement reward function with penalties
# ══════════════════════════════════════════════════════════════════════

print(f"\n=== Reward Shaping ===")
print(f"Revenue:          +${env.unit_price}/unit sold")
print(f"Holding cost:     -${env.holding_cost}/unit/day (penalizes overstocking)")
print(
    f"Stockout penalty: -${env.stockout_penalty}/unit missed (penalizes understocking)"
)
print(
    f"Order cost:       -${env.order_cost}/unit ordered (penalizes frequent ordering)"
)
print(f"\nThe agent must balance: order enough to meet demand, but not so much")
print(f"that holding costs eat into profits. This is the newsvendor problem.")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Train and compare vs heuristic baseline
# ══════════════════════════════════════════════════════════════════════


# Heuristic baseline: order up to a fixed target level
def heuristic_policy(state: list[float], target_ratio: float = 0.6) -> int:
    """Simple (s, S) policy: if stock below s, order up to S."""
    current_stock_ratio = state[0]
    target_stock = int(target_ratio * env.max_stock)
    current_stock = int(current_stock_ratio * env.max_stock)
    if current_stock < target_stock * 0.4:
        return min(target_stock - current_stock, env.max_order)
    return 0


# Evaluate heuristic
random.seed(42)
heuristic_rewards = []
for episode in range(10):
    state = env.reset()
    episode_reward = 0.0
    while True:
        action = heuristic_policy(state)
        state, reward, done = env.step(action)
        episode_reward += reward
        if done:
            break
    heuristic_rewards.append(episode_reward)

avg_heuristic = sum(heuristic_rewards) / len(heuristic_rewards)
print(f"\n=== Heuristic Baseline ===")
print(f"Policy: order up to 60% capacity when stock falls below 24%")
print(f"Average reward over 10 episodes: ${avg_heuristic:.2f}")

# Train PPO
print(f"\n=== PPO Training ===")
result = trainer.train()
print(f"Training complete:")
print(f"  Episodes: {result.n_episodes}")
print(f"  Final avg reward: ${result.avg_reward:.2f}")
print(f"  Best episode reward: ${result.best_reward:.2f}")

# Evaluate PPO policy
ppo_rewards = []
for episode in range(10):
    state = env.reset()
    episode_reward = 0.0
    while True:
        action = result.policy(state)
        state, reward, done = env.step(action)
        episode_reward += reward
        if done:
            break
    ppo_rewards.append(episode_reward)

avg_ppo = sum(ppo_rewards) / len(ppo_rewards)
improvement = ((avg_ppo - avg_heuristic) / abs(avg_heuristic)) * 100

print(f"\n=== Comparison ===")
print(f"Heuristic avg reward: ${avg_heuristic:.2f}")
print(f"PPO avg reward:       ${avg_ppo:.2f}")
print(f"Improvement:          {improvement:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Analyze policy behavior
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

print(f"\n=== Policy Analysis ===")
# Test policy at different stock levels
for stock_ratio in [0.1, 0.3, 0.5, 0.7, 0.9]:
    state = [stock_ratio, 0.5, 0.3]  # mid-week, average demand
    action = result.policy(state)
    print(f"  Stock={stock_ratio*100:.0f}%: order {action} units")

print(f"\nPolicy behavior:")
print(f"  Low stock → large orders (prevent stockouts)")
print(f"  High stock → small/no orders (minimize holding costs)")
print(f"  PPO learns the optimal reorder point adaptively")

print("\n✓ Exercise 4 complete — PPO inventory management vs heuristic baseline")
