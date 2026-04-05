# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT6 — Exercise 3: Reinforcement Learning
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Use RLTrainer with PPO/SAC on an inventory management
#   environment. Compare RL policy vs heuristic baseline.
#
# TASKS:
#   1. Set up Gymnasium environment (inventory management)
#   2. Configure RLTrainer with PPO
#   3. Train RL agent
#   4. Implement heuristic baseline for comparison
#   5. Evaluate and compare policies
#   6. Track with ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from kailash.db.connection import ConnectionManager
from kailash_ml.rl.trainer import RLTrainer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml import ModelVisualizer

from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Custom Gymnasium environment — inventory management
# ══════════════════════════════════════════════════════════════════════


class InventoryEnv(gym.Env):
    """Simplified inventory management environment.

    State: [current_stock, day_of_week, demand_trend]
    Action: order_quantity (0 to max_order)
    Reward: revenue from sales - holding cost - stockout penalty - order cost
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.max_stock = 100
        self.max_order = 50
        self.max_steps = 30  # One month

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([self.max_stock, 6, 2], dtype=np.float32),
        )
        self.action_space = spaces.Discrete(self.max_order + 1)

        self.rng = np.random.default_rng()
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.stock = 50
        self.day = 0
        self.step_count = 0
        self.total_revenue = 0
        self.total_cost = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Demand trend: 0=low, 1=normal, 2=high
        trend = 1 + 0.5 * np.sin(2 * np.pi * self.day / 7)  # Weekly seasonality
        return np.array([self.stock, self.day % 7, trend], dtype=np.float32)

    def step(self, action):
        order_qty = int(action)

        # Receive order (1-day lead time from yesterday's order is simplified away)
        self.stock = min(self.stock + order_qty, self.max_stock)

        # Generate demand (stochastic with weekly pattern)
        base_demand = 15
        day_factor = 1.0 + 0.3 * np.sin(2 * np.pi * self.day / 7)
        demand = max(0, int(self.rng.poisson(base_demand * day_factor)))

        # Fulfill demand
        sold = min(demand, self.stock)
        stockout = demand - sold
        self.stock -= sold

        # Economics
        revenue = sold * 10.0  # $10 per unit sold
        holding_cost = self.stock * 0.50  # $0.50 per unit per day
        stockout_penalty = stockout * 5.0  # $5 per lost sale
        order_cost = order_qty * 3.0  # $3 per unit ordered

        reward = revenue - holding_cost - stockout_penalty - order_cost
        self.total_revenue += revenue
        self.total_cost += holding_cost + stockout_penalty + order_cost

        self.day += 1
        self.step_count += 1
        terminated = self.step_count >= self.max_steps
        truncated = False

        return (
            self._get_obs(),
            reward,
            terminated,
            truncated,
            {
                "sold": sold,
                "demand": demand,
                "stockout": stockout,
                "stock": self.stock,
                "order": order_qty,
            },
        )


# Register environment
env = InventoryEnv()
print(f"=== Inventory Management Environment ===")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Episode length: {env.max_steps} days")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Configure RLTrainer with PPO
# ══════════════════════════════════════════════════════════════════════


async def train_rl():
    trainer = RLTrainer()

    print(f"\n=== Training PPO Agent ===")
    result = await trainer.train(
        env=env,
        algorithm="ppo",
        config={
            "total_timesteps": 50_000,
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,  # PPO clip: L^CLIP = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
            "seed": 42,
        },
    )

    print(f"Training complete:")
    print(f"  Mean reward: {result.mean_reward:.2f}")
    print(f"  Std reward: {result.std_reward:.2f}")
    print(f"  Training time: {result.training_time_seconds:.0f}s")

    return trainer, result


trainer, rl_result = asyncio.run(train_rl())


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Heuristic baseline — (s, S) policy
# ══════════════════════════════════════════════════════════════════════
# Classic inventory policy: if stock < s, order up to S


def evaluate_policy(env, policy_fn, n_episodes=100, seed=42):
    """Evaluate a policy over multiple episodes."""
    rng = np.random.default_rng(seed)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 10000)))
        total_reward = 0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return np.array(rewards)


# Heuristic: (s, S) policy with s=20, S=60
def ss_policy(obs):
    """(s, S) reorder policy: if stock < s, order up to S."""
    stock = obs[0]
    s, S = 20, 60  # Reorder point and order-up-to level
    if stock < s:
        return min(int(S - stock), 50)  # Order up to S
    return 0  # Don't order


# Random baseline
def random_policy(obs):
    return np.random.randint(0, 51)


# Evaluate all policies
heuristic_rewards = evaluate_policy(env, ss_policy)


async def evaluate_rl():
    rl_rewards = await trainer.evaluate(env, n_episodes=100)
    return rl_rewards


rl_rewards = asyncio.run(evaluate_rl())
random_rewards = evaluate_policy(env, random_policy)

print(f"\n=== Policy Comparison ===")
print(f"{'Policy':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("─" * 55)
for name, rewards in [
    ("Random", random_rewards),
    ("(s,S) Heuristic", heuristic_rewards),
    ("PPO", rl_rewards),
]:
    print(
        f"{name:<15} {rewards.mean():>10.1f} {rewards.std():>10.1f} "
        f"{rewards.min():>10.1f} {rewards.max():>10.1f}"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualise training and comparison
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

fig = viz.metric_comparison(
    {
        "Random": {
            "Mean_Reward": random_rewards.mean(),
            "Std_Reward": random_rewards.std(),
        },
        "(s,S) Heuristic": {
            "Mean_Reward": heuristic_rewards.mean(),
            "Std_Reward": heuristic_rewards.std(),
        },
        "PPO": {"Mean_Reward": rl_rewards.mean(), "Std_Reward": rl_rewards.std()},
    }
)
fig.update_layout(title="Inventory Management: Policy Comparison")
fig.write_html("ex5_rl_comparison.html")
print("\nSaved: ex5_rl_comparison.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Track with ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def track_experiment():
    conn = ConnectionManager("sqlite:///ascent06_experiments.db")
    await conn.initialize()
    tracker = ExperimentTracker(conn)
    await tracker.initialize()

    exp_id = await tracker.create_experiment(
        name="ascent06_reinforcement_learning",
        description="RL inventory management — PPO vs heuristic",
    )

    for run_name, rewards, run_params in [
        ("random_baseline", random_rewards, {"policy": "random"}),
        ("ss_heuristic", heuristic_rewards, {"policy": "(s,S)", "s": "20", "S": "60"}),
        ("ppo_agent", rl_rewards, {"algorithm": "PPO", "timesteps": "50000"}),
    ]:
        async with tracker.run(exp_id, run_name=run_name) as run:
            await run.log_params(run_params)
            await run.log_metrics(
                {
                    "mean_reward": float(rewards.mean()),
                    "std_reward": float(rewards.std()),
                    "min_reward": float(rewards.min()),
                    "max_reward": float(rewards.max()),
                }
            )
            await run.set_tag("domain", "rl-inventory")

    print(f"\nLogged 3 runs to ExperimentTracker")
    await conn.close()


asyncio.run(track_experiment())

print("\n✓ Exercise 3 complete — RL inventory management with PPO")
