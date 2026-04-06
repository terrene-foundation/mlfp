# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT06 — Exercise 5: Reinforcement Learning
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

        # TODO: define observation_space as spaces.Box with:
        #   low=np.array([0, 0, 0], dtype=np.float32)
        #   high=np.array([max_stock, 6, 2], dtype=np.float32)
        self.observation_space = ____  # Hint: spaces.Box(low=np.array([0, 0, 0], dtype=np.float32), high=np.array([self.max_stock, 6, 2], dtype=np.float32))

        # TODO: define action_space as spaces.Discrete(max_order + 1)
        self.action_space = ____  # Hint: spaces.Discrete(self.max_order + 1)

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
        # TODO: compute weekly seasonality trend and return float32 observation array
        # Hint: trend = 1 + 0.5 * np.sin(2 * np.pi * self.day / 7)
        #       return np.array([self.stock, self.day % 7, trend], dtype=np.float32)
        trend = ____
        return ____

    def step(self, action):
        # TODO: implement one step of the inventory simulation
        # 1. Receive order: stock = min(stock + order_qty, max_stock)
        # 2. Generate stochastic demand: base=15, day_factor = 1.0 + 0.3*sin(2π*day/7)
        #    demand = max(0, int(rng.poisson(base * day_factor)))
        # 3. Fulfill: sold = min(demand, stock); stockout = demand - sold; stock -= sold
        # 4. Economics per day:
        #    revenue         = sold * 10.0      ($10 per unit sold)
        #    holding_cost    = stock * 0.50     ($0.50 per unit per day)
        #    stockout_penalty= stockout * 5.0   ($5 per lost sale)
        #    order_cost      = order_qty * 3.0  ($3 per unit ordered)
        #    reward = revenue - holding_cost - stockout_penalty - order_cost
        # 5. Advance day; terminated = step_count >= max_steps
        order_qty = int(action)

        self.stock = ____  # Hint: min(self.stock + order_qty, self.max_stock)

        base_demand = 15
        day_factor = ____  # Hint: 1.0 + 0.3 * np.sin(2 * np.pi * self.day / 7)
        demand = ____  # Hint: max(0, int(self.rng.poisson(base_demand * day_factor)))

        sold = ____  # Hint: min(demand, self.stock)
        stockout = ____  # Hint: demand - sold
        self.stock -= sold

        revenue = ____  # Hint: sold * 10.0
        holding_cost = ____  # Hint: self.stock * 0.50
        stockout_penalty = ____  # Hint: stockout * 5.0
        order_cost = ____  # Hint: order_qty * 3.0

        reward = ____  # Hint: revenue - holding_cost - stockout_penalty - order_cost
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
    # TODO: create an RLTrainer and call trainer.train() with the PPO config:
    #   algorithm="ppo", total_timesteps=50_000, learning_rate=3e-4,
    #   n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99,
    #   gae_lambda=0.95, clip_range=0.2, seed=42
    #   clip_range implements L^CLIP = min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)
    trainer = ____  # Hint: RLTrainer()

    print(f"\n=== Training PPO Agent ===")
    result = await ____  # Hint: trainer.train(env=env, algorithm="ppo", config={...})

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


# TODO: implement (s, S) reorder policy: if stock < s=20, order up to S=60
# obs[0] is the current stock level. Return an int in [0, 50].
def ss_policy(obs):
    """(s, S) reorder policy: if stock < s, order up to S."""
    stock = obs[0]
    s, S = 20, 60
    # TODO: return min(int(S - stock), 50) when stock < s, else 0
    return ____  # Hint: min(int(S - stock), 50) if stock < s else 0


# TODO: implement a random baseline policy returning a random int in [0, 50]
def random_policy(obs):
    return ____  # Hint: np.random.randint(0, 51)


# Evaluate all policies
heuristic_rewards = evaluate_policy(env, ss_policy)


async def evaluate_rl():
    # TODO: evaluate the trained RL agent over 100 episodes
    rl_rewards = await ____  # Hint: trainer.evaluate(env, n_episodes=100)
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

# TODO: create a ModelVisualizer and build a metric_comparison chart for all
# three policies, each with Mean_Reward and Std_Reward keys. Save as HTML.
viz = ____  # Hint: ModelVisualizer()
fig = ____  # Hint: viz.metric_comparison({"Random": {"Mean_Reward": ..., "Std_Reward": ...}, ...})
fig.update_layout(title="Inventory Management: Policy Comparison")
fig.write_html("ex5_rl_comparison.html")
print("\nSaved: ex5_rl_comparison.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Track with ExperimentTracker
# ══════════════════════════════════════════════════════════════════════


async def track_experiment():
    # TODO: create a ConnectionManager and ExperimentTracker, then log all
    # three policies as separate runs under experiment "ascent06_reinforcement_learning".
    # Each run should log params, four metrics (mean/std/min/max reward),
    # and set tag "domain"="rl-inventory".
    conn = ____  # Hint: ConnectionManager("sqlite:///ascent06_experiments.db")
    await conn.initialize()
    tracker = ____  # Hint: ExperimentTracker(conn)
    await tracker.initialize()

    exp_id = (
        await ____
    )  # Hint: tracker.create_experiment(name="ascent06_reinforcement_learning", description="RL inventory management — PPO vs heuristic")

    for run_name, rewards, run_params in [
        ("random_baseline", random_rewards, {"policy": "random"}),
        ("ss_heuristic", heuristic_rewards, {"policy": "(s,S)", "s": "20", "S": "60"}),
        ("ppo_agent", rl_rewards, {"algorithm": "PPO", "timesteps": "50000"}),
    ]:
        # TODO: open a run context, log params, log metrics, set tag
        # Hint: async with tracker.run(exp_id, run_name=run_name) as run:
        #           await run.log_params(run_params)
        #           await run.log_metrics({"mean_reward": float(rewards.mean()), ...})
        #           await run.set_tag("domain", "rl-inventory")
        async with ____:  # Hint: tracker.run(exp_id, run_name=run_name) as run
            await ____  # Hint: run.log_params(run_params)
            await ____  # Hint: run.log_metrics({...})
            await ____  # Hint: run.set_tag("domain", "rl-inventory")

    print(f"\nLogged 3 runs to ExperimentTracker")
    await conn.close()


asyncio.run(track_experiment())

print("\n✓ Exercise 5 complete — RL inventory management with PPO")
