# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
Shared infrastructure for Exercise 8 — Reinforcement Learning.

Contains: CartPole setup, reward plotting helpers, ExperimentTracker/ModelRegistry
setup, custom environment base class, evaluation utilities.
Technique-specific code does NOT belong here.
"""
from __future__ import annotations

import asyncio
import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

from kailash.db import ConnectionManager
from kailash_ml import ExperimentTracker, ModelVisualizer
from kailash_ml import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

# ════════════════════════════════════════════════════════════════════════
# ENVIRONMENT SETUP
# ════════════════════════════════════════════════════════════════════════

setup_environment()
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = get_device()

# Output directory for all visualisation artifacts
OUTPUT_DIR = Path("outputs") / "ex8_reinforcement_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════════
# CARTPOLE ENVIRONMENT
# ════════════════════════════════════════════════════════════════════════


def make_cartpole() -> tuple[gym.Env, int, int]:
    """Create CartPole-v1 and return (env, obs_dim, n_actions)."""
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    act_space = env.action_space
    assert (
        isinstance(obs_space, spaces.Box) and obs_space.shape is not None
    ), f"CartPole obs space expected gymnasium Box, got {type(obs_space).__name__}"
    assert isinstance(
        act_space, spaces.Discrete
    ), f"CartPole action space expected Discrete, got {type(act_space).__name__}"
    obs_dim = obs_space.shape[0]
    n_actions = int(act_space.n)
    print(f"CartPole-v1  obs_dim={obs_dim}  n_actions={n_actions}")
    return env, obs_dim, n_actions


# ════════════════════════════════════════════════════════════════════════
# KAILASH ENGINE SETUP
# ════════════════════════════════════════════════════════════════════════


async def _setup_engines():
    """Open kailash-ml 1.1.1 tracker + registry. 5-tuple preserved."""
    # Schema-conflict workaround (kailash-ml 1.5.x): ExperimentTracker
    # and ModelRegistry use incompatible _kml_model_versions schemas.
    # Route them to separate sqlite files until upstream fixes the conflict.
    db = "sqlite:///mlfp05_rl.db"
    registry_db = "sqlite:///mlfp05_rl_registry.db"
    tracker = await ExperimentTracker.create(store_url=db)
    conn = ConnectionManager(registry_db)
    await conn.initialize()
    registry = ModelRegistry(conn)
    return conn, tracker, "m5_reinforcement_learning", registry, True


def setup_engines() -> tuple:
    """Synchronously set up kailash-ml engines."""
    return asyncio.run(_setup_engines())


# ════════════════════════════════════════════════════════════════════════
# REPLAY BUFFER — shared by DQN and custom env training
# ════════════════════════════════════════════════════════════════════════


class ReplayBuffer:
    """Fixed-size buffer storing (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 10_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(list(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(actions, dtype=torch.long, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════════════════
# DQN NETWORK — shared by DQN training and custom env training
# ════════════════════════════════════════════════════════════════════════


class DQN(nn.Module):
    """Deep Q-Network: maps state -> Q-value for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ════════════════════════════════════════════════════════════════════════
# EVALUATION UTILITY
# ════════════════════════════════════════════════════════════════════════


def evaluate_policy(env: gym.Env, policy_fn, n_episodes: int = 30) -> list[float]:
    """Evaluate a policy function over n_episodes. Returns list of total rewards."""
    eval_returns: list[float] = []
    for i in range(n_episodes):
        state, _ = env.reset(seed=1000 + i)
        total = 0.0
        done = False
        while not done:
            action = policy_fn(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = terminated or truncated
        eval_returns.append(total)
    return eval_returns


# ════════════════════════════════════════════════════════════════════════
# REWARD PLOTTING HELPERS
# ════════════════════════════════════════════════════════════════════════


def moving_average(xs: list[float], window: int = 10) -> list[float]:
    """Smooth a time series with a rolling mean."""
    if len(xs) < window:
        return xs
    arr = np.asarray(xs, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return list(np.convolve(arr, kernel, mode="valid"))


def plot_reward_curve(
    viz: ModelVisualizer,
    rewards: list[float],
    title: str,
    filename: str,
    window: int = 20,
    x_label: str = "Episode",
    y_label: str = "Reward",
) -> None:
    """Plot a reward curve with moving average and save to HTML."""
    metrics = {
        f"{title} reward": rewards,
        f"{title} moving avg ({window})": moving_average(rewards, window),
    }
    fig = viz.training_history(metrics=metrics, x_label=x_label, y_label=y_label)
    out_path = OUTPUT_DIR / filename
    fig.write_html(str(out_path))
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════
# RL DIAGNOSTIC CHECKPOINT — kailash_ml.diagnostics.RLDiagnostics
# ════════════════════════════════════════════════════════════════════════
# The DL Prescription Pad (gradient flow / dead neurons / loss trend) is
# built for supervised batches. RL has its own instrument in kailash-ml:
# RLDiagnostics (also what `km.diagnose(algo, kind="rl")` returns). It
# installs no hooks — you feed it the training history you recorded and
# `report()` summarises it. Its automated finding is REWARD COLLAPSE: a
# CRIT alert when the latest reward falls to <10% of the peak after a
# >=50% drop over the rolling window.


def rl_diagnostic_checkpoint(
    title: str,
    algo: str,
    rewards: list[float],
    *,
    lengths: list[int] | None = None,
    q_losses: list[float] | None = None,
    policy_losses: list[float] | None = None,
    value_losses: list[float] | None = None,
    entropies: list[float] | None = None,
    window: int = 20,
) -> dict:
    """Feed a recorded RL training history to RLDiagnostics and print it.

    ``rewards`` holds one entry per episode (DQN) or per PPO iteration
    (the iteration's mean episode return). ``policy_losses`` /
    ``value_losses`` / ``entropies`` are per PPO iteration; ``q_losses``
    per DQN episode (entries of 0.0 = no gradient step yet are skipped).
    Returns the ``report()`` dict.
    """
    from kailash_ml.diagnostics import RLDiagnostics

    diag = RLDiagnostics(algo=algo, window=window)
    for i, reward in enumerate(rewards):
        length = int(lengths[i]) if lengths is not None else 0
        diag.record_episode(reward=float(reward), length=length)
    if policy_losses is not None:
        for i, loss in enumerate(policy_losses):
            entropy = float(entropies[i]) if entropies is not None else None
            diag.record_policy_update(float(loss), entropy=entropy)
            if value_losses is not None:
                diag.record_value_update(float(value_losses[i]))
    n_q = 0
    for loss in q_losses or []:
        if loss > 0.0:
            diag.record_q_update(float(loss))
            n_q += 1
    report = diag.report()

    metrics = report["metrics"]
    unit = "episodes" if algo == "dqn" else "iterations"
    print("=" * 66)
    print(f"  RL Diagnostics — {title}")
    print("=" * 66)
    print(f"  {unit} recorded:            {metrics['episode_count']}")
    print(
        f"  mean reward (last {min(window, len(rewards))} {unit}): "
        f"{metrics['episode_reward_mean']:.2f}"
    )
    print(f"  peak reward:                {metrics['episode_reward_peak']:.2f}")
    if policy_losses is not None:
        print(f"  policy updates recorded:    {metrics['update_count']}")
    if q_losses is not None:
        print(f"  Q-loss entries recorded:    {n_q}")
    if report["findings"]:
        for finding in report["findings"]:
            print(f"  [{finding['severity']}] {finding['category']}: {finding['message']}")
            print(f"        suggestion: {finding['suggestion']}")
    else:
        print("  findings: none — no reward collapse over the rolling window")
    print("=" * 66)
    return report


# ════════════════════════════════════════════════════════════════════════
# MODEL REGISTRATION HELPER
# ════════════════════════════════════════════════════════════════════════


async def _register_rl_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    metrics_dict: dict[str, float],
):
    """Register a single RL policy network in ModelRegistry."""
    from kailash_ml.types import MetricSpec

    model_bytes = pickle.dumps(model.state_dict())
    metric_specs = [MetricSpec(name=k, value=v) for k, v in metrics_dict.items()]
    version = await registry.register_model(
        name=name,
        artifact=model_bytes,
        metrics=metric_specs,
    )
    print(f"  Registered {name}: version={version.version}")
    return version


def register_rl_model(
    registry: ModelRegistry,
    name: str,
    model: nn.Module,
    metrics_dict: dict[str, float],
):
    """Sync wrapper for RL model registration."""
    return asyncio.run(_register_rl_model(registry, name, model, metrics_dict))
