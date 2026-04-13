# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 8: Reinforcement Learning (DQN + PPO)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain the RL framework: agent, environment, state, action, reward
#   - Derive the Bellman expectation and optimality equations
#   - Implement DQN from scratch (experience replay, target network,
#     epsilon-greedy exploration) and train it on CartPole-v1
#   - Implement PPO from scratch (clipped surrogate objective, GAE
#     advantage estimation, actor-critic architecture)
#   - Build 5 custom Gymnasium environments for business applications:
#     customer churn, dynamic pricing, portfolio rebalancing, supply
#     chain inventory, and resource allocation
#   - Track every RL training run with kailash-ml's ExperimentTracker
#   - Register trained policy networks in the ModelRegistry
#   - Visualise and compare reward curves (random vs DQN vs PPO)
#   - Explain how PPO connects to RLHF for LLM alignment (bridge to M6)
#
# PREREQUISITES: M5/ex_2 through M5/ex_4 (PyTorch training loops).
# ESTIMATED TIME: ~120-150 min
# DATASETS: No static dataset — the environment IS the data source.
#   - CartPole-v1 (Gymnasium classic control, 4-D state, 2 actions)
#   - 5 custom business environments (defined in this exercise)
#
# TASKS:
#   1. Set up ExperimentTracker, ModelRegistry, and CartPole environment
#   2. Implement DQN from scratch (replay buffer, target network, epsilon)
#   3. Train DQN on CartPole-v1 and log metrics to ExperimentTracker
#   4. Implement PPO from scratch (actor-critic, GAE, clipped objective)
#   5. Train PPO on CartPole-v1 and log metrics to ExperimentTracker
#   6. Evaluate and compare random vs DQN vs PPO policies
#   7. Build 5 business-themed custom Gymnasium environments
#   8. Train DQN on ChurnPrevention env, register model in ModelRegistry
#   9. Visualise all reward curves and generate comparison plots
#
# ════════════════════════════════════════════════════════════════════════
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
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium import spaces

import polars as pl

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry

from shared.kailash_helpers import get_device, setup_environment

setup_environment()

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Set up kailash-ml engines and CartPole environment
# ════════════════════════════════════════════════════════════════════════
# RL is different from supervised learning: there is no dataset. The agent
# collects data by INTERACTING with the environment. Each (state, action,
# reward, next_state) tuple is a training sample, but the distribution of
# samples changes as the policy improves — this is the non-stationarity
# challenge unique to RL.
#
# Bellman expectation equation:
#   V(s) = E[R_{t+1} + gamma * V(S_{t+1}) | S_t = s]
# "The value of a state equals the expected immediate reward plus the
#  discounted value of the next state."
#
# Bellman optimality equation:
#   Q*(s,a) = E[R_{t+1} + gamma * max_{a'} Q*(S_{t+1}, a') | S_t = s, A_t = a]
# "The optimal Q-value of taking action a in state s equals the expected
#  immediate reward plus the discounted maximum Q-value over all actions
#  in the next state."

cartpole_env = gym.make("CartPole-v1")
obs_dim = cartpole_env.observation_space.shape[0]
n_actions = cartpole_env.action_space.n
print(f"CartPole-v1  obs_dim={obs_dim}  n_actions={n_actions}")


async def setup_engines():
    # TODO: Create ConnectionManager("sqlite:///mlfp05_rl.db"), initialize, create tracker
    # Hint: ConnectionManager("sqlite:///mlfp05_rl.db"), await conn.initialize()
    #       ExperimentTracker(conn), await tracker.create_experiment(name=..., description=...)
    conn = ____  # noqa: F821
    await ____  # noqa: F821

    tracker = ____  # noqa: F821
    exp_name = await tracker.create_experiment(
        name=____,  # noqa: F821
        description=____,  # noqa: F821
    )

    try:
        # TODO: Create ModelRegistry(conn)
        registry = ____  # noqa: F821
        has_registry = True
    except Exception as e:
        registry = None
        has_registry = False
        print(f"  Note: ModelRegistry setup skipped ({e})")

    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert obs_dim == 4, "CartPole has 4-dimensional state"
assert n_actions == 2, "CartPole has 2 actions (left, right)"
assert tracker is not None, "ExperimentTracker should be initialised"
assert exp_name is not None, "Experiment should be created"
# INTERPRETATION: CartPole is the "Hello World" of RL. The agent sees 4
# numbers (cart position, cart velocity, pole angle, angular velocity) and
# chooses left or right. Reward is +1 per timestep the pole stays upright.
# The Bellman equation says V(s) = 1 + gamma*V(s') — the value of a state
# is "1 for surviving this step plus the discounted future."
print("--- Checkpoint 1 passed --- environment and engines ready\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — DQN from scratch
# ════════════════════════════════════════════════════════════════════════
# DQN (Mnih et al. 2015) approximates Q*(s,a) with a neural network.
# Two key innovations prevent training instability:
#   (1) Experience replay: store transitions in a buffer and sample random
#       minibatches — breaks temporal correlation between consecutive samples
#   (2) Target network: a slow-moving copy of Q used to compute targets —
#       prevents the "moving target" problem where Q changes on both sides
#       of the Bellman equation simultaneously.


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


class DQN(nn.Module):
    """Deep Q-Network: maps state -> Q-value for each action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        # TODO: Build a 3-layer MLP: Linear(obs_dim, hidden) -> ReLU ->
        #       Linear(hidden, hidden) -> ReLU -> Linear(hidden, n_actions)
        # Hint: self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), ...)
        self.net = ____  # noqa: F821

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


async def train_dqn_async(
    env: gym.Env,
    obs_d: int,
    n_act: int,
    n_episodes: int = 200,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 10,
    min_replay_size: int = 500,
    run_name: str = "dqn_cartpole",
) -> tuple[DQN, list[float], list[float]]:
    """Train DQN and log to ExperimentTracker via the async context manager."""
    q_net = DQN(obs_d, n_act).to(device)
    target_net = DQN(obs_d, n_act).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=10_000)
    epsilon = epsilon_start

    episode_rewards: list[float] = []
    episode_losses: list[float] = []

    # TODO: Open an ExperimentTracker run context and log hyperparameters
    # Hint: async with tracker.run(experiment_name=exp_name, run_name=run_name) as ctx:
    #           await ctx.log_params({"algorithm": "DQN", "gamma": ..., ...})
    async with tracker.run(experiment_name=exp_name, run_name=run_name) as ctx:
        await ctx.log_params(
            {
                "algorithm": ____,  # noqa: F821
                "gamma": ____,  # noqa: F821
                "lr": ____,  # noqa: F821
                "epsilon_start": ____,  # noqa: F821
                "epsilon_end": ____,  # noqa: F821
                "target_update_freq": ____,  # noqa: F821
                "batch_size": ____,  # noqa: F821
            }
        )

        print(f"\n== Training DQN: {run_name} ==")
        for ep in range(n_episodes):
            state, _ = env.reset(seed=42 + ep)
            total_reward = 0.0
            ep_loss_sum = 0.0
            ep_loss_count = 0
            done = False

            while not done:
                # TODO: Epsilon-greedy action selection
                # Hint: if random.random() < epsilon: action = env.action_space.sample()
                #       else: action = int(q_net(torch.tensor(state, ...)).argmax().item())
                if random.random() < epsilon:
                    action = ____  # noqa: F821
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(state, dtype=torch.float32, device=device)
                        action = ____  # noqa: F821

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                replay.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                # Train on a minibatch from replay buffer
                if len(replay) >= min_replay_size:
                    s_b, a_b, r_b, ns_b, d_b = replay.sample(batch_size)

                    # TODO: Compute current Q-values and Bellman targets
                    # Hint: q_values = q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    #       next_q = target_net(ns_b).max(dim=1).values
                    #       targets = r_b + gamma * next_q * (1.0 - d_b)
                    q_values = ____  # noqa: F821

                    with torch.no_grad():
                        next_q = ____  # noqa: F821
                        targets = ____  # noqa: F821

                    loss = F.mse_loss(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_loss_sum += loss.item()
                    ep_loss_count += 1

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Update target network periodically
            if (ep + 1) % target_update_freq == 0:
                target_net.load_state_dict(q_net.state_dict())

            episode_rewards.append(total_reward)
            avg_loss = ep_loss_sum / max(ep_loss_count, 1)
            episode_losses.append(avg_loss)

            # TODO: Log per-episode metrics using ctx.log_metrics
            # Hint: metrics = {"episode_reward": total_reward, "epsilon": epsilon}
            #       if ep_loss_count > 0: metrics["loss"] = avg_loss
            #       await ctx.log_metrics(metrics, step=ep)
            metrics = {____: total_reward, ____: epsilon}  # noqa: F821
            if ep_loss_count > 0:
                metrics[____] = avg_loss  # noqa: F821
            await ctx.log_metrics(metrics, step=ep)

            if (ep + 1) % 40 == 0:
                avg_20 = float(np.mean(episode_rewards[-20:]))
                print(
                    f"  ep {ep+1:3d}  reward={total_reward:6.1f}  "
                    f"avg20={avg_20:6.1f}  eps={epsilon:.3f}  loss={avg_loss:.4f}"
                )

        await ctx.log_metric(____, float(np.mean(episode_rewards[-20:])))  # noqa: F821

    return q_net, episode_rewards, episode_losses


def train_dqn(
    env: gym.Env,
    obs_d: int,
    n_act: int,
    n_episodes: int = 200,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 10,
    min_replay_size: int = 500,
    run_name: str = "dqn_cartpole",
) -> tuple[DQN, list[float], list[float]]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(
        train_dqn_async(
            env,
            obs_d,
            n_act,
            n_episodes,
            gamma,
            lr,
            batch_size,
            epsilon_start,
            epsilon_end,
            epsilon_decay,
            target_update_freq,
            min_replay_size,
            run_name,
        )
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train DQN on CartPole-v1
# ════════════════════════════════════════════════════════════════════════
dqn_model, dqn_rewards, dqn_losses = train_dqn(cartpole_env, obs_dim, n_actions)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(dqn_rewards) == 200, "DQN should train for 200 episodes"
assert (
    float(np.mean(dqn_rewards[-20:])) > 50.0
), "DQN should achieve avg reward > 50 in last 20 episodes (random baseline ~20)"
# INTERPRETATION: DQN learns Q(s,a) — the expected total reward from
# taking action a in state s and then acting optimally. The replay buffer
# decorrelates samples; the target network stabilises training. If the
# avg reward is climbing, the Q-network is learning to predict which
# action leads to more pole-balancing time.
print("--- Checkpoint 2 passed --- DQN trained on CartPole\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — PPO from scratch (actor-critic + GAE + clipped objective)
# ════════════════════════════════════════════════════════════════════════
# PPO (Schulman 2017) is a policy-gradient method with two key ideas:
#   (1) Actor-critic: policy (actor) + value function (critic) share a
#       trunk. The critic's V(s) provides a baseline for variance reduction.
#   (2) Clipped surrogate objective: prevents catastrophically large
#       policy updates by clipping the probability ratio:
#
#       r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
#       L_clip = E[ min( r_t * A_t,  clip(r_t, 1-eps, 1+eps) * A_t ) ]
#
# GAE (Generalised Advantage Estimation) computes low-variance advantage:
#   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
#   A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}


class ActorCritic(nn.Module):
    """Shared trunk with two heads: policy logits and state value."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        # TODO: Build shared trunk: Linear(obs_dim, hidden) -> Tanh -> Linear(hidden, hidden) -> Tanh
        # Hint: self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), ...)
        self.trunk = ____  # noqa: F821
        # TODO: Build policy_head (hidden -> n_actions) and value_head (hidden -> 1)
        # Hint: self.policy_head = nn.Linear(hidden, n_actions)
        #       self.value_head = nn.Linear(hidden, 1)
        self.policy_head = ____  # noqa: F821
        self.value_head = ____  # noqa: F821

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        s = torch.from_numpy(state.astype(np.float32)).to(device)
        logits, value = self.forward(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a).detach(), value.detach()


def collect_trajectory(env: gym.Env, model: ActorCritic, max_steps: int):
    """Collect a rollout of max_steps for PPO training."""
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    state, _ = env.reset(seed=int(np.random.randint(0, 100_000)))
    for _ in range(max_steps):
        action, log_prob, value = model.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        states.append(state.astype(np.float32))
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(float(reward))
        done = terminated or truncated
        dones.append(done)
        state = next_state
        if done:
            state, _ = env.reset(seed=int(np.random.randint(0, 100_000)))
    return states, actions, log_probs, values, rewards, dones


def compute_gae(rewards, values, dones, gamma: float = 0.99, lam: float = 0.95):
    """Generalised Advantage Estimation: low-variance advantage targets."""
    # TODO: Implement GAE backward pass
    # Hint: advantages = [0.0] * len(rewards); gae = 0.0; next_value = 0.0
    #       for t in reversed(range(len(rewards))):
    #           nonterminal = 1.0 - float(dones[t])
    #           delta = rewards[t] + gamma * next_value * nonterminal - float(values[t])
    #           gae = delta + gamma * lam * nonterminal * gae
    #           advantages[t] = gae; next_value = float(values[t])
    #       returns = [a + float(v) for a, v in zip(advantages, values)]
    advantages = ____  # noqa: F821
    gae = ____  # noqa: F821
    next_value = ____  # noqa: F821
    for t in reversed(range(len(rewards))):
        nonterminal = ____  # noqa: F821
        delta = ____  # noqa: F821
        gae = ____  # noqa: F821
        advantages[t] = gae
        next_value = float(values[t])
    returns = ____  # noqa: F821
    return advantages, returns


async def train_ppo_async(
    env: gym.Env,
    obs_d: int,
    n_act: int,
    n_iters: int = 30,
    steps_per_iter: int = 1024,
    epochs: int = 4,
    minibatch: int = 256,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    run_name: str = "ppo_cartpole",
) -> tuple[ActorCritic, list[float]]:
    """Train PPO; log to ExperimentTracker via the async context manager."""
    model = ActorCritic(obs_d, n_act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    iter_returns: list[float] = []

    # TODO: Open an ExperimentTracker run context and log hyperparameters
    # Hint: async with tracker.run(experiment_name=exp_name, run_name=run_name) as ctx:
    #           await ctx.log_params({"algorithm": "PPO", "gamma": ..., ...})
    async with tracker.run(experiment_name=exp_name, run_name=run_name) as ctx:
        await ctx.log_params(
            {
                "algorithm": ____,  # noqa: F821
                "gamma": ____,  # noqa: F821
                "lambda": ____,  # noqa: F821
                "lr": ____,  # noqa: F821
                "clip_eps": ____,  # noqa: F821
                "steps_per_iter": ____,  # noqa: F821
                "epochs": ____,  # noqa: F821
            }
        )

        print(f"\n== Training PPO ({run_name}) ==")
        for it in range(n_iters):
            (
                states,
                actions,
                old_log_probs,
                values,
                rewards,
                dones,
            ) = collect_trajectory(env, model, steps_per_iter)
            advantages, returns = compute_gae(rewards, values, dones, gamma, lam)

            s_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
            a_t = torch.tensor(actions, dtype=torch.long, device=device)
            old_lp_t = torch.stack(old_log_probs).to(device)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            n = s_t.size(0)
            idxs = np.arange(n)
            for _ in range(epochs):
                np.random.shuffle(idxs)
                for start in range(0, n, minibatch):
                    mb = idxs[start : start + minibatch]
                    logits, vpred = model(s_t[mb])
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(a_t[mb])

                    # TODO: Compute clipped surrogate loss
                    # Hint: ratio = torch.exp(new_lp - old_lp_t[mb])
                    #       surr1 = ratio * adv_t[mb]
                    #       surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t[mb]
                    #       policy_loss = -torch.min(surr1, surr2).mean()
                    ratio = ____  # noqa: F821
                    surr1 = ____  # noqa: F821
                    surr2 = ____  # noqa: F821
                    policy_loss = ____  # noqa: F821
                    value_loss = F.mse_loss(vpred, ret_t[mb])
                    entropy = dist.entropy().mean()
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()

            # Average episode return for this iteration
            ep_rs: list[float] = []
            running = 0.0
            for r, d in zip(rewards, dones):
                running += r
                if d:
                    ep_rs.append(running)
                    running = 0.0
            avg_ep_return = float(np.mean(ep_rs)) if ep_rs else float(running)
            iter_returns.append(avg_ep_return)

            await ctx.log_metric(____, avg_ep_return, step=it)  # noqa: F821

            if (it + 1) % 10 == 0:
                print(f"  iter {it+1:2d}  avg episode return = {avg_ep_return:7.1f}")

        await ctx.log_metric(____, iter_returns[-1])  # noqa: F821

    return model, iter_returns


def train_ppo(
    env: gym.Env,
    obs_d: int,
    n_act: int,
    n_iters: int = 30,
    steps_per_iter: int = 1024,
    epochs: int = 4,
    minibatch: int = 256,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    run_name: str = "ppo_cartpole",
) -> tuple[ActorCritic, list[float]]:
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(
        train_ppo_async(
            env,
            obs_d,
            n_act,
            n_iters,
            steps_per_iter,
            epochs,
            minibatch,
            clip_eps,
            lr,
            gamma,
            lam,
            run_name,
        )
    )


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Train PPO on CartPole-v1
# ════════════════════════════════════════════════════════════════════════
ppo_model, ppo_returns = train_ppo(cartpole_env, obs_dim, n_actions)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(ppo_returns) == 30, "PPO should train for 30 iterations"
assert ppo_returns[-1] > 50.0, "PPO should achieve avg return > 50 by final iteration"
# INTERPRETATION: PPO learns a POLICY directly (probability of each action
# given a state), unlike DQN which learns Q-values and derives a policy.
# The clipped objective prevents the new policy from straying too far from
# the old one — this is the "proximal" in Proximal Policy Optimization.
# In M6, RLHF uses PPO to update an LLM's policy (word probabilities)
# using human preference as the reward signal.
print("--- Checkpoint 3 passed --- PPO trained on CartPole\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Evaluate and compare: random vs DQN vs PPO
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
            total += reward
            done = terminated or truncated
        eval_returns.append(total)
    return eval_returns


def random_policy(state):
    return cartpole_env.action_space.sample()


def dqn_policy(state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device)
        return int(dqn_model(s).argmax().item())


def ppo_policy(state):
    with torch.no_grad():
        s = torch.from_numpy(state.astype(np.float32)).to(device)
        logits, _ = ppo_model(s)
        return int(logits.argmax().item())


random_returns = evaluate_policy(cartpole_env, random_policy)
dqn_eval_returns = evaluate_policy(cartpole_env, dqn_policy)
ppo_eval_returns = evaluate_policy(cartpole_env, ppo_policy)

print("\n== Policy Comparison (CartPole-v1, 30 eval episodes) ==")
print(
    f"  Random : mean={np.mean(random_returns):6.1f} +/- {np.std(random_returns):5.1f}"
)
print(
    f"  DQN    : mean={np.mean(dqn_eval_returns):6.1f} +/- {np.std(dqn_eval_returns):5.1f}"
)
print(
    f"  PPO    : mean={np.mean(ppo_eval_returns):6.1f} +/- {np.std(ppo_eval_returns):5.1f}"
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert float(np.mean(dqn_eval_returns)) > float(
    np.mean(random_returns)
), "DQN should outperform random policy"
assert float(np.mean(ppo_eval_returns)) > float(
    np.mean(random_returns)
), "PPO should outperform random policy"
# INTERPRETATION: Both DQN and PPO should substantially beat the random
# baseline (~20 reward). DQN uses value-based learning (estimate Q-values,
# then act greedily). PPO uses policy-based learning (directly optimise
# the policy). Both converge for discrete, low-dimensional problems; PPO
# tends to be more stable and scales better to complex environments.
print("--- Checkpoint 4 passed --- both algorithms outperform random\n")

cartpole_env.close()


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Custom Gymnasium environments for business applications
# ════════════════════════════════════════════════════════════════════════
# Real RL applications require CUSTOM environments. Each environment must
# define: observation_space, action_space, reset(), step(action).
# We build 5 business-themed environments that model real decision problems.


# ── Environment 1: Customer Churn Prevention ─────────────────────────
class ChurnPreventionEnv(gym.Env):
    """Prevent customer churn through targeted interventions.

    State (4,): [satisfaction_score, usage_frequency, months_active, support_tickets]
    Actions (4): 0=nothing, 1=discount, 2=support_call, 3=feature_upgrade
    Reward: +10 for retaining an at-risk customer, -5 for losing one,
            -1 per intervention (cost), +1 per step customer stays.
    Episode: 30 steps (one month of daily decisions).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.max_steps = 30
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(0.2, 0.8, size=(4,)).astype(np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        satisfaction, usage, tenure, tickets = self.state

        intervention_cost = 0.0
        if action == 1:  # discount
            satisfaction = min(1.0, satisfaction + 0.1)
            usage = min(1.0, usage + 0.05)
            intervention_cost = 1.0
        elif action == 2:  # support call
            tickets = max(0.0, tickets - 0.15)
            satisfaction = min(1.0, satisfaction + 0.05)
            intervention_cost = 0.5
        elif action == 3:  # feature upgrade
            usage = min(1.0, usage + 0.1)
            intervention_cost = 1.5

        satisfaction = max(0.0, satisfaction - 0.02 + self.np_random.normal(0, 0.02))
        usage = max(0.0, min(1.0, usage - 0.01 + self.np_random.normal(0, 0.02)))
        tickets = max(0.0, min(1.0, tickets + 0.02 + self.np_random.normal(0, 0.01)))
        tenure = min(1.0, tenure + 1.0 / self.max_steps)

        self.state = np.array([satisfaction, usage, tenure, tickets], dtype=np.float32)

        churn_prob = max(0.0, 0.3 - satisfaction * 0.4 + tickets * 0.3)
        churned = self.np_random.random() < churn_prob

        if churned:
            reward = -5.0
            terminated = True
        else:
            reward = 1.0 - intervention_cost
            terminated = False

        truncated = self.step_count >= self.max_steps
        if truncated and not terminated:
            reward += 10.0

        return self.state.copy(), reward, terminated, truncated, {}


# ── Environment 2: Dynamic Pricing ───────────────────────────────────
class DynamicPricingEnv(gym.Env):
    """Set prices to maximise revenue under demand uncertainty.

    State (5,): [current_price, inventory, demand_signal, competitor_price, time_remaining]
    Actions (5): 0=big_decrease, 1=small_decrease, 2=hold, 3=small_increase, 4=big_increase
    Reward: units_sold * price - holding_cost
    Episode: 20 steps (pricing decisions over a selling season).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.max_steps = 20
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(
            [
                0.5,
                1.0,
                self.np_random.uniform(0.3, 0.7),
                self.np_random.uniform(0.3, 0.7),
                1.0,
            ],
            dtype=np.float32,
        )
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        price, inventory, demand, comp_price, time_left = self.state

        adjustments = [-0.15, -0.05, 0.0, 0.05, 0.15]
        price = np.clip(price + adjustments[action], 0.1, 1.0)

        base_demand = 0.8 - 0.6 * price + 0.3 * (comp_price - price) + demand * 0.2
        units_sold = max(
            0.0, min(inventory, base_demand + self.np_random.normal(0, 0.05))
        )

        revenue = units_sold * price
        holding_cost = 0.02 * inventory
        reward = revenue - holding_cost

        inventory = max(0.0, inventory - units_sold)
        demand = np.clip(demand + self.np_random.normal(0, 0.05), 0.0, 1.0)
        comp_price = np.clip(comp_price + self.np_random.normal(0, 0.03), 0.1, 1.0)
        time_left = max(0.0, time_left - 1.0 / self.max_steps)

        self.state = np.array(
            [price, inventory, demand, comp_price, time_left], dtype=np.float32
        )

        terminated = inventory <= 0.01
        truncated = self.step_count >= self.max_steps

        return self.state.copy(), reward, terminated, truncated, {}


# ── Environment 3: Portfolio Rebalancing ─────────────────────────────
class PortfolioRebalancingEnv(gym.Env):
    """Rebalance a 3-asset portfolio to maximise risk-adjusted returns.

    State (6,): [weight_stocks, weight_bonds, weight_cash,
                 market_volatility, interest_rate, momentum]
    Actions (27): all combinations of per-asset decrease/hold/increase (3^3).
    Reward: portfolio_return - 0.5 * volatility_penalty - transaction_cost
    Episode: 24 steps (monthly rebalancing over 2 years).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(27)
        self.max_steps = 24
        self.state = None
        self.step_count = 0

    def _decode_action(self, action: int) -> list[int]:
        return [(action // (3**i)) % 3 for i in range(3)]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        weights = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        market = self.np_random.uniform(0.2, 0.5)
        rate = self.np_random.uniform(0.3, 0.6)
        momentum = 0.5
        self.state = np.concatenate([weights, [market, rate, momentum]]).astype(
            np.float32
        )
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        weights = self.state[:3].copy()
        market_vol, interest, momentum = self.state[3], self.state[4], self.state[5]

        decisions = self._decode_action(action)
        shifts = np.array([[-0.05, 0.0, 0.05][d] for d in decisions], dtype=np.float32)
        transaction_cost = 0.005 * np.sum(np.abs(shifts))
        weights = np.clip(weights + shifts, 0.0, 1.0)
        weights = weights / (weights.sum() + 1e-8)

        stock_return = self.np_random.normal(0.01 + momentum * 0.02, market_vol * 0.1)
        bond_return = self.np_random.normal(interest * 0.005, 0.02)
        cash_return = 0.001
        asset_returns = np.array([stock_return, bond_return, cash_return])
        portfolio_return = float(np.dot(weights, asset_returns))

        vol_penalty = market_vol * 0.05 * weights[0]
        reward = portfolio_return - vol_penalty - transaction_cost

        market_vol = np.clip(market_vol + self.np_random.normal(0, 0.03), 0.1, 0.9)
        interest = np.clip(interest + self.np_random.normal(0, 0.02), 0.1, 0.9)
        momentum = np.clip(momentum + self.np_random.normal(0, 0.1), 0.0, 1.0)

        self.state = np.concatenate([weights, [market_vol, interest, momentum]]).astype(
            np.float32
        )
        truncated = self.step_count >= self.max_steps
        return self.state.copy(), reward, False, truncated, {}


# ── Environment 4: Supply Chain Inventory ────────────────────────────
class SupplyChainInventoryEnv(gym.Env):
    """Manage warehouse inventory to minimise stockouts and holding costs.

    State (4,): [inventory_level, demand_forecast, lead_time_remaining, season_phase]
    Actions (4): 0=order_nothing, 1=order_small, 2=order_medium, 3=order_large
    Reward: -stockout_penalty - holding_cost + fulfilled_demand_bonus
    Episode: 52 steps (weekly decisions for one year).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.max_steps = 52
        self.state = None
        self.step_count = 0
        self.pending_orders: list[tuple[int, float]] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([0.5, 0.4, 0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self.pending_orders = []
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        inventory, forecast, lead_time, season = self.state

        order_sizes = [0.0, 0.1, 0.25, 0.5]
        order_cost = [0.0, 0.02, 0.04, 0.07]
        if action > 0:
            lead = self.np_random.integers(2, 5)
            self.pending_orders.append((self.step_count + lead, order_sizes[action]))

        arrived = [qty for (t, qty) in self.pending_orders if t <= self.step_count]
        self.pending_orders = [
            (t, qty) for (t, qty) in self.pending_orders if t > self.step_count
        ]
        inventory = min(1.0, inventory + sum(arrived))

        season = (self.step_count / self.max_steps) % 1.0
        seasonal_factor = 0.5 + 0.3 * np.sin(2 * np.pi * season)
        demand = max(0.0, seasonal_factor * 0.3 + self.np_random.normal(0, 0.05))

        fulfilled = min(inventory, demand)
        stockout = max(0.0, demand - inventory)
        inventory = max(0.0, inventory - demand)

        stockout_penalty = 5.0 * stockout
        holding_cost_val = 0.5 * inventory
        fulfillment_bonus = 2.0 * fulfilled
        reward = (
            fulfillment_bonus - stockout_penalty - holding_cost_val - order_cost[action]
        )

        forecast = np.clip(
            seasonal_factor * 0.3 + self.np_random.normal(0, 0.1), 0.0, 1.0
        )
        lead_remaining = (
            min([t - self.step_count for t, _ in self.pending_orders]) / 5.0
            if self.pending_orders
            else 0.0
        )

        self.state = np.array(
            [inventory, forecast, np.clip(lead_remaining, 0, 1), season],
            dtype=np.float32,
        )
        truncated = self.step_count >= self.max_steps
        return self.state.copy(), reward, False, truncated, {}


# ── Environment 5: Resource Allocation ───────────────────────────────
class ResourceAllocationEnv(gym.Env):
    """Allocate limited compute resources across 3 services to meet SLAs.

    State (6,): [load_svc1, load_svc2, load_svc3,
                 alloc_svc1, alloc_svc2, alloc_svc3]
    Actions (7): 0-5 = shift one unit from service X to service Y, 6 = do_nothing
    Reward: +1 per service meeting SLA, -2 per SLA violation, -0.1 reallocation cost
    Episode: 48 steps (hourly decisions over 2 days).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(7)
        self.max_steps = 48
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        loads = self.np_random.uniform(0.2, 0.6, size=3).astype(np.float32)
        allocs = np.array([0.33, 0.33, 0.34], dtype=np.float32)
        self.state = np.concatenate([loads, allocs]).astype(np.float32)
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        loads = self.state[:3].copy()
        allocs = self.state[3:].copy()

        shift_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        realloc_cost = 0.0
        if action < 6:
            src, dst = shift_pairs[action]
            amount = min(0.05, allocs[src])
            allocs[src] -= amount
            allocs[dst] += amount
            realloc_cost = 0.1

        hour = (self.step_count % 24) / 24.0
        diurnal = 0.3 + 0.5 * np.sin(np.pi * hour)
        for i in range(3):
            loads[i] = np.clip(
                diurnal * (0.3 + 0.2 * i) + self.np_random.normal(0, 0.05),
                0.0,
                1.0,
            )

        sla_met = 0
        sla_violations = 0
        for i in range(3):
            if allocs[i] >= loads[i] * 0.8:
                sla_met += 1
            else:
                sla_violations += 1

        reward = float(sla_met) - 2.0 * sla_violations - realloc_cost

        self.state = np.concatenate([loads, allocs]).astype(np.float32)
        truncated = self.step_count >= self.max_steps
        return self.state.copy(), reward, False, truncated, {}


# ── Checkpoint 5 ─────────────────────────────────────────────────────
# Verify all custom environments conform to the Gymnasium API
env_classes = [
    ("ChurnPrevention", ChurnPreventionEnv),
    ("DynamicPricing", DynamicPricingEnv),
    ("PortfolioRebalancing", PortfolioRebalancingEnv),
    ("SupplyChainInventory", SupplyChainInventoryEnv),
    ("ResourceAllocation", ResourceAllocationEnv),
]
for env_name, env_cls in env_classes:
    test_env = env_cls()
    obs, info = test_env.reset(seed=42)
    assert (
        obs.shape == test_env.observation_space.shape
    ), f"{env_name}: obs shape mismatch"
    obs2, reward, terminated, truncated, info = test_env.step(
        test_env.action_space.sample()
    )
    assert (
        obs2.shape == test_env.observation_space.shape
    ), f"{env_name}: step obs shape mismatch"
    assert isinstance(reward, float), f"{env_name}: reward should be float"
    print(
        f"  {env_name}: obs={obs.shape}, actions={test_env.action_space.n}, reward={reward:.3f}"
    )
    test_env.close()
# INTERPRETATION: Each environment models a real business decision:
# - ChurnPrevention: when to intervene (cost of action vs cost of losing customer)
# - DynamicPricing: price to maximise revenue (demand elasticity, competition)
# - PortfolioRebalancing: asset allocation (risk-return tradeoff, transaction costs)
# - SupplyChainInventory: order timing (lead times, seasonal demand, holding costs)
# - ResourceAllocation: capacity planning (SLA constraints, shifting workloads)
# All follow the Gymnasium API: reset() -> (obs, info), step(a) -> (obs, r, term, trunc, info)
print("--- Checkpoint 5 passed --- all 5 custom environments validated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Train DQN on ChurnPrevention, register in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
# Demonstrate that the DQN implementation generalises to custom environments.

churn_env = ChurnPreventionEnv()
churn_obs_dim = churn_env.observation_space.shape[0]
churn_n_actions = churn_env.action_space.n

churn_dqn = DQN(churn_obs_dim, churn_n_actions).to(device)
churn_target = DQN(churn_obs_dim, churn_n_actions).to(device)
churn_target.load_state_dict(churn_dqn.state_dict())
churn_target.eval()

churn_opt = torch.optim.Adam(churn_dqn.parameters(), lr=1e-3)
churn_replay = ReplayBuffer(capacity=10_000)
churn_rewards_hist: list[float] = []


async def _train_churn_dqn_async():
    """Train DQN on ChurnPrevention under a tracker.run(...) context."""
    churn_epsilon = 1.0

    async with tracker.run(
        experiment_name=exp_name, run_name="dqn_churn_prevention"
    ) as ctx:
        await ctx.log_params(
            {
                "algorithm": "DQN",
                "environment": "ChurnPrevention",
                "gamma": "0.99",
                "episodes": "150",
            }
        )

        print("== Training DQN on ChurnPrevention ==")
        for ep in range(150):
            state, _ = churn_env.reset(seed=ep)
            total_reward = 0.0
            done = False

            while not done:
                if random.random() < churn_epsilon:
                    action = churn_env.action_space.sample()
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(state, dtype=torch.float32, device=device)
                        action = int(churn_dqn(s_t).argmax().item())

                next_state, reward, terminated, truncated, _ = churn_env.step(action)
                done = terminated or truncated
                churn_replay.push(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

                if len(churn_replay) >= 300:
                    s_b, a_b, r_b, ns_b, d_b = churn_replay.sample(64)
                    q_vals = churn_dqn(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q = churn_target(ns_b).max(dim=1).values
                        targets = r_b + 0.99 * next_q * (1.0 - d_b)
                    loss = F.mse_loss(q_vals, targets)
                    churn_opt.zero_grad()
                    loss.backward()
                    churn_opt.step()

            churn_epsilon = max(0.01, churn_epsilon * 0.99)
            if (ep + 1) % 10 == 0:
                churn_target.load_state_dict(churn_dqn.state_dict())

            churn_rewards_hist.append(total_reward)
            await ctx.log_metric("episode_reward", total_reward, step=ep)

            if (ep + 1) % 30 == 0:
                avg_30 = float(np.mean(churn_rewards_hist[-30:]))
                print(
                    f"  ep {ep+1:3d}  reward={total_reward:6.1f}  "
                    f"avg30={avg_30:6.1f}  eps={churn_epsilon:.3f}"
                )

        await ctx.log_metric(
            "final_avg_reward", float(np.mean(churn_rewards_hist[-30:]))
        )


asyncio.run(_train_churn_dqn_async())
churn_env.close()


async def register_rl_models():
    """Register DQN and PPO policy networks in the ModelRegistry."""
    if not has_registry:
        print("  ModelRegistry not available - skipping registration")
        return {}

    from kailash_ml.types import MetricSpec

    model_versions = {}

    # TODO: Serialize dqn_model.state_dict() and register as "m5_dqn_cartpole"
    # Hint: dqn_bytes = pickle.dumps(dqn_model.state_dict())
    #       await registry.register_model(name=..., artifact=..., metrics=[MetricSpec(...)])
    dqn_bytes = ____  # noqa: F821
    dqn_version = await registry.register_model(
        name="m5_dqn_cartpole",
        artifact=____,  # noqa: F821
        metrics=[
            MetricSpec(
                name="avg_reward_last20", value=float(np.mean(dqn_rewards[-20:]))
            ),
            MetricSpec(name="algorithm", value=0.0),
            MetricSpec(name="episodes_trained", value=float(len(dqn_rewards))),
        ],
    )
    model_versions["dqn_cartpole"] = dqn_version
    print(f"  Registered DQN CartPole: version={dqn_version.version}")

    # TODO: Serialize ppo_model.state_dict() and register as "m5_ppo_cartpole"
    ppo_bytes = ____  # noqa: F821
    ppo_version = await registry.register_model(
        name="m5_ppo_cartpole",
        artifact=____,  # noqa: F821
        metrics=[
            MetricSpec(name="avg_return_final", value=float(ppo_returns[-1])),
            MetricSpec(name="algorithm", value=1.0),
            MetricSpec(name="iterations_trained", value=float(len(ppo_returns))),
        ],
    )
    model_versions["ppo_cartpole"] = ppo_version
    print(f"  Registered PPO CartPole: version={ppo_version.version}")

    churn_bytes = pickle.dumps(churn_dqn.state_dict())
    churn_version = await registry.register_model(
        name="m5_dqn_churn_prevention",
        artifact=churn_bytes,
        metrics=[
            MetricSpec(
                name="avg_reward_last30", value=float(np.mean(churn_rewards_hist[-30:]))
            ),
            MetricSpec(name="episodes_trained", value=150.0),
        ],
    )
    model_versions["dqn_churn"] = churn_version
    print(f"  Registered DQN ChurnPrevention: version={churn_version.version}")

    return model_versions


print("\n== Registering RL Models ==")
rl_model_versions = asyncio.run(register_rl_models())

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert len(churn_rewards_hist) == 150, "Churn DQN should train for 150 episodes"
if has_registry:
    assert len(rl_model_versions) == 3, "Should register 3 RL models"
# INTERPRETATION: The ModelRegistry stores policy networks as versioned
# artifacts alongside their training metrics. In production, you would
# promote the best-performing policy to "production" stage and use it
# for real-time decision making (which customers get discounts, how to
# price products, when to reorder inventory).
print("--- Checkpoint 6 passed --- models registered in ModelRegistry\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 9 — Visualise reward curves and generate comparison plots
# ════════════════════════════════════════════════════════════════════════


def moving_average(xs: list[float], window: int = 10) -> list[float]:
    """Smooth a time series with a rolling mean."""
    if len(xs) < window:
        return xs
    arr = np.asarray(xs, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return list(np.convolve(arr, kernel, mode="valid"))


viz = ModelVisualizer()

# TODO: Plot DQN CartPole training curve with moving average
# Hint: viz.training_history(metrics={"DQN episode reward": dqn_rewards,
#         "DQN moving avg (20)": moving_average(dqn_rewards, 20)},
#         x_label="Episode", y_label="Reward")
fig1 = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig1.write_html("ex_8_dqn_cartpole_training.html")

# TODO: Plot PPO CartPole training curve
# Hint: metrics={"PPO avg episode return": ppo_returns}
fig2 = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig2.write_html("ex_8_ppo_cartpole_training.html")

# Policy comparison box plot
comparison_data = pl.DataFrame(
    {
        "Policy": ["Random"] * len(random_returns)
        + ["DQN"] * len(dqn_eval_returns)
        + ["PPO"] * len(ppo_eval_returns),
        "Return": random_returns + dqn_eval_returns + ppo_eval_returns,
    }
)
fig3 = viz.box_plot(comparison_data, "Return", group_by="Policy")
fig3.write_html("ex_8_policy_comparison.html")

# TODO: Plot DQN on ChurnPrevention with moving average
fig4 = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig4.write_html("ex_8_churn_dqn_training.html")

# TODO: Plot DQN loss curve (non-zero losses only)
fig5 = viz.training_history(
    metrics=____,  # noqa: F821
    x_label=____,  # noqa: F821
    y_label=____,  # noqa: F821
)
fig5.write_html("ex_8_dqn_loss_curve.html")

print("\nPlots saved:")
print("  ex_8_dqn_cartpole_training.html")
print("  ex_8_ppo_cartpole_training.html")
print("  ex_8_policy_comparison.html      (random vs DQN vs PPO)")
print("  ex_8_churn_dqn_training.html")
print("  ex_8_dqn_loss_curve.html")

# ── Checkpoint 7 ─────────────────────────────────────────────────────
assert Path("ex_8_dqn_cartpole_training.html").exists(), "DQN plot should be saved"
assert Path("ex_8_policy_comparison.html").exists(), "Comparison plot should be saved"
# INTERPRETATION: The reward curves tell the full training story:
# - DQN: noisy early (random exploration), smoothing upward as Q-values improve
# - PPO: steadier climb because GAE reduces variance and clipping prevents crashes
# - ChurnPrevention: business reward is harder to optimise — multiple competing costs
# The box plot comparison shows the final policy quality on held-out evaluation episodes.
print("--- Checkpoint 7 passed --- all visualisations generated\n")


# Clean up database connection
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Derived the Bellman expectation and optimality equations
  [x] Built DQN from scratch: experience replay, target network, epsilon-greedy
  [x] Built PPO from scratch: actor-critic, GAE advantages, clipped surrogate
  [x] Trained both algorithms on CartPole-v1 and verified they beat random
  [x] Created 5 custom Gymnasium environments for business applications:
      - ChurnPrevention: targeted interventions to retain customers
      - DynamicPricing: revenue maximisation under demand uncertainty
      - PortfolioRebalancing: risk-adjusted asset allocation
      - SupplyChainInventory: order timing with seasonal demand patterns
      - ResourceAllocation: SLA-constrained capacity planning
  [x] Tracked all RL training runs with ExperimentTracker (rewards, epsilon, loss)
  [x] Registered trained policy networks in ModelRegistry with metrics
  [x] Visualised reward curves and compared policies with ModelVisualizer

  KEY TAKEAWAYS:
  - DQN (value-based): learns Q(s,a) — "how good is this action in this state?"
    Best for discrete action spaces with well-defined rewards.
  - PPO (policy-based): learns pi(a|s) directly — "what should I do here?"
    More stable, handles larger action spaces, and scales to complex problems.

  BRIDGE TO M6 (RLHF):
  RLHF uses PPO to fine-tune language models. The "environment" is generating
  text, the "action" is choosing the next token, and the "reward" comes from a
  model trained on human preferences. DPO (Direct Preference Optimization)
  achieves the same alignment without needing the separate reward model — it
  optimises the policy directly from preference pairs.
"""
)
