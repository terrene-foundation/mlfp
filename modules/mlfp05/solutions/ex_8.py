# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 8: Reinforcement Learning (REINFORCE on CartPole)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain the RL framework: agent, environment, state, action, reward
#   - Build a policy network in torch.nn and sample actions with Categorical
#   - Implement the REINFORCE (policy gradient) update rule by hand
#   - Train an agent on Gymnasium's CartPole-v1 from interaction alone
#   - Compare learned policy vs a random baseline
#   - Explain how REINFORCE relates to PPO (same objective family, PPO adds
#     a clipped surrogate objective and advantage estimation)
#
# PREREQUISITES: M5/ex_2 through M5/ex_4 (PyTorch training loops).
# ESTIMATED TIME: ~60 min
# DATASET: No dataset — the environment IS the data source. CartPole-v1
#   provides (state, reward) tuples as the agent acts.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym

from kailash_ml import ModelVisualizer

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# PART 1 — Environment: CartPole-v1
# ════════════════════════════════════════════════════════════════════════
# State  (4,)   : cart position, cart velocity, pole angle, pole angular velocity
# Action discrete(2) : 0 = push left, 1 = push right
# Reward 1.0 per timestep the pole is upright
# Episode ends when pole tips beyond 15 degrees or cart leaves the track.
env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n
print(f"CartPole-v1  obs_dim={obs_dim}  n_actions={n_actions}")


# ════════════════════════════════════════════════════════════════════════
# PART 2 — Policy network
# ════════════════════════════════════════════════════════════════════════
# A small MLP mapping state -> logits over actions. We sample an action
# from the Categorical distribution so the policy is STOCHASTIC — critical
# for REINFORCE because the gradient uses log-probabilities of the sampled
# actions.
class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
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

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor]:
        """Sample an action and return (action, log_prob). The log_prob keeps
        a gradient graph so we can backprop through it later."""
        s = torch.from_numpy(state.astype(np.float32)).to(device)
        logits = self.forward(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a)


policy = PolicyNet(obs_dim, n_actions).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)


# ════════════════════════════════════════════════════════════════════════
# PART 3 — REINFORCE update
# ════════════════════════════════════════════════════════════════════════
# Collect a whole episode, compute the discounted return at every step,
# and update:
#       theta <- theta + alpha * sum_t [ G_t * d/dtheta log pi(a_t | s_t) ]
# We standardise returns (subtract mean, divide by std) for variance
# reduction — this is a simple baseline.
def discount_returns(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    returns = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    return returns_t


def run_episode(policy: PolicyNet, render: bool = False, seed: int | None = None):
    state, _ = env.reset(seed=seed)
    log_probs: list[torch.Tensor] = []
    rewards: list[float] = []
    done = False
    while not done:
        action, log_prob = policy.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(float(reward))
        done = terminated or truncated
    return log_probs, rewards


# ── Training loop ──────────────────────────────────────────────────────
# REINFORCE is sample-inefficient — we need several dozen episodes on
# CartPole. Episode cap 500 timesteps; we aim for a moving average close
# to 150 within ~80 episodes.
N_EPISODES = 120
episode_returns: list[float] = []

print("\n── Training REINFORCE on CartPole-v1 ──")
for ep in range(N_EPISODES):
    log_probs, rewards = run_episode(policy, seed=42 + ep)
    returns = discount_returns(rewards, gamma=0.99)
    loss = torch.stack([-lp * R for lp, R in zip(log_probs, returns)]).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total = float(sum(rewards))
    episode_returns.append(total)
    if (ep + 1) % 20 == 0:
        avg_last_20 = float(np.mean(episode_returns[-20:]))
        print(f"  episode {ep+1:3d}  return={total:6.1f}  avg20={avg_last_20:6.1f}")


# ════════════════════════════════════════════════════════════════════════
# PART 4 — Evaluate trained vs random policy
# ════════════════════════════════════════════════════════════════════════
def evaluate(pol: nn.Module | None, n: int = 30) -> list[float]:
    returns_out: list[float] = []
    for i in range(n):
        state, _ = env.reset(seed=1000 + i)
        total = 0.0
        done = False
        while not done:
            if pol is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s = torch.from_numpy(state.astype(np.float32)).to(device)
                    action = int(pol.forward(s).argmax().item())  # greedy at eval time
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns_out.append(total)
    return returns_out


random_returns = evaluate(None)
trained_returns = evaluate(policy)

print(f"\nRandom policy  : mean return {np.mean(random_returns):.1f} ± {np.std(random_returns):.1f}")
print(f"Trained policy : mean return {np.mean(trained_returns):.1f} ± {np.std(trained_returns):.1f}")


# ════════════════════════════════════════════════════════════════════════
# PART 5 — Visualise the learning curve
# ════════════════════════════════════════════════════════════════════════
# Compute a moving average to smooth the jagged episode-return curve.
def moving_average(xs: list[float], window: int = 10) -> list[float]:
    if len(xs) < window:
        return xs
    arr = np.asarray(xs, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return list(np.convolve(arr, kernel, mode="valid"))


viz = ModelVisualizer()
fig = viz.training_history(
    metrics={
        "episode return": episode_returns,
        "moving avg (10)": moving_average(episode_returns, 10),
    },
    x_label="Episode",
    y_label="Return",
)
fig.write_html("ex_8_training.html")
print("Training history saved to ex_8_training.html")

env.close()


# ── Reflection ─────────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  WHAT YOU'VE MASTERED")
print("═" * 70)
print(
    """
  [x] Built a stochastic policy network with nn.Module + Categorical sampling
  [x] Implemented REINFORCE from scratch (policy gradient + discounted returns)
  [x] Trained an agent on CartPole-v1 purely from environment interaction
  [x] Used return standardisation as a simple variance-reduction baseline
  [x] Compared trained vs random policy on evaluation rollouts

  Bridge to M6: PPO uses the same policy-gradient family but adds
  (1) an advantage estimator (e.g. GAE) for lower variance, and
  (2) a clipped surrogate objective to prevent catastrophically large
  updates. RLHF fine-tunes language models with PPO where the reward
  comes from a learned reward model over human preferences. DPO achieves
  a similar outcome without the separate reward model.
"""
)
