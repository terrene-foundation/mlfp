# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
#
# Note on the Kailash RL engine: this exercise hand-writes DQN on purpose,
# so you see every moving part (replay buffer, target network, epsilon).
# The library path for production RL is `km.rl_train(env, algo="dqn", ...)`
# (also `kailash_ml.rl.RLTrainer` + `RLTrainingConfig`; there is no top-level
# `kailash_ml.RLTrainer`). Its backend is Stable-Baselines3, an optional
# extra (`pip install kailash-ml[rl]`) that the course environment does NOT
# install — without it `km.rl_train` raises ImportError.
#
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 8.1: Deep Q-Network (DQN) from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Explain Q-learning intuition: "learning the value of actions by
#     trial and error"
#   - Implement a replay buffer ("learning from memories") and explain
#     why it breaks temporal correlation
#   - Implement a target network ("aiming at a moving target") and explain
#     why it stabilises the Bellman update
#   - Implement epsilon-greedy exploration ("explore vs exploit") and
#     watch the agent shift from random to purposeful action
#   - Train DQN on CartPole-v1 and track all metrics with ExperimentTracker
#   - Visualise reward curves, epsilon decay, Q-value heatmaps, and
#     episode length progression
#   - Apply DQN to a Singapore retail inventory management problem
#
# PREREQUISITES: M5/ex_2 through M5/ex_4 (PyTorch training loops).
# ESTIMATED TIME: ~40 min
# DATASETS: No static dataset — the environment IS the data source.
#   - CartPole-v1 (Gymnasium classic control, 4-D state, 2 actions)
#
# TASKS:
#   1. Understand DQN theory and the Bellman equation
#   2. Build DQN: replay buffer, Q-network, training loop
#   3. Train DQN on CartPole-v1 with ExperimentTracker logging
#   4. Visualise agent behaviour: reward curve, epsilon decay, Q-value
#      heatmap, episode lengths
#   5. Apply: inventory management for a Singapore retailer
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import gymnasium as gym
import polars as pl

from shared.mlfp05.ex_8 import (
    DQN,
    OUTPUT_DIR,
    ReplayBuffer,
    device,
    evaluate_policy,
    make_cartpole,
    moving_average,
    register_rl_model,
    rl_diagnostic_checkpoint,
    setup_engines,
)
from kailash_ml import ModelVisualizer

# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Theory: Why DQN Exists
# ════════════════════════════════════════════════════════════════════════
# Imagine you're a new hire at a Singapore grocery chain. Every day you
# make decisions — should you restock the shelves? Run a promotion?
# Rearrange the display? You don't know the "right" answers yet, so you
# TRY things and observe the RESULTS (sales go up, complaints go down).
# Over time, you learn which actions are valuable in which situations.
#
# That's Q-learning. Q(s, a) answers: "If I'm in situation s and I take
# action a, how much total future reward can I expect?"
#
# The BELLMAN OPTIMALITY EQUATION is the recursive definition:
#   Q*(s, a) = E[ r + gamma * max_{a'} Q*(s', a') ]
# "The value of an action = immediate reward + best future value"
#
# DQN (Mnih et al., 2015) approximates Q* with a neural network. Two
# innovations prevent training instability:
#
#   (1) REPLAY BUFFER — "Learning from memories"
#       Problem: consecutive (s, a, r, s') samples are correlated — the
#       agent sees similar states in sequence, biasing gradient updates.
#       Solution: store transitions in a buffer, sample RANDOM minibatches.
#       Like a student reviewing flashcards in random order instead of
#       re-reading the textbook front to back.
#
#   (2) TARGET NETWORK — "Aiming at a moving target"
#       Problem: Q appears on BOTH sides of the Bellman equation. Updating
#       Q to match a target that is also Q creates oscillations — you're
#       chasing your own tail.
#       Solution: keep a SLOW-MOVING COPY (target network) for computing
#       targets. Update it periodically, not every step. Like calibrating
#       a scale against a reference weight, not against itself.
#
#   (3) EPSILON-GREEDY — "Explore vs exploit"
#       With probability epsilon, take a RANDOM action (explore new
#       strategies). Otherwise, take the BEST action according to Q
#       (exploit what you've learned). Epsilon starts high (mostly random)
#       and decays over training (mostly purposeful).

print("=" * 70)
print("  TASK 1: DQN Theory — Q-Learning with Neural Networks")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build DQN: training loop with replay buffer + target network
# ════════════════════════════════════════════════════════════════════════

# Set up CartPole and kailash engines
cartpole_env, obs_dim, n_actions = make_cartpole()
conn, tracker, exp_name, registry, has_registry = setup_engines()


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
) -> tuple[DQN, list[float], list[float], list[float], list[int]]:
    """Train DQN and log to ExperimentTracker.

    Returns:
        (q_net, episode_rewards, episode_losses, epsilons, episode_lengths)
    """
    # TODO: Create q_net (DQN) and target_net (DQN copy) on device
    # Hint: target_net starts with the same weights as q_net via load_state_dict
    q_net = ____  # TODO
    target_net = ____  # TODO
    # TODO: Set target_net to eval mode and copy q_net weights
    # Hint: target_net.load_state_dict(q_net.state_dict()); target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=10_000)
    epsilon = epsilon_start

    episode_rewards: list[float] = []
    episode_losses: list[float] = []
    epsilons: list[float] = []
    episode_lengths: list[int] = []

    async with tracker.track(experiment=exp_name, run_name=run_name) as run:
        await run.log_params(
            {
                "algorithm": "DQN",
                "gamma": str(gamma),
                "lr": str(lr),
                "epsilon_start": str(epsilon_start),
                "epsilon_end": str(epsilon_end),
                "target_update_freq": str(target_update_freq),
                "batch_size": str(batch_size),
            }
        )

        print(f"\n== Training DQN: {run_name} ==")
        for ep in range(n_episodes):
            state, _ = env.reset(seed=42 + ep)
            total_reward = 0.0
            ep_loss_sum = 0.0
            ep_loss_count = 0
            steps = 0
            done = False

            while not done:
                # TODO: Epsilon-greedy action selection
                # Hint: with probability epsilon choose random action
                # (env.action_space.sample()), otherwise argmax of Q-values
                # from q_net on the current state tensor
                if random.random() < epsilon:
                    action = ____  # TODO: random action
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(state, dtype=torch.float32, device=device)
                        action = ____  # TODO: argmax of q_net(s_t)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # Store `terminated`, not `done`: a time-limit truncation is
                # not a real ending, so its target must still bootstrap
                # from Q(s'). Only a true termination zeroes the future.
                replay.push(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                steps += 1

                # Train on a minibatch from replay buffer
                if len(replay) >= min_replay_size:
                    s_b, a_b, r_b, ns_b, d_b = replay.sample(batch_size)

                    # TODO: Compute current Q-values for chosen actions
                    # Hint: q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
                    q_values = ____  # TODO

                    # TODO: Compute target Q-values using Bellman equation
                    # Target: r + gamma * max_a' Q_target(s', a') * (1 - done)
                    # Hint: use target_net (not q_net) for next-state Q-values
                    with torch.no_grad():
                        next_q = ____  # TODO: target_net(ns_b).max(dim=1).values
                        targets = ____  # TODO: r_b + gamma * next_q * (1.0 - d_b)

                    # TODO: Compute MSE loss between q_values and targets
                    # Hint: F.mse_loss(q_values, targets)
                    loss = ____  # TODO
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    ep_loss_sum += loss.item()
                    ep_loss_count += 1

            # TODO: Decay epsilon after each episode
            # Hint: epsilon = max(epsilon_end, epsilon * epsilon_decay)
            epsilon = ____  # TODO

            # TODO: Update target network periodically
            # Hint: every target_update_freq episodes, copy q_net weights to target_net
            if (ep + 1) % target_update_freq == 0:
                # TODO: target_net.load_state_dict(q_net.state_dict())
                pass

            episode_rewards.append(total_reward)
            avg_loss = ep_loss_sum / max(ep_loss_count, 1)
            episode_losses.append(avg_loss)
            epsilons.append(epsilon)
            episode_lengths.append(steps)

            metrics = {
                "episode_reward": float(total_reward),
                "epsilon": float(epsilon),
            }
            if ep_loss_count > 0:
                metrics["loss"] = float(avg_loss)
            await run.log_metrics(metrics, step=ep)

            if (ep + 1) % 40 == 0:
                avg_20 = float(np.mean(episode_rewards[-20:]))
                print(
                    f"  ep {ep+1:3d}  reward={total_reward:6.1f}  "
                    f"avg20={avg_20:6.1f}  eps={epsilon:.3f}  loss={avg_loss:.4f}"
                )

        await run.log_metric("final_avg_reward", float(np.mean(episode_rewards[-20:])))

    return q_net, episode_rewards, episode_losses, epsilons, episode_lengths


def train_dqn(
    env: gym.Env,
    obs_d: int,
    n_act: int,
    n_episodes: int = 200,
    **kwargs,
) -> tuple[DQN, list[float], list[float], list[float], list[int]]:
    """Sync wrapper for DQN training."""
    return asyncio.run(train_dqn_async(env, obs_d, n_act, n_episodes, **kwargs))


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train DQN on CartPole-v1
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  TASK 3: Train DQN on CartPole-v1")
print("=" * 70)

dqn_model, dqn_rewards, dqn_losses, dqn_epsilons, dqn_lengths = train_dqn(
    cartpole_env, obs_dim, n_actions
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(dqn_rewards) == 200, "DQN should train for 200 episodes"
assert (
    float(np.mean(dqn_rewards[-20:])) > 50.0
), "DQN should achieve avg reward > 50 in last 20 episodes (random baseline ~20)"
# INTERPRETATION: DQN learns Q(s,a) — the expected total reward from
# taking action a in state s and then acting optimally. The replay buffer
# decorrelates samples; the target network stabilises training. If the
# avg reward is climbing, the Q-network is learning to predict which
# action leads to more pole-balancing time.
print("--- Checkpoint 1 passed --- DQN trained on CartPole\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — RL instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# The DL Prescription Pad (gradient flow, dead neurons, loss trend) reads
# supervised batches. An RL agent is judged on its training HISTORY, so
# kailash-ml has a separate instrument: RLDiagnostics (the object that
# `km.diagnose("dqn", kind="rl")` returns). It installs no hooks — the
# helper feeds it the rewards, episode lengths and Bellman losses we
# recorded above and prints its report().
dqn_rl_report = rl_diagnostic_checkpoint(
    "DQN on CartPole-v1",
    "dqn",
    dqn_rewards,
    lengths=dqn_lengths,
    q_losses=dqn_losses,
    window=20,
)
# HOW TO READ IT (the numbers come from YOUR run; nothing is predicted):
#   mean reward (last 20) vs peak — a late mean far below the peak means
#     the policy found a good behaviour and then lost it.
#   [CRIT] episode_reward_collapse — the LAST episode fell below 10% of
#     the peak after a >=50% drop. Epsilon is still about
#     0.995^200 = 0.37 at episode 200, so a single unlucky exploratory
#     episode can trip it: confirm against the moving average in Task 4
#     before acting. A real DQN collapse usually means the learning rate
#     is too high or the target network is synced too often.
#   findings: none — the agent did not collapse. That does NOT prove it
#     converged; read the reward curve and the Bellman-loss curve below.


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Visualise: reward curve, epsilon decay, Q-value heatmap,
#           episode length progression
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  TASK 4: Visualise DQN Agent Behaviour")
print("=" * 70)

viz = ModelVisualizer()

# ── Plot 1: DQN reward curve with moving average ─────────────────────
# TODO: Use viz.training_history() with DQN episode rewards and moving average
# Hint: metrics dict with "DQN episode reward": dqn_rewards and
#   "DQN moving avg (20)": moving_average(dqn_rewards, 20)
fig1 = ____  # TODO: viz.training_history(metrics={...}, x_label="Episode", y_label="Reward")
fig1.write_html(str(OUTPUT_DIR / "01_dqn_reward_curve.html"))
print(f"  Saved: {OUTPUT_DIR / '01_dqn_reward_curve.html'}")

# ── Plot 2: Epsilon decay over training ──────────────────────────────
# TODO: Plot epsilon values over episodes using viz.training_history()
fig2 = ____  # TODO: viz.training_history(metrics={"Epsilon (exploration rate)": dqn_epsilons}, ...)
fig2.write_html(str(OUTPUT_DIR / "01_dqn_epsilon_decay.html"))
print(f"  Saved: {OUTPUT_DIR / '01_dqn_epsilon_decay.html'}")
print(f"  Final epsilon after {len(dqn_epsilons)} episodes: {dqn_epsilons[-1]:.3f}")
# INTERPRETATION: Epsilon starts at 1.0 (100% random) and decays by x0.995
# per episode toward the 0.01 floor. After 200 episodes it is still about
# 0.995^200 = 0.37 (printed above), so even the last episodes are roughly
# one-third random: the training rewards understate the greedy policy,
# which is why the Apply section evaluates the greedy policy separately.
# The decay rate controls the explore/exploit balance.

# ── Plot 3: Q-value heatmap over state space ─────────────────────────
# Visualise what the DQN has learned: for a grid of (cart_position, pole_angle)
# combinations (with velocity=0), what Q-value does it assign to each action?
cart_positions = np.linspace(-2.4, 2.4, 30)
pole_angles = np.linspace(-0.21, 0.21, 30)
q_values_grid = np.zeros((30, 30))

dqn_model.eval()
# TODO: Fill q_values_grid by querying the DQN for each (cart_pos, pole_angle) pair
# Hint: For each (i, cp) and (j, pa), create state [cp, 0.0, pa, 0.0],
#   pass through dqn_model, and store the max Q-value in q_values_grid[j, i]
for i, cp in enumerate(cart_positions):
    for j, pa in enumerate(pole_angles):
        state = torch.tensor([cp, 0.0, pa, 0.0], dtype=torch.float32, device=device)
        with torch.no_grad():
            q_vals = dqn_model(state)
            q_values_grid[j, i] = ____  # TODO: float(q_vals.max().item())

# Use polars for the heatmap data
heatmap_rows = []
for i, cp in enumerate(cart_positions):
    for j, pa in enumerate(pole_angles):
        heatmap_rows.append(
            {
                "cart_position": float(cp),
                "pole_angle": float(pa),
                "max_Q": q_values_grid[j, i],
            }
        )
q_heatmap_df = pl.DataFrame(heatmap_rows)

# Pivot for heatmap visualisation
import plotly.graph_objects as go

# TODO: Create a heatmap figure using go.Heatmap with q_values_grid
# Hint: go.Figure(data=go.Heatmap(z=q_values_grid, x=cart_positions rounded,
#   y=pole_angles rounded, colorscale="Viridis"))
fig3 = ____  # TODO
fig3.update_layout(
    title="DQN Q-Value Heatmap: Cart Position vs Pole Angle (velocities=0)",
    xaxis_title="Cart Position",
    yaxis_title="Pole Angle (radians)",
)
fig3.write_html(str(OUTPUT_DIR / "01_dqn_qvalue_heatmap.html"))
print(f"  Saved: {OUTPUT_DIR / '01_dqn_qvalue_heatmap.html'}")
# INTERPRETATION: The heatmap is the DQN's learned value landscape. If
# training worked, expect higher max-Q (bright) near the centre (cart
# centred, pole upright — a long future of +1 rewards) and lower max-Q
# (dark) towards large pole angles, where recovery is unlikely. A flat or
# noisy map means the Q-network has not yet separated good from bad states.

# ── Plot 4: Episode length progression ───────────────────────────────
# TODO: Plot episode lengths and their moving average using viz.training_history()
fig4 = ____  # TODO
fig4.write_html(str(OUTPUT_DIR / "01_dqn_episode_lengths.html"))
print(f"  Saved: {OUTPUT_DIR / '01_dqn_episode_lengths.html'}")
# INTERPRETATION: Episode length IS the reward in CartPole (reward=+1 per
# step). Longer episodes = the agent keeps the pole balanced longer. Early
# episodes are short (random flailing). If the agent is learning, the
# moving average climbs; 500 is the cap, because CartPole-v1 truncates
# every episode at 500 steps.

# ── Plot 5: DQN loss curve ───────────────────────────────────────────
# TODO: Plot the Bellman loss curve using viz.training_history()
fig5 = ____  # TODO
fig5.write_html(str(OUTPUT_DIR / "01_dqn_loss_curve.html"))
print(f"  Saved: {OUTPUT_DIR / '01_dqn_loss_curve.html'}")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert Path(
    OUTPUT_DIR / "01_dqn_reward_curve.html"
).exists(), "Reward curve should be saved"
assert Path(
    OUTPUT_DIR / "01_dqn_qvalue_heatmap.html"
).exists(), "Q-value heatmap should be saved"
assert Path(
    OUTPUT_DIR / "01_dqn_epsilon_decay.html"
).exists(), "Epsilon decay should be saved"
print("--- Checkpoint 2 passed --- all DQN visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Apply: Inventory Management for a Singapore Retailer
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: You're the operations manager at a Singapore supermarket chain
# (hypothetical; every number below is illustrative). Every day you
# decide how much stock to order for a perishable product category
# (fresh produce). Order too much -> holding costs and spoilage. Order
# too little -> empty shelves and lost sales.
#
# State: (stock_level, demand_forecast, day_of_week)
#   - stock_level: normalised current inventory [0, 1]
#   - demand_forecast: predicted demand from the POS system [0, 1]
#   - day_of_week: cyclical encoding [0, 1] (0=Mon, 1=Sun)
#
# Actions: order_nothing(0), order_small(1), order_medium(2), order_large(3)
#
# Reward: sales_revenue - holding_cost - stockout_penalty - order_cost
#   Revenue from sales (what you DO sell)
#   minus cost of holding unsold stock (cold storage, spoilage)
#   minus penalty for stockouts (lost customers, reputational damage)
#   minus ordering cost (logistics, supplier handling)

print("=" * 70)
print("  TASK 5: Apply DQN — Singapore Retail Inventory Management")
print("=" * 70)

from gymnasium import spaces


class RetailInventoryEnv(gym.Env):
    """Inventory management for a Singapore supermarket.

    Models weekly ordering decisions for a perishable product category.
    State (3,): [stock_level, demand_forecast, day_of_week]
    Actions (4): 0=order_nothing, 1=order_small, 2=order_medium, 3=order_large
    Reward: sales - holding_cost - stockout_penalty - order_cost
    Episode: 365 steps (daily decisions for one year).
    """

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.max_steps = 365
        self.state = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(
            [0.5, 0.4, 0.0],  # start mid-stock, moderate forecast, Monday
            dtype=np.float32,
        )
        self.step_count = 0
        return self.state.copy(), {}

    def step(self, action):
        self.step_count += 1
        stock, forecast, day_norm = self.state

        # TODO: Implement the environment step logic
        # 1. Apply order quantities based on action:
        #    order_qtys = [0.0, 0.08, 0.18, 0.35]
        #    order_costs = [0.0, 0.01, 0.02, 0.04]
        #    stock = min(1.0, stock + order_qtys[action])
        order_qtys = [0.0, 0.08, 0.18, 0.35]
        order_costs = [0.0, 0.01, 0.02, 0.04]
        stock = ____  # TODO: update stock with order quantity

        # 2. Compute demand with seasonal patterns
        # Hint: weekend boost (+0.08 for day_of_week >= 5)
        # Hint: festive boost (+0.12 for CNY weeks 5-6 and Deepavali weeks 43-44)
        day_of_week = int(day_norm * 7) % 7
        weekend_boost = ____  # TODO
        week = self.step_count // 7
        festive_boost = ____  # TODO
        base_demand = 0.15 + weekend_boost + festive_boost
        demand = max(0.0, base_demand + self.np_random.normal(0, 0.04))

        # 3. Fulfil demand and compute spoilage
        # Hint: sold = min(stock, demand), then subtract demand from stock
        # Hint: spoilage = stock * 0.05 (5% daily for perishable goods)
        sold = ____  # TODO
        stockout = ____  # TODO
        stock = ____  # TODO: remaining stock after demand
        spoiled = ____  # TODO: 5% spoilage
        stock = stock - spoiled

        # 4. Compute reward components
        # Hint: sales_revenue = sold * 3.0, holding_cost = stock * 0.3,
        #   stockout_penalty = stockout * 5.0, spoilage_cost = spoiled * 2.0
        sales_revenue = ____  # TODO
        holding_cost = ____  # TODO
        stockout_penalty = ____  # TODO
        spoilage_cost = ____  # TODO

        reward = (
            sales_revenue
            - holding_cost
            - stockout_penalty
            - spoilage_cost
            - order_costs[action]
        )

        # 5. Update state for next step
        day_norm = ((day_of_week + 1) % 7) / 7.0
        forecast = np.clip(base_demand + self.np_random.normal(0, 0.06), 0.0, 1.0)
        self.state = np.array([stock, forecast, day_norm], dtype=np.float32)

        truncated = self.step_count >= self.max_steps
        return self.state.copy(), reward, False, truncated, {}


# Verify environment API
inv_env = RetailInventoryEnv()
obs, info = inv_env.reset(seed=42)
assert obs.shape == (3,), "Inventory env should have 3-D state"
obs2, r, term, trunc, info = inv_env.step(1)
assert isinstance(r, (int, float)) or hasattr(r, "__float__"), f"Reward should be numeric, got {type(r).__name__}: {r!r}"
print(f"  RetailInventory env: obs={obs.shape}, actions=4, sample_reward={r:.3f}")

# ── Train DQN on inventory environment ───────────────────────────────
inv_dqn, inv_rewards, inv_losses, inv_eps, inv_lens = train_dqn(
    inv_env, 3, 4, n_episodes=200, run_name="dqn_retail_inventory"
)


# ── Fixed-threshold baseline ─────────────────────────────────────────
# Baseline policy: "order medium when stock < 0.3, order small when < 0.5, else nothing"
def fixed_threshold_policy(state):
    stock = state[0]
    if stock < 0.2:
        return 3  # order large
    elif stock < 0.35:
        return 2  # order medium
    elif stock < 0.5:
        return 1  # order small
    return 0  # order nothing


baseline_returns = evaluate_policy(inv_env, fixed_threshold_policy, n_episodes=30)


def dqn_inventory_policy(state):
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device)
        return int(inv_dqn(s).argmax().item())


dqn_inv_returns = evaluate_policy(inv_env, dqn_inventory_policy, n_episodes=30)

print(
    f"\n  Fixed-threshold baseline: mean annual reward = {np.mean(baseline_returns):.1f}"
)
print(f"  DQN learned policy:      mean annual reward = {np.mean(dqn_inv_returns):.1f}")

improvement = float(np.mean(dqn_inv_returns)) - float(np.mean(baseline_returns))
pct_improvement = (
    improvement / abs(float(np.mean(baseline_returns))) * 100
    if float(np.mean(baseline_returns)) != 0
    else 0
)
print(f"  Improvement: {improvement:+.1f} ({pct_improvement:+.1f}%)")

# ── Visualise: DQN vs baseline ───────────────────────────────────────
# TODO: Create a box plot comparing DQN vs fixed-threshold annual rewards
# Hint: pl.DataFrame with "Policy" and "Annual Reward" columns, then
#   viz.box_plot(comparison_df, "Annual Reward", group_by="Policy")
comparison_df = ____  # TODO
fig_apply = ____  # TODO: viz.box_plot(...)
fig_apply.write_html(str(OUTPUT_DIR / "01_dqn_inventory_comparison.html"))
print(f"  Saved: {OUTPUT_DIR / '01_dqn_inventory_comparison.html'}")

# ── Visualise: learned policy actions across stock levels ─────────────
stock_levels = np.linspace(0.0, 1.0, 50)
action_names = ["Nothing", "Small", "Medium", "Large"]
policy_actions = []
inv_dqn.eval()
# TODO: For each stock level, query the DQN with state [sl, 0.3, 0.3]
# and record the chosen action name
for sl in stock_levels:
    state = torch.tensor([sl, 0.3, 0.3], dtype=torch.float32, device=device)
    with torch.no_grad():
        action = ____  # TODO: int(inv_dqn(state).argmax().item())
    policy_actions.append(action_names[action])

policy_df = pl.DataFrame(
    {"Stock Level": stock_levels.tolist(), "Order Action": policy_actions}
)
print("\n  Learned Ordering Policy (demand_forecast=0.3, mid-week):")
# Every 10th of the 50 grid points: stock 0.00, 0.20, 0.41, 0.61, 0.82, 1.00
for row in policy_df.gather_every(10).iter_rows(named=True):
    print(f"    Stock={row['Stock Level']:.2f} -> {row['Order Action']}")
print(f"    Stock=1.00 -> {policy_actions[-1]}")

# INTERPRETATION (computed from this run, not assumed): the fixed rule
# looks only at stock; the DQN also sees the demand forecast and the day
# of week, so it CAN learn weekend and forecast-driven ordering. It cannot
# anticipate the festive weeks directly — the week number is not in the
# state; it only sees them through a higher forecast.
if improvement > 0:
    print(f"  DQN beat the fixed-threshold rule by {pct_improvement:+.1f}% per year.")
else:
    print(
        f"  DQN did NOT beat the fixed-threshold rule ({pct_improvement:+.1f}%). "
        "A hand-tuned rule is a strong baseline; more episodes may be needed."
    )

inv_env.close()

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(inv_rewards) == 200, "Inventory DQN should train for 200 episodes"
assert Path(OUTPUT_DIR / "01_dqn_inventory_comparison.html").exists()
print("--- Checkpoint 3 passed --- DQN applied to retail inventory\n")


# ── Register DQN models ──────────────────────────────────────────────
if has_registry:
    register_rl_model(
        registry,
        "m5_dqn_cartpole",
        dqn_model,
        {
            "avg_reward_last20": float(np.mean(dqn_rewards[-20:])),
            "algorithm": 0.0,
            "episodes_trained": float(len(dqn_rewards)),
        },
    )
    register_rl_model(
        registry,
        "m5_dqn_retail_inventory",
        inv_dqn,
        {
            "avg_reward_last30": float(np.mean(inv_rewards[-30:])),
            "episodes_trained": 200.0,
        },
    )

cartpole_env.close()

# Clean up
asyncio.run(conn.close())


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — DQN")
print("=" * 70)
print(
    """
  [x] Derived the Bellman optimality equation: Q*(s,a) = E[r + gamma * max Q*(s',a')]
  [x] Built DQN from scratch: replay buffer, target network, epsilon-greedy
  [x] Trained DQN on CartPole-v1 — watched it go from random flailing to
      stable pole-balancing
  [x] Visualised the agent's learning:
      - Reward curve: noisy exploration -> smooth convergence
      - Epsilon decay: 100% random -> ~37% random after 200 episodes
      - Q-value heatmap: the agent's learned value landscape
      - Episode lengths: survived longer as it learned
  [x] Applied DQN to Singapore retail inventory management:
      - Built a custom environment with seasonal demand (CNY, Deepavali)
      - Compared the DQN's ordering policy with a fixed-threshold rule
      - Visualised learned policy vs baseline with annual cost comparison

  KEY INSIGHT:
  DQN learns the VALUE of actions (Q-values), then acts greedily. It's
  like learning to evaluate chess positions before deciding the next move.
  Great for discrete action spaces with clear reward signals.

  Next: In 02_ppo.py, you'll learn PPO — an algorithm that learns the
  POLICY directly (what to do) rather than indirectly through values.
"""
)
