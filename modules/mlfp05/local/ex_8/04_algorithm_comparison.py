# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 8.4: Algorithm Comparison — Random vs DQN vs PPO
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this exercise, you will be able to:
#   - Compare Random, DQN, and PPO policies side-by-side on CartPole-v1
#   - Analyse sample efficiency: how many environment interactions does
#     each algorithm need to reach a given performance level?
#   - Measure wall-clock training time for each algorithm
#   - Build a decision framework: which algorithm for which business problem?
#   - Explain how PPO connects to RLHF for LLM alignment (bridge to M6)
#
# PREREQUISITES: M5/ex_8/01_dqn.py and M5/ex_8/02_ppo.py.
# ESTIMATED TIME: ~30 min
# DATASETS: No static dataset — the environment IS the data source.
#   - CartPole-v1 (Gymnasium classic control, 4-D state, 2 actions)
#
# TASKS:
#   1. Train DQN and PPO on CartPole-v1 with timing
#   2. Evaluate Random vs DQN vs PPO side-by-side
#   3. Visualise: reward comparison, sample efficiency, training time
#   4. Apply: decision framework for engineering managers
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium import spaces

import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.mlfp05.ex_8 import (
    DQN,
    OUTPUT_DIR,
    ReplayBuffer,
    device,
    evaluate_policy,
    make_cartpole,
    moving_average,
    rl_diagnostic_checkpoint,
    setup_engines,
)
from shared.mlfp05 import create_visualizer
from kailash_ml import ModelVisualizer


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Train DQN and PPO with Timing
# ════════════════════════════════════════════════════════════════════════
# We retrain both algorithms from scratch with identical episode counts
# so the comparison is fair. We measure wall-clock time for each.

print("=" * 70)
print("  TASK 1: Train DQN and PPO with Timing")
print("=" * 70)

cartpole_env, obs_dim, n_actions = make_cartpole()
conn, tracker, exp_name, registry, has_registry = setup_engines()


# ── ActorCritic for PPO (needed for this comparison file) ────────────
class ActorCritic(nn.Module):
    """SEPARATE actor and critic networks for PPO.

    A shared trunk lets the critic's large value-regression gradients swamp
    the actor's tiny policy gradients, so the policy never moves (CartPole
    stays stuck at ~random return). Independent MLPs let each learn at its
    own scale — see ex_8/02 for the full explanation.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        s = torch.from_numpy(state.astype(np.float32)).to(device)
        logits, value = self.forward(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a).detach(), value.detach()


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE computation for PPO."""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - float(values[t])
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
        next_value = float(values[t])
    returns = [a + float(v) for a, v in zip(advantages, values)]
    return advantages, returns


# ── Train DQN ────────────────────────────────────────────────────────
N_DQN_EPISODES = 200
N_DQN_ENV_STEPS = 0  # count total environment interactions


async def _train_dqn_timed():
    global N_DQN_ENV_STEPS
    q_net = DQN(obs_dim, n_actions).to(device)
    target_net = DQN(obs_dim, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    replay = ReplayBuffer(capacity=10_000)
    epsilon = 1.0
    episode_rewards: list[float] = []
    env_steps = 0

    async with tracker.track(experiment=exp_name, run_name="comparison_dqn") as run:
        await run.log_params({"algorithm": "DQN", "episodes": str(N_DQN_EPISODES)})
        for ep in range(N_DQN_EPISODES):
            state, _ = cartpole_env.reset(seed=42 + ep)
            total_reward = 0.0
            done = False
            while not done:
                # TODO: Epsilon-greedy action selection
                # Hint: same pattern as 01_dqn.py
                if random.random() < epsilon:
                    action = ____  # TODO
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(state, dtype=torch.float32, device=device)
                        action = ____  # TODO
                next_state, reward, terminated, truncated, _ = cartpole_env.step(action)
                done = terminated or truncated
                # `terminated`, not `done`: a 500-step truncation must still
                # bootstrap from Q(s') (see 01_dqn.py).
                replay.push(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                env_steps += 1
                # TODO: DQN training step when replay has enough samples
                # Hint: same pattern as 01_dqn.py — sample, Q-values, targets, MSE loss
                if len(replay) >= 500:
                    s_b, a_b, r_b, ns_b, d_b = replay.sample(64)
                    q_values = ____  # TODO
                    with torch.no_grad():
                        next_q = ____  # TODO
                        targets = ____  # TODO
                    loss = ____  # TODO
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            epsilon = max(0.01, epsilon * 0.995)
            if (ep + 1) % 10 == 0:
                target_net.load_state_dict(q_net.state_dict())
            episode_rewards.append(total_reward)
            await run.log_metric("episode_reward", total_reward, step=ep)
    N_DQN_ENV_STEPS = env_steps
    return q_net, episode_rewards


print("\n  Training DQN (200 episodes)...")
dqn_start = time.time()
dqn_model, dqn_rewards = asyncio.run(_train_dqn_timed())
dqn_time = time.time() - dqn_start
print(
    f"  DQN: {dqn_time:.1f}s, {N_DQN_ENV_STEPS} env steps, final avg20={np.mean(dqn_rewards[-20:]):.1f}"
)


# ── Train PPO ────────────────────────────────────────────────────────
N_PPO_ITERS = 30
STEPS_PER_ITER = 1024
N_PPO_ENV_STEPS = N_PPO_ITERS * STEPS_PER_ITER


async def _train_ppo_timed():
    model = ActorCritic(obs_dim, n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    iter_returns: list[float] = []

    async with tracker.track(experiment=exp_name, run_name="comparison_ppo") as run:
        await run.log_params({"algorithm": "PPO", "iterations": str(N_PPO_ITERS)})
        for it in range(N_PPO_ITERS):
            # Collect trajectory
            states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
            state, _ = cartpole_env.reset(seed=int(np.random.randint(0, 100_000)))
            for _ in range(STEPS_PER_ITER):
                action, log_prob, value = model.act(state)
                next_state, reward, terminated, truncated, _ = cartpole_env.step(action)
                states.append(state.astype(np.float32))
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(float(reward))
                done = terminated or truncated
                dones.append(done)
                state = next_state
                if done:
                    state, _ = cartpole_env.reset(
                        seed=int(np.random.randint(0, 100_000))
                    )

            advantages, returns = compute_gae(rewards, values, dones)
            s_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
            a_t = torch.tensor(actions, dtype=torch.long, device=device)
            old_lp_t = torch.stack(log_probs).to(device)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            n = s_t.size(0)
            idxs = np.arange(n)
            # TODO: PPO update — same clipped surrogate as 02_ppo.py, with the
            # clip range fixed at [0.8, 1.2] (clip_eps = 0.2)
            for _ in range(4):
                np.random.shuffle(idxs)
                for start in range(0, n, 256):
                    mb = idxs[start : start + 256]
                    logits, vpred = model(s_t[mb])
                    dist = Categorical(logits=logits)
                    new_lp = dist.log_prob(a_t[mb])
                    ratio = ____  # TODO
                    surr1 = ____  # TODO
                    surr2 = ____  # TODO
                    policy_loss = ____  # TODO
                    value_loss = ____  # TODO
                    entropy = ____  # TODO
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()

            ep_rs: list[float] = []
            running = 0.0
            for r, d in zip(rewards, dones):
                running += r
                if d:
                    ep_rs.append(running)
                    running = 0.0
            avg_ret = float(np.mean(ep_rs)) if ep_rs else float(running)
            iter_returns.append(avg_ret)
            await run.log_metric("avg_episode_return", avg_ret, step=it)
    return model, iter_returns


print("  Training PPO (30 iterations x 1024 steps)...")
ppo_start = time.time()
ppo_model, ppo_returns = asyncio.run(_train_ppo_timed())
ppo_time = time.time() - ppo_start
print(
    f"  PPO: {ppo_time:.1f}s, {N_PPO_ENV_STEPS} env steps, final return={ppo_returns[-1]:.1f}"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(dqn_rewards) == N_DQN_EPISODES
assert len(ppo_returns) == N_PPO_ITERS
print("--- Checkpoint 1 passed --- both algorithms trained with timing\n")


# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — RL instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# One RLDiagnostics report per algorithm, from the reward history each
# training run recorded (DQN: one entry per episode; PPO: one entry per
# iteration = that iteration's mean episode return).
dqn_rl_report = rl_diagnostic_checkpoint(
    "DQN (comparison run)", "dqn", dqn_rewards, window=20
)
ppo_rl_report = rl_diagnostic_checkpoint(
    "PPO (comparison run)", "ppo", ppo_returns, window=10
)
# HOW TO READ THEM (the numbers come from YOUR run; nothing is predicted):
#   Compare each algorithm's late mean with its own peak — the gap is how
#   much it gave back. A [CRIT] episode_reward_collapse on DQN can be one
#   exploratory episode (epsilon is still ~0.37 at episode 200); check
#   the training curves in Task 3 before calling it a collapse.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Evaluate Random vs DQN vs PPO side-by-side
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  TASK 2: Evaluate Random vs DQN vs PPO")
print("=" * 70)


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


N_EVAL = 50  # greedy episodes per policy (no significance test is run)
random_returns = evaluate_policy(cartpole_env, random_policy, n_episodes=N_EVAL)
dqn_eval_returns = evaluate_policy(cartpole_env, dqn_policy, n_episodes=N_EVAL)
ppo_eval_returns = evaluate_policy(cartpole_env, ppo_policy, n_episodes=N_EVAL)

print(f"\n  Policy Comparison (CartPole-v1, {N_EVAL} eval episodes)")
print(f"  {'Policy':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Median':>8}")
print(f"  {'-'*50}")
for name, returns in [
    ("Random", random_returns),
    ("DQN", dqn_eval_returns),
    ("PPO", ppo_eval_returns),
]:
    print(
        f"  {name:<10} {np.mean(returns):>8.1f} {np.std(returns):>8.1f} "
        f"{np.min(returns):>8.1f} {np.max(returns):>8.1f} {np.median(returns):>8.1f}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert float(np.mean(dqn_eval_returns)) > float(
    np.mean(random_returns)
), "DQN should outperform random"
assert float(np.mean(ppo_eval_returns)) > float(
    np.mean(random_returns)
), "PPO should outperform random"
print("--- Checkpoint 2 passed --- both algorithms outperform random\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Visualise: reward comparison, sample efficiency, training time
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  TASK 3: Comprehensive Comparison Visualisations")
print("=" * 70)

viz = create_visualizer()

# ── Plot 1: Evaluation reward box plot ───────────────────────────────
# TODO: Create box plot comparing Random, DQN, PPO evaluation returns
# Hint: long-format polars DataFrame ("Policy", "Evaluation Return"), then
# ModelVisualizer's box plot grouped by policy
comparison_df = ____  # TODO
fig1 = ____  # TODO
fig1.write_html(str(OUTPUT_DIR / "04_policy_comparison_boxplot.html"))
print(f"  Saved: {OUTPUT_DIR / '04_policy_comparison_boxplot.html'}")
# INTERPRETATION: The box plot shows final policy quality. Random is
# clustered around ~20 (the pole falls quickly). DQN and PPO should be
# much higher and with lower variance — they've learned stable policies.

# ── Plot 2: Training curves on a common x-axis (env steps) ──────────
# Normalise both algorithms to environment interactions for fair comparison
# CartPole pays +1 per step, so an episode's reward IS its length and the
# cumulative sum of DQN episode rewards is the exact env-step count. The
# 20-episode moving average starts at episode 20, so its x-values start
# there too (dqn_ma_steps) — otherwise the curve is shifted left.
dqn_cumulative_steps = np.cumsum(dqn_rewards).tolist()
dqn_ma_rewards = moving_average(dqn_rewards, 20)
dqn_ma_steps = dqn_cumulative_steps[len(dqn_rewards) - len(dqn_ma_rewards) :]
ppo_cumulative_steps = [(i + 1) * STEPS_PER_ITER for i in range(len(ppo_returns))]

# TODO: Line plot of both training curves against env steps: DQN's moving
# average (dqn_ma_rewards at dqn_ma_steps) and PPO's per-iteration returns
# (at ppo_cumulative_steps), plus a dashed horizontal line at the random
# policy's mean return
# Hint: plotly graph_objects Scatter traces; Figure.add_hline
fig2 = ____  # TODO
fig2.write_html(str(OUTPUT_DIR / "04_sample_efficiency.html"))
print(f"  Saved: {OUTPUT_DIR / '04_sample_efficiency.html'}")
# INTERPRETATION: Sample efficiency = how many environment interactions
# an algorithm needs to reach a given return. Read it off YOUR plot: pick
# a return level (say 150) and see which curve crosses it at fewer env
# steps. Expect a trade-off rather than a fixed winner: DQN re-uses every
# transition many times from its replay buffer, while PPO throws each
# 1024-step rollout away after 4 epochs but makes steadier updates. Note
# the training curves include exploration (DQN's epsilon, PPO's sampling);
# the greedy evaluation in Task 2 is the fair final-quality comparison.

# ── Plot 3: Wall-clock training time comparison ──────────────────────
# TODO: Create bar chart comparing DQN and PPO training times
# Hint: plotly graph_objects Bar, labelled with the seconds
fig3 = ____  # TODO
fig3.write_html(str(OUTPUT_DIR / "04_training_time.html"))
print(f"  Saved: {OUTPUT_DIR / '04_training_time.html'}")

# ── Plot 4: Summary dashboard ────────────────────────────────────────
# TODO: Create a 2x2 dashboard with make_subplots
# Subplot (1,1): bar chart of final policy quality (mean +/- std)
# Subplot (1,2): training curves (DQN + PPO scatter)
# Subplot (2,1): training time bars
# Subplot (2,2): algorithm properties table
# Hint: make_subplots(rows=2, cols=2, specs=[[{"type":"bar"},{"type":"scatter"}],
#   [{"type":"bar"},{"type":"table"}]])
fig4 = ____  # TODO
fig4.write_html(str(OUTPUT_DIR / "04_comparison_dashboard.html"))
print(f"  Saved: {OUTPUT_DIR / '04_comparison_dashboard.html'}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert Path(OUTPUT_DIR / "04_policy_comparison_boxplot.html").exists()
assert Path(OUTPUT_DIR / "04_sample_efficiency.html").exists()
assert Path(OUTPUT_DIR / "04_training_time.html").exists()
assert Path(OUTPUT_DIR / "04_comparison_dashboard.html").exists()
print("--- Checkpoint 3 passed --- all comparison visualisations generated\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Apply: Decision Framework for Engineering Managers
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  TASK 4: Which Algorithm for Which Business Problem?")
print("=" * 70)

# Build the decision framework as a polars DataFrame
decision_framework = pl.DataFrame(
    {
        "Business Problem": [
            "Inventory reorder (supermarket)",
            "Surge pricing (ride-hailing)",
            "Customer churn (telco)",
            "Portfolio rebalancing",
            "Queue staffing (airport)",
            "Traffic signals (road authority)",
            "Energy trading (electricity retailer)",
            "LLM alignment (M6)",
        ],
        "Action Space": [
            "Discrete (4 order sizes)",
            "Continuous (price multiplier)",
            "Discrete (4 interventions)",
            "Multi-discrete (27 combos)",
            "Discrete (7 shifts)",
            "Discrete (5 allocations)",
            "Discrete (5 trade sizes)",
            "Discrete (vocabulary tokens)",
        ],
        "Recommended": [
            "DQN",
            "PPO",
            "DQN",
            "DQN or PPO",
            "DQN",
            "DQN",
            "DQN or PPO",
            "PPO (RLHF)",
        ],
        "Why": [
            "Small discrete space, clear reward signal, replay buffer helps with sparse reorders",
            "A continuous price needs a policy network (DQN cannot argmax over a continuum); 02_ppo.py discretises it into 5 levels",
            "Small discrete space, DQN learns value of each intervention",
            "27 joint actions is still fine for DQN; PPO copes better if the joint action set grows combinatorially",
            "Small discrete space, DQN works well",
            "Small discrete space, fast convergence needed",
            "Either works; PPO if extending to continuous trade sizes",
            "Classic RLHF optimiser over a vocabulary of tens of thousands of tokens; a separate KL penalty to the reference model keeps outputs fluent",
        ],
    }
)

print("\n  ALGORITHM SELECTION GUIDE")
print("  " + "=" * 76)
for row in decision_framework.iter_rows(named=True):
    print(f"\n  {row['Business Problem']}")
    print(f"    Action space: {row['Action Space']}")
    print(f"    Recommended:  {row['Recommended']}")
    print(f"    Why: {row['Why']}")

# ── Summary statistics ───────────────────────────────────────────────
print("\n\n  TRAINING SUMMARY")
print("  " + "=" * 60)
print(f"  {'Metric':<30} {'DQN':>12} {'PPO':>12}")
print(f"  {'-'*54}")
print(f"  {'Training episodes/iters':<30} {N_DQN_EPISODES:>12} {N_PPO_ITERS:>12}")
print(f"  {'Total env interactions':<30} {N_DQN_ENV_STEPS:>12} {N_PPO_ENV_STEPS:>12}")
print(f"  {'Wall-clock time (s)':<30} {dqn_time:>12.1f} {ppo_time:>12.1f}")
print(
    f"  {'Eval mean reward':<30} {np.mean(dqn_eval_returns):>12.1f} {np.mean(ppo_eval_returns):>12.1f}"
)
print(
    f"  {'Eval std reward':<30} {np.std(dqn_eval_returns):>12.1f} {np.std(ppo_eval_returns):>12.1f}"
)
print(f"  {'Uses replay buffer':<30} {'Yes':>12} {'No':>12}")
print(f"  {'On/off policy':<30} {'Off-policy':>12} {'On-policy':>12}")
print(f"  {'Continuous actions':<30} {'No':>12} {'Yes*':>12}")
print("  * with a Gaussian policy head; this file's PPO uses a Categorical head")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(decision_framework) == 8, "Decision framework should cover 8 problems"
print("\n--- Checkpoint 4 passed --- decision framework complete\n")

cartpole_env.close()

# Clean up
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — the library path: km.rl_train
# ════════════════════════════════════════════════════════════════════════
# You hand-wrote DQN and PPO so every moving part is visible. In
# production the same comparison is one call per algorithm:
#   km.rl_train("CartPole-v1", algo="dqn", total_timesteps=...)
#   km.rl_train("CartPole-v1", algo="ppo", total_timesteps=...)
# rl_train also accepts "a2c", "sac", "td3" and "ddpg" (SAC/TD3/DDPG need
# a continuous action space), and RLDiagnostics.as_sb3_callback() plugs
# the same diagnostics you used above into that training loop. Its
# backend is Stable-Baselines3, an optional extra
# (`pip install kailash-ml[rl]`); we report whether it is installed
# rather than assume it.
import importlib.util

sb3_installed = importlib.util.find_spec("stable_baselines3") is not None
print("Library path: km.rl_train(env, algo='dqn' | 'ppo' | 'a2c' | ...)")
print(f"  Stable-Baselines3 backend installed here: {sb3_installed}")
if not sb3_installed:
    print("  (install kailash-ml[rl] to run km.rl_train; this file did not use it)")


# ══════════════════════════════════════════════════════════════════════
# REFLECTION
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — Algorithm Comparison")
print("=" * 70)
print(
    """
  [x] Compared Random vs DQN vs PPO on identical CartPole-v1 evaluation
  [x] Measured sample efficiency (env interactions to reach performance)
  [x] Measured wall-clock training time for each algorithm
  [x] Built a decision framework for engineering managers:

      USE DQN WHEN:
        - Small, discrete action space (< 10 actions)
        - Clear, immediate reward signal
        - Data efficiency matters (replay buffer reuses data)
        - Examples: inventory ordering, churn interventions, queue staffing

      USE PPO WHEN:
        - Continuous or large discrete action space
        - Stability matters more than data efficiency
        - You need on-policy guarantees (fresh data each iteration)
        - Examples: pricing, portfolio management, LLM alignment (RLHF)

      USE NEITHER (yet) WHEN:
        - You don't have a good simulator/environment
        - The reward function is unclear or hard to specify
        - Supervised learning can solve the problem (simpler, cheaper)

  BRIDGE TO M6 (RLHF — Reinforcement Learning from Human Feedback):
  The PPO you built here is the optimiser of classic RLHF:
    - Policy = the language model's next-token distribution
    - Action = the next token — a DISCRETE choice from the vocabulary
    - Reward model = trained on human preference rankings, scores
      each finished response
    - DPO (M6) = reaches the preference goal without a reward model
      or PPO at all

  The RLHF loop:
    1. The LLM generates a response, one token (action) at a time
    2. The reward model scores the response
    3. PPO updates the policy to maximise expected reward
    4. TWO separate brakes keep it stable:
       - PPO clipping bounds each update relative to the PREVIOUS
         policy (the same clip_eps you used on CartPole)
       - a KL penalty to the frozen reference (SFT) model keeps the
         LLM close to its starting point, so it does not drift into
         reward-hacking gibberish — clipping alone does not do this

  You now understand RL from first principles. M6 adds what RLHF
  needs on top: a learned reward model and the KL-to-reference term.
"""
)
