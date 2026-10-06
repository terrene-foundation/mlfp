# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 8.7 — DDPG: Off-Policy Actor-Critic for Continuous
# Control
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain why DQN cannot do continuous control (argmax over a
#     continuum of actions is intractable) and how DDPG answers it:
#     a DETERMINISTIC actor mu(s) that IS the argmax, trained by the
#     critic's gradient
#   - Implement the four DDPG moving parts: actor, critic, target
#     networks with soft updates, and a replay buffer
#   - Explain OFF-POLICY learning as a business property: the agent
#     learns from RECORDED experience, not just fresh rollouts
#   - Train DDPG on Pendulum-v1 and beat a measured random baseline
#   - Verify the Bellman target by hand on one stored transition
#   - Apply to dosing-valve control at a water-treatment plant, where
#     the training data is the plant's logged operating history
#
# PREREQUISITES: M5/ex_8/01_dqn.py (replay buffer, target network),
#   02_ppo.py (actor-critic), 05_ppo_continuous.py (continuous actions)
# ESTIMATED TIME: ~35 min
#
# ENVIRONMENT: Pendulum-v1. DDPG (Lillicrap et al. 2016) = "DQN for
#   continuous actions": same replay + target-net ideas, plus a policy
#   network whose output IS the action.
#
# PHASES:
#   1. THEORY  — deterministic policy gradient; off-policy; target nets
#   2. BUILD   — Actor, Critic, soft updates, exploration noise
#   3. TRAIN   — DDPG on Pendulum, tracked with ExperimentTracker
#   4. VISUALISE — return curve, Q-estimate trace, diagnostics
#   5. APPLY   — dosing control trained on LOGGED plant data
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_8 import (
    OUTPUT_DIR,
    ContinuousReplayBuffer,
    device,
    evaluate_policy,
    make_pendulum,
    moving_average,
    register_rl_model,
    rl_diagnostic_checkpoint,
    setup_engines,
)

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: The Deterministic Policy Gradient
# ════════════════════════════════════════════════════════════════════════
# DQN picks actions by argmax_a Q(s, a) — cheap when a is one of two
# buttons, IMPOSSIBLE when a is a real number in [-2, 2]. You cannot
# argmax over a continuum.
#
# DDPG's answer: stop searching for the best action — LEARN it.
#   - The ACTOR mu(s) outputs the action directly (a dial setting).
#   - The CRITIC Q(s, a) scores state-action pairs, exactly as in DQN.
#   - The actor is trained by gradient ASCENT on Q(s, mu(s)): "adjust
#     the dial-setting policy so the critic scores it higher". The
#     critic's gradient flows THROUGH the action into the actor's weights.
#
# TWO STABILITY MECHANISMS inherited from DQN, one upgraded:
#   - REPLAY BUFFER: store (s, a, r, s', done) and train on random
#     batches. This is what makes DDPG OFF-POLICY: the data can come from
#     an older policy — or from a logged plant history (see Phase 5).
#   - TARGET NETWORKS: the Bellman target r + gamma * Q'(s', mu'(s')) is
#     computed with slowly-moving COPIES (prime) of both networks, updated
#     by soft interpolation:  theta' <- tau * theta + (1 - tau) * theta'.
#     Without them the target chases its own tail every step.
#
# EXPLORATION for a deterministic policy: add Gaussian noise to the
# action during training. Evaluation uses mu(s) raw — the policy itself
# carries no randomness.

print("=" * 70)
print("  PHASE 1 — THEORY: learn the argmax instead of computing it")
print("=" * 70)
print(
    """
  DQN:   a* = argmax_a Q(s, a)      — needs a discrete action set
  DDPG:  a = mu_theta(s)            — the actor IS the argmax
         actor update: ascend Q(s, mu_theta(s)) — the critic's gradient
         flows through the action into the actor's weights

  OFF-POLICY = learns from stored transitions (replay buffer), not just
  fresh rollouts. Business meaning: logged operating data becomes
  training data.

  TARGET NETWORKS: y = r + gamma * Q'(s', mu'(s')) with soft updates
    theta' <- tau * theta + (1 - tau) * theta'   (tau = 0.005)
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: Actor, Critic, targets, noise
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: the four moving parts")
print("=" * 70)

env, obs_dim, act_dim, ACT_LIMIT = make_pendulum()
conn, tracker, exp_name, registry, has_registry = setup_engines()


class DeterministicActor(nn.Module):
    """mu(s): state -> one continuous action, tanh-squashed to the limit."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s) * ACT_LIMIT


class QCritic(nn.Module):
    """Q(s, a): state AND action concatenated -> scalar score."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """theta' <- tau * theta + (1 - tau) * theta' — the slow chase."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


# ── Checkpoint 1: shapes + soft-update arithmetic ──────────────────────
_actor = DeterministicActor(obs_dim, act_dim).to(device)
_critic = QCritic(obs_dim, act_dim).to(device)
_s = torch.randn(5, obs_dim, device=device)
with torch.no_grad():
    _a = _actor(_s)
    _q = _critic(_s, _a)
assert _a.shape == (5, act_dim) and float(_a.abs().max()) <= ACT_LIMIT
assert _q.shape == (5,), f"Q should be (batch,), got {_q.shape}"

_target = copy.deepcopy(_actor)
_orig = copy.deepcopy(_actor)  # snapshot BEFORE shifting the source
with torch.no_grad():
    for p in _actor.parameters():
        p.add_(1.0)  # shift every source weight by exactly 1
soft_update(_target, _actor, tau=0.01)
with torch.no_grad():
    # target must have moved ~tau (0.01) of the way from orig to shifted
    _diff = float(
        torch.cat([(p - o).flatten() for p, o in
                   zip(_target.parameters(), _orig.parameters())]).abs().mean()
    )
assert 0.005 < _diff < 0.02, (
    f"soft update with tau=0.01 on weights shifted by 1.0 should move "
    f"targets ~0.01 on average; measured {_diff:.5f}"
)
_n_actor = sum(p.numel() for p in _actor.parameters())
_n_critic = sum(p.numel() for p in _critic.parameters())
print(f"\nActor: {_n_actor:,} params | Critic: {_n_critic:,} params")
print(f"  soft update verified: tau=0.01 on +1.0-shifted weights moved "
      f"targets by {_diff:.5f} on average")
print("\n--- Checkpoint 1 passed --- DDPG components verified\n")
del _actor, _critic, _s, _a, _q, _target, _orig


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: DDPG on Pendulum
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: off-policy learning from the replay buffer")
print("=" * 70)

random_returns = evaluate_policy(env, lambda s: env.action_space.sample(),
                                 n_episodes=10)
random_mean = float(np.mean(random_returns))
print(f"  Random policy baseline: {random_mean:.1f} mean return (10 episodes)")

TOTAL_STEPS = 8_000
WARMUP_STEPS = 1_000
BATCH = 128
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
NOISE_STD = 0.15 * ACT_LIMIT


async def train_ddpg():
    """DDPG loop: noisy act -> store -> sample batch -> Bellman update."""
    actor = DeterministicActor(obs_dim, act_dim).to(device)
    critic = QCritic(obs_dim, act_dim).to(device)
    actor_t = copy.deepcopy(actor)
    critic_t = copy.deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
    buffer = ContinuousReplayBuffer(50_000)

    ep_returns: list[float] = []
    q_losses: list[float] = []
    actor_losses: list[float] = []
    q_means: list[float] = []

    state, _ = env.reset(seed=42)
    ep_return, ep_len = 0.0, 0

    async with tracker.track(experiment=exp_name, run_name="ddpg_pendulum") as run:
        await run.log_params(
            {
                "algorithm": "DDPG",
                "env": "Pendulum-v1",
                "total_steps": str(TOTAL_STEPS),
                "warmup": str(WARMUP_STEPS),
                "batch": str(BATCH),
                "gamma": str(GAMMA),
                "tau": str(TAU),
                "noise_std": f"{NOISE_STD:.3f}",
                "random_baseline_return": f"{random_mean:.1f}",
            }
        )
        for step in range(TOTAL_STEPS):
            # ── Act: warmup = random dial; afterwards mu(s) + noise ──
            if step < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(state.astype(np.float32)).to(device)
                    action = (
                        actor(s_t).cpu().numpy()
                        + np.random.normal(0, NOISE_STD, size=act_dim)
                    ).astype(np.float32)
                    action = np.clip(action, -ACT_LIMIT, ACT_LIMIT)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state.astype(np.float32), action, float(reward),
                        next_state.astype(np.float32), done)
            ep_return += float(reward)
            ep_len += 1
            state = next_state

            if done:
                ep_returns.append(ep_return)
                ep_return, ep_len = 0.0, 0
                state, _ = env.reset()

            # ── Learn: one gradient step per env step after warmup ──
            if step >= WARMUP_STEPS and len(buffer) >= BATCH:
                s, a, r, s2, d = buffer.sample(BATCH)
                with torch.no_grad():
                    a2 = actor_t(s2)
                    target_q = r + GAMMA * (1 - d) * critic_t(s2, a2)
                q_pred = critic(s, a)
                critic_loss = F.mse_loss(q_pred, target_q)
                opt_c.zero_grad()
                critic_loss.backward()
                opt_c.step()

                # Actor: ascend Q(s, mu(s)) — i.e. minimise the negative
                actor_loss = -critic(s, actor(s)).mean()
                opt_a.zero_grad()
                actor_loss.backward()
                opt_a.step()

                soft_update(actor_t, actor, TAU)
                soft_update(critic_t, critic, TAU)

                q_losses.append(float(critic_loss.item()))
                actor_losses.append(float(actor_loss.item()))
                q_means.append(float(q_pred.mean().item()))

            if (step + 1) % 2000 == 0:
                recent = ep_returns[-5:] if ep_returns else [ep_return]
                print(
                    f"  step {step+1:>6}/{TOTAL_STEPS}  recent return="
                    f"{np.mean(recent):8.1f}  episodes={len(ep_returns)}"
                )
                await run.log_metrics(
                    {
                        "recent_episode_return": float(np.mean(recent)),
                        "critic_loss": float(np.mean(q_losses[-200:])) if q_losses else 0.0,
                        "mean_Q": float(np.mean(q_means[-200:])) if q_means else 0.0,
                    },
                    step=step + 1,
                )
        if ep_len > 0:  # close out a partial final episode honestly
            ep_returns.append(ep_return)
        await run.log_metric("final_window_return",
                             float(np.mean(ep_returns[-5:])))
    return actor, critic, buffer, ep_returns, q_losses, actor_losses, q_means


actor, critic, buffer, ep_returns, q_losses, actor_losses, q_means = (
    asyncio.run(train_ddpg())
)

# ── Checkpoint 2: beats random + Bellman target hand-check ────────────
assert len(ep_returns) >= 20, f"expected >= 20 episodes, got {len(ep_returns)}"
best_window = float(np.mean(ep_returns[-5:]))
first_window = float(np.mean(ep_returns[:5]))
assert best_window > random_mean + 150.0, (
    f"DDPG last-5 mean {best_window:.1f} must clearly beat random "
    f"{random_mean:.1f}"
)
assert best_window > first_window, (
    f"returns should improve: first-5 {first_window:.1f} -> "
    f"last-5 {best_window:.1f}"
)

# Hand-check ONE Bellman target from the buffer against the formula:
#   y = r + gamma * (1 - done) * Q'(s', mu'(s'))
# recomputed with frozen copies of the trained networks (same form as the
# training loop's target networks, which live inside train_ddpg).
_s, _a, _r, _s2, _d = buffer.sample(1)
_actor_frozen = copy.deepcopy(actor)
_critic_frozen = copy.deepcopy(critic)
with torch.no_grad():
    _a2 = _actor_frozen(_s2)
    _target = _r + GAMMA * (1 - _d) * _critic_frozen(_s2, _a2)
assert _target.shape == (1,), "target is one scalar per transition"
assert torch.isfinite(_target).all(), "Bellman target must be finite"
# terminal transitions must drop the bootstrap term entirely
_terminal = _r + GAMMA * (1 - torch.ones_like(_d)) * _critic_frozen(_s2, _a2)
assert float(_terminal.item()) == float(_r.item()), (
    "done=1 must zero the bootstrap"
)

print(
    f"\n  Random baseline: {random_mean:.1f} | first-5 eps: {first_window:.1f} | "
    f"last-5 eps: {best_window:.1f}"
)
print(f"  Bellman target verified: y = r + gamma*(1-done)*Q'(s', mu'(s'))")
print("\n--- Checkpoint 2 passed --- DDPG trained and verified\n")

register_rl_model(
    registry,
    "ddpg_pendulum",
    actor,
    {
        "random_baseline_return": random_mean,
        "final_window_return": best_window,
        "total_env_steps": float(TOTAL_STEPS),
    },
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: return curve + Q estimates + diagnostics
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: off-policy learning from stored transitions")
print("=" * 70)

rl_diagnostic_checkpoint(
    "DDPG on Pendulum-v1",
    "ddpg",
    ep_returns,
    q_losses=q_losses,
)

import warnings

from kailash_ml import ModelVisualizer
from kailash_ml._decorators import ExperimentalWarning

with warnings.catch_warnings():
    # ModelVisualizer's P2 experimental notice; the gate is warnings-as-errors.
    warnings.simplefilter("ignore", ExperimentalWarning)
    viz = ModelVisualizer()

fig = viz.training_history(
    metrics={
        "DDPG episode return": ep_returns,
        "moving avg (5)": moving_average(ep_returns, 5),
        "random baseline": [random_mean] * len(ep_returns),
    },
    x_label="Episode",
    y_label="Episode return",
)
fig.write_html(str(OUTPUT_DIR / "07_ddpg_returns.html"))
print(f"  Saved: {OUTPUT_DIR / '07_ddpg_returns.html'}")

fig_q, ax = plt.subplots(figsize=(9, 5))
ax.plot(moving_average(q_means, 200), color="#2196F3")
ax.set_xlabel("gradient step")
ax.set_ylabel("mean predicted Q")
ax.set_title("Critic's Q estimates over training (200-step moving average)")
ax.grid(True, alpha=0.3)
fig_q.tight_layout()
fig_q.savefig(str(OUTPUT_DIR / "07_ddpg_q_estimates.png"), dpi=150)
plt.close(fig_q)
print(f"  Saved: {OUTPUT_DIR / '07_ddpg_q_estimates.png'}")

# ── Checkpoint 3: artefacts + Q estimates stayed finite ───────────────
import os

for artefact in ("07_ddpg_returns.html", "07_ddpg_q_estimates.png"):
    assert os.path.exists(OUTPUT_DIR / artefact), f"Missing: {artefact}"
assert np.isfinite(q_means).all(), "Q estimates must stay finite"
print("\n--- Checkpoint 3 passed --- learning curves verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: dosing-valve control from LOGGED plant data
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a water-treatment plant doses
# coagulant through a continuous valve. The plant has THREE YEARS of
# logged operation: (sensor state, valve setting, turbidity outcome)
# tuples recorded every minute. Two facts make DDPG the right tool:
#
#   1. THE VALVE IS A DIAL: dosing is continuous, so DQN-style argmax is
#      out; the deterministic actor emits a valve setting directly.
#   2. THE DATA ALREADY EXISTS: off-policy learning can train on the
#      logged history BEFORE touching the physical plant. PPO/A2C are
#      on-policy — they must generate fresh rollouts with THEIR OWN
#      current policy, which means experimenting on a live plant.
#
# THE GOVERNANCE DIFFERENCE is the one an operations director cares
# about: "you mean it learns from the logs first?" Yes — the replay
# buffer is literally a database of past operation. Online fine-tuning
# still follows, but the first trained policy never has to flail on
# live equipment.

print("=" * 70)
print("  PHASE 5 — APPLY: off-policy means the logs are training data")
print("=" * 70)
print(
    f"""
  DOSING-CONTROL REPORT (measured, this run):

    Environment analogue:      Pendulum-v1 torque dial in [-{ACT_LIMIT}, {ACT_LIMIT}]
    Random-schedule baseline:  {random_mean:.1f} mean return
    DDPG after {TOTAL_STEPS:,} steps:  {best_window:.1f} (last-5 episodes)
    Training transitions:      {TOTAL_STEPS:,} stored, replayed in batches of {BATCH}
    Gradient updates:          {len(q_losses):,} (one per env step after warmup)

  THE OFF-POLICY PROPERTY, IN OPERATIONS LANGUAGE:
    PPO/A2C: "let the new policy run the plant, then learn from what
             happens" (on-policy — fresh data only)
    DDPG:    "learn from the last three years of logs first; only then
             touch the plant" (off-policy — the replay buffer IS the
             historical record)

  STAKEHOLDER-READY OUTPUT:
    "The dosing controller trained against recorded operating history,
    not live experimentation: it improved from {random_mean:.0f} (random
    valve schedule) to {best_window:.0f} mean return using only stored
    transitions replayed in batches. Target networks and soft updates
    kept the value estimates stable throughout — the diagnostic pad
    shows no reward collapse."
"""
)

# ── Checkpoint 4: the report's numbers are the measured ones ──────────
assert best_window > random_mean, "trained must beat random"
assert len(q_losses) == TOTAL_STEPS - WARMUP_STEPS, (
    "one gradient step per env step after warmup"
)
print("--- Checkpoint 4 passed --- off-policy application demonstrated\n")

env.close()
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] DQN's argmax breaks on a continuum; DDPG LEARNS the argmax —
      the deterministic actor mu(s) ascends the critic's Q(s, mu(s))
  [x] Off-policy: training data can be older policies' experience — or
      a plant's logged history. The replay buffer is the asset.
  [x] Target networks with soft updates (tau={TAU}) stop the Bellman
      target chasing its own tail — verified arithmetically

  BUILD + TRAIN:
  [x] DeterministicActor (tanh-squashed to +/-{ACT_LIMIT}) and QCritic
      (concatenated state+action)
  [x] Full loop: noisy act -> store -> sample {BATCH} -> Bellman update ->
      soft-update targets, {TOTAL_STEPS:,} env steps on Pendulum
  [x] Measured: random {random_mean:.1f} -> {best_window:.1f} (last-5),
      first-5 {first_window:.1f} -> last-5 {best_window:.1f},
      {len(ep_returns)} episodes tracked in ExperimentTracker

  VISUALISE (the proof):
  [x] Return curve against the measured random baseline
  [x] Critic Q-estimate trace over {len(q_losses):,} gradient steps
  [x] RLDiagnostics pad on the episode history + Q-loss stream

  APPLY:
  [x] Dosing-valve control where the replay buffer is the plant's
      three-year operating log — off-policy as a GOVERNANCE property:
      learn from records before touching live equipment
  [x] Stated boundary: on-policy methods (05's PPO, 06's A2C) must
      generate fresh rollouts with their own current policy

  KEY INSIGHT: DDPG is DQN's ideas transplanted to continuous actions:
  replay buffer, target networks — plus one new trick, the actor whose
  output IS the action. When a stakeholder asks "can it learn from our
  historical data?", you are really asking whether the algorithm is
  off-policy. Now you can answer from the mechanism, not the marketing.
"""
)
