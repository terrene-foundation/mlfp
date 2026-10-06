# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 8.8 — SAC: Soft Actor-Critic (maximum-entropy,
# twin-critic, off-policy)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - State the maximum-entropy objective: maximise E[return] + alpha *
#     H(pi) — "get the reward AND keep your options open"
#   - Explain the TWO failure fixes SAC makes to DDPG: stochastic actor
#     (principled exploration) and twin critics with min (clipped
#     double-Q against overestimation)
#   - Implement the reparameterised tanh-Gaussian policy WITH the
#     change-of-variables log-prob correction
#   - Auto-tune the temperature alpha against a target entropy
#   - Train SAC on Pendulum-v1 and beat a measured random baseline
#   - Apply to energy-optimised HVAC control, where entropy is insurance
#     against premature commitment to a noisy operating point
#
# PREREQUISITES: M5/ex_8/05_ppo_continuous.py (Gaussian policy),
#   07_ddpg.py (off-policy actor-critic, replay, target nets)
# ESTIMATED TIME: ~35 min
#
# ENVIRONMENT: Pendulum-v1, 8K steps — the same budget as 07_ddpg.py so
#   the ExperimentTracker receipts compare like for like.
#
# PHASES:
#   1. THEORY  — max-entropy RL; twin critics; reparameterisation
#   2. BUILD   — TanhGaussianActor, TwinCritic, alpha auto-tune
#   3. TRAIN   — SAC on Pendulum, tracked with ExperimentTracker
#   4. VISUALISE — return curve, alpha schedule, diagnostics
#   5. APPLY   — energy-optimised HVAC with entropy as insurance
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch

# Deterministic seeds — RL outcomes must reproduce run to run, or the
# checkpoints are unverifiable and students chase noise.
SEED = 2026
np.random.seed(SEED)
torch.manual_seed(SEED)

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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
from shared.mlfp05 import create_visualizer

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Maximum-Entropy RL and SAC's Two Fixes
# ════════════════════════════════════════════════════════════════════════
# DDPG's actor is DETERMINISTIC: one action per state. Two failure modes
# follow:
#   1. BRITTLE Q OVERESTIMATION: the critic is trained on its own max —
#      small errors compound into optimistic Q values, and the actor
#      climbs toward the optimism, not the reward.
#   2. NO PRINCIPLED EXPLORATION: exploration is bolted-on noise, not
#      part of the objective.
#
# SAC (Haarnoja et al. 2018) changes the OBJECTIVE, not just the
# machinery:
#     maximise  E[ sum_t r_t ]  +  alpha * H(pi)
#     "get the reward AND keep the policy as random as you can afford"
#
#   - THE STOCHASTIC ACTOR is the exploration: a tanh-Gaussian policy
#     whose spread is learned per state. Because entropy is IN the
#     objective, exploration is not a hack — it's what's being optimised.
#   - TWIN CRITICS (clipped double-Q): train two independent Q networks,
#     and use the MINIMUM in the Bellman target. Overestimates rarely
#     survive two independent estimates; the min is the pessimist's
#     price, and pessimism is what you want from a value estimate.
#   - THE TEMPERATURE alpha is auto-tuned: alpha rises when entropy runs
#     below the target (more exploration) and falls when above. The
#     target entropy for Pendulum's 1-D action is -1 (the negative
#     action dimension — the convention from the paper).
#
# REPARAMETERISATION: to train the actor through the sampled action, we
# write a = tanh(mu + std * eps) with eps ~ N(0, I) — the randomness is
# an INPUT, so the path mu -> a is differentiable. The tanh squash
# changes the density, so log_prob needs the change-of-variables
# correction:  log pi(a) = log N(u) - sum log(1 - tanh(u)^2).

print("=" * 70)
print("  PHASE 1 — THEORY: reward + entropy, twin critics, reparameterise")
print("=" * 70)
print(
    """
  SAC objective:  maximise E[return] + alpha * H(pi)
    "get the reward AND keep your options open"

  Fix 1 — STOCHASTIC ACTOR: tanh-Gaussian; the spread is learned and
          the entropy term pays for keeping it honest.
  Fix 2 — TWIN CRITICS: Bellman target uses min(Q1', Q2') — independent
          overestimates rarely coincide; the min is the pessimist's price.
  Temperature alpha: auto-tuned to hold entropy at the target (-act_dim).

  Reparameterisation: a = tanh(mu + std * eps), eps ~ N(0, I)
    log pi(a) = log N(u) - sum log(1 - tanh(u)^2)   (tanh correction)
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: TanhGaussianActor + TwinCritic + auto-alpha
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: the stochastic actor and the pessimist twins")
print("=" * 70)

env, obs_dim, act_dim, ACT_LIMIT = make_pendulum()
conn, tracker, exp_name, registry, has_registry = setup_engines()

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class TanhGaussianActor(nn.Module):
    """Stochastic policy: mu(s), log_std(s) -> tanh-squashed Gaussian.

    sample() returns (action, log_prob) with the tanh change-of-variables
    correction; act() is the deterministic EVALUATION policy (tanh(mu)).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(s)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return self.mu_head(h), log_std

    def sample(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised sample + corrected log-probability."""
        mu, log_std = self.forward(s)
        dist = Normal(mu, log_std.exp())
        # TODO: REPARAMETERISED sample (differentiable through mu and std)
        # Hint: dist.rsample(), NOT dist.sample()
        u = ____
        a = torch.tanh(u)
        # TODO: log-prob with the tanh change-of-variables correction:
        #   Normal log_prob summed over action dims, MINUS the tanh
        #   compression term sum(log(1 - tanh(u)^2 + 1e-6))
        log_prob = ____
        return a * ACT_LIMIT, log_prob

    def act(self, state: np.ndarray) -> np.ndarray:
        """Deterministic evaluation: tanh(mu), no sampling."""
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).to(device)
            mu, _ = self.forward(s)
            # TODO: tanh-squash the mean and scale to the action limit
            return ____


class TwinCritic(nn.Module):
    """Two independent Q networks; the Bellman target uses their MIN."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()

        def _q() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(obs_dim + act_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        self.q1 = _q()
        self.q2 = _q()

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: concatenate state+action, run BOTH Q networks, return both
        #   scalars squeezed to (batch,)
        sa = ____
        return ____


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """theta' <- tau * theta + (1 - tau) * theta' (same as 07_ddpg)."""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.mul_(1 - tau).add_(sp, alpha=tau)


# ── Checkpoint 1: shapes, tanh correction, twin independence ──────────
_actor = TanhGaussianActor(obs_dim, act_dim).to(device)
_critic = TwinCritic(obs_dim, act_dim).to(device)
_s = torch.randn(8, obs_dim, device=device)
with torch.no_grad():
    _a, _lp = _actor.sample(_s)
    _q1, _q2 = _critic(_s, _a)
assert _a.shape == (8, act_dim) and float(_a.abs().max()) <= ACT_LIMIT
assert _lp.shape == (8,), f"log_prob should be (batch,), got {_lp.shape}"
assert torch.isfinite(_lp).all(), "log_prob must be finite"

# Tanh correction verified: for a DEGENERATE (zero-std) policy the
# correction is exact — a delta mass at tanh(mu) has no density to
# correct, so we check the ALGEBRA on a known case instead: u = 0 gives
# tanh(0) = 0 and correction log(1 - 0) = 0, so log_prob equals the
# Normal log_prob at the mean.
with torch.no_grad():
    _mu = torch.zeros(1, act_dim, device=device)
    _ls = torch.zeros(1, act_dim, device=device)
    _d0 = Normal(_mu, _ls.exp())
    _u0 = torch.zeros(1, act_dim, device=device)
    _raw = _d0.log_prob(_u0).sum(-1)
    _corr = _raw - torch.log(1 - torch.tanh(_u0).pow(2) + 1e-6).sum(-1)
assert abs(float(_corr - _raw)) < 1e-5, "at u=0 the tanh correction is zero"
# and at a nonzero u the correction is POSITIVE (tanh compresses density)
_u1 = torch.full((1, act_dim), 0.8, device=device)
_corr_term = -torch.log(1 - torch.tanh(_u1).pow(2) + 1e-6).sum(-1)
assert float(_corr_term) > 0, "tanh correction must raise the log-prob"

# Twin critics must start INDEPENDENT (different random init)
assert not torch.allclose(_q1, _q2), "twin critics must be independent"
_n_actor = sum(p.numel() for p in _actor.parameters())
_n_critic = sum(p.numel() for p in _critic.parameters())
print(f"\nActor: {_n_actor:,} params | TwinCritic: {_n_critic:,} params")
print(f"  tanh correction: zero at u=0, +{float(_corr_term):.3f} at u=0.8")
print(f"  twin critics independent at init: max |Q1-Q2| = "
      f"{float((_q1 - _q2).abs().max()):.4f}")
print("\n--- Checkpoint 1 passed --- SAC components verified\n")
del _actor, _critic, _s, _a, _lp, _q1, _q2


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: SAC on Pendulum
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: maximum-entropy off-policy learning")
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
SAC_LR = 3e-4
TARGET_ENTROPY = -float(act_dim)  # paper convention: -dim(A)


async def train_sac():
    """SAC loop: stochastic act -> store -> twin-critic Bellman update."""
    actor = TanhGaussianActor(obs_dim, act_dim).to(device)
    critic = TwinCritic(obs_dim, act_dim).to(device)
    critic_t = copy.deepcopy(critic)
    opt_a = torch.optim.Adam(actor.parameters(), lr=SAC_LR)
    opt_c = torch.optim.Adam(critic.parameters(), lr=SAC_LR)
    # Auto-tuned temperature: alpha = exp(log_alpha)
    log_alpha = torch.zeros(1, device=device, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=SAC_LR)
    buffer = ContinuousReplayBuffer(50_000)

    ep_returns: list[float] = []
    q_losses: list[float] = []
    actor_losses: list[float] = []
    alpha_hist: list[float] = []
    entropy_hist: list[float] = []

    state, _ = env.reset(seed=42)
    ep_return = 0.0

    async with tracker.track(experiment=exp_name, run_name="sac_pendulum") as run:
        await run.log_params(
            {
                "algorithm": "SAC",
                "env": "Pendulum-v1",
                "total_steps": str(TOTAL_STEPS),
                "batch": str(BATCH),
                "gamma": str(GAMMA),
                "tau": str(TAU),
                "lr": str(SAC_LR),
                "target_entropy": str(TARGET_ENTROPY),
                "random_baseline_return": f"{random_mean:.1f}",
            }
        )
        for step in range(TOTAL_STEPS):
            # ── Act: warmup random; afterwards the stochastic policy ──
            if step < WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_t = torch.from_numpy(state.astype(np.float32)).to(device)
                    action, _ = actor.sample(s_t.unsqueeze(0))
                    action = action.squeeze(0).cpu().numpy().astype(np.float32)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state.astype(np.float32), action, float(reward),
                        next_state.astype(np.float32), done)
            ep_return += float(reward)
            state = next_state
            if done:
                ep_returns.append(ep_return)
                ep_return = 0.0
                state, _ = env.reset()

            # ── Learn ────────────────────────────────────────────────
            if step >= WARMUP_STEPS and len(buffer) >= BATCH:
                s, a, r, s2, d = buffer.sample(BATCH)
                alpha = log_alpha.exp().detach()

                # Critic: twin Bellman target with entropy bonus
                with torch.no_grad():
                    a2, logp2 = actor.sample(s2)
                    q1_t, q2_t = critic_t(s2, a2)
                    # TODO: soft Bellman target — r + gamma*(1-done) *
                    #   (MIN of the twin target critics MINUS alpha*logp2)
                    # Hint: torch.min(q1_t, q2_t) - alpha.squeeze() * logp2
                    target_q = ____
                q1, q2 = critic(s, a)
                # TODO: critic loss — MSE of EACH twin against the shared
                #   target, summed
                critic_loss = ____
                opt_c.zero_grad()
                critic_loss.backward()
                opt_c.step()

                # Actor: maximise min(Q1, Q2) - alpha * log_prob
                a_new, logp_new = actor.sample(s)
                q1_new, q2_new = critic(s, a_new)
                # TODO: actor loss = mean(alpha * log_prob - min(Q1, Q2))
                #   (we MINIMISE the negative of the soft Q objective)
                actor_loss = ____
                opt_a.zero_grad()
                actor_loss.backward()
                opt_a.step()

                # Temperature: steer entropy toward the target
                # TODO: alpha loss = -log_alpha * (logp_detached +
                #   TARGET_ENTROPY), meaned — rises when entropy runs low
                alpha_loss = ____
                opt_alpha.zero_grad()
                alpha_loss.backward()
                opt_alpha.step()

                # TODO: soft-update the TARGET twin critic at rate TAU
                ____

                q_losses.append(float(critic_loss.item()))
                actor_losses.append(float(actor_loss.item()))
                alpha_hist.append(float(log_alpha.exp().item()))
                entropy_hist.append(float(-logp_new.mean().item()))

            if (step + 1) % 2000 == 0:
                recent = ep_returns[-5:] if ep_returns else [ep_return]
                print(
                    f"  step {step+1:>6}/{TOTAL_STEPS}  recent return="
                    f"{np.mean(recent):8.1f}  alpha={alpha_hist[-1]:.3f}  "
                    f"episodes={len(ep_returns)}"
                )
                await run.log_metrics(
                    {
                        "recent_episode_return": float(np.mean(recent)),
                        "critic_loss": float(np.mean(q_losses[-200:])) if q_losses else 0.0,
                        "alpha": alpha_hist[-1] if alpha_hist else 1.0,
                        "policy_entropy": entropy_hist[-1] if entropy_hist else 0.0,
                    },
                    step=step + 1,
                )
        if ep_return != 0.0:
            ep_returns.append(ep_return)
        await run.log_metric("final_window_return",
                             float(np.mean(ep_returns[-5:])))
    return actor, critic, ep_returns, q_losses, actor_losses, alpha_hist, entropy_hist


actor, critic, ep_returns, q_losses, actor_losses, alpha_hist, entropy_hist = (
    asyncio.run(train_sac())
)

# ── Checkpoint 2: beats random; alpha stayed finite and positive ───────
assert len(ep_returns) >= 20, f"expected >= 20 episodes, got {len(ep_returns)}"
best_window = float(np.mean(ep_returns[-5:]))
first_window = float(np.mean(ep_returns[:5]))
assert best_window > random_mean + 150.0, (
    f"SAC last-5 mean {best_window:.1f} must clearly beat random "
    f"{random_mean:.1f}"
)
assert best_window > first_window, (
    f"returns should improve: first-5 {first_window:.1f} -> last-5 {best_window:.1f}"
)
assert all(0.0 < al < 50.0 for al in alpha_hist), "alpha must stay positive/finite"

print(
    f"\n  Random baseline: {random_mean:.1f} | first-5 eps: {first_window:.1f} | "
    f"last-5 eps: {best_window:.1f}"
)
print(
    f"  Temperature schedule: alpha {alpha_hist[0]:.3f} -> {alpha_hist[-1]:.3f} "
    f"(auto-tuned toward target entropy {TARGET_ENTROPY})"
)
print("  Same 8K-step budget as 07_ddpg.py — compare the receipts in "
      "ExperimentTracker.")
print("\n--- Checkpoint 2 passed --- SAC trained and verified\n")

register_rl_model(
    registry,
    "sac_pendulum",
    actor,
    {
        "random_baseline_return": random_mean,
        "final_window_return": best_window,
        "final_alpha": alpha_hist[-1],
        "total_env_steps": float(TOTAL_STEPS),
    },
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: return curve, alpha schedule, diagnostics
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: entropy-managed exploration")
print("=" * 70)

rl_diagnostic_checkpoint(
    "SAC on Pendulum-v1",
    "sac",
    ep_returns,
    q_losses=q_losses,
)

import warnings

from kailash_ml import ModelVisualizer
from kailash_ml._decorators import ExperimentalWarning

with warnings.catch_warnings():
    # ModelVisualizer's P2 experimental notice; the gate is warnings-as-errors.
    warnings.simplefilter("ignore", ExperimentalWarning)
    viz = create_visualizer()

fig = viz.training_history(
    metrics={
        "SAC episode return": ep_returns,
        "moving avg (5)": moving_average(ep_returns, 5),
        "random baseline": [random_mean] * len(ep_returns),
    },
    x_label="Episode",
    y_label="Episode return",
)
fig.write_html(str(OUTPUT_DIR / "08_sac_returns.html"))
print(f"  Saved: {OUTPUT_DIR / '08_sac_returns.html'}")

fig_alpha, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(moving_average(alpha_hist, 200), color="#9C27B0")
ax1.set_ylabel("alpha (entropy temperature)")
ax1.set_title("Auto-tuned temperature: rising = more exploration paid for")
ax1.grid(True, alpha=0.3)
ax2.plot(moving_average(entropy_hist, 200), color="#2196F3")
ax2.axhline(TARGET_ENTROPY, color="gray", linestyle="--",
            label=f"target entropy ({TARGET_ENTROPY})")
ax2.set_ylabel("policy entropy (nats)")
ax2.set_xlabel("gradient step")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
fig_alpha.tight_layout()
fig_alpha.savefig(str(OUTPUT_DIR / "08_sac_alpha_entropy.png"), dpi=150)
plt.close(fig_alpha)
print(f"  Saved: {OUTPUT_DIR / '08_sac_alpha_entropy.png'}")

# ── Checkpoint 3: artefacts + entropy tracked toward its target ───────
import os

for artefact in ("08_sac_returns.html", "08_sac_alpha_entropy.png"):
    assert os.path.exists(OUTPUT_DIR / artefact), f"Missing: {artefact}"
assert np.isfinite(entropy_hist).all(), "entropy trace must be finite"
print("\n--- Checkpoint 3 passed --- learning curves verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: energy-optimised HVAC with entropy as insurance
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a commercial building's HVAC
# plant sets a continuous chilled-water valve. Energy is paid per
# minute; comfort complaints arrive per degree-hour of drift. Two
# properties make SAC the right tool:
#
#   1. OFF-POLICY (as in 07): the first training pass runs against the
#      building's logged telemetry, not on the live plant.
#   2. ENTROPY AS INSURANCE: the building's sensors are noisy and the
#      load pattern shifts seasonally. A deterministic policy can commit
#      to a brittle operating point that is optimal ONLY under the
#      training distribution. SAC's objective pays the policy to keep
#      probability mass on near-optimal alternatives — measured here as
#      the entropy trace and the auto-tuned alpha.
#
# THE GOVERNANCE READING: alpha is the audit dial. If alpha collapses to
# ~0, the policy has stopped hedging; if it stays high, the environment
# has not been learned confidently. Either reading is actionable before
# deployment.

print("=" * 70)
print("  PHASE 5 — APPLY: entropy as operational insurance")
print("=" * 70)
print(
    f"""
  HVAC-VALVE REPORT (measured, this run):

    Environment analogue:      Pendulum-v1 torque dial in [-{ACT_LIMIT}, {ACT_LIMIT}]
    Random-schedule baseline:  {random_mean:.1f} mean return
    SAC after {TOTAL_STEPS:,} steps:   {best_window:.1f} (last-5 episodes)
    Temperature alpha:         {alpha_hist[0]:.3f} -> {alpha_hist[-1]:.3f} (auto-tuned)
    Policy entropy (last):     {entropy_hist[-1]:.3f} nats vs target {TARGET_ENTROPY}

  THE INSURANCE READING:
    alpha starts at 1.0 and is STEERED by the entropy target — rising
    alpha means the optimiser is paying for exploration because the
    policy ran too confident; falling alpha means reward is being
    banked. The final alpha and entropy are deployment-gate numbers,
    not diagnostics to ignore.

  STAKEHOLDER-READY OUTPUT:
    "The valve controller trained off-policy on logged telemetry and
    improved from {random_mean:.0f} (random schedule) to
    {best_window:.0f} mean return. Unlike the deterministic controller,
    this one is paid to keep fallback options open: its entropy
    temperature was auto-tuned to {alpha_hist[-1]:.2f} against a target
    of {TARGET_ENTROPY:.0f} nats, so it retains measured hedging rather
    than collapsing onto a single brittle setpoint."
"""
)

# ── Checkpoint 4: the report's numbers are the measured ones ──────────
assert best_window > random_mean
assert 0.0 < alpha_hist[-1] < 50.0
print("--- Checkpoint 4 passed --- insurance application demonstrated\n")

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
  [x] Maximum-entropy objective: E[return] + alpha * H(pi) — reward AND
      options kept open
  [x] SAC's two fixes to DDPG: stochastic tanh-Gaussian actor
      (principled exploration) + twin critics with min (pessimism
      against Q overestimation)
  [x] Reparameterisation with the tanh change-of-variables correction —
      verified: zero at u=0, positive for |u| > 0

  BUILD + TRAIN:
  [x] TanhGaussianActor ({_n_actor:,} params) + independent TwinCritic
      ({_n_critic:,} params)
  [x] Auto-tuned temperature: alpha {alpha_hist[0]:.3f} ->
      {alpha_hist[-1]:.3f} steering entropy toward target {TARGET_ENTROPY}
  [x] {TOTAL_STEPS:,} env steps on Pendulum: random {random_mean:.1f} ->
      SAC {best_window:.1f} (last-5), tracked in ExperimentTracker
      alongside 07's DDPG receipts at the SAME budget

  VISUALISE (the proof):
  [x] Return curve against the measured random baseline
  [x] Alpha schedule + entropy-vs-target trace over
      {len(q_losses):,} gradient steps
  [x] RLDiagnostics pad on the episode history + Q-loss stream

  APPLY:
  [x] HVAC valve control: off-policy training on logged telemetry PLUS
      entropy as insurance against brittle setpoints under noisy sensors
  [x] Alpha as a deployment-gate number: collapse = stopped hedging;
      high = environment not yet learned

  KEY INSIGHT: DDPG, SAC, A2C and PPO are the same skeleton — collect,
  estimate advantage or Q, update actor and critic — with different
  answers to three questions: fresh data or replayed? clipped or
  unclipped? deterministic or stochastic? You can now read any RL
  algorithm as a point in that design space, and defend the choice for
  a given plant, simulator, or market from first principles.
"""
)
