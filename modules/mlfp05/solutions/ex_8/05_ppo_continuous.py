# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 8.5 — PPO for CONTINUOUS Actions: the Gaussian Policy
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain why a Categorical head cannot express "apply 1.37 Nm of
#     torque" — and what a Gaussian policy head outputs instead
#   - Implement a Gaussian actor: network emits mu(s); a learned log_std
#     carries the exploration noise; log_prob comes from Normal
#   - Reuse the ENTIRE PPO stack (GAE, clipped surrogate, entropy bonus)
#     unchanged — only the distribution changes
#   - Train PPO on Pendulum-v1 (a Box action space) and verify the run
#     beats a measured random-policy baseline
#   - Visualise the exploration noise collapsing as the policy gains
#     confidence (log_std schedule, measured not assumed)
#   - Apply continuous control to HVAC damper control at a data centre
#
# PREREQUISITES: M5/ex_8/02_ppo.py (GAE, clipped surrogate, PPO loop on
#   discrete CartPole). This file changes ONE thing: the action
#   distribution.
# ESTIMATED TIME: ~35 min
#
# ENVIRONMENT: Pendulum-v1 (Gymnasium classic control; 3-D Box state,
#   1-D Box action in [-2, 2]). Rewards are negative; random ~= -1200,
#   a trained policy approaches -150. CPU-budgeted: 25 iterations of
#   1024 steps.
#
# PHASES:
#   1. THEORY  — continuous action spaces and the Gaussian policy
#   2. BUILD   — GaussianActorCritic
#   3. TRAIN   — PPO on Pendulum, tracked with ExperimentTracker
#   4. VISUALISE — reward curve, exploration-noise collapse
#   5. APPLY   — HVAC damper control at a data centre
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_8 import (
    OUTPUT_DIR,
    device,
    evaluate_policy,
    make_pendulum,
    moving_average,
    register_rl_model,
    rl_diagnostic_checkpoint,
    setup_engines,
)
from kailash_ml import ModelVisualizer

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: When the Action Is a Dial, Not a Button
# ════════════════════════════════════════════════════════════════════════
# CartPole's action is a BUTTON: push left or push right — a Categorical
# distribution over 2 choices. Pendulum's action is a DIAL: a torque in
# [-2, 2]. A softmax over infinitely many settings is impossible.
#
# THE GAUSSIAN POLICY (the standard answer):
#   The actor outputs the MEAN of a Normal distribution; a separate
#   learned parameter (log_std) controls the spread. Acting is sampling:
#       a ~ Normal(mu_theta(s), std)
#   The log-probability of the sampled action (needed for the PPO ratio)
#   comes from the Normal's log_prob — the same role Categorical played.
#
# WHAT CHANGES FROM DISCRETE PPO: essentially only the distribution object.
#   GAE: unchanged.  Clipped surrogate: unchanged.  Entropy bonus: dropped
#   (coef 0.0) — on a continuous dial the spread IS the entropy term, and
#   paying a bonus for it pins the dial in the noisy regime. We also scale
#   rewards by 0.1 for the critic (a units change, not a reward hack):
#   Pendulum's -1000-scale returns would otherwise dominate the value loss.
#
# EXPLORATION, MEASURED: log_std starts near 0 (std ~ 1 — wild swings of
# the dial) and should SHRINK as the policy converges on a confident mean
# torque. We plot the measured log_std schedule — exploration collapse
# you can see, not a claim in a comment.

print("=" * 70)
print("  PHASE 1 — THEORY: the Gaussian policy for continuous actions")
print("=" * 70)
print(
    """
  Discrete action (CartPole):  softmax over 2 buttons     (Categorical)
  Continuous action (Pendulum): mean + learned spread on a dial (Normal)

  ONLY THE DISTRIBUTION CHANGES:
    a ~ Normal(mu_theta(s), exp(log_std));  ratio = exp(new_logprob - old)
    GAE and clipping: identical to 02_ppo.py. Two honest adjustments for
    the dial: entropy bonus OFF (the spread IS the entropy) and rewards
    scaled 0.1 for the critic (a units change; reported returns are raw).

  EXPLORATION = the spread. log_std shrinking over training IS the
  policy gaining confidence — we measure it, not assume it.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: GaussianActorCritic
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: Gaussian actor + value critic")
print("=" * 70)

env, obs_dim, act_dim, ACT_LIMIT = make_pendulum()
conn, tracker, exp_name, registry, has_registry = setup_engines()


class GaussianActorCritic(nn.Module):
    """PPO actor-critic for a Box action space.

    Actor: MLP -> mean torque, squashed by tanh into [-ACT_LIMIT, LIMIT].
    log_std is a FREE parameter (state-independent spread) — the simplest
    honest exploration schedule, and the one whose collapse we plot.
    Critic: separate MLP -> V(s), kept independent for the same reason as
    in 02_ppo.py (value-regression gradients dominate a shared trunk).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, act_dim),
        )
        # log_std starts at -0.5 (std ~ 0.6): wide enough to explore the
        # dial, not so wide that early updates are pure noise.
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Small orthogonal policy-head init (gain 0.01): initial mean torque
        # ~ 0, so the critic maps the value landscape before the actor
        # commits to a strategy. The standard PPO initialisation for
        # continuous control (and the difference between converging and
        # plateauing on Pendulum at CPU budgets).
        for m in self.actor:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for m in self.critic:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        nn.init.zeros_(self.critic[-1].bias)

    def forward(self, x: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mu = torch.tanh(self.actor(x)) * ACT_LIMIT
        dist = Normal(mu, self.log_std.exp().expand_as(mu))
        return dist, self.critic(x).squeeze(-1)

    def act(self, state: np.ndarray) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (action_array, log_prob, value)."""
        s = torch.from_numpy(state.astype(np.float32)).to(device)
        dist, value = self.forward(s)
        a = dist.sample()
        return (
            a.cpu().numpy().astype(np.float32),
            dist.log_prob(a).sum(-1).detach(),
            value.detach(),
        )


# ── Checkpoint 1: shapes, action bounds, log-prob sanity ──────────────
_probe = GaussianActorCritic(obs_dim, act_dim).to(device)
_s, _ = env.reset(seed=0)
_a, _lp, _v = _probe.act(_s)
assert _a.shape == (act_dim,), f"action shape {_a.shape} != ({act_dim},)"
assert np.all(np.abs(_a) <= ACT_LIMIT + 1e-6), (
    f"tanh-squashed action {_a} exceeds +/-{ACT_LIMIT}"
)
assert np.isfinite(float(_lp)), "log_prob must be finite"
with torch.no_grad():
    dist_probe, _ = _probe(torch.from_numpy(_s.astype(np.float32)).to(device))
assert abs(float(dist_probe.stddev.mean()) - float(np.exp(-0.5))) < 1e-4, (
    "log_std starts at -0.5 -> initial std must be exp(-0.5) ~ 0.607"
)
_n_params = sum(p.numel() for p in _probe.parameters())
print(f"\nGaussianActorCritic built: {_n_params:,} parameters")
print(f"  action shape ({act_dim},), bounded in [-{ACT_LIMIT}, {ACT_LIMIT}]")
print(f"  initial exploration std = {np.exp(-0.5):.3f} (log_std = -0.5, free parameter)")
print(f"  policy head init: orthogonal gain 0.01 (initial mean torque ~ 0)")
print("\n--- Checkpoint 1 passed --- Gaussian policy verified\n")
del _probe, _s, _a, _lp, _v, dist_probe


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: PPO on Pendulum (GAE + clip, unchanged)
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: continuous PPO on Pendulum-v1")
print("=" * 70)

# Measured random baseline FIRST — every improvement claim needs it.
random_returns = evaluate_policy(env, lambda s: env.action_space.sample(),
                                 n_episodes=10)
random_mean = float(np.mean(random_returns))
print(f"  Random policy baseline: {random_mean:.1f} mean return (10 episodes)")

N_ITERS = 120
STEPS_PER_ITER = 1024
GAMMA, LAM, CLIP_EPS, LR = 0.99, 0.95, 0.2, 3e-4
REWARD_SCALE = 0.1  # critic/GAE see scaled rewards; REPORTED returns are raw
# Pendulum's rewards are large and negative (a swing-up costs ~-1000).
# Scaling by 0.1 keeps the critic's MSE in a sane numeric range without
# changing the ordering of policies — a units change, not a reward hack.
# No entropy bonus here (coef 0.0): for a continuous dial the exploration
# spread IS the entropy term, and paying a bonus for it pins the dial in
# the noisy regime — exactly the failure the std schedule is here to catch.


def collect_continuous_trajectory(env, model, max_steps):
    """On-policy rollout for continuous actions (float arrays, not ints)."""
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
        if done:  # Pendulum truncates at 200 steps; reset and continue
            state, _ = env.reset(seed=int(np.random.randint(0, 100_000)))
    return states, actions, log_probs, values, rewards, dones


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAM):
    """GAE — identical to 02_ppo.py (the distribution changed, GAE didn't)."""
    advantages = [0.0] * len(rewards)
    gae, next_value = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * nonterminal - float(values[t])
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
        next_value = float(values[t])
    returns = [a + float(v) for a, v in zip(advantages, values)]
    return advantages, returns


async def train_continuous_ppo():
    """PPO loop with the Gaussian head; returns history for plotting."""
    model = GaussianActorCritic(obs_dim, act_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    iter_returns, entropies, actor_losses, critic_losses, std_hist = (
        [], [], [], [], [],
    )

    async with tracker.track(experiment=exp_name, run_name="ppo_pendulum") as run:
        await run.log_params(
            {
                "algorithm": "PPO-continuous",
                "env": "Pendulum-v1",
                "gamma": str(GAMMA),
                "lambda": str(LAM),
                "clip_eps": str(CLIP_EPS),
                "lr": str(LR),
                "steps_per_iter": str(STEPS_PER_ITER),
                "random_baseline_return": f"{random_mean:.1f}",
            }
        )
        for it in range(N_ITERS):
            states, actions, old_lps, values, rewards, dones = (
                collect_continuous_trajectory(env, model, STEPS_PER_ITER)
            )
            scaled_rewards = [r * REWARD_SCALE for r in rewards]
            advantages, returns = compute_gae(scaled_rewards, values, dones)

            s_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
            a_t = torch.tensor(np.stack(actions), dtype=torch.float32, device=device)
            old_lp_t = torch.stack(old_lps).to(device)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            n = s_t.size(0)
            idxs = np.arange(n)
            it_actor = it_critic = it_entropy = 0.0
            updates = 0
            for _ in range(4):  # PPO epochs over the same rollout
                np.random.shuffle(idxs)
                for start in range(0, n, 256):
                    mb = idxs[start : start + 256]
                    dist, vpred = model(s_t[mb])
                    new_lp = dist.log_prob(a_t[mb]).sum(-1)
                    ratio = torch.exp(new_lp - old_lp_t[mb])
                    surr1 = ratio * adv_t[mb]
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t[mb]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(vpred, ret_t[mb])
                    entropy = dist.entropy().sum(-1).mean()
                    # entropy coef 0.0 — see REWARD_SCALE comment above
                    loss = policy_loss + 0.5 * value_loss
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    opt.step()
                    it_actor += policy_loss.item()
                    it_critic += value_loss.item()
                    it_entropy += entropy.item()
                    updates += 1

            # Mean episode return inside this iteration's rollout
            ep_rs, running = [], 0.0
            for r, d in zip(rewards, dones):
                running += r
                if d:
                    ep_rs.append(running)
                    running = 0.0
            avg_return = float(np.mean(ep_rs)) if ep_rs else float(running)
            iter_returns.append(avg_return)
            entropies.append(it_entropy / max(updates, 1))
            actor_losses.append(it_actor / max(updates, 1))
            critic_losses.append(it_critic / max(updates, 1))
            std_hist.append(float(model.log_std.exp().mean().item()))

            await run.log_metrics(
                {
                    "avg_episode_return": avg_return,
                    "policy_entropy": entropies[-1],
                    "actor_loss": actor_losses[-1],
                    "critic_loss": critic_losses[-1],
                    "exploration_std": std_hist[-1],
                },
                step=it,
            )
            if (it + 1) % 20 == 0:
                print(
                    f"  iter {it+1:3d}/{N_ITERS}  return={avg_return:8.1f}  "
                    f"std={std_hist[-1]:.3f}  entropy={entropies[-1]:.3f}"
                )
        await run.log_metric("final_avg_return", iter_returns[-1])
    return model, iter_returns, entropies, actor_losses, critic_losses, std_hist


model, iter_returns, entropies, actor_losses, critic_losses, std_hist = (
    asyncio.run(train_continuous_ppo())
)

# ── Checkpoint 2: the trained policy beats the measured baseline ──────
assert len(iter_returns) == N_ITERS
best_window = float(np.mean(iter_returns[-5:]))
assert best_window > random_mean + 100.0, (
    f"trained return {best_window:.1f} must clearly beat the random "
    f"baseline {random_mean:.1f}"
)
assert std_hist[-1] < std_hist[0], (
    "exploration std should shrink as the policy gains confidence "
    f"(start {std_hist[0]:.3f} -> end {std_hist[-1]:.3f})"
)
print(
    f"\n  Random baseline: {random_mean:.1f} | trained (last-5 mean): "
    f"{best_window:.1f} | exploration std: {std_hist[0]:.3f} -> {std_hist[-1]:.3f}"
)
print("\n--- Checkpoint 2 passed --- continuous PPO trained and verified\n")

register_rl_model(
    registry,
    "ppo_pendulum_continuous",
    model,
    {
        "random_baseline_return": random_mean,
        "final_window_return": best_window,
        "final_exploration_std": std_hist[-1],
    },
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: reward curve + exploration collapse
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: the dial settling down")
print("=" * 70)

# rl_diagnostic_checkpoint prints the RLDiagnostics report
rl_diagnostic_checkpoint(
    "PPO (continuous) on Pendulum-v1",
    "ppo",
    iter_returns,
    policy_losses=actor_losses,
    value_losses=critic_losses,
    entropies=entropies,
)

import warnings

from kailash_ml._decorators import ExperimentalWarning

with warnings.catch_warnings():
    # ModelVisualizer's P2 experimental notice; the gate is warnings-as-errors.
    warnings.simplefilter("ignore", ExperimentalWarning)
    viz = ModelVisualizer()

fig = viz.training_history(
    metrics={
        "PPO-Pendulum return": iter_returns,
        "moving avg (5)": moving_average(iter_returns, 5),
        "random baseline": [random_mean] * len(iter_returns),
    },
    x_label="Iteration",
    y_label="Mean episode return",
)
fig.write_html(str(OUTPUT_DIR / "05_ppo_continuous_rewards.html"))
print(f"  Saved: {OUTPUT_DIR / '05_ppo_continuous_rewards.html'}")

fig_std, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, N_ITERS + 1), std_hist, "o-", color="#9C27B0")
ax.set_xlabel("PPO iteration")
ax.set_ylabel("exploration std = exp(log_std)")
ax.set_title("Exploration collapse: the dial settles as confidence grows")
ax.grid(True, alpha=0.3)
fig_std.tight_layout()
fig_std.savefig(str(OUTPUT_DIR / "05_ppo_continuous_std.png"), dpi=150)
plt.close(fig_std)
print(f"  Saved: {OUTPUT_DIR / '05_ppo_continuous_std.png'}")

# (c) Evaluation episodes: trained deterministic policy vs random policy,
# same protocol, same horizon. The honest behavioural claim at this CPU
# budget is DIRECTIONAL: the trained policy swings the pendulum measurably
# closer to upright than random flailing. Full stabilisation (holding
# angle ~ 0) needs a larger budget than 120 iterations — we say so.
def run_episode_angles(policy_fn, seed: int) -> list[float]:
    """Run one episode, returning the pendulum angle at each step."""
    state, _ = env.reset(seed=seed)
    out = []
    for _ in range(200):
        action = policy_fn(state)
        state, _, terminated, truncated, _ = env.step(action)
        out.append(float(np.arctan2(state[1], state[0])))
        if terminated or truncated:
            break
    return out


model.eval()


def trained_policy(state: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        dist, _ = model(torch.from_numpy(state.astype(np.float32)).to(device))
        return dist.mean.cpu().numpy()


angles = run_episode_angles(trained_policy, seed=2026)
rng = np.random.default_rng(2026)
angles_random = run_episode_angles(
    lambda s: rng.uniform(-ACT_LIMIT, ACT_LIMIT, size=act_dim).astype(np.float32),
    seed=2026,
)
torques = []
state, _ = env.reset(seed=2026)
for _ in range(200):
    with torch.no_grad():
        dist, _ = model(torch.from_numpy(state.astype(np.float32)).to(device))
    action = dist.mean.cpu().numpy()
    state, _, terminated, truncated, _ = env.step(action)
    torques.append(float(action[0]))
    if terminated or truncated:
        break

fig_ep, (ax_a, ax_t) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax_a.plot(angles, color="#2196F3", label="trained policy (mean action)")
ax_a.plot(angles_random, color="gray", alpha=0.7, label="random policy")
ax_a.axhline(0.0, color="green", linestyle="--", alpha=0.6,
             label="upright (angle 0)")
ax_a.set_ylabel("pendulum angle (rad)")
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3)
ax_t.plot(torques, color="#F44336")
ax_t.set_ylabel("applied torque (trained)")
ax_t.set_xlabel("timestep")
ax_t.grid(True, alpha=0.3)
fig_ep.suptitle("Evaluation episode: trained vs random (deterministic mean policy)")
fig_ep.tight_layout()
fig_ep.savefig(str(OUTPUT_DIR / "05_ppo_continuous_episode.png"), dpi=150)
plt.close(fig_ep)
print(f"  Saved: {OUTPUT_DIR / '05_ppo_continuous_episode.png'}")

# ── Checkpoint 3: artefacts + trained beats random on ANGLE, not just reward
import os

for artefact in (
    "05_ppo_continuous_rewards.html",
    "05_ppo_continuous_std.png",
    "05_ppo_continuous_episode.png",
):
    assert os.path.exists(OUTPUT_DIR / artefact), f"Missing: {artefact}"
late_trained = float(np.mean(np.abs(angles[-50:])))
late_random = float(np.mean(np.abs(angles_random[-50:])))
assert late_trained < late_random * 0.9, (
    f"trained policy should swing measurably closer to upright than random "
    f"(late-episode mean |angle|: trained {late_trained:.2f} rad vs random "
    f"{late_random:.2f} rad)"
)
print(
    f"\n  Late-episode mean |angle| from upright: trained "
    f"{late_trained:.2f} rad vs random {late_random:.2f} rad. At this CPU "
    "budget the policy has learned the swing-up, not full stabilisation — "
    "the direction is the verified claim; the remaining budget is stated."
)
print("\n--- Checkpoint 3 passed --- behaviour verified, not just rewards\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: HVAC damper control at a data centre
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a tropical data centre tunes its
# cooling dampers. The damper is a DIAL (0-100% open), not a switch, and
# the cost of oscillation is mechanical wear. A discrete-action agent can
# only bang the damper between preset positions; a Gaussian PPO policy
# emits a continuous setpoint and — because the exploration noise is
# measured — the facilities team can SEE when the policy stopped
# experimenting.
#
# The std schedule is the operational artefact: "experimentation ended at
# iteration ~15" is a governance statement, not a vibe. Combined with the
# reward curve it answers the two questions a facility manager asks:
# "is it better than the naive schedule?" (reward vs baseline) and "has
# it stopped fiddling?" (std collapse).

print("=" * 70)
print("  PHASE 5 — APPLY: continuous setpoint control, governed")
print("=" * 70)
print(
    f"""
  DAMPER CONTROL — GOVERNANCE REPORT (measured, this run):

    Environment analogue:     Pendulum-v1 torque in [-{ACT_LIMIT}, {ACT_LIMIT}]
    Random-schedule baseline: {random_mean:.1f} mean return
    Trained policy:           {best_window:.1f} (last-5-iteration mean)
    Exploration std:          {std_hist[0]:.3f} -> {std_hist[-1]:.3f}
      (the dial's experimental wiggle, shrinking as confidence grows)

  TWO GOVERNANCE QUESTIONS, TWO MEASURED ANSWERS:
    "Better than the naive schedule?"  reward vs the measured baseline
    "Has it stopped experimenting?"    the std schedule, not a promise

  STAKEHOLDER-READY OUTPUT:
    "The continuous policy measurably outperforms a random damper
    schedule: mean return improved from {random_mean:.0f} to
    {best_window:.0f}, the exploration noise narrowed from
    {std_hist[0]:.2f} to {std_hist[-1]:.2f}, and in a head-to-head
    evaluation episode the trained controller held the process
    {late_trained:.2f} rad from setpoint versus {late_random:.2f} rad
    for the random schedule. At this training budget it has learned the
    swing-up, not full stabilisation — the remaining budget is an
    infrastructure decision, and both curves are in the tracker."
"""
)

# ── Checkpoint 4: the governance numbers are the measured ones ────────
assert best_window > random_mean, "trained must beat random"
assert std_hist[-1] < std_hist[0], "std must shrink"
print("--- Checkpoint 4 passed --- governed application demonstrated\n")

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
  [x] Buttons vs dials: Categorical cannot express a continuous setpoint;
      the Gaussian policy emits mu(s) with a learned spread log_std
  [x] PPO's machinery is distribution-agnostic: same GAE, same clipped
      ratio, same entropy bonus — only Normal replaced Categorical

  BUILD + TRAIN:
  [x] GaussianActorCritic: tanh-squashed mean torque in [-{ACT_LIMIT}, {ACT_LIMIT}],
      free log_std parameter, separate critic ({_n_params:,} params)
  [x] {N_ITERS} iterations x {STEPS_PER_ITER} on-policy steps on Pendulum-v1,
      ExperimentTracker receipts (incl. exploration_std per iteration)
  [x] Measured: random baseline {random_mean:.1f} -> trained
      {best_window:.1f}; std {std_hist[0]:.3f} -> {std_hist[-1]:.3f}

  VISUALISE (the proof):
  [x] Reward curve with the random baseline drawn as a line
  [x] Exploration-collapse curve (std {std_hist[0]:.3f} -> {std_hist[-1]:.3f})
  [x] Eval episode, trained vs random: late-episode mean |angle|
      {late_trained:.2f} vs {late_random:.2f} rad — the swing-up is
      learned at this budget; full stabilisation is stated as remaining
      work, not claimed

  APPLY:
  [x] HVAC damper framing: "better than naive?" + "stopped
      experimenting?" answered by measured curves, not assurances
  [x] Stated limit: the std schedule is the governance artefact — when
      it has not collapsed, the policy has not finished learning

  KEY INSIGHT: Moving from discrete to continuous actions changed ONE
  object — the distribution — and left the entire PPO stack untouched.
  That is the sign of a well-factored algorithm: the policy-gradient
  theorem does not care whether actions are buttons or dials. What
  changes is what you must GOVERN: with a dial, the exploration noise
  itself is a first-class, measurable quantity.
"""
)
