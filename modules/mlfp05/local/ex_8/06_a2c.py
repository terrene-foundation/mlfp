# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 8.6 — A2C: Advantage Actor-Critic, the Unclipped
# Original
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Place A2C in the actor-critic family tree: REINFORCE (no critic)
#     -> A2C (critic baseline, one update per batch) -> PPO (clipped,
#     multi-epoch)
#   - Explain what the ADVANTAGE buys: the critic's V(s) subtracts the
#     expected return, so the actor only learns "better than expected"
#   - Implement A2C: GAE advantages + ONE gradient step per rollout —
#     no ratio, no clipping, no minibatch epochs
#   - Explain what PPO's clip buys and what it costs (data efficiency
#     vs implementation simplicity)
#   - Train A2C on CartPole-v1 and beat a measured random baseline
#   - Apply to queue-staffing simulation screening, where fast simple
#     iterations beat final-point performance
#
# PREREQUISITES: M5/ex_8/02_ppo.py (actor-critic, GAE, clipped surrogate)
# ESTIMATED TIME: ~30 min
#
# ENVIRONMENT: CartPole-v1 (discrete). A2C predates PPO; we train on the
#   same environment so the comparison against 02's receipts is direct.
#
# PHASES:
#   1. THEORY  — the actor-critic family tree; what the advantage buys
#   2. BUILD   — ActorCritic + the single-update A2C step
#   3. TRAIN   — A2C on CartPole, tracked with ExperimentTracker
#   4. VISUALISE — reward curve, entropy, diagnostic pad
#   5. APPLY   — queue-staffing simulation screening
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from shared.mlfp05.ex_8 import (
    OUTPUT_DIR,
    device,
    evaluate_policy,
    make_cartpole,
    moving_average,
    register_rl_model,
    rl_diagnostic_checkpoint,
    setup_engines,
)

# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: The Actor-Critic Family Tree
# ════════════════════════════════════════════════════════════════════════
# REINFORCE (the grandparent): push up the log-probability of every action
# by the episode's total return. Problem: the total return is a NOISY
# credit assignment — a lucky ride on a bad policy looks like brilliance.
#
# A2C (Advantage Actor-Critic): subtract what the critic EXPECTED:
#     A(s, a) = Q(s, a) - V(s)  ~=  "how much better than par was that?"
# The actor climbs the advantage, not the raw return. Variance collapses
# because average actions get ~zero push, good actions get positive push,
# and BAD actions get pushed DOWN (negative advantage) — REINFORCE with
# only positive returns can never actively discourage a bad action.
#
# THE UPDATE RULE: collect a fresh on-policy rollout, compute GAE
# advantages, take ONE gradient step:
#     loss = -mean( log pi(a|s) * A_norm ) + 0.5 * MSE(V, returns)
#            - 0.01 * entropy
# No importance ratio (one update means the data IS from the current
# policy), no clipping (nothing to clip against). That is the whole
# algorithm — PPO is A2C plus a mechanism for safely REUSING the rollout
# for several epochs.
#
# THE TRADE-OFF, honestly: A2C is simpler (no ratio bookkeeping) and each
# iteration is cheaper (one update), but it throws away data after one
# use. PPO's clip buys data reuse at the price of machinery. When
# simulation is cheap and fast, A2C's simplicity wins; when data is
# expensive, PPO's reuse wins. There is no free lunch — only a priced one.

print("=" * 70)
print("  PHASE 1 — THEORY: REINFORCE -> A2C -> PPO")
print("=" * 70)
print(
    """
  REINFORCE:  push log-prob by TOTAL return — noisy, never negative
  A2C:        push log-prob by ADVANTAGE (return minus what the critic
              expected); one gradient step per fresh rollout
  PPO:        A2C + importance ratio + clipping -> reuse the rollout
              for several epochs safely

  A2C update (the entire algorithm):
    loss = -mean(log pi(a|s) * A_norm) + 0.5 * MSE(V, R) - 0.01 * H(pi)

  The trade: A2C is simpler and cheaper per iteration; PPO reuses data.
  Cheap simulator -> A2C. Expensive data -> PPO.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: ActorCritic + single-update A2C
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: the unclipped actor-critic")
print("=" * 70)

env, obs_dim, n_actions = make_cartpole()
conn, tracker, exp_name, registry, has_registry = setup_engines()


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs (same reasoning as 02_ppo.py:
    value-regression gradients dominate a shared trunk on CartPole)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            # TODO: actor output layer — one logit per ACTION
            nn.Linear(____),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # TODO: return (actor logits, critic value squeezed to (batch,))
        return ____

    def act(self, state: np.ndarray) -> tuple[int, torch.Tensor, torch.Tensor]:
        s = torch.from_numpy(state.astype(np.float32)).to(device)
        logits, value = self.forward(s)
        dist = Categorical(logits=logits)
        a = dist.sample()
        # TODO: return the int action, its DETACHED log-probability, and
        #   the detached value estimate
        return ____


def collect_rollout(env, model, n_steps: int):
    """Fresh on-policy rollout: states, actions, rewards, dones, values."""
    states, actions, values, rewards, dones = [], [], [], [], []
    state, _ = env.reset(seed=int(np.random.randint(0, 100_000)))
    for _ in range(n_steps):
        action, _, value = model.act(state)
        # TODO: step the environment and capture the five-tuple
        # Hint: gymnasium returns (next_state, reward, terminated, truncated, info)
        next_state, reward, terminated, truncated, _ = ____
        states.append(state.astype(np.float32))
        actions.append(action)
        values.append(value)
        rewards.append(float(reward))
        done = terminated or truncated
        dones.append(done)
        state = next_state
        if done:
            state, _ = env.reset(seed=int(np.random.randint(0, 100_000)))
    return states, actions, values, rewards, dones


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """GAE — identical to PPO's. The algorithms share credit assignment;
    they differ in what the UPDATE is allowed to do with it."""
    advantages = [0.0] * len(rewards)
    gae, next_value = 0.0, 0.0
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - float(dones[t])
        # TODO: TD error delta = r_t + gamma * V(s_{t+1}) * nonterminal - V(s_t)
        delta = ____
        # TODO: GAE recursion — delta plus the decayed running advantage
        # Hint: gae = delta + gamma * lam * nonterminal * gae
        gae = ____
        advantages[t] = gae
        next_value = float(values[t])
    returns = [a + float(v) for a, v in zip(advantages, values)]
    return advantages, returns


# ── Checkpoint 1: shapes + advantage sanity ────────────────────────────
_probe = ActorCritic(obs_dim, n_actions).to(device)
_s, _ = env.reset(seed=0)
_a, _lp, _v = _probe.act(_s)
assert 0 <= _a < n_actions, f"action {_a} outside [0, {n_actions})"
_logits, _value = _probe(torch.from_numpy(_s.astype(np.float32)).to(device))
assert _logits.shape == (n_actions,), f"logits {_logits.shape} != ({n_actions},)"
# GAE sanity: a rollout of all-zero rewards, zero values, no dones must
# give zero advantages (nothing better or worse than expected)
_adv_zero, _ = compute_gae([0.0] * 10, [torch.tensor(0.0)] * 10, [False] * 10)
assert all(abs(a) < 1e-6 for a in _adv_zero), "zero deltas -> zero advantages"
# a single +1 reward at the LAST step propagates backwards with decay
_adv_one, _ = compute_gae(
    [0.0] * 9 + [1.0], [torch.tensor(0.0)] * 10, [False] * 10
)
assert abs(_adv_one[-1] - 1.0) < 1e-6, "final-step delta = reward itself"
assert abs(_adv_one[0] - (0.99 * 0.95) ** 9) < 1e-4, (
    "GAE decay: step 0 sees the final reward scaled by (gamma*lam)^9"
)
_n_params = sum(p.numel() for p in _probe.parameters())
print(f"\nActorCritic built: {_n_params:,} parameters")
print(f"  GAE verified: zero deltas -> zero advantages; +1 final reward -> "
      f"A[0] = {(0.99 * 0.95) ** 9:.4f} = (0.99*0.95)^9")
print("\n--- Checkpoint 1 passed --- A2C machinery verified\n")
del _probe, _s, _a, _lp, _v, _logits, _value


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: A2C on CartPole (one update per rollout)
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: A2C on CartPole-v1")
print("=" * 70)

random_returns = evaluate_policy(env, lambda s: env.action_space.sample(),
                                 n_episodes=10)
random_mean = float(np.mean(random_returns))
print(f"  Random policy baseline: {random_mean:.1f} mean return (10 episodes)")

N_ITERS = 80
STEPS_PER_ITER = 512
A2C_LR = 1e-3
ENTROPY_COEF = 0.01  # discrete buttons still need an exploration nudge


async def train_a2c():
    """A2C: collect rollout -> GAE -> ONE gradient step -> discard data."""
    model = ActorCritic(obs_dim, n_actions).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=A2C_LR)
    iter_returns, entropies, actor_losses, critic_losses = [], [], [], []

    async with tracker.track(experiment=exp_name, run_name="a2c_cartpole") as run:
        await run.log_params(
            {
                "algorithm": "A2C",
                "env": "CartPole-v1",
                "lr": str(A2C_LR),
                "steps_per_iter": str(STEPS_PER_ITER),
                "updates_per_rollout": "1",  # the defining trait
                "entropy_coef": str(ENTROPY_COEF),
                "random_baseline_return": f"{random_mean:.1f}",
            }
        )
        for it in range(N_ITERS):
            states, actions, values, rewards, dones = collect_rollout(
                env, model, STEPS_PER_ITER
            )
            advantages, returns = compute_gae(rewards, values, dones)

            s_t = torch.tensor(np.stack(states), dtype=torch.float32, device=device)
            a_t = torch.tensor(actions, dtype=torch.long, device=device)
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # ONE gradient step over the whole rollout — no ratio, no clip
            logits, vpred = model(s_t)
            dist = Categorical(logits=logits)
            # TODO: actor loss — negative mean of log-prob times the
            #   NORMALISED advantage (this is the whole policy update)
            policy_loss = ____
            # TODO: critic loss — MSE between value predictions and the
            #   GAE return targets
            value_loss = ____
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - ENTROPY_COEF * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            ep_rs, running = [], 0.0
            for r, d in zip(rewards, dones):
                running += r
                if d:
                    ep_rs.append(running)
                    running = 0.0
            avg_return = float(np.mean(ep_rs)) if ep_rs else float(running)
            iter_returns.append(avg_return)
            entropies.append(float(entropy.item()))
            actor_losses.append(float(policy_loss.item()))
            critic_losses.append(float(value_loss.item()))

            await run.log_metrics(
                {
                    "avg_episode_return": avg_return,
                    "policy_entropy": entropies[-1],
                    "actor_loss": actor_losses[-1],
                    "critic_loss": critic_losses[-1],
                },
                step=it,
            )
            if (it + 1) % 20 == 0:
                print(
                    f"  iter {it+1:3d}/{N_ITERS}  return={avg_return:6.1f}  "
                    f"entropy={entropies[-1]:.3f}"
                )
        await run.log_metric("final_avg_return", iter_returns[-1])
    return model, iter_returns, entropies, actor_losses, critic_losses


model, iter_returns, entropies, actor_losses, critic_losses = asyncio.run(
    train_a2c()
)

# ── Checkpoint 2: beats the measured random baseline ───────────────────
assert len(iter_returns) == N_ITERS
best_window = float(np.mean(iter_returns[-10:]))
assert best_window > 3.0 * random_mean, (
    f"A2C last-10 mean {best_window:.1f} should be > 3x the random "
    f"baseline {random_mean:.1f}"
)
assert best_window > 75.0, (
    f"A2C on CartPole should average > 75 by iteration {N_ITERS}; "
    f"got {best_window:.1f}"
)
print(
    f"\n  Random baseline: {random_mean:.1f} | A2C last-10 mean: "
    f"{best_window:.1f} ({best_window / random_mean:.1f}x random)"
)
print("  Compare with 02_ppo.py's receipts in ExperimentTracker: PPO reuses")
print("  each rollout 4x; A2C uses it once. Same environment, honest split.")
print("\n--- Checkpoint 2 passed --- A2C trained and verified\n")

register_rl_model(
    registry,
    "a2c_cartpole",
    model,
    {
        "random_baseline_return": random_mean,
        "final_window_return": best_window,
        "updates_per_rollout": 1.0,
    },
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: reward curve + entropy + diagnostics
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: learning without a safety clip")
print("=" * 70)

rl_diagnostic_checkpoint(
    "A2C on CartPole-v1",
    "a2c",
    iter_returns,
    policy_losses=actor_losses,
    value_losses=critic_losses,
    entropies=entropies,
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
        "A2C return": iter_returns,
        "moving avg (10)": moving_average(iter_returns, 10),
        "random baseline": [random_mean] * len(iter_returns),
    },
    x_label="Iteration (one gradient step each)",
    y_label="Mean episode return",
)
fig.write_html(str(OUTPUT_DIR / "06_a2c_rewards.html"))
print(f"  Saved: {OUTPUT_DIR / '06_a2c_rewards.html'}")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_ent, ax = plt.subplots(figsize=(9, 5))
ax.plot(range(1, N_ITERS + 1), entropies, color="#9C27B0")
ax.set_xlabel("Iteration")
ax.set_ylabel("policy entropy (nats)")
ax.set_title(
    f"Entropy over training (max = ln 2 = {np.log(2):.3f}): "
    "the policy committing to actions"
)
ax.grid(True, alpha=0.3)
fig_ent.tight_layout()
fig_ent.savefig(str(OUTPUT_DIR / "06_a2c_entropy.png"), dpi=150)
plt.close(fig_ent)
print(f"  Saved: {OUTPUT_DIR / '06_a2c_entropy.png'}")

# ── Checkpoint 3: artefacts + entropy moved (policy committed) ─────────
import os

for artefact in ("06_a2c_rewards.html", "06_a2c_entropy.png"):
    assert os.path.exists(OUTPUT_DIR / artefact), f"Missing: {artefact}"
assert entropies[-1] < entropies[0], (
    "entropy should decline as the policy commits "
    f"({entropies[0]:.3f} -> {entropies[-1]:.3f})"
)
print("\n--- Checkpoint 3 passed --- learning curves verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: queue-staffing simulation screening
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a logistics firm screens staffing
# policies for a parcel-sorting hub. The simulator is CHEAP (thousands of
# simulated shifts per minute on one core) and the policy space is small
# (how many sorters per belt per hour-band). The engineering question is
# not "the best possible policy" — it is "screen 50 candidate staffing
# shapes by Friday and hand operations the top three".
#
# A2C IS THE RIGHT TOOL HERE, and the reason is the trade-off measured in
# this file:
#   - Data is free: the simulator regenerates a rollout in milliseconds,
#     so PPO's data-reuse machinery buys nothing.
#   - Iterations must be fast and the code reviewable: one gradient step
#     per rollout, no ratio bookkeeping, no clipping — an ops engineer
#     can read the whole update in five lines.
#   - The cost of a slightly-suboptimal screened candidate is low: the
#     final choice gets a full evaluation anyway.
#
# WHEN THIS ANSWER CHANGES: if the simulator were a physical robot cell
# (each rollout costs battery, wear, and a supervisor's time), the same
# problem flips to PPO — data reuse suddenly matters.

print("=" * 70)
print("  PHASE 5 — APPLY: cheap-simulator screening chooses A2C")
print("=" * 70)
print(
    f"""
  STAFFING-SCREEN REPORT (measured, this run):

    Random staffing baseline:  {random_mean:.1f} mean return
    A2C screened policy:       {best_window:.1f} (last-10 mean)
    Updates per rollout:       1 (vs PPO's 4 epochs over 4 minibatches)
    Policy entropy:            {entropies[0]:.3f} -> {entropies[-1]:.3f}
                               (max {np.log(2):.3f}; the policy committed)

  DECISION RULE:
    cheap, fast simulator + review-by-Friday  ->  A2C
    expensive data (robots, users, markets)   ->  PPO (data reuse)
    continuous setpoints                      ->  05's Gaussian PPO,
                                                  or 07/08's off-policy
                                                  actor-critics

  STAKEHOLDER-READY OUTPUT:
    "We screened the staffing policies with the simplest actor-critic
    that works: each simulated week is used for exactly one policy
    update, then discarded — the simulator is cheap, so data reuse
    machinery would only add review surface. The screened policy beats
    the random staffing baseline by {best_window / random_mean:.0f}x on
    the tracked metric."
"""
)

# ── Checkpoint 4: the screening numbers are the measured ones ──────────
assert best_window > random_mean, "screened policy must beat random"
print("--- Checkpoint 4 passed --- screening application demonstrated\n")

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
  [x] The family tree: REINFORCE (no critic) -> A2C (critic baseline,
      one update) -> PPO (clipped, multi-epoch reuse)
  [x] The advantage A = Q - V: average actions get ~zero push, bad
      actions get pushed DOWN — what raw-return REINFORCE cannot do
  [x] The priced trade: A2C discards data after one use; PPO's clip is
      the price of reusing it

  BUILD + TRAIN:
  [x] ActorCritic (separate trunks, {_n_params:,} params) + GAE with two
      closed-form checks (zero deltas -> zero advantages; decay
      (gamma*lam)^t)
  [x] The entire A2C update in one loss line — no ratio, no clip
  [x] {N_ITERS} iterations x {STEPS_PER_ITER} steps on CartPole:
      random {random_mean:.1f} -> A2C {best_window:.1f}
      ({best_window / random_mean:.1f}x), tracked in ExperimentTracker

  VISUALISE (the proof):
  [x] Reward curve against the measured random baseline
  [x] Entropy curve: {entropies[0]:.3f} -> {entropies[-1]:.3f} — the
      policy committing to actions
  [x] RLDiagnostics pad: no reward collapse over the rolling window

  APPLY:
  [x] Cheap-simulator screening: the decision rule for when A2C beats
      PPO is a DATA-PRICE argument, not a performance slogan
  [x] Stated boundary: expensive data flips the answer to PPO;
      continuous dials flip it to 05/07/08

  KEY INSIGHT: A2C and PPO share the credit-assignment machinery (GAE)
  and differ ONLY in how the update treats the data: use once vs reuse
  with a clip. Choosing between them is an economics question — what
  does a rollout cost you? — and that is a question an engineering
  manager can answer without reading a single line of torch.
"""
)
