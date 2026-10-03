# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 2.1: EM Algorithm From Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Explain EM as coordinate ascent on the ELBO (lower bound on log-lik)
#   - Implement the E-step (posterior responsibilities) with log-sum-exp
#   - Implement the M-step (weighted MLE for pi, mu, Sigma)
#   - Verify that log-likelihood is non-decreasing across EM iterations
#   - Check your EM against sklearn's GaussianMixture on the same data
#   - Plot the convergence curve as visual proof the algorithm works
#
# PREREQUISITES:
#   - MLFP04 Exercise 1 (clustering — GMM used as a black box there)
#   - MLFP02 Lesson 2.1 (Bayesian thinking)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — EM as ELBO maximisation (why it's guaranteed to improve)
#   2. Build — E-step, M-step, log-likelihood, full EM loop
#   3. Train — fit a 3-component GMM on synthetic 2D data and compare it
#      with sklearn's GaussianMixture
#   4. Visualise — convergence curve + recovered vs true parameters
#   5. Apply — Singapore property-marketplace lead scoring with soft
#      assignments
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from kailash_ml import ModelVisualizer

# Cross-exercise import: tracker helpers live in ex_1.shared so every M4
# unsupervised technique logs to the same `m4_clustering_zoo` experiment.
from shared.mlfp04.ex_1 import setup_engines, teardown_engines, track_run
from shared.mlfp04.ex_2 import (
    N_SYNTH,
    TRUE_COVS,
    TRUE_MEANS,
    TRUE_WEIGHTS,
    make_synthetic_gmm,
    out_path,
    safe_silhouette,
)

# ── Kailash-ML ExperimentTracker — every clustering run logs here ─────────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — EM as ELBO Maximisation
# ════════════════════════════════════════════════════════════════════════
# Goal: maximise log P(X|theta) = Sum_i log Sum_k pi_k N(x_i|mu_k,Sigma_k).
# The log-of-a-sum makes this intractable in closed form.
#
# EM introduces Q(Z) over the hidden component assignments and maximises:
#
#     log P(X|theta)  >=  E_Q[log P(X,Z|theta)] + H(Q)    (Jensen)
#
#   E-step: Q*(Z) = P(Z|X,theta)  — the posterior responsibilities
#   M-step: theta* = argmax E_{Q*}[log P(X,Z|theta)]  — weighted MLE
#
# KEY GUARANTEE: each E+M pair cannot decrease log P(X|theta). We will
# verify this numerically in Task 4.
#
# For GMMs the updates are:
#   E: r_{ik} = pi_k N(x_i|mu_k,Sigma_k) / Sum_j pi_j N(x_i|mu_j,Sigma_j)
#   M: N_k = Sum_i r_{ik}
#      pi_k = N_k / N
#      mu_k = (Sum_i r_{ik} x_i) / N_k
#      Sigma_k = (Sum_i r_{ik} (x_i-mu_k)(x_i-mu_k)') / N_k  (+ ridge)


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: E-step, M-step, log-likelihood, full EM loop
# ════════════════════════════════════════════════════════════════════════


def e_step(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """E-step: posterior responsibilities R[i,k] = P(z_i=k | x_i, theta)."""
    n_samples = X.shape[0]
    n_components = len(weights)
    log_probs = np.zeros((n_samples, n_components))

    # TODO: for each component k, fill log_probs[:, k] with the log of
    # pi_k * N(x | mu_k, Sigma_k).
    # Hint: scipy.stats.multivariate_normal(mean=..., cov=..., allow_singular=True)
    # has a .logpdf(X) method; add 1e-300 inside np.log(weights[k] ...) for safety.
    # No try/except: the M-step ridge keeps covariances valid, so a failure is a bug.
    for k in range(n_components):
        ____

    # Log-sum-exp trick for numerical stability
    # TODO: subtract the per-row max, exponentiate, sum, and re-add the max
    # Hint: log_probs_max = log_probs.max(axis=1, keepdims=True)
    log_probs_max = ____
    log_norm = ____
    return np.exp(log_probs - log_norm)


def m_step(
    X: np.ndarray,
    R: np.ndarray,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """M-step: update (means, covs, weights) from responsibilities R."""
    n_samples, n_features = X.shape
    n_components = R.shape[1]

    # TODO: implement the M-step updates from the THEORY block:
    # N_k (column sums of R, plus 1e-300), pi_k, and mu_k.
    # Hint: mu for all k at once is a single matrix product R.T @ X, divided
    # row-wise by N_k
    N_k = ____
    weights = ____
    means = ____

    covs = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        # TODO: responsibility-weighted outer products of diff, divided by
        # N_k[k], plus the ridge reg_covar * np.eye(n_features)
        covs[k] = ____

    return means, covs, weights


def compute_log_likelihood(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Full data log-likelihood under the current GMM parameters."""
    n_samples = X.shape[0]
    log_l = np.full(n_samples, -np.inf)
    for k in range(len(weights)):
        dist = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
        log_l = np.logaddexp(log_l, np.log(weights[k] + 1e-300) + dist.logpdf(X))
    return float(log_l.sum())


def fit_gmm_em(
    X: np.ndarray,
    n_components: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict:
    """Full EM loop with random initialisation. Returns a result dict."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    idx = rng.choice(n_samples, n_components, replace=False)
    means = X[idx].copy()
    covs = np.array([np.eye(n_features)] * n_components)
    weights = np.ones(n_components) / n_components

    log_likelihoods: list[float] = []
    R = np.zeros((n_samples, n_components))

    for iteration in range(max_iter):
        # TODO: call e_step then m_step to update R, means, covs, weights
        R = ____
        means, covs, weights = ____
        ll = compute_log_likelihood(X, means, covs, weights)
        log_likelihoods.append(ll)

        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    return {
        "means": means,
        "covs": covs,
        "weights": weights,
        "responsibilities": R,
        "labels": R.argmax(axis=1),
        "log_likelihoods": log_likelihoods,
        "n_iter": len(log_likelihoods),
        "final_ll": log_likelihoods[-1],
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: fit the from-scratch EM on synthetic 2D data
# ════════════════════════════════════════════════════════════════════════

X_synth, z_true = make_synthetic_gmm()

print("=" * 70)
print("  Synthetic GMM Data")
print("=" * 70)
print(f"Samples: {N_SYNTH}  Components: 3")
print(f"True weights: {TRUE_WEIGHTS}")

# Sanity-check the E-step against the known-good parameters
R_true_params = e_step(X_synth, TRUE_MEANS, TRUE_COVS, TRUE_WEIGHTS)
accuracy_true = (R_true_params.argmax(axis=1) == z_true).mean()
print(f"\nE-step with TRUE params -> assignment accuracy: {accuracy_true:.4f}")

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert R_true_params.shape == (N_SYNTH, 3), "R should be (n_samples, n_components)"
assert abs(R_true_params.sum(axis=1).mean() - 1.0) < 1e-6, "rows must sum to 1"
assert R_true_params.min() >= 0, "responsibilities must be non-negative"
assert accuracy_true > 0.8, "true params should give high assignment accuracy"
print("[ok] Checkpoint 1 passed — E-step behaves as a proper posterior")

# Sanity-check the M-step: hard ground-truth labels should recover the
# true parameters almost exactly.
R_hard = np.zeros((N_SYNTH, 3))
R_hard[np.arange(N_SYNTH), z_true] = 1.0
means_hat, covs_hat, weights_hat = m_step(X_synth, R_hard)

print("\nM-step with hard ground-truth labels:")
print(f"  weights_hat = {weights_hat.round(3)}  (true {TRUE_WEIGHTS})")
for k in range(3):
    print(f"  mean_{k}   = {means_hat[k].round(3)}  (true {TRUE_MEANS[k]})")

# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert abs(weights_hat.sum() - 1.0) < 1e-6, "weights must sum to 1"
for k in range(3):
    assert (
        np.linalg.norm(means_hat[k] - TRUE_MEANS[k]) < 1.0
    ), f"recovered mean {k} too far from ground truth"
print("[ok] Checkpoint 2 passed — M-step is weighted MLE")

# Run the full EM loop from a random initialisation
print("\nRunning EM from random init...")
# TODO: call fit_gmm_em with n_components=3, max_iter=100, tol=1e-4
em = ____

print(f"\nIterations: {em['n_iter']}")
print(f"Final log-likelihood: {em['final_ll']:.2f}")
print(f"Recovered weights: {em['weights'].round(3)}  (true {TRUE_WEIGHTS})")

sil = safe_silhouette(X_synth, em["labels"])
print(f"Silhouette on recovered labels: {sil:.4f}")

# Verify the non-decreasing log-likelihood property numerically
lls = em["log_likelihoods"]
deltas = [lls[i] - lls[i - 1] for i in range(1, len(lls))]
n_down = sum(1 for d in deltas if d < -0.1)
print(
    f"\nLog-likelihood deltas: min={min(deltas):.4f} max={max(deltas):.4f} "
    f"n_decreases>0.1: {n_down}"
)

# ── Checkpoint 3 ────────────────────────────────────────────────────────
assert em["n_iter"] > 1, "EM should take more than one iteration"
for i in range(1, len(lls)):
    assert lls[i] >= lls[i - 1] - 0.1, f"log-likelihood decreased at iter {i}"
assert len(set(em["labels"])) >= 2, "EM should use at least 2 components"
print("[ok] Checkpoint 3 passed — EM converged with non-decreasing log-likelihood")

# Compare with the library. sklearn's GaussianMixture runs the same EM
# (k-means initialisation instead of random points, same ridge reg_covar).
# Component ORDER is arbitrary, so match components by nearest means
# (Hungarian assignment) before comparing weights and means.
# TODO: fit a GaussianMixture with n_components=3, covariance_type="full",
# reg_covar=1e-6, tol=1e-6, max_iter=500, random_state=42 on X_synth.
sk_gmm = ____
# TODO: total log-likelihood — GaussianMixture.score() returns the
# per-sample MEAN, so scale it by N_SYNTH.
sk_ll = ____
mean_dist = np.linalg.norm(
    em["means"][:, None, :] - sk_gmm.means_[None, :, :], axis=-1
)
# TODO: match your components to sklearn's by minimising total mean distance.
# Hint: scipy.optimize.linear_sum_assignment returns (row_idx, col_idx)
row, col = ____
max_mean_gap = float(mean_dist[row, col].max())
max_weight_gap = float(np.abs(em["weights"][row] - sk_gmm.weights_[col]).max())
ll_gap_per_sample = abs(em["final_ll"] - sk_ll) / N_SYNTH
ari_vs_sklearn = adjusted_rand_score(em["labels"], sk_gmm.predict(X_synth))

print("\nFrom-scratch EM vs sklearn GaussianMixture (same data, K=3):")
print(f"  log-likelihood: yours={em['final_ll']:.2f}  sklearn={sk_ll:.2f}")
print(f"  |Δ log-lik| per sample : {ll_gap_per_sample:.6f}")
print(f"  max |Δ weight|         : {max_weight_gap:.4f}")
print(f"  max distance of means  : {max_mean_gap:.4f}")
print(f"  ARI between labelings  : {ari_vs_sklearn:.4f}")
if ll_gap_per_sample < 1e-3:
    print("  Same optimum — your EM and the library agree.")
else:
    print(
        "  Different optimum — EM only finds a LOCAL maximum, and the two\n"
        "  implementations started from different initial means."
    )

# ── Checkpoint 4 ────────────────────────────────────────────────────────
assert ll_gap_per_sample < 0.05, "your EM should reach a log-likelihood close to sklearn's"
assert max_weight_gap < 0.05, "matched mixture weights should agree with sklearn's"
print("[ok] Checkpoint 4 passed — from-scratch EM agrees with sklearn's GMM")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: convergence curve
# ════════════════════════════════════════════════════════════════════════
# The convergence plot is the VISUAL PROOF of the ELBO theorem: the
# log-likelihood is a staircase that only goes up.

viz = ModelVisualizer()
# TODO: call viz.training_history with {"Log-Likelihood": em["log_likelihoods"]}
# and x_label="EM Iteration"
fig = ____
fig.update_layout(title="EM from scratch — log-likelihood per iteration")
fig.write_html(str(out_path("ex2_em_convergence.html")))
print(f"\nSaved: {out_path('ex2_em_convergence.html')}")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore Property-Marketplace Lead Scoring
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore online property marketplace's inside-sales team
# gets (assume) ~8,000 new lead signals per week but
# can only call ~1,500. Every unqualified call is a call they could
# have made to a genuine buyer.
#
# Why a GMM beats hard clustering here:
#   A lead with r = [0.45, 0.55] is genuinely between two intents —
#   maybe a "window shopper" warming up into an "active buyer". Hard
#   assignment buries that signal. Soft responsibilities let the sales
#   platform score every lead on EVERY segment and sort by expected
#   revenue = sum_k r_k * E[deal_value | segment_k].
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures):
#   - Weekly closed deals: ~55 @ avg S$9,200 agent fee = S$506K/week
#   - Assume soft-GMM scoring lifts close rate by ~11% on ambiguous leads
#     (an assumption to confirm with your own A/B test)
#   - 11% * S$506K/week = S$55.7K/week = S$2.9M/year extra commission
#     from the same headcount — no extra spend.

responsibilities = em["responsibilities"]
# TODO: build a mask of ambiguous rows where 0.4 < max responsibility < 0.6
# Hint: use responsibilities.max(axis=1)
ambiguous_mask = ____
n_ambiguous = int(ambiguous_mask.sum())
print("\n" + "=" * 70)
print("  APPLY — Property-Marketplace Lead Scoring")
print("=" * 70)
print(
    f"Of {N_SYNTH} synthetic leads, {n_ambiguous} "
    f"({n_ambiguous / N_SYNTH:.1%}) are ambiguous (max responsibility 0.4-0.6)."
)
print("Hard clustering would lose the between-segment signal on every one.")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# The from-scratch EM joins the m4_clustering_zoo experiment alongside
# the ex_1 clustering runs. series_metrics captures the log-likelihood
# staircase for the convergence proof.

# TODO: call track_run with run_name "em_from_scratch". scalar_metrics
# include final_log_likelihood (em["final_ll"]), n_iter (em["n_iter"]),
# true_param_assignment_accuracy (accuracy_true), recovered_silhouette
# (float(sil) — track_run skips it if undefined), ambiguous_count,
# ambiguous_pct. series_metrics is {"log_likelihood_per_iter": em["log_likelihoods"]}.
track_run(
    tracker,
    exp_name,
    run_name=____,
    params={
        "n_components": 3,
        "n_synth": N_SYNTH,
        "max_iter": 100,
        "tol": 1e-4,
        "init": "random",
    },
    scalar_metrics={
        "final_log_likelihood": float(em["final_ll"]),
        "n_iter": float(em["n_iter"]),
        "true_param_assignment_accuracy": float(accuracy_true),
        "recovered_silhouette": ____,
        "sklearn_ll_gap_per_sample": float(ll_gap_per_sample),
        "sklearn_label_ari": float(ari_vs_sklearn),
        "ambiguous_count": float(n_ambiguous),
        "ambiguous_pct": float(n_ambiguous) / float(N_SYNTH),
    },
    series_metrics={"log_likelihood_per_iter": ____},
)
print(f"  [tracked] EM convergence + assignment metrics logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — ClusteringEngine.fit(algorithm='gmm')
# ════════════════════════════════════════════════════════════════════════
# You built EM from scratch — E-step responsibilities, M-step weighted MLE,
# the log-likelihood staircase, the convergence proof. kailash-ml's
# ClusteringEngine wraps the same EM (sklearn's GaussianMixture, which you
# just matched): one sync call returns labels + silhouette/CH/inertia.

import polars as pl

from kailash_ml.engines.clustering import ClusteringEngine

synth_df = pl.from_numpy(X_synth, schema=["x0", "x1"])

# TODO: instantiate ClusteringEngine and call .fit on synth_df with
# algorithm='gmm' and n_clusters=3.
clustering = ____
fit_result = ____
print(
    f"  ClusteringEngine.fit(gmm, K=3): "
    f"silhouette={(fit_result.silhouette_score or 0.0):.4f}  "
    f"n_clusters={fit_result.n_clusters}"
)
print(
    "  Same EM you derived by hand — encapsulated under the same fit()"
    " surface that backs kmeans/dbscan/spectral.\n"
)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] EM as coordinate ascent on the ELBO — each step must improve
  [x] E-step: soft posterior responsibilities via log-sum-exp
  [x] M-step: weighted MLE for pi, mu, Sigma
  [x] Numerical proof that log-likelihood is monotone non-decreasing
  [x] Matched your EM against sklearn's GaussianMixture (log-likelihood,
      weights, means after label matching)
  [x] Property-marketplace lead scoring: soft assignments turn into revenue

  KEY INSIGHT: EM is a template, not a GMM-specific trick. Hidden
  Markov Models, topic models (LDA), and missing-data imputation all
  use the same E / M structure.

  Next: 02_sklearn_gmm.py — now that you trust the library, use it on
  real customers and select K via BIC/AIC.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
