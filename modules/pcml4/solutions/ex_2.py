# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT4 — Exercise 2: EM Algorithm and Gaussian Mixture Models
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Derive the EM algorithm from first principles, implement
#   the E-step and M-step manually on 2D synthetic data, then compare
#   with sklearn GMM and AutoMLEngine on real e-commerce customer data.
#
# TASKS:
#   1. Derive EM intuitively — soft assignments and parameter updates
#   2. Implement E-step (posterior responsibilities)
#   3. Implement M-step (update means, covariances, weights)
#   4. Run EM loop and visualise convergence
#   5. Compare manual EM with sklearn GMM on real data
#   6. AutoMLEngine for automated GMM comparison
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio

import numpy as np
import polars as pl
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from kailash_ml import ModelVisualizer
from kailash_ml.interop import to_sklearn_input

from shared import ASCENTDataLoader


# ── Synthetic Data for Manual EM ─────────────────────────────────────

rng = np.random.default_rng(42)

# Three well-separated Gaussians in 2D — ground truth for EM validation
true_means = np.array([[0.0, 0.0], [5.0, 2.0], [2.0, 6.0]])
true_covs = np.array(
    [
        [[1.0, 0.3], [0.3, 0.8]],
        [[0.8, -0.2], [-0.2, 1.2]],
        [[1.5, 0.0], [0.0, 0.5]],
    ]
)
true_weights = np.array([0.4, 0.35, 0.25])

n_synth = 600
# Generate samples from each component proportionally
n_per_component = (true_weights * n_synth).astype(int)
n_per_component[-1] = n_synth - n_per_component[:-1].sum()  # Ensure total = n_synth

X_synth_parts = []
z_true = []
for k, (mean, cov, n) in enumerate(zip(true_means, true_covs, n_per_component)):
    X_synth_parts.append(rng.multivariate_normal(mean, cov, n))
    z_true.extend([k] * n)

X_synth = np.vstack(X_synth_parts)
z_true = np.array(z_true)

# Shuffle
idx = rng.permutation(n_synth)
X_synth, z_true = X_synth[idx], z_true[idx]

print(f"=== Synthetic 2D GMM Data ===")
print(f"Samples: {n_synth}, Components: 3")
print(f"True weights: {true_weights}")
for k, (m, n) in enumerate(zip(true_means, n_per_component)):
    print(f"  Component {k}: mean={m}, n={n}")


# ══════════════════════════════════════════════════════════════════════
# TASK 1: EM Algorithm — intuition and derivation
# ══════════════════════════════════════════════════════════════════════
# Problem: observed data X, latent assignments Z, parameters θ = {π, μ, Σ}
#
# Maximum Likelihood: max_{θ} log P(X|θ) = max_{θ} Σ_i log Σ_k π_k N(x_i|μ_k,Σ_k)
#   → Intractable: log of a sum
#
# EM insight: introduce Z, maximise a lower bound (ELBO):
#   E-step: compute Q(Z|X, θ^old) = P(Z|X, θ^old)  [posterior responsibilities]
#   M-step: θ^new = argmax_{θ} E_Q[log P(X,Z|θ)]    [weighted ML update]
#
# Why it works: each step is guaranteed to increase log P(X|θ).
# Converges to a local maximum of the likelihood.

print(f"\n=== EM Algorithm Derivation ===")
print(
    """
EM as coordinate ascent on the ELBO:

  log P(X|θ) ≥ ELBO = E_Q[log P(X,Z|θ)] + H(Q)   (Jensen's inequality)

  E-step: fix θ, maximise ELBO over Q
    → Q*(Z) = P(Z|X, θ)  [just compute the posterior]

  M-step: fix Q, maximise ELBO over θ
    → θ* = argmax E_{Q*}[log P(X,Z|θ)]  [weighted MLE]

For GMMs:
  E-step: r_{ik} = π_k N(x_i|μ_k,Σ_k) / Σ_j π_j N(x_i|μ_j,Σ_j)
  M-step:
    N_k = Σ_i r_{ik}
    π_k^new = N_k / N
    μ_k^new = (Σ_i r_{ik} x_i) / N_k
    Σ_k^new = (Σ_i r_{ik} (x_i - μ_k^new)(x_i - μ_k^new)') / N_k
"""
)


# ══════════════════════════════════════════════════════════════════════
# TASK 2: E-step — compute posterior responsibilities
# ══════════════════════════════════════════════════════════════════════


def e_step(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    E-step: compute responsibility matrix R.

    R[i, k] = P(z_i = k | x_i, θ)
             = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_j | μ_j, Σ_j)

    Returns:
        R: (n_samples, n_components) responsibility matrix
    """
    n_samples = X.shape[0]
    n_components = len(weights)
    log_probs = np.zeros((n_samples, n_components))

    for k in range(n_components):
        # Log probability under component k (more numerically stable)
        try:
            dist = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
            log_probs[:, k] = np.log(weights[k] + 1e-300) + dist.logpdf(X)
        except Exception:
            log_probs[:, k] = -np.inf

    # Normalise in log space (log-sum-exp trick for numerical stability)
    log_probs_max = log_probs.max(axis=1, keepdims=True)
    log_normaliser = (
        np.log(np.exp(log_probs - log_probs_max).sum(axis=1, keepdims=True))
        + log_probs_max
    )
    R = np.exp(log_probs - log_normaliser)

    return R


# Test E-step on true parameters
R_init = e_step(X_synth, true_means, true_covs, true_weights)
print(f"\n=== E-step Test (true parameters) ===")
print(f"Responsibility matrix shape: {R_init.shape}")
print(f"Row sums (should all be 1): {R_init.sum(axis=1)[:5].round(4)}")
print(f"Average max responsibility: {R_init.max(axis=1).mean():.4f}")
print(f"  (≈1 = high confidence; ≈0.33 = maximum uncertainty)")

# Soft assignments from true parameters
soft_assignments = R_init.argmax(axis=1)
accuracy_true_params = (soft_assignments == z_true).mean()
print(f"Assignment accuracy (true params): {accuracy_true_params:.4f}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: M-step — update parameters from responsibilities
# ══════════════════════════════════════════════════════════════════════


def m_step(
    X: np.ndarray,
    R: np.ndarray,
    reg_covar: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    M-step: update GMM parameters given responsibility matrix R.

    N_k = Σ_i r_{ik}
    π_k = N_k / N
    μ_k = (Σ_i r_{ik} x_i) / N_k
    Σ_k = (Σ_i r_{ik} (x_i - μ_k)(x_i - μ_k)') / N_k  +  reg_covar * I

    Returns:
        means:   (n_components, n_features)
        covs:    (n_components, n_features, n_features)
        weights: (n_components,)
    """
    n_samples, n_features = X.shape
    n_components = R.shape[1]

    N_k = R.sum(axis=0) + 1e-300  # Effective sample counts per component
    weights = N_k / n_samples  # Normalised mixing weights

    means = (R.T @ X) / N_k[:, np.newaxis]  # (K, D)

    covs = np.zeros((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]  # (N, D)
        # Weighted outer products: Σ_k = (1/N_k) Σ_i r_{ik} (x_i - μ_k)(x_i - μ_k)'
        covs[k] = (R[:, k : k + 1] * diff).T @ diff / N_k[k]
        covs[k] += reg_covar * np.eye(n_features)  # Regularise for stability

    return means, covs, weights


# Test M-step: given true responsibilities, should recover near-true params
R_true = np.zeros((n_synth, 3))
for i, k in enumerate(z_true):
    R_true[i, k] = 1.0  # Hard assignments from ground truth

means_recovered, covs_recovered, weights_recovered = m_step(X_synth, R_true)

print(f"\n=== M-step Test (hard assignments from ground truth) ===")
print(f"Recovered weights: {weights_recovered.round(3)} (true: {true_weights})")
for k in range(3):
    print(
        f"  Component {k}: mean={means_recovered[k].round(3)} (true: {true_means[k]})"
    )


# ══════════════════════════════════════════════════════════════════════
# TASK 4: EM loop — iterate until convergence
# ══════════════════════════════════════════════════════════════════════


def compute_log_likelihood(
    X: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute log-likelihood: Σ_i log Σ_k π_k N(x_i | μ_k, Σ_k)."""
    n_samples = X.shape[0]
    n_components = len(weights)
    log_likelihoods = np.zeros(n_samples)

    for k in range(n_components):
        try:
            dist = multivariate_normal(mean=means[k], cov=covs[k], allow_singular=True)
            log_likelihoods = np.logaddexp(
                log_likelihoods,
                np.log(weights[k] + 1e-300) + dist.logpdf(X),
            )
        except Exception:
            pass

    return log_likelihoods.sum()


def fit_gmm_em(
    X: np.ndarray,
    n_components: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
) -> dict:
    """
    Fit GMM using manual EM loop.

    Initialises with K-means++, then alternates E/M steps
    until log-likelihood converges.
    """
    rng_em = np.random.default_rng(seed)
    n_samples, n_features = X.shape

    # Initialise with random assignment (K-means++ style)
    idx = rng_em.choice(n_samples, n_components, replace=False)
    means = X[idx].copy()
    covs = np.array([np.eye(n_features)] * n_components)
    weights = np.ones(n_components) / n_components

    log_likelihoods = []

    for iteration in range(max_iter):
        # E-step
        R = e_step(X, means, covs, weights)

        # M-step
        means, covs, weights = m_step(X, R)

        # Track log-likelihood
        ll = compute_log_likelihood(X, means, covs, weights)
        log_likelihoods.append(ll)

        # Convergence check
        if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
            print(f"  Converged at iteration {iteration + 1}")
            break

    labels = R.argmax(axis=1)
    return {
        "means": means,
        "covs": covs,
        "weights": weights,
        "labels": labels,
        "log_likelihoods": log_likelihoods,
        "n_iter": len(log_likelihoods),
        "final_ll": log_likelihoods[-1],
    }


print(f"\n=== Running Manual EM on Synthetic Data ===")
em_result = fit_gmm_em(X_synth, n_components=3, max_iter=100, tol=1e-4)

print(f"Iterations: {em_result['n_iter']}")
print(f"Final log-likelihood: {em_result['final_ll']:.2f}")
print(f"Recovered weights: {em_result['weights'].round(3)} (true: {true_weights})")
for k in range(3):
    print(
        f"  Component {k}: mean={em_result['means'][k].round(3)} (true: {true_means[k]})"
    )

# Evaluate cluster recovery
# Match components by closest mean (true label assignment may differ)
em_labels = em_result["labels"]
if len(set(em_labels)) > 1:
    sil = silhouette_score(X_synth, em_labels)
    print(f"Silhouette score: {sil:.4f}")

# Log-likelihood convergence plot
viz = ModelVisualizer()
fig = viz.training_history(
    {"Log-Likelihood": em_result["log_likelihoods"]},
    x_label="EM Iteration",
)
fig.update_layout(title="EM Convergence: Log-Likelihood per Iteration")
fig.write_html("ex2_em_convergence.html")
print("Saved: ex2_em_convergence.html")


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Compare manual EM with sklearn GMM on real data
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
customers = loader.load("ascent04", "ecommerce_customers.parquet")

feature_cols = [
    c
    for c, d in zip(customers.columns, customers.dtypes)
    if d in (pl.Float64, pl.Float32, pl.Int64, pl.Int32) and c not in ("customer_id",)
]

X_real, _, _ = to_sklearn_input(
    customers.drop_nulls(subset=feature_cols),
    feature_columns=feature_cols,
)

scaler = StandardScaler()
X_real_scaled = scaler.fit_transform(X_real)

print(f"\n=== Real Data: E-commerce Customers ===")
print(f"Shape: {X_real_scaled.shape}")

# Test different numbers of components using BIC/AIC model selection
print(f"\n=== GMM Model Selection (BIC/AIC) ===")
print(f"{'K':>4} {'BIC':>12} {'AIC':>12} {'Log-L':>12} {'Silhouette':>12}")
print("─" * 56)

bic_scores = {}
for k in range(2, 9):
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        random_state=42,
        max_iter=200,
    )
    gmm.fit(X_real_scaled)
    labels = gmm.predict(X_real_scaled)

    bic = gmm.bic(X_real_scaled)
    aic = gmm.aic(X_real_scaled)
    ll = gmm.score(X_real_scaled) * X_real_scaled.shape[0]

    sil = silhouette_score(X_real_scaled, labels) if len(set(labels)) > 1 else -1.0
    bic_scores[k] = {"bic": bic, "aic": aic, "ll": ll, "silhouette": sil, "gmm": gmm}

    print(f"{k:>4} {bic:>12.0f} {aic:>12.0f} {ll:>12.0f} {sil:>12.4f}")

best_k_bic = min(bic_scores.items(), key=lambda x: x[1]["bic"])
best_k_sil = max(bic_scores.items(), key=lambda x: x[1]["silhouette"])
print(f"\nBest K by BIC: {best_k_bic[0]}")
print(f"Best K by Silhouette: {best_k_sil[0]}")

print("\nBIC vs AIC vs Silhouette:")
print("  BIC: penalises model complexity (prefer smaller K)")
print("  AIC: less conservative than BIC (can favour larger K)")
print("  Silhouette: cluster separation (domain-interpretable)")
print("  Choose K based on both statistical criteria AND business meaning")

# Best model: fit and profile
k_final = best_k_bic[0]
gmm_best = bic_scores[k_final]["gmm"]
labels_best = gmm_best.predict(X_real_scaled)
soft_probs = gmm_best.predict_proba(X_real_scaled)

print(f"\n=== Best GMM (K={k_final}) ===")
print(f"Component weights: {gmm_best.weights_.round(3)}")
print(f"Average max probability: {soft_probs.max(axis=1).mean():.4f}")
print(f"  (high = crisp separation, low = overlapping clusters)")

# Profile each component
customers_with_clusters = customers.drop_nulls(subset=feature_cols).with_columns(
    pl.Series("gmm_cluster", labels_best)
)

print(f"\n=== Customer Segment Profiles ===")
for k in range(k_final):
    subset = customers_with_clusters.filter(pl.col("gmm_cluster") == k)
    print(f"\nSegment {k} (n={subset.height:,}, weight={gmm_best.weights_[k]:.3f}):")
    for col in feature_cols[:4]:
        mean_val = subset[col].mean()
        overall_mean = customers_with_clusters[col].mean()
        diff_pct = (mean_val - overall_mean) / (abs(overall_mean) + 1e-9) * 100
        indicator = "HIGH" if diff_pct > 15 else "low" if diff_pct < -15 else "avg"
        print(f"  {col:<28} {mean_val:>10.2f}  [{indicator:>4}] {diff_pct:+.1f}%")


# ══════════════════════════════════════════════════════════════════════
# TASK 6: AutoMLEngine for automated comparison
# ══════════════════════════════════════════════════════════════════════

from kailash_ml.engines.automl_engine import AutoMLEngine, AutoMLConfig


async def automl_gmm():
    """Use AutoMLEngine to automate GMM hyperparameter search."""
    config = AutoMLConfig(
        task_type="clustering",
        metric_to_optimize="bic",
        direction="minimize",
        search_strategy="random",
        search_n_trials=15,
        agent=False,  # Double opt-in: set True + install kailash-ml[agents]
        max_llm_cost_usd=0.5,
    )

    print(f"\n=== AutoMLEngine Config ===")
    print(f"Task: {config.task_type}")
    print(f"Optimising: {config.metric_to_optimize} ({config.direction})")
    print(
        f"Search strategy: {config.search_strategy} ({config.search_n_trials} trials)"
    )
    print(f"Agent LLM guidance: {config.agent} (double opt-in)")

    print("\nAutoMLEngine would search:")
    print("  n_components: 2..10")
    print("  covariance_type: full, tied, diag, spherical")
    print("  init_params: kmeans, k-means++, random")
    print("  → Selects configuration minimising BIC across CV folds")

    return config


asyncio.run(automl_gmm())

# Summary comparison
comparison = {
    f"GMM K={k}": {"BIC": v["bic"], "Silhouette": v["silhouette"]}
    for k, v in bic_scores.items()
}
fig_cmp = viz.metric_comparison(comparison)
fig_cmp.update_layout(title="GMM: BIC and Silhouette vs Number of Components")
fig_cmp.write_html("ex2_gmm_comparison.html")
print("\nSaved: ex2_gmm_comparison.html")

print("\n✓ Exercise 2 complete — EM algorithm from scratch + sklearn GMM")
print("  Key takeaways:")
print("  1. EM = coordinate ascent on ELBO; each step raises log-likelihood")
print("  2. Soft assignments r_{ik} are the 'hidden variable' estimates")
print("  3. BIC penalises complexity → objective model selection criterion")
print("  4. GMM = probabilistic generalisation of K-means with soft boundaries")
