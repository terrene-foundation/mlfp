# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.4: Matrix Factorisation with ALS (from scratch)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Implement biased Alternating Least Squares (ALS) matrix factorisation
#   - Recover true latent structure from a sparse rating matrix
#   - Track convergence — verify the regularised loss is non-increasing
#   - Visualise learned user + item embeddings in 2D
#   - Understand the SVD++ extension for implicit feedback
#   - Articulate THE PIVOT: optimisation drives feature discovery
#
# PREREQUISITES: Exercises 7.1-7.3; MLFP04 Ex 3 (PCA/SVD)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why R ≈ mu + b_u + b_j + U V^T and why ALS converges
#   2. Build — ALS update equations from scratch
#   3. Train — run the alternating least squares loop
#   4. Visualise — user + item embeddings (PCA projection) + convergence
#   5. Apply — music-streaming recommendation at scale
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import polars as pl
from kailash_ml import ModelVisualizer
from sklearn.decomposition import PCA

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    N_LATENT_TRUE,
    N_USERS,
    build_rating_dataset,
    holdout_rmse,
    print_baselines,
    print_method_scores,
    print_warm_comparison,
    save_html,
)
from shared.mlfp04 import create_visualizer

K_LATENT = N_LATENT_TRUE  # we search for as many factors as the data has
LAMBDA_REG = 5.0
N_ITERATIONS = 30


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why matrix factorisation works
# ════════════════════════════════════════════════════════════════════════
# The central assumption: every user and every item can be represented by
# a short vector of latent factors (k = 5 here), plus a per-user and a
# per-item bias. A rating is the global mean plus both biases plus the
# inner product of the user's factors and the item's factors:
#
#   R[u, j] ≈ mu + b_u + b_j + U[u] · V[j]
#
# The biases soak up "this user is generous" and "this SKU is simply good";
# the factors capture taste — WHICH kinds of items this user likes.
#
# If we could see the full R, SVD would give us U and V in closed form.
# Since R is sparse (only ~30% observed), we minimise a regularised loss:
#
#   L = sum_{(u,j) observed} (R[u,j] - mu - b_u - b_j - U[u] · V[j])^2
#     + lambda * (||U||^2 + ||V||^2 + ||b_user||^2 + ||b_item||^2)
#
# ALS: fix the item side, solve for each user's [U[u], b_u] (a ridge
# regression). Then fix the user side, solve for each item's [V[j], b_j].
# Each sub-problem is solved EXACTLY in closed form, so the regularised
# loss L can never go up from one half-step to the next — L is monotone
# non-increasing by construction. The plain training RMSE is NOT
# guaranteed to fall every iteration (the penalty term can trade a little
# fit for smaller weights), so we track L for the convergence check.
#
# Why regularise hard? Each user has only ~25 training ratings. Without a
# strong lambda, k factors per user can memorise those ratings (tiny train
# RMSE) and generalise terribly (large holdout RMSE) — overfitting.
#
# Connection to SVD: when R is fully observed and lambda = 0, ALS
# converges to the truncated SVD. Sparse + regularised ALS is a GENERALISED
# SVD that handles missing values gracefully.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD the ALS update step
# ════════════════════════════════════════════════════════════════════════


def als_matrix_factorisation(
    R: np.ndarray,
    obs_mask: np.ndarray,
    k: int,
    lam: float,
    n_iter: int,
    rng: np.random.Generator,
) -> dict:
    """Biased Alternating Least Squares matrix factorisation from scratch.

    Returns a dict with U, V, b_user, b_item, mu, predictions,
    loss_history (regularised objective) and rmse_history (train RMSE).
    Items with no training ratings get NaN predictions: ALS has nothing to
    place them with.
    """
    n_users, n_items = R.shape
    mu = float(R[obs_mask].mean())
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    b_user = np.zeros(n_users)
    b_item = np.zeros(n_items)
    R_safe = np.nan_to_num(R, nan=0.0)
    penalty = lam * np.eye(k + 1)
    loss_history: list[float] = []
    rmse_history: list[float] = []

    for iteration in range(n_iter):
        # Fix the item side, solve a ridge regression for each user.
        # Design matrix = [V_j, 1] so the solve returns [U[u], b_u].
        for u in range(n_users):
            rated = np.where(obs_mask[u])[0]
            if len(rated) == 0:
                continue
            X_u = np.hstack([V[rated], np.ones((len(rated), 1))])
            y_u = R_safe[u, rated] - mu - b_item[rated]
            A = X_u.T @ X_u + penalty
            b = X_u.T @ y_u
            solution = np.linalg.solve(A, b)
            U[u], b_user[u] = solution[:k], solution[k]

        # Fix the user side, solve a ridge regression for each item.
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            X_j = np.hstack([U[raters], np.ones((len(raters), 1))])
            y_j = R_safe[raters, j] - mu - b_user[raters]
            A = X_j.T @ X_j + penalty
            b = X_j.T @ y_j
            solution = np.linalg.solve(A, b)
            V[j], b_item[j] = solution[:k], solution[k]

        R_hat = mu + b_user[:, None] + b_item[None, :] + U @ V.T
        residuals = (R_safe - R_hat)[obs_mask]
        loss = float(
            (residuals**2).sum()
            + lam * ((U**2).sum() + (V**2).sum() + (b_user**2).sum() + (b_item**2).sum())
        )
        rmse = float(np.sqrt(np.mean(residuals**2)))
        loss_history.append(loss)
        rmse_history.append(rmse)
        if iteration % 5 == 0 or iteration == n_iter - 1:
            print(f"  iter {iteration:3d}: loss = {loss:10.2f}  train RMSE = {rmse:.4f}")

    predictions = np.clip(mu + b_user[:, None] + b_item[None, :] + U @ V.T, 1.0, 5.0)
    predictions[:, obs_mask.sum(axis=0) == 0] = np.nan  # unseen items
    return {
        "U": U,
        "V": V,
        "b_user": b_user,
        "b_item": b_item,
        "mu": mu,
        "predictions": predictions,
        "loss_history": loss_history,
        "rmse_history": rmse_history,
    }


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN via ALS
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ALS Matrix Factorisation on SG E-commerce Ratings")
print("=" * 70)
print(f"k={K_LATENT}, lambda={LAMBDA_REG}, iterations={N_ITERATIONS}")

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
U_true = data["U_true"]
V_true = data["V_true"]

als = als_matrix_factorisation(
    R_train,
    train_mask,
    k=K_LATENT,
    lam=LAMBDA_REG,
    n_iter=N_ITERATIONS,
    rng=data["rng"],
)
U_learned, V_learned = als["U"], als["V"]
loss_history, rmse_history = als["loss_history"], als["rmse_history"]
als_predictions = als["predictions"]
als_rmse, als_cov = holdout_rmse(als_predictions, R_observed, holdout_mask)


# ── Checkpoint ──────────────────────────────────────────────────────────
assert U_learned.shape == (N_USERS, K_LATENT), "U must be (N_USERS, K_LATENT)"
assert V_learned.shape == (N_ITEMS, K_LATENT), "V must be (N_ITEMS, K_LATENT)"
for i in range(1, len(loss_history)):
    assert loss_history[i] <= loss_history[i - 1] * (1 + 1e-9), (
        f"Regularised ALS loss must be non-increasing (violated at step {i})"
    )
print(
    f"\n[ok] Checkpoint passed — ALS loss non-increasing: "
    f"{loss_history[0]:.1f} -> {loss_history[-1]:.1f}; train RMSE "
    f"{rmse_history[-1]:.4f}, holdout RMSE={als_rmse:.4f}, coverage={als_cov:.0%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE embeddings + convergence
# ════════════════════════════════════════════════════════════════════════
# The 5-dimensional learned factors can be projected to 2D via PCA. The
# result is a scatter plot where geometric proximity = similar taste.

viz = create_visualizer()

pca_users = PCA(n_components=2, random_state=42)
U_2d = pca_users.fit_transform(U_learned)
user_df = pl.DataFrame(
    {
        "user_id": data["user_ids"],
        "pc1": U_2d[:, 0].tolist(),
        "pc2": U_2d[:, 1].tolist(),
        "learned_user_bias": als["b_user"].tolist(),
    }
)
fig_u = viz.scatter(user_df, x="pc1", y="pc2", color="learned_user_bias")
fig_u.update_layout(title="ALS User Embeddings (2D PCA projection)")
save_html(fig_u, "04_als_user_embeddings.html")

pca_items = PCA(n_components=2, random_state=42)
V_2d = pca_items.fit_transform(V_learned)
item_df = pl.DataFrame(
    {
        "item_id": data["item_ids"],
        "pc1": V_2d[:, 0].tolist(),
        "pc2": V_2d[:, 1].tolist(),
        "learned_item_bias": als["b_item"].tolist(),
    }
)
fig_i = viz.scatter(item_df, x="pc1", y="pc2", color="learned_item_bias")
fig_i.update_layout(title="ALS Item Embeddings (2D PCA projection)")
save_html(fig_i, "04_als_item_embeddings.html")

fig_conv = viz.training_history(
    {"regularised_loss": loss_history}, x_label="ALS iteration"
)
fig_conv.update_layout(title="ALS Convergence (regularised objective)")
save_html(fig_conv, "04_als_convergence.html")


# Subspace alignment — did ALS recover the true latent structure?
def subspace_similarity(A: np.ndarray, B: np.ndarray) -> float:
    """Mean cosine of the principal angles between span(A) and span(B).

    1.0 = identical subspaces. Uses ALL columns of both matrices, so the
    arbitrary rotation/order of learned factors does not matter.
    """
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)
    sigmas = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.mean(np.minimum(sigmas, 1.0)))


warm_items = ~data["cold_items"]  # cold SKUs were never trained
user_align = subspace_similarity(U_learned, U_true)
item_align = subspace_similarity(V_learned[warm_items], V_true[warm_items])
chance_rng = np.random.default_rng(0)
user_chance = subspace_similarity(chance_rng.normal(size=U_true.shape), U_true)
item_chance = subspace_similarity(
    chance_rng.normal(size=V_true[warm_items].shape), V_true[warm_items]
)
print(
    f"\nSubspace recovery (1.0 = perfect): user={user_align:.3f} "
    f"(random-subspace baseline {user_chance:.3f}), item={item_align:.3f} "
    f"(baseline {item_chance:.3f})"
)
recovered = user_align > user_chance + 0.3 and item_align > item_chance + 0.3
if recovered:
    print(
        "  -> ALS found the hidden taste space far better than chance, "
        "without ever being told what the factors mean."
    )
else:
    print(
        "  -> Recovery is close to the random baseline on this run — the "
        "factors fit the ratings but do NOT match the true taste space."
    )

print_baselines(R_train, train_mask, R_observed, holdout_mask)
print_method_scores("ALS MF", als_predictions, R_observed, holdout_mask)
print_warm_comparison(
    "ALS MF", als_predictions, R_train, train_mask, R_observed, holdout_mask,
    data["cold_items"],
)
overfit_gap = als_rmse - rmse_history[-1]
print(
    f"  Train RMSE {rmse_history[-1]:.4f} vs holdout RMSE {als_rmse:.4f} "
    f"(gap {overfit_gap:+.4f}). A large gap would mean lambda is too small."
)


# ════════════════════════════════════════════════════════════════════════
# SVD++ — conceptual extension (no code, just the intuition)
# ════════════════════════════════════════════════════════════════════════
# Plain MF: r_hat(u, i) = U[u] · V[i]
#
# SVD++ adds implicit feedback (clicks, views, not just ratings):
#
#   r_hat(u, i) = mu + b_u + b_i
#               + V[i] · (U[u] + |N(u)|^(-1/2) * sum_{j in N(u)} y[j])
#
# where:
#   mu, b_u, b_i  = global/user/item biases
#   N(u)          = items user u interacted with (implicit set)
#   y[j]          = implicit feedback vectors
#
# The term after U[u] is "if you clicked these items, your taste drifts
# in this direction." It captures information that pure rating-based MF
# misses. SVD++ (Koren, KDD 2008) was one of the core models in the
# BellKor team's Netflix Prize blends.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Music-Streaming Recommendation at Scale
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Southeast Asian music streaming service has 600M "plays"
# per day across 80M tracks and 12M users.
# Explicit ratings (thumbs up/down) are rare — <3% of plays. Implicit
# feedback (skipped after 5s, played to completion, added to playlist) is
# the only usable signal at volume.
#
# Why matrix factorisation is the right tool:
#   - Implicit feedback is ~97% of the signal — user-CF and item-CF
#     struggle with binary play/skip data; MF with a confidence weighting
#     (Hu et al. 2008) handles it natively
#   - Embeddings compress 12M x 80M = 960T pairs down to (12M + 80M) x
#     128 float32 numbers ~ 47GB — big, but it fits in the RAM of one
#     large server, unlike the 960-trillion-cell matrix
#   - Once trained, scoring a (user, track) pair is a single 128-number
#     dot product — cheap enough to score thousands of candidates per
#     request
#   - The same embeddings can power "similar tracks", a personalised daily
#     mix and a weekly discovery playlist — train once, serve several
#     products
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures): on a
# market with S$180M annual streaming revenue, if better discovery
# recommendations cut churn enough to retain 2% of revenue, that is
# ~S$3.6M/year. The size of the real effect has to be measured with an
# online A/B test; the offline holdout metrics above only tell you whether
# ALS ranks held-out items better than the baselines.
#
# LIMITATIONS:
#   - Cold-start tracks: a brand-new upload has no listens = no factors
#     (solution: hybrid with content-based audio embeddings)
#   - Popularity bias: top-streamed tracks dominate every recommendation
#   - Filter bubble: a low-dimensional taste space collapses the long tail
#
# THE PIVOT: optimisation drives feature discovery.
#
#   We never told ALS what the 5 latent dimensions mean. It learned them
#   by minimising reconstruction error. Compare:
#       MF:              min ||R - U V^T||^2
#       Neural net:      min ||y - f(W x + b)||^2
#   SAME PRINCIPLE. Neural nets just add non-linearities.
#
# The next technique (05_hybrid_evaluation.py) combines all four
# recommenders. ALS cannot score the brand-new SKUs (no ratings, no
# factors) — a hybrid can route those to content-based filtering.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Implemented biased ALS from scratch — ridge regression per user + per item
  [x] Verified the regularised loss is non-increasing ({loss_history[0]:.1f} -> {loss_history[-1]:.1f})
  [x] Measured subspace recovery against a random baseline
      (user {user_align:.3f} vs {user_chance:.3f}, item {item_align:.3f} vs {item_chance:.3f})
  [x] Visualised 5-dim embeddings in 2D via PCA
  [x] Understood SVD++ as the implicit-feedback extension of MF
  [x] Sized an (illustrative) SEA streaming scenario for MF

  THE PIVOT: Matrix factorisation DISCOVERS latent factors by optimising
  a loss. This is the same principle as neural network training — the
  hidden layer is just an embedding, learned by backpropagation.

  Next: 05_hybrid_evaluation.py — combine all four recommenders and see
  why Netflix Prize teams won by blending many models.
"""
)
