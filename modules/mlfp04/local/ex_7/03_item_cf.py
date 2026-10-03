# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.3: Item-Based Collaborative Filtering
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Flip CF from user-similarity to item-similarity
#   - Understand why item similarity is more stable than user similarity
#   - Implement item-item cosine similarity with mean-centring per item
#   - See why item-to-item CF is the classic choice for large catalogues
#
# PREREQUISITES: Exercise 7.2 (user-based CF)
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — "items that were co-rated by the same people"
#   2. Build — item similarity + weighted sum predictor
#   3. Train — precompute the item x item matrix once
#   4. Visualise — item similarity heatmap + most-similar items
#   5. Apply — Amazon-style "customers who bought this also bought..."
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.express as px
from kailash_ml import ModelVisualizer  # noqa: F401

from shared.mlfp04.ex_7 import (
    N_ITEMS,
    build_rating_dataset,
    holdout_rmse,
    print_baselines,
    print_method_scores,
    print_warm_comparison,
    save_html,
)

K_NEIGHBOURS = 20


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why item-based CF dominates at scale
# ════════════════════════════════════════════════════════════════════════
# User-based CF asks: "who rates like me?" and is unstable — users change
# tastes, new users have no history, and the N-user set grows with every
# signup (often into the millions).
#
# Item-based CF asks: "which items were rated the same way?" When the
# actively-sold catalogue is smaller than the user base, the item-item
# matrix is the cheaper one to compute. More importantly, item-item
# relationships ("people who bought A also bought B") change slowly, so the
# precompute can run nightly and still be accurate, while an individual
# user's neighbourhood shifts every time they rate something.
#
# Key trick: mean-centre PER ITEM (not per user). This removes the
# "everyone loves this item" bias and compares how items RANK in each
# user's preference order.
#
# Item-to-item CF was popularised by Amazon's 2003 paper (Linden, Smith &
# York, "Amazon.com Recommendations: Item-to-Item Collaborative
# Filtering", IEEE Internet Computing) behind the "Customers who bought
# this also bought..." feature.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD item similarity + predictor
# ════════════════════════════════════════════════════════════════════════


def item_similarity_matrix(
    R: np.ndarray, obs_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise cosine similarity between items on mean-centred ratings."""
    n_items = R.shape[1]
    sim = np.zeros((n_items, n_items))
    item_means = np.array(
        [
            float(np.nanmean(R[obs_mask[:, j], j])) if obs_mask[:, j].any() else 0.0
            for j in range(n_items)
        ]
    )
    R_centred = R.copy()
    for j in range(n_items):
        R_centred[obs_mask[:, j], j] -= item_means[j]
    R_centred[~obs_mask] = 0.0

    for i in range(n_items):
        for j in range(i, n_items):
            both = obs_mask[:, i] & obs_mask[:, j]
            if not both.any():
                continue
            ri, rj = R_centred[both, i], R_centred[both, j]
            denom = np.linalg.norm(ri) * np.linalg.norm(rj)
            if denom < 1e-10:
                continue
            # TODO: cosine similarity of the co-rated, item-centred vectors
            s = ____
            sim[i, j] = s
            sim[j, i] = s
    return sim, item_means


def item_based_cf_predict(
    R: np.ndarray,
    obs_mask: np.ndarray,
    item_sim: np.ndarray,
    k: int = K_NEIGHBOURS,
) -> np.ndarray:
    """Weighted-deviation predictor over the top-k most similar items.

    For user u and target item j:
      prediction = mean_j + sum(sim(j, i) * (r(u, i) - mean_i))
                            / sum(|sim(j, i)|)
    where i ranges over the top-k positively-similar items user u already
    rated. Working in deviations from each item's mean mirrors the centring
    used to compute the similarities.
    """
    item_means = np.array(
        [
            float(np.nanmean(R[obs_mask[:, j], j])) if obs_mask[:, j].any() else 0.0
            for j in range(R.shape[1])
        ]
    )
    n_users, n_items = R.shape
    predictions = np.full((n_users, n_items), np.nan)

    for u in range(n_users):
        rated_items = np.where(obs_mask[u])[0]
        if len(rated_items) == 0:
            continue
        for j in range(n_items):
            sims = item_sim[j, rated_items]
            if len(sims) > k:
                top_idx = np.argsort(sims)[-k:]
            else:
                top_idx = np.arange(len(sims))
            pos_idx = top_idx[sims[top_idx] > 0]
            if len(pos_idx) == 0:
                continue
            weights = sims[pos_idx]
            denom = np.abs(weights).sum()
            if denom < 1e-10:
                continue
            # TODO: user u's deviations from each neighbour item's mean,
            # weighted by similarity / denom, added to item j's mean
            deviations = ____
            predictions[u, j] = ____

    return np.clip(predictions, 1.0, 5.0)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" (precompute item-item matrix once)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Item-Based CF on SG E-commerce Ratings")
print("=" * 70)

data = build_rating_dataset()
R_train = data["R_train"]
R_observed = data["R_observed"]
train_mask = data["train_mask"]
holdout_mask = data["holdout_mask"]
item_ids = data["item_ids"]

# TODO: Precompute the item similarity matrix on the training ratings,
# then predict with your top-k item-CF function
item_sim, item_means = ____
ibcf_predictions = ____


# ── Checkpoint ──────────────────────────────────────────────────────────
ibcf_rmse, ibcf_cov = holdout_rmse(ibcf_predictions, R_observed, holdout_mask)
assert item_sim.shape == (N_ITEMS, N_ITEMS), "Item similarity must be M x M"
assert np.allclose(item_sim, item_sim.T), "Item similarity must be symmetric"
assert ibcf_rmse > 0, "Item-CF RMSE should be positive"
print(
    f"\n[ok] Checkpoint passed — Item-CF holdout RMSE={ibcf_rmse:.4f}, "
    f"coverage={ibcf_cov:.1%}\n"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE item similarity structure
# ════════════════════════════════════════════════════════════════════════
# Two visual signals matter:
#   1. The similarity matrix — does it show block structure? (= categories)
#   2. For a chosen "anchor" item, which items are most similar?
#      These are the "customers who bought this also bought..." candidates.

order = np.argsort(item_means)
item_sim_sorted = item_sim[np.ix_(order, order)]
fig_heat = px.imshow(
    item_sim_sorted,
    color_continuous_scale="RdBu",
    zmin=-1,
    zmax=1,
    title="Item-Item Similarity Heatmap (sorted by mean rating)",
    labels={"x": "item j", "y": "item i", "color": "cosine sim"},
)
save_html(fig_heat, "03_item_similarity_heatmap.html")

# Pick the most-rated item and show its top-5 neighbours
rated_counts = train_mask.sum(axis=0)
anchor = int(np.argmax(rated_counts))
# TODO: indices of the 5 most similar items to the anchor (skip itself)
top5 = ____
print(f"\nAnchor item: {item_ids[anchor]} (rated by {rated_counts[anchor]} users)")
print("Top-5 most similar items ('customers also bought'):")
for j in top5:
    print(f"  {item_ids[j]}  sim={item_sim[anchor, j]:+.3f}")

print_baselines(R_train, train_mask, R_observed, holdout_mask)
print_method_scores("Item-CF", ibcf_predictions, R_observed, holdout_mask)
print_warm_comparison(
    "Item-CF", ibcf_predictions, R_train, train_mask, R_observed, holdout_mask,
    data["cold_items"],
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Amazon-Style "Customers Who Bought This Also Bought..."
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A regional cross-border e-commerce platform serves 1.8M
# active users. Its long-tail catalogue lists millions of SKUs, but the
# "you may also like" carousel only needs neighbours for the ~60K SKUs
# that are actively sold in a given month.
#
# Why item-CF fits this setting:
#   - The similarity matrix is O(M^2) in items, not O(N^2) in users. With
#     ~60K active SKUs vs 1.8M users, the item side is the smaller one.
#     (If you had to cover every one of millions of listed SKUs, that size
#     advantage would disappear — item-CF wins on SIZE only when M < N.)
#   - Item relationships are stable: "phone + phone case" stays true for
#     years, while user taste shifts monthly
#   - Precompute nightly, cache: the sparse top-50 neighbours per item is
#     a small record per SKU, so a page load is a cache lookup
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures): on a
# platform with S$4.2B annual GMV, better cross-sell that adds just 0.1%
# to GMV is worth ~S$4.2M/year. A tuning effort costing ~S$250K/year in
# engineering + infrastructure pays back if it adds ~0.006% of GMV
# (S$250K / S$4.2B) — which is why carousel ranking is tested so heavily.
#
# LIMITATIONS:
#   - Niche items (long tail) have sparse similarity rows
#   - Items with wildly different rating distributions still leak through
#   - Cold-start NEW items still need content features (back to Ex 7.1)
#
# The next technique (04_matrix_factorisation.py) takes a completely
# different approach: learn dense user and item embeddings by minimising
# a loss — the bridge from recommenders to deep learning.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Flipped CF from user-similarity to item-similarity
  [x] Understood why items are more stable than users at scale
  [x] Built the item-to-item "customers also bought" predictor
  [x] Inspected top-5 neighbours for the most-rated item
  [x] Sized an (illustrative) cross-sell scenario for SG e-commerce

  KEY INSIGHT: Item-CF pays off when item relationships are more stable
  than user tastes and the active catalogue is smaller than the user base
  — then the item-item matrix is cheap to precompute and cache.

  Next: 04_matrix_factorisation.py — abandon similarity entirely and
  learn dense embeddings by optimisation. This is the bridge from
  classical recommenders to neural networks.
"""
)
