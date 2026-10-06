# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 7.5: Hybrid Recommender + Full Evaluation
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Blend four recommenders into a hybrid with weights tuned on a
#     validation split (never on the test holdout)
#   - Compare every method on RMSE, coverage, precision@k, and MAP —
#     against no-skill baselines
#   - Understand why ranking metrics matter more than RMSE in production
#   - Explain implicit vs explicit feedback and when each applies
#   - See why Netflix Prize teams won by blending many models
#
# PREREQUISITES: Exercises 7.1, 7.2, 7.3, 7.4
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — when (and why) blends beat a single model
#   2. Build — re-run all four base recommenders and the hybrid blender
#   3. Train — tune the blend weights on the validation split
#   4. Visualise — side-by-side method comparison (RMSE + MAP)
#   5. Apply — Singapore news aggregator front-page personalisation
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
from kailash_ml import ModelVisualizer
from scipy.optimize import nnls

from shared.mlfp04.ex_7 import (
    N_LATENT_TRUE,
    baseline_predictions,
    build_rating_dataset,
    evaluate_method,
    save_html,
)
from shared.mlfp004 import create_visualizer

K_LATENT = N_LATENT_TRUE
LAMBDA_REG = 5.0
N_ITERATIONS = 30
K_NEIGHBOURS = 20


# ════════════════════════════════════════════════════════════════════════
# THEORY — When hybrids beat any single method
# ════════════════════════════════════════════════════════════════════════
# Every recommender in this exercise has a failure mode:
#   - Content-based: limited by feature quality, filter bubble
#   - User-CF: O(N^2) compute, cold-start users AND cold-start items
#   - Item-CF: niche-item sparsity, cold-start items
#   - ALS MF: popularity bias, cold-start everything
#
# A hybrid combines their predictions. It helps when the components make
# DIFFERENT mistakes — e.g. CF methods cannot score a brand-new SKU at all,
# while content-based can. If one component is better than the others
# everywhere, a well-tuned blend simply puts (almost) all its weight on
# that component — and a badly-tuned blend makes things worse.
#
# Blend weights are model parameters, so they must be tuned on data the
# final evaluation never sees. We split the training ratings into a FIT
# part (base models learn from it) and a VALIDATION part (blend weights
# are chosen on it). The test holdout is used exactly once, at the end.
#
# Here the weights come from non-negative least squares (NNLS): find
# w >= 0 minimising sum_val (r - sum_m w_m * pred_m)^2. That is a tiny
# "stacking" model. The Netflix Prize teams used far richer stacked
# blends: the 2007 Progress Prize entry combined 107 models, and the 2009
# winning entry blended several hundred.
#
# RMSE vs ranking metrics:
#   RMSE measures rating prediction accuracy. Precision@k and MAP measure
#   whether the TOP recommendations are actually good. For production
#   systems, ranking matters more — users never see predictions 10-50.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: rerun the four base models and the blender
# ════════════════════════════════════════════════════════════════════════
# To keep this file standalone we inline the four algorithms from
# 01-04 (same maths, compact form) so it runs top-to-bottom.


def content_based_predict(R, item_feats, obs_mask):
    n_users, n_items = R.shape
    preds = np.full((n_users, n_items), np.nan)
    feats_c = item_feats - item_feats.mean(axis=0)
    for u in range(n_users):
        rated = np.where(obs_mask[u])[0]
        if len(rated) == 0:
            continue
        mean_u = float(R[u, rated].mean())
        std_u = float(R[u, rated].std()) + 1e-9
        profile = ((R[u, rated] - mean_u)[:, None] * feats_c[rated]).sum(axis=0)
        pn = np.linalg.norm(profile)
        if pn < 1e-10:
            continue
        profile /= pn
        for j in range(n_items):
            fn = np.linalg.norm(feats_c[j])
            if fn < 1e-10:
                continue
            preds[u, j] = mean_u + 2.0 * std_u * (profile @ feats_c[j] / fn)
    return np.clip(preds, 1.0, 5.0)


def user_cf_predict(R, obs_mask, k=K_NEIGHBOURS):
    n_users, n_items = R.shape
    user_means = np.array(
        [
            float(np.nanmean(R[u, obs_mask[u]])) if obs_mask[u].any() else 0.0
            for u in range(n_users)
        ]
    )
    Rc = R.copy()
    for u in range(n_users):
        Rc[u, obs_mask[u]] -= user_means[u]
    Rc[~obs_mask] = 0.0
    sim = np.zeros((n_users, n_users))
    for u in range(n_users):
        for v in range(u, n_users):
            both = obs_mask[u] & obs_mask[v]
            if not both.any():
                continue
            ru, rv = Rc[u, both], Rc[v, both]
            d = np.linalg.norm(ru) * np.linalg.norm(rv)
            if d < 1e-10:
                continue
            s = float(ru @ rv / d)
            sim[u, v] = s
            sim[v, u] = s
    preds = np.full((n_users, n_items), np.nan)
    for u in range(n_users):
        s = sim[u].copy()
        s[u] = -np.inf
        top = np.argsort(s)[-k:]
        top = top[s[top] > 0]
        if len(top) == 0:
            continue
        for j in range(n_items):
            rated_n = top[obs_mask[top, j]]
            if len(rated_n) == 0:
                continue
            w = sim[u, rated_n]
            denom = np.abs(w).sum()
            if denom < 1e-10:
                continue
            preds[u, j] = (
                user_means[u] + (w @ (R[rated_n, j] - user_means[rated_n])) / denom
            )
    return np.clip(preds, 1.0, 5.0)


def item_cf_predict(R, obs_mask, k=K_NEIGHBOURS):
    n_users, n_items = R.shape
    item_means = np.array(
        [
            float(np.nanmean(R[obs_mask[:, j], j])) if obs_mask[:, j].any() else 0.0
            for j in range(n_items)
        ]
    )
    Rc = R.copy()
    for j in range(n_items):
        Rc[obs_mask[:, j], j] -= item_means[j]
    Rc[~obs_mask] = 0.0
    sim = np.zeros((n_items, n_items))
    for i in range(n_items):
        for j in range(i, n_items):
            both = obs_mask[:, i] & obs_mask[:, j]
            if not both.any():
                continue
            ri, rj = Rc[both, i], Rc[both, j]
            d = np.linalg.norm(ri) * np.linalg.norm(rj)
            if d < 1e-10:
                continue
            s = float(ri @ rj / d)
            sim[i, j] = s
            sim[j, i] = s
    preds = np.full((n_users, n_items), np.nan)
    for u in range(n_users):
        rated = np.where(obs_mask[u])[0]
        if len(rated) == 0:
            continue
        for j in range(n_items):
            sims = sim[j, rated]
            top_idx = np.argsort(sims)[-k:] if len(sims) > k else np.arange(len(sims))
            pos = top_idx[sims[top_idx] > 0]
            if len(pos) == 0:
                continue
            w = sims[pos]
            denom = np.abs(w).sum()
            if denom < 1e-10:
                continue
            dev = R[u, rated[pos]] - item_means[rated[pos]]
            preds[u, j] = item_means[j] + (w @ dev) / denom
    return np.clip(preds, 1.0, 5.0)


def als_predict(R, obs_mask, k, lam, n_iter, rng):
    n_users, n_items = R.shape
    mu = float(R[obs_mask].mean())
    U = rng.normal(0, 0.1, size=(n_users, k))
    V = rng.normal(0, 0.1, size=(n_items, k))
    b_user = np.zeros(n_users)
    b_item = np.zeros(n_items)
    R_safe = np.nan_to_num(R, nan=0.0)
    penalty = lam * np.eye(k + 1)
    for _ in range(n_iter):
        for u in range(n_users):
            rated = np.where(obs_mask[u])[0]
            if len(rated) == 0:
                continue
            X = np.hstack([V[rated], np.ones((len(rated), 1))])
            y = R_safe[u, rated] - mu - b_item[rated]
            sol = np.linalg.solve(X.T @ X + penalty, X.T @ y)
            U[u], b_user[u] = sol[:k], sol[k]
        for j in range(n_items):
            raters = np.where(obs_mask[:, j])[0]
            if len(raters) == 0:
                continue
            X = np.hstack([U[raters], np.ones((len(raters), 1))])
            y = R_safe[raters, j] - mu - b_user[raters]
            sol = np.linalg.solve(X.T @ X + penalty, X.T @ y)
            V[j], b_item[j] = sol[:k], sol[k]
    preds = np.clip(mu + b_user[:, None] + b_item[None, :] + U @ V.T, 1.0, 5.0)
    preds[:, obs_mask.sum(axis=0) == 0] = np.nan
    return preds


def fit_blend_weights(
    all_preds: dict, R_true: np.ndarray, val_mask: np.ndarray, fill_value: float
) -> dict:
    """Non-negative least-squares blend weights fitted on the VALIDATION split.

    Missing component predictions are filled with ``fill_value`` (the
    global training mean) for the fit only.
    """
    names = list(all_preds)
    X_val = np.column_stack(
        [np.nan_to_num(all_preds[n][val_mask], nan=fill_value) for n in names]
    )
    y_val = R_true[val_mask]
    w, _ = nnls(X_val, y_val)
    return dict(zip(names, w))


def blend_hybrid(all_preds: dict, weights: dict, fallback: np.ndarray) -> np.ndarray:
    """Weighted blend over the components that CAN score each pair.

    For each (user, item): average the available component predictions
    with the tuned weights (renormalised over the available ones). If no
    weighted component can score the pair — e.g. a cold-start SKU that
    only content-based can see — use ``fallback`` instead.
    """
    names = list(all_preds)
    stack = np.stack([all_preds[n] for n in names])
    w = np.array([weights[n] for n in names])[:, None, None]
    available = ~np.isnan(stack)
    w_avail = w * available
    denom = w_avail.sum(axis=0)
    blended = (w_avail * np.nan_to_num(stack)).sum(axis=0) / np.where(
        denom > 0, denom, 1.0
    )
    return np.where(denom > 0, blended, fallback)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — FIT base models, TUNE blend on validation, EVALUATE on holdout
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  Hybrid Recommender + Full Evaluation")
print("=" * 70)

data = build_rating_dataset()
R_observed = data["R_observed"]
fit_mask = data["fit_mask"]
val_mask = data["val_mask"]
holdout_mask = data["holdout_mask"]
item_features = data["item_features"]
cold_items = data["cold_items"]
R_fit = np.where(fit_mask, R_observed, np.nan)
global_mean = float(R_fit[fit_mask].mean())

print(
    f"\nSplit sizes: fit={int(fit_mask.sum())}  validation={int(val_mask.sum())}  "
    f"test holdout={int(holdout_mask.sum())} ratings "
    f"({int(cold_items.sum())} cold-start SKUs appear only in the holdout)"
)

print("\nRunning base recommenders on the FIT split...")
cb = content_based_predict(R_fit, item_features, fit_mask)
ucf = user_cf_predict(R_fit, fit_mask)
icf = item_cf_predict(R_fit, fit_mask)
als = als_predict(R_fit, fit_mask, K_LATENT, LAMBDA_REG, N_ITERATIONS, data["rng"])

all_predictions = {
    "Content-Based": cb,
    "User-CF": ucf,
    "Item-CF": icf,
    "ALS MF": als,
}

blend_weights = fit_blend_weights(all_predictions, R_observed, val_mask, global_mean)
print("\nBlend weights (NNLS on the validation split):")
for name, w in blend_weights.items():
    print(f"  {name:<18} {w:.3f}")
hybrid_preds = blend_hybrid(all_predictions, blend_weights, fallback=cb)

eval_results: dict = {}
for name, preds in baseline_predictions(R_fit, fit_mask).items():
    eval_results[f"[baseline] {name}"] = evaluate_method(
        preds, R_observed, holdout_mask
    )
for name, preds in all_predictions.items():
    eval_results[name] = evaluate_method(preds, R_observed, holdout_mask)
eval_results["Hybrid"] = evaluate_method(hybrid_preds, R_observed, holdout_mask)

print(f"\n{'Method':<24} {'RMSE':>8} {'Coverage':>10} {'P@5':>8} {'MAP':>8}")
print("─" * 62)
for name, r in eval_results.items():
    print(
        f"{name:<24} {r['RMSE']:>8.4f} {r['Coverage']:>9.1%} "
        f"{r['P@5']:>8.4f} {r['MAP']:>8.4f}"
    )

base_methods = list(all_predictions)
best_single = max(base_methods, key=lambda n: eval_results[n]["MAP"])
best_single_map = eval_results[best_single]["MAP"]
hybrid_map = eval_results["Hybrid"]["MAP"]
lift = hybrid_map - best_single_map
chance_map = eval_results["[baseline] Random"]["MAP"]
print(f"\nBest single method by MAP: {best_single} ({best_single_map:.4f})")
print(f"Hybrid MAP:                {hybrid_map:.4f}  (lift {lift:+.4f})")
print(f"Random-ranking MAP:        {chance_map:.4f}")

# Where does the hybrid's lift come from? Score warm SKUs on their own.
warm_holdout = holdout_mask & ~cold_items[None, :]
warm_map = {
    name: evaluate_method(preds, R_observed, warm_holdout)["MAP"]
    for name, preds in all_predictions.items()
}
best_warm = max(warm_map, key=warm_map.get)
hybrid_warm_map = evaluate_method(hybrid_preds, R_observed, warm_holdout)["MAP"]
warm_gain = hybrid_warm_map - warm_map[best_warm]
top_component = max(blend_weights, key=blend_weights.get)
print(
    f"Warm SKUs only: hybrid MAP {hybrid_warm_map:.4f} vs best component "
    f"{best_warm} {warm_map[best_warm]:.4f} (gain {warm_gain:+.4f}); "
    f"largest blend weight: {top_component} ({blend_weights[top_component]:.3f})"
)
if lift > 0.005 and abs(warm_gain) <= 0.01:
    print(
        "  -> The hybrid beats every single method, but on warm SKUs it only "
        f"matches {best_warm}: the gain comes from COVERAGE — the hybrid "
        "hands the cold-start SKUs that CF/MF cannot score to content-based."
    )
elif lift > 0.005:
    print(
        "  -> The hybrid beats every single method, and it also improves on "
        "the best component for warm SKUs — the components' errors are "
        "complementary there too."
    )
elif lift > -0.005:
    print(
        "  -> The hybrid ties the best single method: blending adds little "
        "on this data."
    )
else:
    print(
        "  -> The hybrid is WORSE than the best single method on this run — "
        "blending diluted the winner. Check the weights and the split sizes."
    )


# ── Checkpoint ──────────────────────────────────────────────────────────
assert len(eval_results) == 8, "Should have 3 baselines + 4 base methods + hybrid"
assert eval_results["Hybrid"]["Coverage"] == 1.0, "Hybrid must score every pair"
assert all(w >= 0 for w in blend_weights.values()), "NNLS weights are non-negative"
assert 0.0 <= eval_results["Hybrid"]["P@5"] <= 1.0, "Precision@k must be in [0, 1]"
assert 0.0 <= hybrid_map <= 1.0, "MAP must be in [0, 1]"
assert best_single_map > chance_map, "Best recommender must beat random ranking"
print("\n[ok] Checkpoint passed — all methods evaluated on RMSE, P@5, MAP\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: side-by-side comparison
# ════════════════════════════════════════════════════════════════════════
# A metric comparison chart makes trade-offs obvious: a method with low
# RMSE on the pairs it covers can still have poor MAP if it cannot score
# the cold-start SKUs at all.

viz = create_visualizer()
comparison_metrics = {
    name: {"RMSE": r["RMSE"], "MAP": r["MAP"], "Coverage": r["Coverage"]}
    for name, r in eval_results.items()
    if name != "[baseline] Random"  # random-score RMSE dwarfs the axis
}
fig_cmp = viz.metric_comparison(comparison_metrics)
fig_cmp.update_layout(title="Recommender Method Comparison (RMSE, MAP, coverage)")
save_html(fig_cmp, "05_method_comparison.html")


# ════════════════════════════════════════════════════════════════════════
# IMPLICIT vs EXPLICIT FEEDBACK — conceptual summary
# ════════════════════════════════════════════════════════════════════════
print(
    """
Explicit feedback: users provide ratings (1-5 stars).
  + clear signal
  - sparse (most users rate few items)
  - absence of rating is ambiguous

Implicit feedback: clicks, views, purchases, time spent.
  + abundant
  - noisy (click != like)
  - no NEGATIVE signal (no click could mean "never seen")

ALS for implicit data (Hu, Koren & Volinsky, 2008):
  - Treat ALL pairs as observed
  - p[u,i] = 1 if user interacted, 0 otherwise
  - c[u,i] = 1 + alpha * count  (confidence)
  - Loss: sum_all c(u,i) * (p(u,i) - U[u] V[i])^2 + lambda * (||U||^2 + ||V||^2)

Most large consumer platforms have far more implicit than explicit
feedback, so implicit-feedback MF and its neural successors are common.
"""
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Singapore News Aggregator Front-Page Personalisation
# ════════════════════════════════════════════════════════════════════════
# SCENARIO: A Singapore news aggregator runs personalised front pages for
# ~3M daily active users. The front page has 8 slots. Every refresh must
# rank hundreds of articles by predicted relevance in <50ms.
#
# Why a HYBRID is the right tool:
#   - BREAKING NEWS = cold-start items (0 reads). Content-based kicks in
#     using tags, section, author, entity mentions — exactly the role it
#     played for the cold-start SKUs above
#   - REGULAR CONTENT = CF / MF win via co-read patterns
#   - A blend tuned on held-out validation data picks how much to trust
#     each lever; your measured weights above show what that looks like
#
# BUSINESS IMPACT (illustrative assumptions, not measured figures): on an
# aggregator with S$35M annual ad revenue, if personalisation lifted
# clicks-per-session by 10% and ad revenue scaled with clicks, that would
# be ~S$3.5M/year — against perhaps S$400K/year of ML infrastructure and
# engineering. Only an online A/B test can confirm the lift; offline MAP
# tells you which candidate to put into that test.
#
# LIMITATIONS of simple global blending:
#   - Weights are global; a per-user or per-segment gate (learn which
#     method to trust for WHICH user or item type) can do better
#   - A richer stacking model (e.g. gradient boosting on component scores
#     plus user/item counts) can capture interactions NNLS cannot
#   - At scale, a 4-method blend costs 4x inference — teams often distil
#     the ensemble into a single neural model (Exercise 8 builds the
#     neural-network foundations)


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  [x] Evaluated four recommenders + three baselines on RMSE, coverage, P@5, MAP
  [x] Tuned blend weights on a validation split, never on the test holdout
  [x] Measured the hybrid lift vs best single method ({lift:+.4f} MAP)
  [x] Explained implicit vs explicit feedback and ALS-implicit
  [x] Sized an (illustrative) SEA news scenario for hybrid recommenders

  KEY INSIGHT: A hybrid only helps when its components make different
  mistakes. Here the clearest difference is coverage — collaborative
  methods are blind to brand-new items, content-based is not.

  Exercise 7 complete — you now understand the full recommender stack.

  Next: Exercise 8 — neural networks generalise matrix factorisation by
  adding non-linear activations. The hidden layer IS an embedding,
  learned by the same principle.
"""
)
