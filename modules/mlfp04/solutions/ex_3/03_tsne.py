# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.3: t-SNE for local structure
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand what t-SNE optimises (KL divergence of neighbourhoods)
#   - Tune the perplexity hyperparameter
#   - Recognise the three classic t-SNE pitfalls
#   - Know when t-SNE is a visualisation tool, not a feature extractor
#
# PREREQUISITES: 01_pca.py — we pre-reduce with PCA before t-SNE.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — t-SNE as a neighbourhood-preserving map
#   2. Build — PCA pre-reduction + t-SNE at 4 perplexity values
#   3. Train — KL divergence + trustworthiness + silhouette per perplexity
#   4. Visualise — 2D embedding scatter per perplexity + metric comparison
#   5. Apply — passenger-journey micro-segments at an airport hub
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    evaluate_embedding,
    load_customer_matrix,
    setup_engines,
    subsample_indices,
    teardown_engines,
    track_run,
)

# ── Kailash-ML ExperimentTracker — every dim-reduction run logs here ─────
tracker, exp_name = setup_engines()


# ════════════════════════════════════════════════════════════════════════
# THEORY — what t-SNE actually does
# ════════════════════════════════════════════════════════════════════════
# t-SNE builds two probability distributions:
#   1. High-dim P: for every pair (i, j), P_ij is a Gaussian over
#      distances, so nearby points have high probability.
#   2. Low-dim Q: a heavy-tailed Student-t distribution over the 2D
#      coordinates that we OPTIMISE.
#
# We minimise KL(P || Q) by gradient descent on the low-dim positions.
# The result: points that were close in high-dim stay close in 2D.
#
# PERPLEXITY is the effective number of "nearest neighbours" each point
# considers. Small perplexity (5) gives micro-clusters; large perplexity
# (50) smooths the layout.
#
# THREE PITFALLS to memorise:
#   A. Cluster SIZES in the 2D picture are meaningless — t-SNE equalises
#      density. A tiny dense cluster and a huge diffuse one look similar.
#   B. Distances BETWEEN clusters are meaningless — the layout only
#      preserves local neighbourhoods.
#   C. t-SNE has NO out-of-sample transform. Every new point forces a
#      full refit. This is why t-SNE is a visualisation tool, not a
#      feature extractor for production.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: PCA pre-reduction + subsample
# ════════════════════════════════════════════════════════════════════════
# t-SNE is O(n log n) with Barnes-Hut but has a large constant. Two
# standard preparations:
#   - Subsample to ~3K rows (the visible embedding size anyway)
#   - Pre-reduce with PCA to ~10-50 dims on WIDE data (speeds t-SNE and
#     denoises distances). Our customer matrix has only 7 features, so
#     min(10, 7) keeps all 7 components — here the PCA step is just a
#     rotation, kept so the pipeline matches what you would run on wide data.

X, feature_cols, df_customers = load_customer_matrix()
n_samples, n_features = X.shape

pca_pre = PCA(n_components=min(10, n_features), random_state=42)
X_pca = pca_pre.fit_transform(X)

idx = subsample_indices(n_samples, n_target=3000)
X_tsne_input = X_pca[idx]
print(f"=== t-SNE input ===  n={X_tsne_input.shape[0]:,}, d={X_tsne_input.shape[1]}")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sweep perplexity
# ════════════════════════════════════════════════════════════════════════

tsne_results: dict[int, dict] = {}
perplexities = [5, 15, 30, 50]

print("\n=== t-SNE perplexity sweep ===")
print(
    f"{'perplexity':>12}{'KL div':>10}{'trust':>10}{'kNN ovl':>10}"
    f"{'silhouette':>12}{'time (s)':>10}"
)
print("-" * 64)

for perplexity in perplexities:
    t0 = time.time()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    embedding = tsne.fit_transform(X_tsne_input)
    elapsed = time.time() - t0

    # Structure preservation is judged against the ORIGINAL features
    quality = evaluate_embedding(X[idx], embedding)
    tsne_results[perplexity] = {
        "embedding": embedding,
        "kl": float(tsne.kl_divergence_),
        **quality,
        "time_s": elapsed,
    }
    print(
        f"{perplexity:>12}{tsne.kl_divergence_:>10.4f}"
        f"{quality['trustworthiness']:>10.4f}{quality['knn_overlap']:>10.4f}"
        f"{quality['silhouette']:>12.4f}{elapsed:>10.1f}"
    )

# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(tsne_results) == 4, "Must test 4 perplexity values"
for perp, res in tsne_results.items():
    assert res["embedding"].shape[1] == 2, "t-SNE must produce 2D output"
    assert res["kl"] > 0, "KL divergence must be positive"
print("\n[ok] Checkpoint 1 — 2D embeddings across 4 perplexity settings")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: perplexity comparison
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# (a) The embeddings themselves — one panel per perplexity, coloured by
# churn status. `churned` was NOT a reducer input, so any region where
# churners concentrate is structure t-SNE found in behaviour alone.
churned_sub = df_customers["churned"].to_numpy()[idx]
fig_scatter = make_subplots(
    rows=1,
    cols=len(perplexities),
    subplot_titles=[f"perplexity={p}" for p in perplexities],
)
for col, perplexity in enumerate(perplexities, start=1):
    emb = tsne_results[perplexity]["embedding"]
    for flag, colour, name in [(0, "#636EFA", "retained"), (1, "#EF553B", "churned")]:
        mask = churned_sub == flag
        fig_scatter.add_trace(
            go.Scatter(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                marker=dict(size=3, color=colour, opacity=0.6),
                name=name,
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
fig_scatter.update_layout(
    title="t-SNE embeddings by perplexity (colour = churned, not a model input)",
    height=420,
    width=320 * len(perplexities),
)
scatter_path = OUTPUT_DIR / "03_tsne_embeddings.html"
fig_scatter.write_html(str(scatter_path))
print(f"\nSaved: {scatter_path}")

# (b) Metric comparison across perplexities
fig_perp = viz.metric_comparison(
    {
        f"perplexity={p}": {
            "Trustworthiness": r["trustworthiness"],
            "kNN overlap": r["knn_overlap"],
            "Silhouette": r["silhouette"],
        }
        for p, r in tsne_results.items()
    }
)
fig_perp.update_layout(title="t-SNE: structure preservation vs clusterability")
perp_path = OUTPUT_DIR / "03_tsne_perplexity.html"
fig_perp.write_html(str(perp_path))
print(f"\nSaved: {perp_path}")

print("\nPerplexity guidance:")
print("  5  — micro-clusters, very local structure (fragile)")
print("  15 — fine local structure (good for dense datasets)")
print("  30 — balanced default recommendation")
print("  50 — smoother, fewer isolated clusters")
print("\nCaution: KL values are NOT comparable across perplexities (each")
print("perplexity defines a different P), and silhouette in a t-SNE map is")
print("inflated by construction — t-SNE pulls points into tight blobs.")
print("Judge structure by trustworthiness, then inspect the picture.")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Passenger-Journey Micro-Segments at an Airport Hub
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a major Asian airport hub instruments passenger
# journeys through one terminal with dozens of touchpoints: check-in time,
# dwell-time per retail zone, dwell at gates, food-court visits, e-gate
# transits, inter-terminal train usage. The retail team wants to SEE the
# micro-segments hiding inside the "transit passenger" macro-group —
# families with small kids, business travellers on short layovers,
# premium-cabin passengers heading straight to the lounge, budget
# travellers lingering in the food court. These are LOCAL patterns.
#
# WHY t-SNE:
#   - Captures LOCAL neighbourhood structure — the retail team wants a
#     picture of the micro-clusters, not features for a downstream model.
#   - An afternoon's few thousand passengers is well within Barnes-Hut
#     t-SNE's reach after PCA pre-reduction.
#   - The output drives a static dashboard for merchandising planners, so
#     the lack of an out-of-sample transform is not a blocker.
#
# HOW PERPLEXITY IS USED: perplexity is a granularity knob. Low values
# fragment the map into many micro-clusters; high values merge them.
# Choose among perplexities whose trustworthiness is comparably high, then
# pick the granularity the audience can act on — not the most blob-like.
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): if a
# re-planned retail mix lifted food-and-beverage conversion by a few
# percent on a terminal doing hundreds of millions of dollars a year,
# the gain would dwarf the cost — a few laptop-hours of t-SNE a month.
# Measure the lift with a controlled experiment before claiming it.
#
# PITFALL TO AVOID: never feed t-SNE coordinates into a downstream churn
# model or LTV regression. The coordinates are picture-only; feeding them
# into a model bakes in randomness from the initialisation and breaks
# every time the job re-runs.

best_p, best_r = max(
    tsne_results.items(), key=lambda kv: kv[1]["trustworthiness"]
)
blobbiest_p = max(tsne_results, key=lambda p: tsne_results[p]["silhouette"])
print("\n=== Airport micro-segment projection ===")
print(f"  Best perplexity (trustworthiness) : {best_p}")
print(f"  Trustworthiness                   : {best_r['trustworthiness']:.4f}")
print(f"  Silhouette (clusterability only)  : {best_r['silhouette']:.4f}")
if blobbiest_p != best_p:
    print(
        f"  Note: perplexity={blobbiest_p} has the highest silhouette but lower"
        " trustworthiness — the most blob-like picture is not the most faithful."
    )


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Per-perplexity KL/silhouette/time scalars + parallel series go into the
# m4_dimreduction_zoo experiment for side-by-side comparison.

perplexities_sorted = sorted(tsne_results.keys())
track_run(
    tracker,
    exp_name,
    run_name=f"tsne_perp_{best_p}",
    params={
        "algorithm": "tsne",
        "n_components": 2,
        "n_subsample": int(X_tsne_input.shape[0]),
        "pca_pre_components": int(X_tsne_input.shape[1]),
        "perplexities": ",".join(str(p) for p in perplexities_sorted),
        "best_perplexity": best_p,
    },
    scalar_metrics={
        "best_trustworthiness": float(best_r["trustworthiness"]),
        "best_silhouette": float(best_r["silhouette"]),
        "best_kl": float(best_r["kl"]),
    }
    | {
        f"perp_{p}_trustworthiness": float(r["trustworthiness"])
        for p, r in tsne_results.items()
    }
    | {f"perp_{p}_silhouette": float(r["silhouette"]) for p, r in tsne_results.items()}
    | {f"perp_{p}_kl": float(r["kl"]) for p, r in tsne_results.items()}
    | {f"perp_{p}_time_s": float(r["time_s"]) for p, r in tsne_results.items()},
    series_metrics={
        "sweep_trustworthiness": [
            float(tsne_results[p]["trustworthiness"]) for p in perplexities_sorted
        ],
        "sweep_silhouette": [
            float(tsne_results[p]["silhouette"]) for p in perplexities_sorted
        ],
        "sweep_kl": [float(tsne_results[p]["kl"]) for p in perplexities_sorted],
    },
)
print(f"  [tracked] perplexity sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — DimReductionEngine.reduce(algorithm='tsne')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's DimReductionEngine wraps sklearn t-SNE under the same
# `reduce` surface that backed PCA in lesson 01. The engine handles the
# polars→numpy conversion, runs t-SNE, returns a DimReductionResult with
# the embedding and KL divergence in the metrics dict — one sync call.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

# Engine takes the standardised features (not the PCA-pre-reduced matrix)
# and runs the whole pipeline; we pass the perplexity chosen above.
sub_idx = idx
cust_df = pl.from_numpy(X[sub_idx], schema=feature_cols)
dimreduce = DimReductionEngine()

reduce_result = dimreduce.reduce(
    cust_df, algorithm="tsne", n_components=2, perplexity=best_p
)
print(
    f"  DimReductionEngine.reduce(tsne, perplexity={best_p}): "
    f"embedding shape=({len(reduce_result.transformed)}, "
    f"{reduce_result.n_components})  "
    f"kl={reduce_result.metrics.get('kl_divergence', float('nan')):.4f}"
)
print()
print("  Same t-SNE you swept by hand — wrapped under the engine surface")
print("  that backs pca / tsne / umap / nmf. The leaderboard now compares")
print("  this perplexity with the PCA baseline from lesson 01.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Ran t-SNE at 4 perplexity values and measured KL, trustworthiness
      and silhouette — and saw why silhouette is the wrong ruler here
  [x] Plotted the 2D embeddings and coloured them by a held-out profile
  [x] Pre-reduced with PCA before t-SNE (standard practice)
  [x] Recognised the three pitfalls: cluster size, inter-cluster
      distance, no out-of-sample transform
  [x] Framed t-SNE for an airport retail dashboard where the output is a
      visual, not a feature (illustrative)

  KEY INSIGHT: t-SNE is not dimensionality reduction in the production
  sense — it is a PICTURE generator. When your deliverable is an insight
  for a human, t-SNE is brilliant. When your deliverable is a feature
  for another model, use PCA or UMAP instead.

  Next: 04_umap.py keeps the neighbourhood idea but adds an out-of-
  sample transform and preserves global structure too.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
