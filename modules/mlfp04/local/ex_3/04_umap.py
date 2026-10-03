# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 3.4: UMAP for production dim reduction
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Understand UMAP's fuzzy topological formulation vs t-SNE
#   - Tune n_neighbors (local vs global) and min_dist (tight vs spread)
#   - Use .transform() for out-of-sample embedding — the production path
#   - Choose UMAP over t-SNE when feature extraction is the goal
#
# PREREQUISITES: 01_pca.py, 03_tsne.py.
#
# ESTIMATED TIME: ~30 min
#
# TASKS:
#   1. Theory — UMAP as a weighted k-NN graph layout
#   2. Build — fit UMAP with 4 hyperparameter configurations
#   3. Train — fit on a subsample, .transform() held-out rows (OOS)
#   4. Visualise — 2D scatter per configuration + quality comparison
#   5. Apply — AML entity screening at a Singapore bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import time

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

from kailash_ml import ModelVisualizer

from shared.mlfp04.ex_3 import (
    OUTPUT_DIR,
    evaluate_embedding,
    holdout_indices,
    load_customer_matrix,
    setup_engines,
    subsample_indices,
    teardown_engines,
    track_run,
)

# ── Kailash-ML ExperimentTracker — every dim-reduction run logs here ─────
tracker, exp_name = setup_engines()

# UMAP is an optional extra — fall back gracefully so the exercise stays
# runnable on machines without umap-learn installed (e.g. Colab cold
# start). See rules/dependencies.md "Optional Extras with Loud Failure".
try:
    import umap as umap_lib  # type: ignore

    UMAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    umap_lib = None
    UMAP_AVAILABLE = False
    print("[warn] umap-learn not installed — install with: pip install umap-learn")
    print("       Falling back to PCA 2D for the APPLY phase only.")


# ════════════════════════════════════════════════════════════════════════
# THEORY — UMAP in one paragraph
# ════════════════════════════════════════════════════════════════════════
# UMAP models the data as a weighted k-NN graph, then optimises a
# low-dimensional layout whose own k-NN graph matches the high-dim one.
# Compared to t-SNE:
#   + preserves BOTH local neighbours AND the global skeleton
#   + supports .transform() for new points (trained embedder becomes
#     a function from R^p to R^2) — t-SNE cannot; PCA and Kernel PCA can
#   + faster: ~O(n) amortised, scales to ~1M rows
#   + embeds into arbitrary dimensions, not just 2D
#
# Two key hyperparameters:
#   - n_neighbors: size of the local neighbourhood. Small (5) = local
#     detail, large (50) = global structure.
#   - min_dist: minimum distance between points in the embedding. Small
#     (0.0) = tight clusters, large (1.0) = spread out for visualisation.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: data + PCA pre-reduction
# ════════════════════════════════════════════════════════════════════════

X, feature_cols, df_customers = load_customer_matrix()
n_samples, n_features = X.shape

# TODO: PCA pre-reduction to min(10, n_features) dims (same recipe as t-SNE).
pca_pre = ____
X_pca = ____

# Fit on a 3K subsample; transform a 10K slice of OTHER rows to showcase a
# genuine out-of-sample transform. Transforming all ~47K remaining rows x 4
# configs is slow on a laptop; 10K still demonstrates the OOS workflow.
# TODO: subsample 3,000 rows on which to FIT the UMAP reducer, then draw
# the out-of-sample slice from the REMAINING rows only.
# Hint: subsample_indices(...) for the fit rows; holdout_indices(n_samples,
# exclude=fit_idx, n_target=TRANSFORM_TARGET) for the transform rows.
fit_idx = ____
TRANSFORM_TARGET = 10_000
transform_idx = ____
n_transform = len(transform_idx)
X_pca_transform = X_pca[transform_idx]
print("=== UMAP inputs ===")
print(f"  fit on  : {len(fit_idx):,} rows")
print(
    f"  transform: {n_transform:,} rows (out-of-sample, sub-sampled from {n_samples:,})"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: sweep n_neighbors x min_dist
# ════════════════════════════════════════════════════════════════════════

umap_configs = [
    {"n_neighbors": 5, "min_dist": 0.1, "label": "local (n=5, d=0.1)"},
    {"n_neighbors": 15, "min_dist": 0.1, "label": "default (n=15, d=0.1)"},
    {"n_neighbors": 30, "min_dist": 0.1, "label": "broad (n=30, d=0.1)"},
    {"n_neighbors": 50, "min_dist": 0.5, "label": "global (n=50, d=0.5)"},
]

umap_results: dict[str, dict] = {}

print("\n=== UMAP hyperparameter sweep ===")
print(f"{'config':<28}{'trust':>10}{'silhouette':>12}{'time (s)':>10}")
print("-" * 60)

if UMAP_AVAILABLE:
    for cfg in umap_configs:
        t0 = time.time()
        # TODO: build umap_lib.UMAP with n_components=2, this config's
        # n_neighbors, min_dist, random_state=42, metric='euclidean'.
        reducer = ____
        # TODO: fit on the subsample, then .transform() the OOS slice.
        # Hint: reducer.fit(X_pca[fit_idx]); reducer.transform(X_pca_transform)
        reducer.fit(X_pca[fit_idx])
        embedding_full = ____
        elapsed = time.time() - t0

        # Quality is judged on the HELD-OUT rows against their original features
        # TODO: score the OOS embedding against the original rows X[transform_idx]
        quality = ____
        umap_results[cfg["label"]] = {
            "embedding": embedding_full,
            **quality,
            "time_s": elapsed,
        }
        print(
            f"{cfg['label']:<28}{quality['trustworthiness']:>10.4f}"
            f"{quality['silhouette']:>12.4f}{elapsed:>10.1f}"
        )
else:
    # PCA 2D fallback — keeps the exercise runnable in minimal envs.
    pca_2d = PCA(n_components=2, random_state=42)
    embedding_full = pca_2d.fit_transform(X_pca_transform)
    quality = evaluate_embedding(X[transform_idx], embedding_full)
    umap_results["PCA-2D-fallback"] = {
        "embedding": embedding_full,
        **quality,
        "time_s": 0.0,
    }
    print(
        f"{'PCA-2D-fallback':<28}{quality['trustworthiness']:>10.4f}"
        f"{quality['silhouette']:>12.4f}{0.0:>10.1f}"
    )


# ── Checkpoint 1 ────────────────────────────────────────────────────────
assert len(umap_results) >= 1, "Must produce at least one UMAP result"
assert len(set(fit_idx) & set(transform_idx)) == 0, "OOS rows must exclude fit rows"
for label, res in umap_results.items():
    assert res["embedding"].shape == (n_transform, 2), (
        f"UMAP {label} must return ({n_transform}, 2) 2D embedding "
        f"(out-of-sample transform on a {n_transform}-row slice)"
    )
print(f"\n[ok] Checkpoint 1 — out-of-sample transform produced {n_transform}-row 2D")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: silhouette across configurations
# ════════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()
fig = viz.metric_comparison(
    {
        label: {
            "Trustworthiness": r["trustworthiness"],
            "kNN overlap": r["knn_overlap"],
            "Silhouette": r["silhouette"],
        }
        for label, r in umap_results.items()
    }
)
fig.update_layout(title="UMAP: structure preservation vs clusterability")
umap_path = OUTPUT_DIR / "04_umap_sweep.html"
fig.write_html(str(umap_path))
print(f"\nSaved: {umap_path}")

# The held-out embeddings themselves, one panel per configuration,
# coloured by churn status (NOT a reducer input).
churned_oos = df_customers["churned"].to_numpy()[transform_idx]
labels_in_order = list(umap_results.keys())
fig_scatter = make_subplots(
    rows=1, cols=len(labels_in_order), subplot_titles=labels_in_order
)
for col, label in enumerate(labels_in_order, start=1):
    emb = umap_results[label]["embedding"]
    for flag, colour, name in [(0, "#636EFA", "retained"), (1, "#EF553B", "churned")]:
        mask = churned_oos == flag
        fig_scatter.add_trace(
            go.Scatter(
                x=emb[mask, 0],
                y=emb[mask, 1],
                mode="markers",
                marker=dict(size=2, color=colour, opacity=0.5),
                name=name,
                showlegend=(col == 1),
            ),
            row=1,
            col=col,
        )
fig_scatter.update_layout(
    title="UMAP out-of-sample embeddings (colour = churned, not a model input)",
    height=420,
    width=320 * len(labels_in_order),
)
scatter_path = OUTPUT_DIR / "04_umap_embeddings.html"
fig_scatter.write_html(str(scatter_path))
print(f"Saved: {scatter_path}")

print("\nUMAP hyperparameter guide:")
print("  n_neighbors small -> fine local detail, fractured clusters")
print("  n_neighbors large -> smoother, more global structure")
print("  min_dist   small -> tight clusters (good for downstream KMeans)")
print("  min_dist   large -> spread out (good for visual inspection)")
print("\nOut-of-sample recipe: reducer.fit(train); reducer.transform(new_X)")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: AML Entity Screening at a Singapore Bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore bank's AML analytics team embeds
# every customer entity into a 2D UMAP space built from ~60 features
# (transaction velocity, counterparty diversity, cross-border ratio, cash
# intensity, sector code, device-fingerprint counts). Analysts use the
# map to decide which alerts to escalate; in Singapore, suspicious
# transaction reports are filed with the Suspicious Transaction Reporting
# Office (STRO) of the Singapore Police Force. The space is refit WEEKLY
# on a stable training slice and applied every NIGHT to new activity.
#
# WHY UMAP IS A GOOD FIT:
#   - Out-of-sample .transform() — new entities land on the existing map
#     without a refit, so the axes stay stable between weekly refits.
#     (PCA and Kernel PCA can transform new points too; t-SNE cannot.)
#   - Preserves local neighbourhoods while keeping more of the global
#     layout than t-SNE — check trustworthiness on held-out rows, as above.
#   - Scales to hundreds of thousands of entities on commodity hardware.
#
# BUSINESS IMPACT (illustrative assumptions, not reported figures): if
# alert triage costs ~S$1,200 of analyst time per false positive and the
# map lets analysts close a fraction of false-positive alerts faster, the
# saving is (alerts/year) x (false-positive share avoided) x S$1,200.
# At 18,000 alerts a year, avoiding one in ten false positives is worth
# ~S$2.2M a year. Validate the avoided share in a pilot before relying on it.
#
# WHY NOT t-SNE HERE: t-SNE cannot place new points without a refit, and a
# refit reshuffles the axes. The analysts' mental map ("unusual entities
# sit in the upper-left") would reset every night.

if UMAP_AVAILABLE and umap_results:
    best_label, best = max(
        umap_results.items(), key=lambda kv: kv[1]["trustworthiness"]
    )
    print("\n=== AML entity-map projection (UMAP, held-out rows) ===")
    print(f"  Best config (trust) : {best_label}")
    print(f"  Trustworthiness     : {best['trustworthiness']:.4f}")
    print(f"  Silhouette          : {best['silhouette']:.4f}")
    print(f"  Fit wall time   : {best['time_s']:.1f}s")
    print(f"  Output shape    : {best['embedding'].shape}")
else:
    print("\n[note] Install umap-learn to run the full AML scenario.")


# ════════════════════════════════════════════════════════════════════════
# TRACK — Log this lesson's run to the kailash-ml ExperimentTracker
# ════════════════════════════════════════════════════════════════════════
# Per-config silhouette + wall-time scalars + parallel series go into the
# m4_dimreduction_zoo experiment for side-by-side comparison.

config_labels = list(umap_results.keys())
best_label_for_run = (
    max(umap_results.items(), key=lambda kv: kv[1]["trustworthiness"])[0]
    if umap_results
    else "none"
)


def _finite(x: float) -> float:
    """NaN-guard: tracker rejects non-finite metric values. Silhouette can
    return NaN when the embedding collapses to a single cluster."""
    return float(x) if x == x else 0.0


track_run(
    tracker,
    exp_name,
    # TODO: run_name=f"umap_{best_label_for_run.split()[0].replace('-', '_')}";
    # merge per-config silhouette / time_s dicts with |, and pass the sweep
    # lists as series_metrics (wrap metric values in _finite).
    run_name=____,
    params={
        "algorithm": "umap",
        "n_components": 2,
        "n_fit_subsample": int(len(fit_idx)),
        "n_transform_oos": int(n_transform),
        "pca_pre_components": int(X_pca.shape[1]),
        "umap_available": str(UMAP_AVAILABLE),
        "best_config": best_label_for_run,
    },
    scalar_metrics={
        "best_trustworthiness": (
            _finite(umap_results[best_label_for_run]["trustworthiness"])
            if umap_results
            else 0.0
        ),
        "best_silhouette": (
            _finite(umap_results[best_label_for_run]["silhouette"])
            if umap_results
            else 0.0
        ),
    }
    | {
        f"cfg{i}_trustworthiness": _finite(umap_results[label]["trustworthiness"])
        for i, label in enumerate(config_labels)
    }
    | ____
    | ____,
    series_metrics={
        "sweep_trustworthiness": [
            _finite(umap_results[label]["trustworthiness"]) for label in config_labels
        ],
        "sweep_silhouette": ____,
        "sweep_time_s": ____,
    },
)
print(f"  [tracked] UMAP sweep logged to {exp_name}\n")


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — DimReductionEngine.reduce(algorithm='umap')
# ════════════════════════════════════════════════════════════════════════
# kailash-ml's DimReductionEngine wraps UMAP under the same `reduce`
# surface that backed PCA in lesson 01 and t-SNE in lesson 03. The engine
# handles polars→numpy and returns a DimReductionResult — embedding +
# n_neighbors + min_dist surfaced on the metrics dict.

import polars as pl

from kailash_ml.engines.dim_reduction import DimReductionEngine

cust_df = pl.from_numpy(X[fit_idx], schema=feature_cols)
# TODO: instantiate DimReductionEngine and call .reduce on cust_df with
# algorithm='umap' and n_components=2.
dimreduce = ____
reduce_result = ____
print(
    f"  DimReductionEngine.reduce(umap, n_components=2): "
    f"embedding shape=({len(reduce_result.transformed)}, "
    f"{reduce_result.n_components})  "
    f"n_neighbors={reduce_result.metrics.get('n_neighbors', 'n/a')}  "
    f"min_dist={reduce_result.metrics.get('min_dist', 'n/a')}"
)
print()
print("  Same UMAP you swept by hand — wrapped under the engine surface")
print("  that backs pca / tsne / umap / nmf. The leaderboard now compares")
print("  this fit with the PCA + t-SNE runs from lessons 01 and 03.\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Fit UMAP on a training subsample and transformed held-out rows
      out-of-sample — the production workflow
  [x] Swept n_neighbors and min_dist across 4 configurations
  [x] Judged each configuration by trustworthiness on held-out rows and
      plotted the embeddings themselves
  [x] Framed UMAP for a weekly bank AML entity-embedding pipeline
      (illustrative)

  KEY INSIGHT: The out-of-sample transform is what separates UMAP from
  t-SNE's picture generator. PCA and Kernel PCA can also embed new
  points; UMAP's draw is combining that with nonlinear neighbourhood
  structure. Its inverse_transform is only approximate — for exact
  reconstruction, PCA remains the tool.

  Next: 05_comparison.py pits all five techniques against each other on
  one neighbourhood-preservation ruler and estimates the intrinsic dimensionality
  of the customer feature space.
"""
)


# Drain the aiosqlite worker threads so Py_Finalize doesn't hang.
teardown_engines(tracker)
