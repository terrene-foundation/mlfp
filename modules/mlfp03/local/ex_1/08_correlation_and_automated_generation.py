# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.8: Correlation-Threshold Filtering and Automated
#                         Feature Generation (FeatureEngineer)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - The correlation-threshold filter: drop one of every redundant pair
#   - How it differs from mutual-information filtering (02): linear
#     redundancy vs monotone relevance
#   - Automated candidate generation: interactions, squares, binning
#   - kailash-ml's FeatureEngineer.generate + .select — the Kailash path
#     for the "generate candidates, then select" pattern the spec mandates
#
# PREREQUISITES: 02_filter_selection.py (MI filter), 07 (wrapper selection)
# ESTIMATED TIME: ~25 min
#
# 5-PHASE STRUCTURE:
#   Theory   — redundancy vs relevance; generate-then-select
#   Build    — correlation-threshold filter by hand
#   Train    — FeatureEngineer generate + select on the credit frame
#   Visualise — correlation heatmap before/after + selected candidates
#   Apply    — a bureau-data cost review at a retail lender
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from kailash_ml import FeatureEngineer
from kailash_ml.types import FeatureField, FeatureSchema
from shared import MLFPDataLoader
from shared.kailash_helpers import split_then_preprocess
from shared.mlfp03.ex_1 import OUTPUT_DIR
from shared.mlfp03.ex_5 import CREDIT_NON_FEATURE_COLUMNS

# ── THEORY — redundancy is not irrelevance ───────────────────────────────
# 02's mutual-information filter answers: "does this feature know anything
# about the TARGET?" (relevance). The correlation-threshold filter answers
# a different question: "does this feature duplicate ANOTHER FEATURE?"
# (redundancy). A feature can be highly relevant AND highly redundant —
# months_employed and employment_years both predict default, but together
# they tell the model the same thing twice, doubling the governance and
# data-cost burden for no signal.
#
# The filter: compute pairwise |Pearson correlation|; for every pair above
# the threshold, drop one member (the one LESS correlated with the target,
# or simply the second one). Thresholds: 0.95 is conservative, 0.90 common,
# 0.80 aggressive. Pearson sees only LINEAR redundancy — two features can
# be Pearson-uncorrelated and still duplicates through a monotone bend.
#
# AUTOMATED GENERATION is the mirror move: instead of removing columns,
# manufacture candidates (a×b interactions, squares, bins) and let a
# SELECTION step decide which earn their place. kailash-ml's FeatureEngineer
# packages exactly that pattern: generate(schema, strategies) →
# select(candidates, target, top_k).
#
# SDK literacy: FeatureEngineer is marked EXPERIMENTAL (P2) by kailash-ml —
# the constructor emits ExperimentalWarning. That is the SDK telling you the
# API may change. We acknowledge it explicitly (a narrow, category-matched
# filter at the construction site) rather than letting it fire unexplained —
# and you should expect the same notice when you adopt P2 engines.

credit = MLFPDataLoader().load("mlfp02", "sg_credit_scoring.parquet").drop(
    CREDIT_NON_FEATURE_COLUMNS
)
result = split_then_preprocess(
    credit, target="default", test_size=0.2, seed=42,
    normalize=False, categorical_encoding="ordinal",
)
train = result.train_data
feature_cols = [c for c in train.columns if c != "default"]
X = train.select(feature_cols).to_numpy()
y = train["default"].to_numpy().astype(float)
print(f"  Credit train frame: {train.height:,} loans × {len(feature_cols)} features")

# ── BUILD — the correlation-threshold filter, by hand ────────────────────
THRESHOLD = 0.90
corr = train.select(feature_cols).corr()
# |corr with target| per feature; nan_to_num guards zero-variance columns
# TODO: |Pearson corr with target| per feature (guard zero-variance with nan_to_num)
# Hint: np.corrcoef(train[c].to_numpy(), y)[0, 1]
target_corr = ____

dropped, kept = [], []
for i, a in enumerate(feature_cols):
    if a in dropped:
        continue
    kept.append(a)
    for b in feature_cols[i + 1 :]:
        if b in dropped:
            continue
        # TODO: |corr(a, b)| — corr is square, same column order as feature_cols
        # Hint: corr[b][i] indexes the matrix entry for row i (feature a)
        r = ____
        if r >= THRESHOLD:
            # TODO: drop the member LESS correlated with the target
            victim = ____
            if victim == a:
                kept.remove(a)
                dropped.append(a)
                break
            dropped.append(b)

pairs_dropped = len(dropped)
print(f"\n  |r| ≥ {THRESHOLD}: dropped {pairs_dropped} redundant features")
for d in dropped:
    print(f"    dropped {d}")

# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert kept and not set(dropped) & set(kept), "kept/dropped must be disjoint"
assert len(kept) + len(dropped) == len(feature_cols)
print(f"[ok] Checkpoint 1 — {len(kept)} kept, {pairs_dropped} dropped at |r|≥{THRESHOLD}")

# ── VISUALISE — the correlation matrix before/after ──────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"Before: all {len(feature_cols)} features",
        f"After: {len(kept)} features (|r| ≥ {THRESHOLD} pruned)",
    ),
)
fig.add_trace(go.Heatmap(z=corr.to_numpy(), x=feature_cols, y=feature_cols,
                         zmin=-1, zmax=1, colorscale="RdBu", reversescale=True),
              row=1, col=1)
fig.add_trace(go.Heatmap(z=train.select(kept).corr().to_numpy(), x=kept, y=kept,
                         zmin=-1, zmax=1, colorscale="RdBu", reversescale=True),
              row=1, col=2)
fig.update_layout(title="Correlation-threshold filter — redundancy made visible", height=520)
fig.write_html(str(OUTPUT_DIR / "ex1_08_correlation_filter.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex1_08_correlation_filter.html'}")

# ── TRAIN — FeatureEngineer: generate candidates, then select ────────────
# Keep the demo honest: generate on a 5-column numeric core (interactions
# explode quadratically — 33 columns would make 528 pairs).
CORE = ["income_sgd", "employment_years", "credit_utilization", "avg_balance_utilization", "age"]
core_frame = train.select(CORE + ["default"])
schema = FeatureSchema(
    name="credit_core",
    features=[FeatureField(name=c, dtype="float64") for c in CORE],
    entity_id_column="age",  # required by the schema type; unused for generation
)

import warnings as _warnings
from kailash_ml._decorators import ExperimentalWarning as _ExperimentalWarning

with _warnings.catch_warnings():
    # Documented above: the SDK's P2 experimental-API notice, acknowledged.
    _warnings.simplefilter("ignore", _ExperimentalWarning)
    # TODO: construct FeatureEngineer with max_features=50
    engine = ____
# TODO: generate candidates — strategies=["interactions", "polynomial"]
# Hint: engine.generate(core_frame, schema, strategies=[...])
candidates = ____
print(f"\n  FeatureEngineer generated {candidates.total_candidates} candidates "
      f"from {len(CORE)} core features")

# IMPORTANT: select against candidates.data (original + generated columns),
# not the raw frame — generated columns only exist there.
# TODO: select top 15 by importance against candidates.data (not core_frame!)
selected = ____
print(f"  Selected top {selected.n_selected} by importance:")
for rank in selected.rankings[:10]:
    print(f"    {rank.column_name:<44} score {rank.score:.4f}")

# ── Checkpoint 2 ──────────────────────────────────────────────────────────
assert candidates.total_candidates >= len(CORE), "generation must add candidates"
assert 0 < selected.n_selected <= 15
assert any("_x_" in c.name for c in candidates.generated_columns), "interactions expected"
print("[ok] Checkpoint 2 — generate + select ran the spec's pattern")

# Does generation EARN its place? The honest test is not "raw vs selected"
# (importance picked only raw columns — itself a finding) but: does ADDING
# the best generated candidates beat the raw core alone?
generated_names = {c.name for c in candidates.generated_columns}
# TODO: ranking entries whose column is a GENERATED name, top 5
# Hint: [r.column_name for r in selected.rankings if r.column_name in generated_names]
gen_ranked = ____
top_generated = ____
print(f"  Best generated candidates: {top_generated}")
eval_frame = candidates.data
rng = np.random.default_rng(42)
eval_idx = rng.choice(eval_frame.height, size=20000, replace=False)

def cv_auc(frame: pl.DataFrame, cols: list[str]) -> float:
    Xs = frame.select(cols).to_numpy()[eval_idx]
    ys = frame["default"].to_numpy()[eval_idx]
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    return float(cross_val_score(est, Xs, ys,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                 scoring="roc_auc", n_jobs=8).mean())

# TODO: CV AUC of the raw core columns
auc_core = ____
# TODO: CV AUC of core + top generated (fall back to auc_core if none)
auc_aug = ____
delta = auc_aug - auc_core
print(f"\n  5-fold CV AUC — raw 5 core: {auc_core:.4f} · raw + top generated: {auc_aug:.4f}")
print(
    "  NOTE the disagreement: importance ranks three generated candidates at "
    "the top, but out-of-sample AUC does not move. Importance is measured "
    "inside the training fit; CV AUC is out-of-sample. When they disagree, "
    "the out-of-sample number wins."
)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=["raw 5 core features", "raw + top generated"],
                      y=[auc_core, auc_aug],
                      marker_color=["#64748B", "#0D9488"],
                      text=[f"{auc_core:.4f}", f"{auc_aug:.4f}"], textposition="auto"))
fig2.update_layout(title="Do generated candidates earn their place?",
                   yaxis_title="5-fold CV AUC", yaxis_range=[0.5, 1.0])
fig2.write_html(str(OUTPUT_DIR / "ex1_08_generated_vs_raw.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex1_08_generated_vs_raw.html'}")

# ── APPLY — a bureau-data cost review at a retail lender ─────────────────
print(
    "\n  APPLY: bureau features cost real money per pull. The correlation "
    "filter is the first pass of the cost review — redundant pairs are free "
    "savings. Automated generation is the second half of the meeting: "
    "FeatureEngineer manufactures the obvious interactions, importance "
    "selection keeps the few that earn their pull, and the AUC comparison "
    "tells the committee whether the engineered set beat the raw columns. "
    f"Today it moved {auc_core:.4f} → {auc_aug:.4f} ({delta:+.4f}), "
    "and that honest near-zero IS the answer — generate-then-select exists "
    "to MEASURE whether manufactured features help, not to assume it."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ Correlation-threshold filtering — redundancy vs 02's relevance filter
    ✓ Drop the member of each redundant pair LESS correlated with the target
    ✓ Automated generation: interactions + squares via FeatureEngineer
      .generate, kept honest by .select (importance, top_k)
    ✓ The quadratic explosion: 5 cores → a dozen+ candidates; 33 → hundreds

  This completes Exercise 1's toolkit: hand-built domain features (01),
  relevance filters (02), wrappers (03, 07), embedded (04), tracking (05),
  temporal features (06), and automated generate-then-select (08).
"""
)
