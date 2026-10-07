# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 1.7: Forward and Backward Selection — Wrappers
#                         Beyond RFE
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Forward selection: start empty, add the feature that helps most
#   - Backward elimination: start full, drop the feature that hurts least
#   - How both differ from RFE (recursive importance-pruning)
#   - Why wrapper answers disagree — and why that's information, not noise
#
# PREREQUISITES: 03_wrapper_selection.py (RFE)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — greedy search direction changes WHAT is found
#   Build    — SequentialFeatureSelector forward and backward
#   Train    — run both searches, compare with RFE's picks
#   Visualise — overlap diagram + CV-AUC of each selected set
#   Apply    — feature-governance review at a clinical registry
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shared import MLFPDataLoader
from shared.kailash_helpers import split_then_preprocess
from shared.mlfp03.ex_1 import OUTPUT_DIR
from shared.mlfp03.ex_5 import CREDIT_NON_FEATURE_COLUMNS

# ── THEORY — greedy direction changes the answer ─────────────────────────
# RFE (03) fits once on ALL features, reads the estimator's importance
# ranking, and prunes the weakest — it trusts the importance numbers.
# Sequential search trusts only OUT-OF-SAMPLE SCORE:
#
#   FORWARD — start with nothing; repeatedly ADD the feature whose
#   addition raises CV score the most. Cheap, finds the "money features"
#   first, but can lock in a redundant pair early.
#
#   BACKWARD — start with everything; repeatedly DROP the feature whose
#   removal hurts CV score least (or helps most). Keeps synergistic groups
#   longer, costs more fits, and can retain noise that a forward search
#   never admits.
#
# The three wrappers often select different sets. That disagreement IS the
# lesson: selection is estimator- and direction-dependent, so the honest
# deliverable is the OVERLAP plus the CV score of each set — not a crown
# for one method.

# The credit-scoring frame — real signal (ex_5 family baseline AUC ≈ 0.80),
# so the selection scoreboard actually discriminates. Same leak-free
# split-then-preprocess discipline as the rest of the module.
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
y = train["default"].to_numpy()
print(f"  Feature matrix: {X.shape[0]:,} loans × {X.shape[1]} features "
      f"(train split; positive rate {y.mean():.2%})")

# ── BUILD — the search estimator and both selectors ──────────────────────
# Logistic regression is the right search estimator here: fast enough for
# the O(n_features²) fit count, and its linear bias makes the selections
# complement 03's tree-based RFE. Standardisation is part of the pipeline
# so the CV folds never see the full-data scaling.
selector_est = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42),
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
N_SELECT = 12

# Search on a seeded subsample (wrapper searches are O(d²) fits); the FINAL
# evaluation of every selected set runs on the full frame.
rng = np.random.default_rng(42)
search_idx = rng.choice(len(y), size=min(8000, len(y)), replace=False)
X_search, y_search = X[search_idx], y[search_idx]

# ── TRAIN — forward, backward, and RFE for comparison ────────────────────
print(f"\n  Forward selection to {N_SELECT} features (3-fold CV, search subsample)...")
sfs_fwd = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000, random_state=42),
    n_features_to_select=N_SELECT,
    direction="forward",
    cv=cv,
    n_jobs=8,
)
# SFS has no pipeline wrapper for scaling per fold; scale once on the
# SEARCH subsample only (never the full frame) to keep the demo honest.
scaler = StandardScaler().fit(X_search)
Xs = scaler.transform(X_search)
sfs_fwd.fit(Xs, y_search)
fwd_set = [c for c, keep in zip(feature_cols, sfs_fwd.get_support()) if keep]
print(f"    forward picked: {fwd_set[:4]} … ({len(fwd_set)} total)")

print(f"  Backward elimination to {N_SELECT} features...")
sfs_bwd = SequentialFeatureSelector(
    LogisticRegression(max_iter=1000, random_state=42),
    n_features_to_select=N_SELECT,
    direction="backward",
    cv=cv,
    n_jobs=8,
)
sfs_bwd.fit(Xs, y_search)
bwd_set = [c for c, keep in zip(feature_cols, sfs_bwd.get_support()) if keep]
print(f"    backward picked: {bwd_set[:4]} … ({len(bwd_set)} total)")

print(f"  RFE to {N_SELECT} features (same estimator, for a fair three-way)...")
rfe = RFE(
    LogisticRegression(max_iter=1000, random_state=42),
    n_features_to_select=N_SELECT,
    step=5,
)
rfe.fit(Xs, y_search)
rfe_set = [c for c, keep in zip(feature_cols, rfe.support_) if keep]

# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(fwd_set) == len(bwd_set) == len(rfe_set) == N_SELECT
overlap_fb = set(fwd_set) & set(bwd_set)
print(f"\n  forward ∩ backward: {len(overlap_fb)} of {N_SELECT} features agree")
print("[ok] Checkpoint 1 — three wrapper answers on the table")

# ── VISUALISE — overlap + full-frame CV AUC of each set ──────────────────
# Final scoring on a seeded 20k-row evaluation subsample of the train split —
# large enough that CV AUC differences of ~0.005 are real, small enough to run.
eval_idx = rng.choice(len(y), size=min(20000, len(y)), replace=False)
X_eval, y_eval = X[eval_idx], y[eval_idx]

def cv_auc(cols: list[str]) -> float:
    idx = [feature_cols.index(c) for c in cols]
    est = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    )
    scores = cross_val_score(
        est, X_eval[:, idx], y_eval,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        scoring="roc_auc", n_jobs=8,
    )
    return float(scores.mean())

sets = {"forward": fwd_set, "backward": bwd_set, "RFE": rfe_set}
aucs = {name: cv_auc(cols) for name, cols in sets.items()}
aucs[f"all {len(feature_cols)} (reference)"] = cv_auc(feature_cols)

print("\n  5-fold CV AUC on the evaluation subsample (the honest scoreboard):")
for name, auc in aucs.items():
    print(f"    {name:<22} AUC {auc:.4f}  ({len(sets.get(name, feature_cols))} features)")

fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=list(aucs.keys()),
        y=list(aucs.values()),
        marker_color=["#0D9488", "#6366F1", "#F59E0B", "#64748B"],
        text=[f"{a:.4f}" for a in aucs.values()],
        textposition="auto",
    )
)
fig.update_layout(
    title="Three wrappers, three answers — the full-frame CV scoreboard",
    yaxis_title="5-fold CV AUC",
    yaxis_range=[0.5, 1.0],
)
fig.write_html(str(OUTPUT_DIR / "ex1_07_forward_backward.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex1_07_forward_backward.html'}")

# ── Checkpoint 2 ──────────────────────────────────────────────────────────
ref_key = f"all {len(feature_cols)} (reference)"
best = max(a for k, a in aucs.items() if k != ref_key)
assert best >= aucs[ref_key] - 0.02, (
    f"a {N_SELECT}-feature wrapper set should stay within 0.02 AUC of all {len(feature_cols)}"
)
print(f"[ok] Checkpoint 2 — {N_SELECT} features carry (nearly) the full signal")

# ── APPLY — feature-governance review at a retail lender ────────────────
print(
    "\n  APPLY: a retail lender's model-governance team must justify every "
    "feature in the scorecard (privacy review, bureau data cost). The wrapper "
    "trio is the review "
    "instrument: features all three methods pick are uncontroversial keeps; "
    "features only ONE direction picks get a human look — forward-only picks "
    "are usually 'first strong signal', backward-only picks are usually "
    "'redundant but harmless'. The deliverable is the agreement table plus "
    f"the CV cost of dropping to {N_SELECT} — not a single ordained list."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ Forward selection vs backward elimination — direction changes the answer
    ✓ How both differ from RFE's importance-pruning (out-of-sample score
      vs fitted importances)
    ✓ Searching on a subsample but SCORING the final sets on the full frame
    ✓ Reading method disagreement as governance information

  Next: 08_correlation_and_automated_generation.py — the correlation-
  threshold filter and kailash-ml's FeatureEngineer generate + select.
"""
)
