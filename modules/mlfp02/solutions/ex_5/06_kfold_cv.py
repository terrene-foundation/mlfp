# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 5.6: K-Fold Cross-Validation from Scratch
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why one train/test split is a noisy verdict: the split itself is
#     a draw from a random process
#   - Implement k-fold CV from scratch: seeded shuffle, k folds, every
#     row tested exactly once
#   - Read a CV DISTRIBUTION (mean ± sd across folds), not one number
#   - Use CV for model selection: does a squared floor-area term
#     generalise, or does it only win in-sample?
#   - Choose k: 5 vs 10 vs leave-one-out, and the bias-variance trade
#
# PREREQUISITES: 01_ols_from_scratch.py; 05_log_price_model.py helps
#   (model comparison on a common scale)
#
# ESTIMATED TIME: ~40 min
#
# TASKS:
#   1. Theory — what CV estimates and what it does not
#   2. Build — seeded fold assignment on the cleaned design matrix
#   3. Train — 5-fold and 10-fold CV; base vs squared-term model
#   4. Visualise — fold R² distributions + 20 random single splits
#   5. Apply — reporting a model's generalisation error to a stakeholder
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from shared.mlfp02.ex_5 import (
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    TARGET,
    build_design_matrix,
    fit_ols,
    kfold_indices,
    load_hdb_clean,
    ols_r2_on,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — What CV Estimates and What It Does Not
# ════════════════════════════════════════════════════════════════════════
# A single 80/20 split answers "how does THIS fitted model do on THAT
# held-out set?" The answer depends on the draw: reshuffle and you get a
# different R². Split variance is pure noise in your decision process.
#
# K-FOLD CV removes the draw: shuffle once (seeded), cut into k folds,
# and take turns holding each fold out:
#
#   fold 1: train on folds 2..k, test on fold 1
#   fold 2: train on folds 1,3..k, test on fold 2
#   ...
#   Every row is tested EXACTLY once and trained on k-1 times.
#
#   CV estimate = mean of the k out-of-fold R² values
#   CV spread   = sd of the same — the uncertainty of the estimate
#
# Choosing k: larger k → each fit sees more data (less bias) but the
# folds overlap more (more variance across repeats) and cost k fits.
# Practice: k = 5 or 10. LOO (k = n) is unbiased but high-variance and
# costs n fits — fine for n = 200, absurd for n = 25,000.
#
# WHAT CV DOES NOT DO: it does not fix leakage. If a feature was built
# using the whole dataset (a target-encoded column, a scaler fit on all
# rows), every fold already contains the answer. Feature pipelines belong
# INSIDE the fold loop — Exercise 8's point-in-time discipline is the
# same idea for time.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: Cleaned design matrix + seeded folds
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  MLFP02 Exercise 5.6: K-Fold Cross-Validation")
print("=" * 70)

hdb_all = load_hdb_clean()
hdb = hdb_all.filter(
    (pl.col(TARGET) >= 100_000) & (pl.col(TARGET) <= 5_000_000)
)
X, y, names = build_design_matrix(hdb)
n = len(y)
print(f"\n  Design matrix: {n:,} rows × {X.shape[1]} columns "
      f"(sentinel prices removed: {hdb_all.height - hdb.height:,})")

folds5 = kfold_indices(n, k=5, seed=42)
print(f"  5-fold split: fold sizes {[len(t[1]) for t in folds5]}")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert len(folds5) == 5, "Need exactly 5 folds"
assert sum(len(t[1]) for t in folds5) == n, "Folds must partition the data"
assert len(set(np.concatenate([t[1] for t in folds5]).tolist())) == n, (
    "Every row must be tested exactly once"
)
print("\n--- Checkpoint 1 passed --- folds partition the data\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: CV distributions + model selection
# ════════════════════════════════════════════════════════════════════════

# How noisy is ONE split? 20 random 80/20 splits of the same model
rng = np.random.default_rng(7)
single_split_r2 = []
for rep in range(20):
    order = rng.permutation(n)
    cut = int(0.8 * n)
    single_split_r2.append(ols_r2_on(X, y, order[:cut], order[cut:]))
single_split_r2 = np.array(single_split_r2)
print("=== The single-split lottery (20 random 80/20 splits) ===")
print(
    f"  R² range: [{single_split_r2.min():.4f}, {single_split_r2.max():.4f}]  "
    f"sd = {single_split_r2.std(ddof=1):.4f}"
)

# 5-fold and 10-fold CV for the base model
def cv_r2(Xm: np.ndarray, ym: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    return np.array(
        [ols_r2_on(Xm, ym, tr, te) for tr, te in kfold_indices(len(ym), k, seed)]
    )


cv5_base = cv_r2(X, y, k=5)
cv10_base = cv_r2(X, y, k=10)
print(f"\n=== Base model CV (features: {', '.join(names[1:])}) ===")
print(f"  5-fold:  mean {cv5_base.mean():.4f} ± {cv5_base.std(ddof=1):.4f}  "
      f"folds {np.round(cv5_base, 4).tolist()}")
print(f"  10-fold: mean {cv10_base.mean():.4f} ± {cv10_base.std(ddof=1):.4f}")

# Candidate enrichment: add floor_area_sqm² (non-linearity from the spec)
area = hdb["floor_area_sqm"].to_numpy().astype(np.float64)
X_sq = np.column_stack([X, area**2])
names_sq = [*names, "floor_area_sqm²"]

fit_base_full = fit_ols(X, y)
fit_sq_full = fit_ols(X_sq, y)
print(f"\n=== In-sample fit (all rows) ===")
print(f"  Base R²:   {fit_base_full['R2']:.4f}")
print(f"  +area² R²: {fit_sq_full['R2']:.4f}  "
      f"(in-sample ALWAYS favours the bigger model — adj-R² "
      f"{fit_base_full['adj_R2']:.4f} vs {fit_sq_full['adj_R2']:.4f})")

cv5_sq = cv_r2(X_sq, y, k=5)
print(f"\n=== Out-of-sample verdict (5-fold CV) ===")
print(f"  Base:   {cv5_base.mean():.4f} ± {cv5_base.std(ddof=1):.4f}")
print(f"  +area²: {cv5_sq.mean():.4f} ± {cv5_sq.std(ddof=1):.4f}")
lift = cv5_sq.mean() - cv5_base.mean()
pooled_sd = np.sqrt(cv5_base.var(ddof=1) / 5 + cv5_sq.var(ddof=1) / 5)
print(
    f"  CV lift from the squared term: {lift:+.4f} "
    f"(fold sd of each ≈ {pooled_sd:.4f})"
)
if lift > 2 * pooled_sd:
    print("  → The squared term GENERALISES: keep it.")
elif lift < -2 * pooled_sd:
    print("  → The squared term overfits: drop it.")
else:
    print("  → Within noise: keep the simpler model (parsimony wins ties).")

# ── Log the comparison to ExperimentTracker ──────────────────────────
run_id = track_train_run(
    experiment="mlfp02_ex5_06_kfold_cv",
    run_name="base_vs_area_squared_5fold",
    params={"k": "5,10", "candidate_feature": "floor_area_sqm^2", "seed": "42"},
    metrics={
        "single_split_sd": float(single_split_r2.std(ddof=1)),
        "cv5_base_mean": float(cv5_base.mean()),
        "cv5_base_sd": float(cv5_base.std(ddof=1)),
        "cv10_base_mean": float(cv10_base.mean()),
        "cv5_sq_mean": float(cv5_sq.mean()),
        "cv_lift": float(lift),
        "insample_r2_base": float(fit_base_full["R2"]),
        "insample_r2_sq": float(fit_sq_full["R2"]),
    },
)
print(f"\nLogged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert cv5_base.mean() > 0.5, "CV R² should be well above zero on this data"
assert cv5_base.mean() < fit_base_full["R2"] + 1e-9, (
    "CV (out-of-sample) should not beat the in-sample fit"
)
assert len(cv5_base) == 5 and len(cv10_base) == 10, "Fold counts must match k"
print("\n--- Checkpoint 2 passed --- CV estimates computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Fold distributions + the single-split lottery
# ════════════════════════════════════════════════════════════════════════

fig = go.Figure()
fig.add_trace(
    go.Box(y=single_split_r2, name="20 random 80/20 splits", boxpoints="all",
           marker_color="grey")
)
fig.add_trace(
    go.Box(y=cv5_base, name="5-fold CV (base)", boxpoints="all",
           marker_color="blue")
)
fig.add_trace(
    go.Box(y=cv5_sq, name="5-fold CV (+area²)", boxpoints="all",
           marker_color="green")
)
fig.update_layout(
    title="Out-of-Sample R²: Split Noise vs CV Estimates",
    yaxis_title="Out-of-sample R²",
    height=450,
)
fig_path = OUTPUT_DIR / "kfold_distributions.html"
fig.write_html(str(fig_path))
print(f"Saved: {fig_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert fig_path.exists(), "Figure must be written"
print("\n--- Checkpoint 3 passed --- visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Reporting Generalisation Error to a Stakeholder
# ════════════════════════════════════════════════════════════════════════
# A valuation desk head (anonymised) asks: "If we ship this model, how
# well will it do on NEXT quarter's transactions?" The honest report is
# a CV estimate with its spread — and its limits:
#
#   - CV mean ± sd: the expected out-of-sample R² for THIS procedure
#     on data from the SAME period and process.
#   - It is NOT a promise about a different market regime — that needs
#     out-of-TIME validation (Exercise 8).
#   - Report both numbers: the mean is the estimate; the sd is how much
#    the fold composition still moves it.

print("=== APPLICATION: The Stakeholder Report ===")
print(
    f"\n  Model: price ~ {', '.join(names[1:])}"
    + (" + floor_area²" if lift > 2 * pooled_sd else "")
)
print(f"  Expected out-of-sample R²: {cv5_base.mean():.3f} "
      f"± {cv5_base.std(ddof=1):.3f} (5-fold CV)")
print(f"  A single split would have said anywhere in "
      f"[{single_split_r2.min():.3f}, {single_split_r2.max():.3f}] — "
      "that is the noise CV removes.")
print(
    "\n  Caveats for the desk head:\n"
    "    • Same-period estimate only — regime shifts need out-of-time\n"
    "      validation (ex_8).\n"
    "    • CV does not detect leakage built before the split; feature\n"
    "      pipelines belong inside the fold loop.\n"
    "    • k=5 vs k=10 differ by far less than one split's noise here."
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert cv5_base.std(ddof=1) < single_split_r2.std(ddof=1) + 0.05, (
    "CV spread should be comparable to or smaller than split-to-split noise"
)
print("\n--- Checkpoint 4 passed --- stakeholder report complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (5.6 — K-Fold Cross-Validation)")
print("═" * 70)
print(
    """
  ✓ One split is a lottery: 20 random 80/20 splits of the SAME model
    span a visible R² range — that noise is the split, not the model
  ✓ K-fold from scratch: seeded shuffle, partition, each fold tested
    once — CV mean ± sd replaces the single draw
  ✓ Model selection by CV: in-sample R² always favours the bigger
    model; the CV lift decides whether area² generalises
  ✓ k = 5 or 10 is the practical band; LOO costs n fits for little gain
  ✓ CV estimates same-process generalisation — it does not fix leakage
    and it does not see regime change (that is out-of-time's job)

  NEXT: In 07_geo_features.py, you'll give the model a map — town
  centroids, haversine distance to the CBD, and what location adds.
"""
)

print("\n✓ Exercise 5.6 complete — K-Fold Cross-Validation")
