# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 5.2: Sampling Strategies — SMOTE vs Cost-Sensitive
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - How SMOTE generates synthetic minority samples (k-NN interpolation)
#   - The three failure modes of SMOTE (Lipschitz, noise, dimensionality)
#   - Cost-sensitive learning via scale_pos_weight and sample weights
#   - How each strategy changes ranking (AUC-PR) AND calibration (Brier)
#   - How to catch SMOTE fabricating impossible rows in your own data
#
# PREREQUISITES: 01_metrics_and_baseline.py (saves the baseline — this
#                file reads it back for the calibration comparison)
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — SMOTE intuition + failure taxonomy, then cost-sensitive loss
#   Build    — imblearn SMOTE pipeline + LightGBM with sample_weight
#   Train    — fit both strategies on the same splits
#   Visualise — side-by-side metrics table + class-balance diagram
#   Apply    — an illustrative card-fraud scenario where SMOTE misleads
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
import polars as pl
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE

from shared.mlfp03.ex_5 import (
    DEFAULT_COSTS,
    OUTPUT_DIR,
    load_credit_splits,
    load_strategy_proba,
    metrics_row,
    print_metrics_table,
    save_strategy_proba,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — SMOTE and its three failure modes
# ════════════════════════════════════════════════════════════════════════
# SMOTE (Synthetic Minority Over-sampling TEchnique) fixes imbalance by
# MAKING UP new minority samples. For each minority row it:
#   1. Finds its k nearest minority neighbours
#   2. Picks one neighbour at random
#   3. Creates a new synthetic row on the line segment between them
#
# This works beautifully in toy 2-D plots. Then it goes to production and
# fails for three distinct reasons.
#
#   FAILURE 1 — Lipschitz violation. Interpolation assumes the decision
#     boundary is smooth between the two real points. In credit scoring,
#     `age=20, income=S$3k, tenure=1yr` and `age=60, income=S$3k, tenure=1yr`
#     may both be "default=yes" but the midpoint is a completely different
#     customer profile that doesn't match ANY real applicant. The synthetic
#     row trains the model to believe in customers that don't exist.
#
#   FAILURE 2 — Noise amplification. Real defaulters include mislabelled
#     rows, data-entry errors, and unusual edge cases. SMOTE copies those
#     errors multiple times. Your model now fits the noise better than
#     the signal.
#
#   FAILURE 3 — High-dimensional collapse. In >20 features, nearest
#     neighbours become nearly equidistant (curse of dimensionality).
#     "Between two neighbours" loses meaning. The interpolated row is
#     just a random blob in feature space.
#
# SMOTE is hugely popular in the research literature (Fernández et al.,
# 2018, review its first 15 years) — but by construction it trains the
# model on a 50/50 world, so its raw probabilities no longer describe the
# real ~13% default rate unless you correct them afterwards.
#
# COST-SENSITIVE ALTERNATIVE: instead of faking new data, we tell the
# loss function how much each mistake costs. LightGBM supports two
# mechanisms:
#   - `scale_pos_weight = n_neg / n_pos` (class-balanced)
#   - `sample_weight = cost_matrix[y]`   (from the business cost matrix)
# The second form is more general: you can encode any asymmetric cost,
# not just the class ratio. BUT reweighting has a price too: up-weighting
# defaulters tells the model defaults are more common than they are, so
# it OVER-predicts default probability. Better recall at 0.5, worse
# calibration — which 5.5 repairs with Platt/isotonic calibration.


# ════════════════════════════════════════════════════════════════════════
# BUILD — SMOTE and cost-sensitive classifiers
# ════════════════════════════════════════════════════════════════════════

X_train, y_train, X_test, y_test, pos_rate = load_credit_splits()

print("\n" + "=" * 70)
print("  Exercise 5.2 — Sampling Strategies (SMOTE vs Cost-Sensitive)")
print("=" * 70)

# --- SMOTE pipeline -----------------------------------------------------
smote = SMOTE(random_state=42)
# TODO: Fit imblearn SMOTE on X_train, y_train
# Hint: smote.fit_resample(X, y) returns (X_resampled, y_resampled)
X_smote, y_smote = ____

smote_model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)

# --- Cost-sensitive (A) scale_pos_weight --------------------------------
# TODO: Derive scale_weight from pos_rate
# Hint: (1 - p) / p — this equals n_neg / n_pos
scale_weight = ____
cost_a_model = lgb.LGBMClassifier(
    n_estimators=300,
    scale_pos_weight=____,
    random_state=42,
    verbose=-1,
)

# --- Cost-sensitive (B) sample_weight from cost matrix ------------------
# TODO: Build per-row sample_weights from the business cost matrix
# Hint: np.where(condition, value_if_default, value_if_repaid) with DEFAULT_COSTS
sample_weights = ____
cost_b_model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)


# ════════════════════════════════════════════════════════════════════════
# TRAIN — fit all three
# ════════════════════════════════════════════════════════════════════════

# TODO: Fit smote_model on the SMOTE-resampled data
____
# TODO: Fit cost_a_model on the ORIGINAL X_train, y_train
____
# TODO: Fit cost_b_model on X_train, y_train, passing sample_weight=...
____

y_proba_smote = smote_model.predict_proba(X_test)[:, 1]
y_proba_cost_a = cost_a_model.predict_proba(X_test)[:, 1]
y_proba_cost_b = cost_b_model.predict_proba(X_test)[:, 1]

save_strategy_proba("smote", y_proba_smote)
save_strategy_proba("cost_sensitive_scale", y_proba_cost_a)
save_strategy_proba("cost_sensitive_matrix", y_proba_cost_b)


# ── Checkpoint 2 ────────────────────────────────────────────────────────
assert len(y_smote) > len(y_train), "SMOTE must increase dataset size"
assert y_smote.mean() > pos_rate, "SMOTE must rebalance minority class"
assert scale_weight > 1.0, "scale_pos_weight must up-weight the minority class"
assert sample_weights[y_train == 1][0] == DEFAULT_COSTS.fn, "FN weight mismatch"
print("[ok] Checkpoint 2 — three imbalance strategies trained\n")


# ════════════════════════════════════════════════════════════════════════
# VISUALISE — side-by-side metrics + class-balance diagram
# ════════════════════════════════════════════════════════════════════════

rows = [
    metrics_row("SMOTE", y_test, y_proba_smote),
    metrics_row("Cost-sens (scale_pos)", y_test, y_proba_cost_a),
    metrics_row("Cost-sens (matrix)", y_test, y_proba_cost_b),
]
print_metrics_table(rows, "Sampling strategy comparison (threshold=0.5)")

print("\n  Class balance after each strategy:")
print(f"    {'Strategy':<24} {'n_neg':>8} {'n_pos':>8} {'pos_rate':>10}")
print("    " + "─" * 52)
print(
    f"    {'Original training':<24} {int((y_train == 0).sum()):>8,} "
    f"{int((y_train == 1).sum()):>8,} {pos_rate:>10.2%}"
)
print(
    f"    {'After SMOTE':<24} {int((y_smote == 0).sum()):>8,} "
    f"{int((y_smote == 1).sum()):>8,} {y_smote.mean():>10.2%}"
)
print(
    f"    {'Cost-sens (weights)':<24} {int((y_train == 0).sum()):>8,} "
    f"{int((y_train == 1).sum()):>8,} {pos_rate:>10.2%}"
)
print("    (cost-sensitive changes the LOSS, not the data — no fake rows)")

metrics_df = pl.DataFrame(rows)
metrics_df.write_parquet(OUTPUT_DIR / "sampling_metrics.parquet")
print(f"\n  Saved: {OUTPUT_DIR / 'sampling_metrics.parquet'}")

# ── Visual: Precision-Recall comparison across strategies ────────────────
strategy_names = [r["strategy"] for r in rows]
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=strategy_names,
        y=[r["auc_pr"] for r in rows],
        name="AUC-PR",
        marker_color="#6366f1",
    )
)
fig.add_trace(
    go.Bar(
        x=strategy_names,
        y=[r["brier"] for r in rows],
        name="Brier score",
        marker_color="#f43f5e",
    )
)
fig.update_layout(
    title="Sampling Strategy Comparison: AUC-PR (higher = better ranking) vs Brier (lower = better probabilities)",
    barmode="group",
    yaxis_title="Score",
    height=450,
    legend=dict(orientation="h", y=-0.2),
)
viz_path = OUTPUT_DIR / "ex5_02_sampling_comparison.html"
fig.write_html(str(viz_path))
print(f"  Saved: {viz_path}")

# ── Visual: SMOTE vs Original data distribution ─────────────────────────
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=X_train[:500, 0],
        y=X_train[:500, 1],
        mode="markers",
        marker=dict(
            color=y_train[:500].astype(float), colorscale="RdBu", size=4, opacity=0.5
        ),
        name="Original",
    )
)
fig2.add_trace(
    go.Scatter(
        x=X_smote[-500:, 0],
        y=X_smote[-500:, 1],
        mode="markers",
        marker=dict(
            color=y_smote[-500:].astype(float),
            colorscale="Sunset",
            size=4,
            opacity=0.5,
            symbol="diamond",
        ),
        name="SMOTE synthetic",
    )
)
fig2.update_layout(
    title="SMOTE vs Original: Feature-space scatter (first two features)",
    xaxis_title="Feature 0",
    yaxis_title="Feature 1",
    height=450,
)
viz_path2 = OUTPUT_DIR / "ex5_02_smote_scatter.html"
fig2.write_html(str(viz_path2))
print(f"  Saved: {viz_path2}")

# ── Calibration at a glance: does each strategy's AVERAGE predicted
# probability still match the real default rate? (A calibrated model's
# mean prediction ≈ the base rate.)
y_proba_base = load_strategy_proba("baseline")
calib_rows = [("Baseline (01)", y_proba_base)] + [
    (r["strategy"], p)
    for r, p in zip(rows, [y_proba_smote, y_proba_cost_a, y_proba_cost_b])
]
print(f"\n  Real default rate in test: {y_test.mean():.3f}")
print(f"  {'Strategy':<24} {'mean p':>8} {'Brier':>8} {'AUC-PR':>8}")
for name, p in calib_rows:
    r = metrics_row(name, y_test, p)
    print(f"  {name:<24} {p.mean():>8.3f} {r['brier']:>8.4f} {r['auc_pr']:>8.4f}")

# ── SMOTE forensics: can the model tell a synthetic row from a real one?
# 22 of our columns only ever hold whole numbers (counts, ordinal-encoded
# categories). SMOTE interpolates between two rows, so it writes values
# like gender = 0.37 that no real applicant can have.
integer_cols = [j for j in range(X_train.shape[1]) if np.all(np.mod(X_train[:, j], 1) == 0)]
X_synthetic = X_smote[len(X_train) :]
# TODO: Share of synthetic rows with ANY fractional value in an integer column
# Hint: np.mod(values, 1) != 0 marks fractions; np.any(..., axis=1).mean()
impossible_share = ____
p_synthetic = float(smote_model.predict_proba(X_synthetic[:2000])[:, 1].mean())
p_real_defaulters = float(smote_model.predict_proba(X_train[y_train == 1][:2000])[:, 1].mean())
print(f"\n  Synthetic rows with an impossible fractional value: {impossible_share:.1%}")
print(f"  SMOTE model's mean P(default) on synthetic rows:   {p_synthetic:.3f}")
print(f"  SMOTE model's mean P(default) on REAL defaulters:  {p_real_defaulters:.3f}")
# INTERPRETATION: read the two tables above. If the synthetic rows score
# far higher than real defaulters, the model has partly learned to spot
# SMOTE's fingerprints (fractional category codes) rather than default
# risk. If the weighted models' mean p sits well above the real default
# rate, reweighting has inflated the probabilities — fine for ranking,
# wrong for pricing until recalibrated.


# ════════════════════════════════════════════════════════════════════════
# APPLY — Card-fraud detection (illustrative): where SMOTE misleads
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore card issuer scores every tap/swipe
# in real time; ~0.2% of transactions are fraudulent. A team "fixes" the
# imbalance with SMOTE and ships the model.
#
# The forensics you just ran show the risks in miniature:
#   - Offline metrics can look fine while the model has partly learned
#     SMOTE's fingerprints (impossible interpolated values) instead of
#     fraud behaviour — patterns that never occur in live traffic.
#   - The model was trained on a 50/50 world, so its scores cannot be
#     read as fraud probabilities without recalibration.
#   - Every synthetic row is a "customer" an auditor cannot trace back to
#     a real transaction.
#
# Class weighting avoids the fabricated rows (nothing to audit) but, as
# the mean-p column shows, inflates the probabilities: the issuer would
# still need 5.4 (re-choose the threshold) and 5.5 (recalibrate) before
# using the scores to set per-merchant decline rules.

best = min(rows, key=lambda r: r["brier"])
base_brier = metrics_row("baseline", y_test, y_proba_base)["brier"]
print("\n  Card-fraud implication (computed from the tables above):")
print(f"    Baseline Brier (no correction):      {base_brier:.4f}")
for r in rows:
    verdict = "better" if r["brier"] < base_brier else "worse"
    print(f"    {r['strategy']:<24} Brier {r['brier']:.4f} ({verdict} than baseline)")
print(f"    Best-calibrated imbalance strategy:  {best['strategy']}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED — 5.2")
print("=" * 70)
print(
    """
  [x] Ran SMOTE via imblearn and observed the class-balance change
  [x] Trained cost-sensitive LightGBM via scale_pos_weight (class-balanced)
  [x] Trained cost-sensitive LightGBM via explicit sample_weight (matrix)
  [x] Compared all three strategies on the complete metrics taxonomy
  [x] Measured how each strategy moves ranking (AUC-PR) AND calibration
      (Brier, mean predicted probability vs the real default rate)
  [x] Caught SMOTE fabricating impossible rows in this very dataset
  [x] Mapped SMOTE's failure modes onto an illustrative card-fraud case

  KEY INSIGHT: Neither trick gives you probabilities you can price from.
  SMOTE fabricates rows (and here the model learned their fingerprints);
  class weighting fabricates nothing but inflates the scores. Weighting
  is the auditable choice — then re-choose the threshold (5.4) and
  recalibrate (5.5).

  Next: 03_loss_functions.py — focal loss goes further, down-weighting
  easy examples automatically with a single gamma parameter.
"""
)
