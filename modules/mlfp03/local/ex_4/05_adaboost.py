# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 4.5: AdaBoost — The Warm-Up Act of Boosting
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - AdaBoost (1997): reweight the ROWS, not the gradient — stump by stump
#   - Build it from scratch in ~25 lines to see the weight update
#   - Why exponential loss makes AdaBoost fragile to label noise
#   - Where AdaBoost still wins: instant, interpretable stump baselines
#
# PREREQUISITES: 01_boosting_theory.py
# ESTIMATED TIME: ~30 min
#
# 5-PHASE STRUCTURE:
#   Theory   — reweighting vs gradient descent: the genealogy of boosting
#   Build    — AdaBoost from scratch with decision stumps
#   Train    — from-scratch vs sklearn AdaBoostClassifier vs LightGBM
#   Visualise — staged AUC: the overfitting bend
#   Apply    — a collections team's same-day triage baseline
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from shared.mlfp03.ex_4 import (
    OUTPUT_DIR,
    evaluate_classifier,
    prepare_credit_split,
    print_metrics,
)

# ── THEORY — the genealogy ───────────────────────────────────────────────
# AdaBoost (Freund & Schapire, 1997) is where boosting began. No gradients:
# fit a weak learner, measure its weighted error, give it a vote
# α = ½ ln((1−err)/err), then MULTIPLY the weight of every row it got wrong
# by e^α and renormalise. The next stump sees a different dataset — one
# where the hard rows shout louder. Final vote: Σ α_t · stump_t(x).
#
# Gradient boosting (02–04) descends a LOSS FUNCTION instead: fit each new
# tree to the negative gradient (pseudo-residuals). AdaBoost's reweighting
# is secretly the same move for the EXPONENTIAL loss e^(−y·f(x)) — and that
# loss is exactly why AdaBoost is fragile: exponential penalties explode on
# mislabeled rows, so noisy labels dominate the fit. Squared/log losses in
# modern boosters bend instead of exploding.

split = prepare_credit_split()
X_train, y_train = split["X_train"], split["y_train"]
X_test, y_test = split["X_test"], split["y_test"]
feature_names = split["feature_names"]
print(f"  Credit split: {X_train.shape[0]:,} train / {X_test.shape[0]:,} test, "
      f"{len(feature_names)} features, default rate {split['default_rate']:.2%}")

# AdaBoost fits stumps fast on a modest sample; keep the from-scratch demo
# honest but quick with a seeded 10k-row subsample.
rng = np.random.default_rng(42)
sub = rng.choice(len(y_train), size=10_000, replace=False)
Xs, ys = X_train[sub], y_train[sub]

# ── BUILD — AdaBoost from scratch (stumps + row reweighting) ─────────────
def fit_stump(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[int, float, int]:
    """Best single-threshold stump under row weights w (±1 labels).

    Returns (feature_index, threshold, sign) minimising the WEIGHTED error.
    """
    best = (0, 0.0, 1)
    # TODO: start the best weighted error at +infinity
    best_err = ____
    for j in range(X.shape[1]):
        thresholds = np.quantile(X[:, j], np.linspace(0.05, 0.95, 19))
        for t in thresholds:
            pred_left = np.where(X[:, j] <= t, 1, -1)
            err_left = w[pred_left != y].sum()
            err_right = w[-pred_left != y].sum()
            if err_left <= err_right:
                err, pred, sign = err_left, pred_left, 1
            else:
                err, pred, sign = err_right, -pred_left, -1
            if err < best_err:
                best_err, best = err, (j, t, sign)
    return best

def adaboost_scratch(X: np.ndarray, y01: np.ndarray, n_rounds: int = 50):
    """AdaBoost.M1 with stumps. Returns (stumps, alphas) for ±1 labels."""
    y = np.where(y01 > 0, 1, -1)
    w = np.full(len(y), 1.0 / len(y))
    stumps, alphas = [], []
    for _ in range(n_rounds):
        j, t, sign = fit_stump(X, y, w)
        pred = np.where(X[:, j] <= t, sign, -sign)
        err = w[pred != y].sum()
        err = np.clip(err, 1e-10, 1 - 1e-10)
        # TODO: stump vote alpha = ½ ln((1−err)/err)
        alpha = ____
        # TODO: reweight rows: w *= exp(−alpha · y · pred), then renormalise
        # Hint: correct rows shrink, wrong rows grow; w /= w.sum() at the end
        ____
        ____
        stumps.append((j, t, sign))
        alphas.append(alpha)
    return stumps, alphas

def predict_scratch(X: np.ndarray, stumps, alphas) -> np.ndarray:
    score = np.zeros(X.shape[0])
    for (j, t, sign), a in zip(stumps, alphas):
        score += a * np.where(X[:, j] <= t, sign, -sign)
    return score  # margin: sign gives the class, magnitude the confidence

# ── TRAIN — from-scratch vs sklearn vs LightGBM ──────────────────────────
print("\n  From-scratch AdaBoost (50 stumps, 10k-row teaching subsample)...")
# TODO: fit 50 stumps on the teaching subsample
stumps, alphas = ____
# TODO: test margins, then logistic-of-margin to probabilities
scratch_margin = ____
scratch_proba = ____
m_scratch = evaluate_classifier(y_test, scratch_proba)

print("  sklearn AdaBoostClassifier (same subsample, 200 stumps)...")
# TODO: sklearn AdaBoostClassifier — stump base (max_depth=1), 200 rounds,
#       learning_rate=0.5, fitted on the SAME subsample (Xs, ys)
ada = ____
m_ada = evaluate_classifier(y_test, ada.predict_proba(X_test)[:, 1])

print("  LightGBM reference (full train split, course defaults)...")
lgbm = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1).fit(X_train, y_train)
m_lgbm = evaluate_classifier(y_test, lgbm.predict_proba(X_test)[:, 1])

print()
for name, m in [("AdaBoost from scratch", m_scratch),
                ("AdaBoost (sklearn)", m_ada),
                ("LightGBM (reference)", m_lgbm)]:
    print_metrics(name, m)

# ── Checkpoint 1 ──────────────────────────────────────────────────────────
assert len(stumps) == 50 and len(alphas) == 50
assert m_scratch["auc_roc"] > 0.6, "from-scratch AdaBoost must beat chance decisively"
assert abs(m_scratch["auc_roc"] - m_ada["auc_roc"]) < 0.05, (
    "from-scratch and sklearn AdaBoost should land close together"
)
assert m_lgbm["auc_pr"] > m_ada["auc_pr"], "modern boosting wins on the noisy real data"
print("[ok] Checkpoint 1 — scratch ≈ sklearn ≈ LightGBM within noise on CLEAN data;")
print("      AdaBoost's fragility appears under label noise, not here — that is the point")

# ── VISUALISE — staged AUC: where AdaBoost bends ─────────────────────────
# TODO: staged AUC-ROC per iteration — positive-class column of each
#       staged_predict_proba output, scored with evaluate_classifier
staged_auc = ____
every = 4
xs = list(range(1, len(staged_auc) + 1, every))
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=xs, y=staged_auc[::every], mode="lines", name="AdaBoost (staged AUC)",
    line=dict(color="#F59E0B"),
))
fig.add_hline(y=m_lgbm["auc_roc"], line_dash="dash", line_color="#6366F1",
              annotation_text="LightGBM final AUC")
fig.update_layout(
    title="AdaBoost staged AUC — the bend where stumps start fitting noise",
    xaxis_title="iteration", yaxis_title="held-out AUC-ROC", height=440,
)
fig.write_html(str(OUTPUT_DIR / "ex4_05_adaboost_staged.html"))
print(f"  Saved: {OUTPUT_DIR / 'ex4_05_adaboost_staged.html'}")
print("[ok] Checkpoint 2 — staged curve saved")

# ── APPLY — a collections team's same-day triage baseline ────────────────
# TODO: first iteration index where staged AUC peaks (1-based)
best_iter = ____
print(
    f"\n  APPLY: a collections team needs a same-day call-priority baseline "
    f"before the modern stack ships. AdaBoost-on-stumps is the honest answer: "
    f"each stump is a one-line rule an agent can read ('months_employed ≤ 14 "
    f"→ risk'), the staged curve says where to stop (iteration {best_iter} "
    f"here), and nobody pretends 50 rules are a production model. When label "
    f"noise arrives — misrecorded defaults — the exponential loss will chase "
    f"the bad labels; that is your cue to graduate to LightGBM, not to tune "
    f"AdaBoost harder."
)

# REFLECTION
print(
    """
  What you've mastered:
    ✓ AdaBoost reweights ROWS (α = ½ ln((1−err)/err)); gradient boosting
      reweights the LOSS. Same genealogy, different machinery
    ✓ 25 lines of numpy reproduce sklearn's AdaBoost within noise
    ✓ Exponential loss = the overfitting bend on the staged curve
    ✓ AdaBoost's real 2026 role: instant interpretable baselines, not the
      production model

  Next: 06_catboost_native.py — CatBoost on RAW categoricals: no ordinal
  encoding, no invented order, and the ordered-target-statistics leak guard.
"""
)
