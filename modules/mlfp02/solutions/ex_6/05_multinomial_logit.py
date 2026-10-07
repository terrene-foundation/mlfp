# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP02 — Exercise 6.5: Multinomial Logistic Regression
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Generalise binary logistic to K classes: softmax + reference class
#   - Why one class is the reference (identification: only K-1 logit
#     vectors are free — the Kth is fixed at zero)
#   - Fit multinomial MLE from scratch (NLL + analytical gradient, L-BFGS)
#   - Read odds ratios against the reference class: exp(β) per feature
#   - Compare against sklearn's multinomial oracle and a one-vs-rest
#     baseline (the framing alternative from the lesson)
#
# PREREQUISITES: 01_logistic_from_scratch.py (sigmoid, Bernoulli NLL)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — from two classes to K: softmax and the reference class
#   2. Build — 3/4/5-room transactions with context features (no area!)
#   3. Train — multinomial MLE from scratch + sklearn oracle agreement
#   4. Visualise — class-probability profiles across price
#   5. Apply — an agent infers flat type when the listing omits area
#
# ─── FRAMEWORK-FIRST EXEMPTION ──────────────────────────────────────────
# As in 01_logistic_from_scratch.py, sklearn appears ONCE as a
# correctness oracle for the from-scratch optimiser. M2 teaches logistic
# models as inference; production engines arrive in M3.
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression  # exemption: oracle only
from sklearn.metrics import accuracy_score  # exemption: stateless utility

from shared.mlfp02.ex_6 import (
    FLAT_TYPE_CLASSES,
    MULTINOMIAL_FEATURES,
    OUTPUT_DIR,
    load_flat_type_frame,
    multinomial_gradient,
    neg_log_likelihood_multinomial,
    softmax_rows,
    track_train_run,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — From Two Classes to K: Softmax and the Reference Class
# ════════════════════════════════════════════════════════════════════════
# Binary logistic models ONE logit: log(P₁/P₀) = xβ. With K classes we
# need K-1 logits against a REFERENCE class (class 0):
#
#   log(P_k / P_0) = xβ_k     for k = 1, …, K-1
#   P_k = exp(xβ_k) / (1 + Σ_j exp(xβ_j))        ← the softmax
#
# The reference class is not optional: adding a constant vector c to ALL
# β_k leaves every probability unchanged, so the model is unidentified
# until one class is pinned at β₀ = 0. Software does this silently;
# doing it by hand makes the odds-ratio reading possible:
#
#   exp(β_kj) = how the ODDS of class k vs the reference multiply for
#               each unit of feature j, holding the others constant
#
# Why no floor_area_sqm in the features? Flat type is DEFINED by floor
# area band — an area feature would score ~100% and teach nothing about
# estimation. The honest task: infer the type from MARKET CONTEXT
# (price, storey, lease, distance) — the situation an agent faces when a
# listing omits the floor size.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: The flat-type frame
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  MLFP02 Exercise 6.5: Multinomial Logistic Regression")
print("=" * 70)

frame = load_flat_type_frame()
K = len(FLAT_TYPE_CLASSES)
print(f"\n  Rows: {frame.height:,} (2020+, sentinel prices removed)")
for cls in FLAT_TYPE_CLASSES:
    share = frame.filter(frame["flat_type"] == cls).height / frame.height
    print(f"    {cls}: {share:.1%}")
print(f"  Features (market context, no floor area): {MULTINOMIAL_FEATURES}")
print(f"  Reference class: {FLAT_TYPE_CLASSES[0]} (β fixed at 0)")

X_raw = frame.select(MULTINOMIAL_FEATURES).to_numpy().astype(np.float64)
y_idx = np.array(
    [FLAT_TYPE_CLASSES.index(v) for v in frame["flat_type"].to_list()]
)

# Standardise (same discipline as 6.1) and append the intercept
X_mean = X_raw.mean(axis=0)
X_std = X_raw.std(axis=0)
X = np.column_stack([np.ones(len(y_idx)), (X_raw - X_mean) / X_std])
n_feat = X.shape[1]

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X.shape == (len(y_idx), len(MULTINOMIAL_FEATURES) + 1), "Design shape"
assert set(np.unique(y_idx).tolist()) == {0, 1, 2}, "All three classes present"
assert np.isfinite(X).all(), "Design matrix must be finite"
print("\n✓ Checkpoint 1 passed — multinomial frame built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: Multinomial MLE from scratch + oracle agreement
# ════════════════════════════════════════════════════════════════════════

beta0 = np.zeros(n_feat * (K - 1))
result = minimize(
    neg_log_likelihood_multinomial,
    beta0,
    args=(X, y_idx, K),
    jac=multinomial_gradient,
    method="L-BFGS-B",
    options={"maxiter": 500},
)
assert result.success, f"Multinomial optimisation failed: {result.message}"

B_hat = np.zeros((n_feat, K))
B_hat[:, 1:] = result.x.reshape(n_feat, K - 1)
P_hat = softmax_rows(X @ B_hat)
y_pred = P_hat.argmax(axis=1)
acc_scratch = float((y_pred == y_idx).mean())
base_rate = float(np.bincount(y_idx).max() / len(y_idx))

print("=== Multinomial MLE (from scratch) ===")
print(f"  converged: {result.success}, final NLL: {result.fun:,.1f}")
print(f"  in-sample accuracy: {acc_scratch:.3f}  (base rate: {base_rate:.3f})")

# sklearn oracle, matched to the from-scratch model: same standardised design
# and NO regularisation (sklearn defaults to L2, which would bias the
# comparison). multinomial is the default for lbfgs.
X_std_design = (X_raw - X_mean) / X_std
oracle = LogisticRegression(max_iter=2000, C=1e12, fit_intercept=True)
oracle.fit(X_std_design, y_idx)
acc_oracle = float(accuracy_score(y_idx, oracle.predict(X_std_design)))
print(f"  sklearn oracle accuracy: {acc_oracle:.3f}")

# Agreement on predicted class labels (evaluate on the same design the
# oracle was fitted on — the standardised one, not raw).
P_oracle = oracle.predict_proba(X_std_design)  # columns follow oracle.classes_
oracle_classes = list(oracle.classes_)
P_oracle_aligned = np.column_stack(
    [P_oracle[:, oracle_classes.index(k)] for k in range(K)]
)
pred_oracle = P_oracle_aligned.argmax(axis=1)
label_agreement = float((pred_oracle == y_pred).mean())
print(f"  label agreement with oracle: {label_agreement:.1%}")

# Odds ratios vs the reference class, per feature (standardised scale)
print(f"\n=== Odds ratios vs {FLAT_TYPE_CLASSES[0]} (per SD of feature) ===")
print(f"{'Feature':<18} {'OR 4-room':>10} {'OR 5-room':>10}")
print("-" * 42)
for j, feat in enumerate(["intercept", *MULTINOMIAL_FEATURES]):
    if feat == "intercept":
        continue
    or_4 = float(np.exp(B_hat[j, 1]))
    or_5 = float(np.exp(B_hat[j, 2]))
    print(f"{feat:<18} {or_4:>10.3f} {or_5:>10.3f}")
print(
    "\n  OR > 1: one SD more of the feature multiplies the odds of that\n"
    "  class vs 3-room by that factor. OR < 1: the odds shrink. Price\n"
    "  and distance carry the type signal when area is hidden."
)

# One-vs-rest comparison (the lesson's framing alternative): three
# separate binary logits, then normalise their probabilities
ovr_probs = np.zeros_like(P_hat)
for k in range(K):
    from shared.mlfp02.ex_6 import neg_log_likelihood_logistic, neg_ll_gradient

    y_bin = (y_idx == k).astype(np.float64)
    res_k = minimize(
        neg_log_likelihood_logistic,
        np.zeros(n_feat),
        args=(X, y_bin),
        jac=neg_ll_gradient,
        method="L-BFGS-B",
    )
    z_k = X @ res_k.x
    from shared.mlfp02.ex_6 import sigmoid

    ovr_probs[:, k] = sigmoid(z_k)
ovr_probs = ovr_probs / ovr_probs.sum(axis=1, keepdims=True)
acc_ovr = float((ovr_probs.argmax(axis=1) == y_idx).mean())
print(f"\n  One-vs-rest accuracy: {acc_ovr:.3f}")
print(
    "  One-vs-rest fits K independent binaries and renormalises — simple,\n"
    "  but the binaries are not jointly calibrated. Softmax shares the\n"
    "  denominator across classes, which is why it is the default."
)

# ── Log the fit to ExperimentTracker ─────────────────────────────────
run_id = track_train_run(
    experiment="mlfp02_ex6_05_multinomial",
    run_name="flat_type_context_softmax",
    params={
        "classes": ",".join(FLAT_TYPE_CLASSES),
        "features": ",".join(MULTINOMIAL_FEATURES),
        "reference_class": FLAT_TYPE_CLASSES[0],
        "optimiser": "L-BFGS-B analytical gradient",
    },
    metrics={
        "nll_final": float(result.fun),
        "accuracy_scratch": acc_scratch,
        "accuracy_oracle": acc_oracle,
        "accuracy_ovr": acc_ovr,
        "label_agreement_oracle": label_agreement,
        "base_rate": base_rate,
    },
)
print(f"\nLogged training run to ExperimentTracker (run {run_id})")

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert acc_scratch > base_rate, "Model must beat the majority-class base rate"
assert abs(acc_scratch - acc_oracle) < 0.02, (
    "From-scratch and oracle accuracies must agree within 2 points"
)
assert label_agreement > 0.95, "Label agreement with oracle must exceed 95%"
assert np.allclose(P_hat.sum(axis=1), 1.0), "Softmax rows must sum to 1"
print("\n✓ Checkpoint 2 passed — multinomial fit agrees with the oracle\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: Class-probability profiles across price
# ════════════════════════════════════════════════════════════════════════

# Sweep resale_price across its observed range, other features at medians
price_grid_raw = np.linspace(X_raw[:, 0].min(), np.percentile(X_raw[:, 0], 99), 200)
medians_raw = np.median(X_raw, axis=0)
profiles = np.tile(medians_raw, (len(price_grid_raw), 1))
profiles[:, 0] = price_grid_raw
profiles_std = (profiles - X_mean) / X_std
P_prof = softmax_rows(np.column_stack([np.ones(len(profiles)), profiles_std]) @ B_hat)

fig = go.Figure()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for k, cls in enumerate(FLAT_TYPE_CLASSES):
    fig.add_trace(
        go.Scatter(
            x=price_grid_raw,
            y=P_prof[:, k],
            mode="lines",
            name=f"P({cls})",
            line={"color": colors[k]},
        )
    )
fig.update_layout(
    title=(
        "Class Probabilities vs Resale Price "
        "(storey, lease, distance held at medians)"
    ),
    xaxis_title="Resale price ($)",
    yaxis_title="Predicted probability",
    height=450,
)
fig_path = OUTPUT_DIR / "multinomial_profiles.html"
fig.write_html(str(fig_path))
print(f"Saved: {fig_path}")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert fig_path.exists(), "Figure must be written"
assert np.allclose(P_prof.sum(axis=1), 1.0), "Profile rows must sum to 1"
print("\n✓ Checkpoint 3 passed — visualisation saved\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Inferring Flat Type When the Listing Omits Area
# ════════════════════════════════════════════════════════════════════════
# A property portal (anonymised) ingests listings from partners whose
# feeds sometimes drop the floor-area field. Before the listing goes
# live, the portal wants a type suggestion with calibrated confidence —
# and a rule for when to suppress the suggestion entirely.

print("=== APPLICATION: Listing with missing floor area ===")
new_listing_raw = np.array([[620_000.0, 10.0, 75.0, 12.5]])  # price, storey, lease, km
new_std = (new_listing_raw - X_mean) / X_std
p_new = softmax_rows(np.column_stack([[1.0], new_std]) @ B_hat)[0]
print(
    f"  Listing: $620K, storey ~10, lease 75y, 12.5 km from CBD\n"
    f"  Suggestion: "
    + ", ".join(
        f"P({cls}) = {p_new[k]:.2f}" for k, cls in enumerate(FLAT_TYPE_CLASSES)
    )
)
confidence = float(p_new.max())
threshold = 0.60
print(f"\n  Confidence threshold for display: {threshold:.0%}")
if confidence >= threshold:
    print(
        f"  → Suggest '{FLAT_TYPE_CLASSES[int(p_new.argmax())]}' "
        f"(confidence {confidence:.0%})"
    )
else:
    print(
        f"  → Top class only reaches {confidence:.0%} — SUPPRESS the "
        "suggestion and ask the partner to supply the area."
    )

# Calibration honesty: how often is the model's top class right at
# different confidence levels?
conf_all = P_hat.max(axis=1)
correct_all = (P_hat.argmax(axis=1) == y_idx)
print(f"\n  Empirical reliability on 2020+ data:")
for lo, hi in ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)):
    m = (conf_all >= lo) & (conf_all < hi)
    if m.sum() > 0:
        print(
            f"    confidence {lo:.1f}–{min(hi, 1.0):.1f}: "
            f"accuracy {correct_all[m].mean():.2f} on {m.sum():,} rows"
        )

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert 0 <= confidence <= 1, "Confidence must be a probability"
assert abs(float(p_new.sum()) - 1.0) < 1e-9, "Class probabilities must sum to 1"
print("\n✓ Checkpoint 4 passed — application complete\n")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("═" * 70)
print("  WHAT YOU'VE MASTERED (6.5 — Multinomial Logistic Regression)")
print("═" * 70)
print(
    """
  ✓ Softmax generalises the sigmoid: K-1 free logits against a pinned
    reference class — identification is not optional
  ✓ From-scratch multinomial MLE (NLL + analytical gradient, L-BFGS)
    matches the sklearn oracle within 2 accuracy points and >95% label
    agreement
  ✓ Odds ratios vs the reference class read exactly as in the binary
    case: exp(β) multiplies the odds per unit (here, per SD)
  ✓ One-vs-rest is the simpler framing but its probabilities are not
    jointly calibrated — softmax shares the denominator
  ✓ Withholding floor area turned a tautology into a real inference
    task — and the confidence-threshold rule (suppress below 60%)
    connects probability quality to product behaviour

  NEXT: Exercise 7 returns to experiments — CUPED variance reduction,
  Bayesian A/B decisions, sequential testing, and DiD.
"""
)

print("\n✓ Exercise 6.5 complete — Multinomial Logistic Regression")
