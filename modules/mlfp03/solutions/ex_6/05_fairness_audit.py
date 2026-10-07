# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP03 — Exercise 6.5: Fairness Audit (Disparate Impact, Equalized Odds,
#                         Calibration Parity, Impossibility Theorem)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Measure the disparate impact ratio (the "four-fifths rule")
#   - Measure equalized odds (TPR and FPR parity across groups)
#   - Measure calibration parity (are probabilities equally reliable?)
#   - Work the Chouldechova/Kleinberg impossibility theorem with THIS
#     model's real base rates
#   - Produce a fairness audit report for race, gender and age
#
# PREREQUISITES:
#   - 01_shap_global.py (same model, same SHAP bundle)
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why fairness has MULTIPLE incompatible definitions
#   2. Build — decode protected groups + per-group rate machinery
#   3. Train — no training; AUDIT the trained credit model
#   4. Visualise — per-group tables, charts, impossibility worked example
#   5. Apply — fairness disclosure for an (illustrative) retail bank
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv

from kailash_ml import diagnose

from shared.mlfp03.ex_6 import (
    OUTPUT_DIR,
    PROTECTED_ATTRIBUTES,
    build_shap_explainer,
    decode_group,
    feature_index,
    print_section,
    rank_features_by_mean_abs_shap,
)

load_dotenv()


# ════════════════════════════════════════════════════════════════════════
# THEORY — Why Fairness Has MULTIPLE Incompatible Definitions
# ════════════════════════════════════════════════════════════════════════
# There is no single mathematical definition of "fair". Three families:
#
#   1. INDEPENDENCE (demographic parity):
#        P(decline | G = a) == P(decline | G = b)
#      Equal decision rates regardless of group.
#
#   2. SEPARATION (equalized odds):
#        TPR_a == TPR_b   AND   FPR_a == FPR_b
#      Equal error rates across groups.
#
#   3. SUFFICIENCY (calibration parity):
#        P(Y = 1 | score = p, G = a) == P(Y = 1 | score = p, G = b)
#      A score means the same default risk whatever your group.
#
# IMPOSSIBILITY (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan
# 2016): when the groups' BASE RATES differ, these criteria are pairwise
# incompatible — no two of them can hold together, except for a perfect
# classifier or one that ignores the data. It is not "pick any two"; you
# choose which ONE to prioritise and document the trade-off.
#
# DISPARATE IMPACT RATIO = (approval rate of a group) / (approval rate of
# the most-approved group). The "four-fifths rule" (ratio < 0.8 = adverse
# impact) comes from the US EEOC Uniform Guidelines on Employee Selection
# Procedures (1978) — an employment-law screening heuristic that is widely
# borrowed as a rule of thumb in credit fairness audits.
#
# WHICH ATTRIBUTES: the dataset has race, gender and age. The model was
# trained WITH them as features (a choice many lenders avoid); the audit
# below measures outcomes per group either way — removing a column does not
# remove its proxies.


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD group-level audit machinery
# ════════════════════════════════════════════════════════════════════════

bundle = build_shap_explainer()
model = bundle["model"]
X_test = bundle["X_test"]
y_test = bundle["y_test"]
y_pred = bundle["y_pred"]  # 1 = predicted default = DECLINE
y_proba = bundle["y_proba"]
feature_names: list[str] = bundle["feature_names"]
shap_vals = bundle["shap_vals"]
ordinal_mappings = bundle["ordinal_mappings"]


def group_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, groups: np.ndarray
) -> dict[str, dict[str, float]]:
    """Per-group rates for the three fairness families."""
    report: dict[str, dict[str, float]] = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        pos, neg = m & (y_true == 1), m & (y_true == 0)
        declined = m & (y_pred == 1)
        report[g] = {
            "n": int(m.sum()),
            "base_rate": float(y_true[m].mean()),
            "approval_rate": float(1.0 - y_pred[m].mean()),
            "tpr": float(y_pred[pos].mean()) if pos.any() else float("nan"),
            "fpr": float(y_pred[neg].mean()) if neg.any() else float("nan"),
            "ppv": float(y_true[declined].mean()) if declined.any() else float("nan"),
            "mean_p": float(y_proba[m].mean()),
            # observed / predicted: 1.0 = calibrated "in the large" for this group
            "obs_over_pred": float(y_true[m].mean() / y_proba[m].mean()),
        }
    return report


def disparate_impact(report: dict[str, dict[str, float]]) -> dict[str, float]:
    """Approval-rate ratio of each group vs the most-approved group."""
    best = max(r["approval_rate"] for r in report.values())
    return {g: r["approval_rate"] / best for g, r in report.items()}


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — "TRAIN" = run the audit against the trained model
# ════════════════════════════════════════════════════════════════════════

reports: dict[str, dict[str, dict[str, float]]] = {}
di_by_attr: dict[str, dict[str, float]] = {}
group_labels: dict[str, np.ndarray] = {}
for attr in PROTECTED_ATTRIBUTES:
    groups = decode_group(X_test, feature_names, attr, ordinal_mappings)
    group_labels[attr] = groups
    reports[attr] = group_report(y_test, y_pred, y_proba, groups)
    di_by_attr[attr] = disparate_impact(reports[attr])

print_section("Fairness Audit — per-group rates (decline if P(default) >= 0.5)")
for attr in PROTECTED_ATTRIBUTES:
    print(f"\n  --- {attr} ---")
    print(
        f"  {'group':<10} {'n':>6} {'base':>6} {'approve':>8} {'DI':>6} "
        f"{'TPR':>6} {'FPR':>6} {'PPV':>6} {'obs/pred':>9}"
    )
    for g, r in reports[attr].items():
        di = di_by_attr[attr][g]
        flag = " <0.8" if di < 0.8 else ""
        print(
            f"  {g:<10} {r['n']:>6} {r['base_rate']:>6.3f} {r['approval_rate']:>8.3f} "
            f"{di:>6.3f} {r['tpr']:>6.3f} {r['fpr']:>6.3f} {r['ppv']:>6.3f} "
            f"{r['obs_over_pred']:>9.3f}{flag}"
        )

# Summary gaps per attribute (max - min across groups)
summary: dict[str, dict[str, float]] = {}
for attr, rep in reports.items():
    vals = list(rep.values())
    summary[attr] = {
        "min_di": min(di_by_attr[attr].values()),
        "tpr_gap": max(v["tpr"] for v in vals) - min(v["tpr"] for v in vals),
        "fpr_gap": max(v["fpr"] for v in vals) - min(v["fpr"] for v in vals),
        "calib_gap": max(v["obs_over_pred"] for v in vals) - min(v["obs_over_pred"] for v in vals),
        "base_rate_gap": max(v["base_rate"] for v in vals) - min(v["base_rate"] for v in vals),
    }

# ── Checkpoint ──────────────────────────────────────────────────────────
assert set(reports) == set(PROTECTED_ATTRIBUTES), "Task 3: every protected attribute audited"
assert all(len(rep) >= 2 for rep in reports.values()), "Task 3: each attribute needs >= 2 groups"
assert all(
    sum(r["n"] for r in rep.values()) == len(y_test) for rep in reports.values()
), "Task 3: groups must partition the test set"
print("\n[ok] Checkpoint — per-group audit computed for race, gender and age\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE: charts + the impossibility theorem on THIS model
# ════════════════════════════════════════════════════════════════════════

for attr in PROTECTED_ATTRIBUTES:
    rep = reports[attr]
    names = list(rep)
    fig = go.Figure()
    for metric, colour in [("approval_rate", "#10b981"), ("tpr", "#6366f1"), ("fpr", "#f43f5e")]:
        fig.add_trace(go.Bar(x=names, y=[rep[g][metric] for g in names], name=metric, marker_color=colour))
    fig.update_layout(
        title=f"Fairness by {attr}: approval rate, TPR, FPR",
        barmode="group",
        yaxis_title="Rate",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    path = OUTPUT_DIR / f"ex6_05_fairness_{attr}.html"
    fig.write_html(str(path))
    print(f"  Saved: {path}")

# Per-group reliability (calibration parity) for the attribute with the
# largest base-rate gap
focus_attr = max(summary, key=lambda a: summary[a]["base_rate_gap"])
fig_cal = go.Figure()
fig_cal.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(dash="dash", color="#9ca3af")))
edges = np.linspace(0, 1, 11)
for g in reports[focus_attr]:
    m = group_labels[focus_attr] == g
    bin_idx = np.clip(np.digitize(y_proba[m], edges) - 1, 0, 9)
    xs, ys = [], []
    for b in range(10):
        sel = bin_idx == b
        if sel.sum() >= 30:
            xs.append(float(y_proba[m][sel].mean()))
            ys.append(float(y_test[m][sel].mean()))
    fig_cal.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=g))
fig_cal.update_layout(
    title=f"Calibration parity by {focus_attr}: predicted vs observed default rate",
    xaxis_title="Mean predicted P(default) in bin",
    yaxis_title="Observed default rate",
    height=450,
)
cal_path = OUTPUT_DIR / f"ex6_05_calibration_parity_{focus_attr}.html"
fig_cal.write_html(str(cal_path))
print(f"  Saved: {cal_path}")

# ── Impossibility theorem, worked on this model ─────────────────────────
# Chouldechova's identity links the three error rates in ANY group:
#
#     FPR = p / (1 - p) * (1 - PPV) / PPV * TPR        (p = base rate)
#
# Take the two groups of `focus_attr` with the most different base rates.
# Suppose the low-base-rate group had the SAME PPV (sufficiency-style
# parity) and the SAME TPR as the high-base-rate group. The identity then
# FIXES its FPR — and that FPR cannot equal the high group's FPR, so
# equalized odds must fail. Different base rates leave no way out.


def chouldechova_fpr(base_rate: float, ppv: float, tpr: float) -> float:
    return base_rate / (1.0 - base_rate) * (1.0 - ppv) / ppv * tpr


rep = reports[focus_attr]
hi_g = max(rep, key=lambda g: rep[g]["base_rate"])
lo_g = min(rep, key=lambda g: rep[g]["base_rate"])
hi, lo = rep[hi_g], rep[lo_g]
fpr_hi_check = chouldechova_fpr(hi["base_rate"], hi["ppv"], hi["tpr"])
fpr_lo_forced = chouldechova_fpr(lo["base_rate"], hi["ppv"], hi["tpr"])

print_section(f"Impossibility theorem — {focus_attr}: {hi_g} vs {lo_g}", char="─")
print(f"  Base rates: {hi_g} = {hi['base_rate']:.3f}, {lo_g} = {lo['base_rate']:.3f}")
print(f"  {hi_g}: TPR={hi['tpr']:.3f}  PPV={hi['ppv']:.3f}  FPR={hi['fpr']:.3f}")
print(f"  Identity check for {hi_g}: FPR from formula = {fpr_hi_check:.3f}")
print(
    f"  If {lo_g} matched {hi_g}'s PPV and TPR, its FPR would be forced to "
    f"{fpr_lo_forced:.3f} — not {hi['fpr']:.3f}."
)
print("  -> With unequal base rates, equal PPV + equal TPR rules out equal FPR.")

# ── SHAP contribution of protected attributes ──────────────────────────
importance_ranking = rank_features_by_mean_abs_shap(shap_vals, feature_names)
ranked_names = [n for n, _ in importance_ranking]
print_section("SHAP contribution of protected attributes", char="─")
for attr in PROTECTED_ATTRIBUTES:
    attr_shap = shap_vals[:, feature_index(feature_names, attr)]
    rank = ranked_names.index(attr) + 1
    action = "INVESTIGATE — in top 10" if rank <= 10 else "outside top 10"
    print(f"  {attr:<8} mean|SHAP|={np.abs(attr_shap).mean():.4f}  rank #{rank}/{len(feature_names)}  ({action})")

# ── Checkpoint ──────────────────────────────────────────────────────────
assert abs(fpr_hi_check - hi["fpr"]) < 1e-9, "Chouldechova identity must hold exactly"
assert hi["base_rate"] > lo["base_rate"], "Worked example needs unequal base rates"
print("\n[ok] Checkpoint — impossibility identity verified on the model's own rates\n")

# INTERPRETATION: read the per-group table. Disparate impact flags groups
# whose approval rate is < 80% of the most-approved group. Large TPR/FPR
# gaps mean the model's errors land unevenly. obs/pred far from 1 means
# the probabilities are off for that group; compare the RATIO across
# groups — this model was trained with scale_pos_weight, so it
# over-predicts for everyone (see 5.5) and parity is about consistency.


# ════════════════════════════════════════════════════════════════════════
# FAIRNESS AUDIT REPORT — every line computed above
# ════════════════════════════════════════════════════════════════════════
print_section("FAIRNESS AUDIT REPORT — Credit Default Model")
for attr in PROTECTED_ATTRIBUTES:
    s = summary[attr]
    worst = min(di_by_attr[attr], key=di_by_attr[attr].get)
    status = "FAILS four-fifths screen" if s["min_di"] < 0.8 else "passes four-fifths screen"
    print(
        f"  {attr:<7} min DI={s['min_di']:.3f} ({worst}; {status}) | "
        f"TPR gap={s['tpr_gap']:.3f} FPR gap={s['fpr_gap']:.3f} | "
        f"obs/pred gap={s['calib_gap']:.3f} | base-rate gap={s['base_rate_gap']:.3f}"
    )
failing = [a for a in PROTECTED_ATTRIBUTES if summary[a]["min_di"] < 0.8]
print(
    f"\n  Attributes needing escalation: {', '.join(failing) if failing else 'none'}"
    f"\n  Worked impossibility example used: {focus_attr} ({hi_g} vs {lo_g})"
)


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Fairness disclosure for an (illustrative) retail bank
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): a Singapore retail bank's model-risk committee
# asks for a fairness section in every credit model's documentation. The
# MAS FEAT principles (2018, non-binding) ask firms to assess AI-driven
# decisions for fairness and to justify the choices made; they do not
# prescribe a metric or a numeric threshold. A defensible disclosure:
#
#   1. Disparate impact per protected attribute (the table above)
#   2. TPR/FPR parity (equalized odds) per attribute
#   3. Calibration parity per attribute
#   4. WHICH criterion the bank prioritises, and why — the impossibility
#      example shows it cannot have all three when base rates differ
#   5. An escalation path when a screen fails
#
# What THIS audit shows: read the report lines above. Where an attribute
# fails the four-fifths screen together with a large base-rate gap (as age
# does if its line says so), the gap partly reflects genuinely different
# default rates — and the committee must decide whether the age effect
# is a legitimate risk factor or a proxy it should not use. That is a
# human decision; the pipeline's job is to make it visible.
#
# LIMITATION: a fairness audit identifies SYMPTOMS, not causes. Fixes —
# removing the attribute and its proxies, re-weighting, group-specific
# thresholds, a different model — each trade one criterion for another.


# ════════════════════════════════════════════════════════════════════════
# DESTINATION-FIRST CLOSE — km.diagnose
# ════════════════════════════════════════════════════════════════════════
# This lesson built the fairness audit from primitives. kailash-ml packages
# the standard (non-group) diagnostic surface — per-class metrics,
# severity heuristics, confusion matrix — into one call; the group
# metrics above sit on top of it.

report = diagnose(model, kind="classical_classifier", data=(X_test, y_test), show=False)
print()
print("  km.diagnose model    : audited credit-default classifier")
print(f"  km.diagnose metrics  : {report.metrics}")
print(f"  km.diagnose severity : {report.severity}")


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print_section("WHAT YOU'VE MASTERED")
print(
    """
  [x] Decoded race, gender and age bands from the encoded feature matrix
  [x] Measured disparate impact with the four-fifths screen
  [x] Measured TPR/FPR parity (equalized odds) and calibration parity
  [x] Verified Chouldechova's identity and used it to show, on this
      model's own base rates, why the criteria cannot all hold
  [x] Audited the SHAP contribution of protected attributes
  [x] Produced a computed fairness audit report

  KEY INSIGHT: Fairness is not a single number. It is a FAMILY of
  definitions that are mutually incompatible whenever base rates differ.
  The ML engineer's job is to measure all of them, show the trade-off,
  and make the choice legible to the people accountable for it.

  END OF EXERCISE 6. Next: Exercise 7 scales from a single model to a
  full Kailash Workflow with feature engineering, training, evaluation,
  and persistence.
"""
)
