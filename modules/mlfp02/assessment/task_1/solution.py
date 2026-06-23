# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP02 — Assessment Task 1: Probability, Bayes & Experiment Validation
(Reference Solution)

Reference implementation. Withheld from students. Verified to pass grader.py.
"""
from __future__ import annotations

import polars as pl
from scipy import stats

from shared import MLFPDataLoader

# --- Fixed problem constants (see problem.md) ---
COHORT = ["control", "treatment_a"]
CONVERT_THRESHOLD = 50.0          # converted := metric_value >= 50.0
FRAUD_BASE_RATE = 0.02            # P(fraud)
FRAUD_SENSITIVITY = 0.95          # P(flagged | fraud)
FRAUD_FPR = 0.03                  # P(flagged | not fraud)
BETA_PRIOR_ALPHA = 2.0
BETA_PRIOR_BETA = 20.0


def solve() -> dict:
    """Probability, Bayes, and A/B-experiment validity checks.

    Returns a dict with these exact keys (full precision floats):
        p_convert_overall        P(converted) over the control+treatment_a cohort
        p_convert_control        P(converted | control)
        p_convert_treatment      P(converted | treatment_a)
        p_treatment_given_convert  Bayes inversion P(treatment_a | converted)
        srm_chi2                 chi-square statistic, 50/50 split goodness-of-fit
        srm_p_value              chi-square p-value (df=1)
        srm_flag                 bool, True if srm_p_value < 1e-3
        p_fraud_given_flagged    base-rate-fallacy Bayes scalar
        beta_post_alpha          posterior alpha = prior + successes (treatment_a)
        beta_post_beta           posterior beta  = prior + failures  (treatment_a)
        posterior_mean           alpha/(alpha+beta)
        cred_int_low             2.5%  Beta posterior quantile
        cred_int_high            97.5% Beta posterior quantile
    """
    loader = MLFPDataLoader()
    df = loader.load("mlfp02", "experiment_data.parquet")

    # Restrict to the primary A/B comparison and label the conversion event.
    co = df.filter(pl.col("experiment_group").is_in(COHORT)).with_columns(
        (pl.col("metric_value") >= CONVERT_THRESHOLD).alias("converted")
    )

    n_total = co.height
    n_control = co.filter(pl.col("experiment_group") == "control").height
    n_treatment = co.filter(pl.col("experiment_group") == "treatment_a").height

    # --- 1a: priors, conditionals, and the Bayes inversion ---
    p_convert_overall = co["converted"].mean()
    p_convert_control = co.filter(pl.col("experiment_group") == "control")[
        "converted"
    ].mean()
    p_convert_treatment = co.filter(pl.col("experiment_group") == "treatment_a")[
        "converted"
    ].mean()

    # P(treatment | converted) = P(converted | treatment) * P(treatment) / P(converted)
    p_treatment = n_treatment / n_total
    p_treatment_given_convert = (
        p_convert_treatment * p_treatment
    ) / p_convert_overall

    # --- 1b: Sample Ratio Mismatch (designed 50/50) ---
    expected = n_total / 2.0
    srm_chi2 = ((n_control - expected) ** 2 / expected) + (
        (n_treatment - expected) ** 2 / expected
    )
    srm_p_value = float(stats.chi2.sf(srm_chi2, df=1))
    srm_flag = bool(srm_p_value < 1e-3)

    # --- 1c: base-rate fallacy (fraud detector), pure scalar Bayes ---
    # P(fraud | flagged) = P(flagged|fraud)P(fraud) /
    #                      [P(flagged|fraud)P(fraud) + P(flagged|¬fraud)P(¬fraud)]
    p_flagged = FRAUD_SENSITIVITY * FRAUD_BASE_RATE + FRAUD_FPR * (
        1 - FRAUD_BASE_RATE
    )
    p_fraud_given_flagged = (FRAUD_SENSITIVITY * FRAUD_BASE_RATE) / p_flagged

    # --- 1d: Beta-Binomial conjugate update on the treatment_a arm ---
    treatment = co.filter(pl.col("experiment_group") == "treatment_a")
    successes = int(treatment["converted"].sum())
    failures = int(treatment.height - successes)
    beta_post_alpha = BETA_PRIOR_ALPHA + successes
    beta_post_beta = BETA_PRIOR_BETA + failures
    posterior_mean = beta_post_alpha / (beta_post_alpha + beta_post_beta)
    cred_int_low = float(stats.beta.ppf(0.025, beta_post_alpha, beta_post_beta))
    cred_int_high = float(stats.beta.ppf(0.975, beta_post_alpha, beta_post_beta))

    return {
        "p_convert_overall": float(p_convert_overall),
        "p_convert_control": float(p_convert_control),
        "p_convert_treatment": float(p_convert_treatment),
        "p_treatment_given_convert": float(p_treatment_given_convert),
        "srm_chi2": float(srm_chi2),
        "srm_p_value": srm_p_value,
        "srm_flag": srm_flag,
        "p_fraud_given_flagged": float(p_fraud_given_flagged),
        "beta_post_alpha": float(beta_post_alpha),
        "beta_post_beta": float(beta_post_beta),
        "posterior_mean": float(posterior_mean),
        "cred_int_low": cred_int_low,
        "cred_int_high": cred_int_high,
    }


if __name__ == "__main__":
    r = solve()
    print(f"P(converted)               = {r['p_convert_overall']:.4f}")
    print(f"P(converted | control)     = {r['p_convert_control']:.4f}")
    print(f"P(converted | treatment_a) = {r['p_convert_treatment']:.4f}")
    print(f"P(treatment_a | converted) = {r['p_treatment_given_convert']:.4f}")
    print(f"SRM chi2 = {r['srm_chi2']:.2f}, p = {r['srm_p_value']:.3e}, flag = {r['srm_flag']}")
    print(f"P(fraud | flagged)         = {r['p_fraud_given_flagged']:.4f}")
    print(
        f"Posterior Beta({r['beta_post_alpha']:.0f}, {r['beta_post_beta']:.0f}), "
        f"mean = {r['posterior_mean']:.4f}, "
        f"95% CrI = [{r['cred_int_low']:.4f}, {r['cred_int_high']:.4f}]"
    )
