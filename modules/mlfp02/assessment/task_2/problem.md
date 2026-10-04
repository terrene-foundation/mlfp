# MLFP02 — Task 2: Experiment Read-out — Allocation, Power, Inference, CUPED

**Weight**: 25 marks · **Outcomes**: 2.3 (bootstrap, hypothesis and permutation tests, multiple testing), 2.4 (power analysis, SRM, ExperimentTracker), 2.7 (CUPED)
**Data**: `mlfp02/experiment_data.parquet` (same log as Task 1).

## Scenario

The four-arm experiment was **designed** to send 40% of users to `control`,
35% to `treatment_a`, 15% to `treatment_b` and 10% to `variant_c`. Product
wants a decision on `treatment_a` versus `control`. Your read-out code will be
re-run by the analytics platform on future experiments, so it must work on any
log with this schema — including logs where something has gone wrong.

Business definitions and policy:

- A user **converts** when their order value (`metric_value`) is at least $50.
- `pre_metric_value` was measured **before** assignment; every other numeric
  column was measured during the experiment.
- The allocation alarm fires when the allocation test gives p < 0.01.
- Hypothesis tests are two-sided at the 5% level; family-wise and
  false-discovery corrections both use 5%.
- **Ship rule**: ship only if the allocation of the two arms being compared is
  trustworthy _and_ the treatment's conversion rate is significantly higher
  than control's.

## What to submit

`starter.py` with these functions (signatures fixed):

| Function                                           | Returns                                                                                                                                                                                                                                  |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `srm_check(orders, design)`                        | dict: `chi2`, `p_value`, `srm` (bool alarm), `worst_arm` (the arm whose count deviates most from its design, relative to that arm's expected count's sampling noise). `design` maps every arm to its designed share.                     |
| `sample_size_per_arm(baseline, mde, alpha, power)` | int: users per arm needed to detect an absolute conversion lift of `mde` over `baseline` with a two-sided test.                                                                                                                          |
| `analyse_ab(orders, treatment, design)`            | dict (below) for `control` versus `treatment`                                                                                                                                                                                            |
| `segment_tests(orders, treatment)`                 | dict: `p_values` (keyed `"<segment>                                                                                                                                                                                                      | <platform>"`, one conversion-lift test per subgroup), `bonferroni_significant`, `bh_significant` (sorted lists of the keys each correction declares significant) |
| `log_to_tracker(results, store_url)`               | str: the run id of a **finished** Kailash `ExperimentTracker` run in the store at `store_url` that holds every numeric value of an `analyse_ab` result as a metric (same key names) and the decision as a run parameter named `decision` |

`analyse_ab` keys:

| Key                                                                                 | Meaning                                                                                                                                                                            |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pair_srm_p`                                                                        | allocation test of the two compared arms against their designed relative shares                                                                                                    |
| `conv_control`, `conv_treatment`, `conv_lift`                                       | conversion rates and their difference (treatment − control)                                                                                                                        |
| `conv_ci_low`, `conv_ci_high`, `conv_p`                                             | large-sample 95% CI and two-sided p-value for the lift                                                                                                                             |
| `mean_diff`, `boot_ci_low`, `boot_ci_high`                                          | difference in mean order value and its 95% bootstrap percentile CI                                                                                                                 |
| `perm_p`                                                                            | two-sided permutation-test p-value for `mean_diff`                                                                                                                                 |
| `cuped_theta`, `cuped_var_reduction`, `cuped_diff`, `cuped_ci_low`, `cuped_ci_high` | CUPED adjustment of order value: the coefficient (one value estimated on both arms together), the fraction of order-value variance removed, the adjusted difference and its 95% CI |
| `decision`                                                                          | `"SHIP"` or `"DO NOT SHIP"` under the ship rule                                                                                                                                    |

## Acceptance criteria

- The grader runs your functions on experiments **you have not seen**: random
  subsets of this log, allocation tables with a deliberately faulty arm, a
  log damaged by a tracking bug, and an experiment with no true effect. It
  recomputes every answer itself.
- Exact quantities (counts, rates, χ², CUPED θ to 2%) must match. Resampling
  results are accepted within Monte-Carlo tolerance — use at least 2,000
  resamples. Sample sizes must be within 3% of the normal-approximation
  answer.
- Each correction's significant set must be exactly what that procedure
  implies for your p-values.
- The decision must be right on every experiment, including the damaged one.
- The tracker run is read back from the store; it must contain the grader's
  reference values, not just any numbers.

## Rules

- Polars for data handling (no pandas); numpy / scipy for statistics;
  `kailash_ml.ExperimentTracker` for the record (it is async — see the
  module 2 materials).
- Allow ~1 minute for a full grading run (the tracker store start-up is slow).
