# MLFP02 — Task 4: Difference-in-Differences Policy Evaluation

**Weight**: 15 marks · **Outcomes**: 2.7 (DiD and the ATT, parallel-trends test, placebo test, judging when a causal claim is credible)
**Data**: policy panels in the format below. `starter.py` generates an
illustrative development panel; the grader uses its own.

## Scenario

A housing agency introduced a (hypothetical) property cooling measure that
applied to one region only. Randomisation was impossible, so the agency wants
a difference-in-differences estimate of the measure's effect on transaction
prices in the treated region. It has been burnt before by analyses that
reported an effect when the two regions were already drifting apart, so every
estimate must come with the diagnostics that say whether it can be believed.

Each panel is a polars DataFrame, one row per transaction:

| Column    | Meaning                                                              |
| --------- | -------------------------------------------------------------------- |
| `period`  | integer time period (0, 1, 2, …)                                     |
| `treated` | 1 if the transaction is in the region the measure applies to, else 0 |
| `post`    | 1 if the period is on or after the measure took effect, else 0       |
| `y`       | transaction price                                                    |

Panels differ in length, policy timing, effect size and the number of
transactions per period (cells are unbalanced). Pre/post averages are taken
over **transactions**, not over periods.

## What to submit

`did_analysis(panel) -> dict` in `starter.py`, returning:

| Key                                   | Meaning                                                                                                                                                 |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `att`                                 | DiD estimate of the average treatment effect on the treated                                                                                             |
| `se`, `ci_low`, `ci_high`             | its standard error and a 95% normal-approximation CI                                                                                                    |
| `pre_trend_slope_diff`, `pre_trend_p` | difference in linear pre-period time trends (treated minus control) and the two-sided p-value for "no difference"                                       |
| `placebo_att`, `placebo_p`            | DiD estimate and two-sided p-value for a fake measure that starts at the middle pre-period (`sorted pre periods[len // 2]`), using pre-period data only |
| `credible`                            | bool — the estimate can be trusted only if neither diagnostic rejects at the 5% level                                                                   |

## Acceptance criteria

- The grader simulates panels you have not seen, with secret true effects;
  some have parallel trends and some have a planted pre-existing trend gap.
- `att`, `pre_trend_slope_diff` and `placebo_att` must match exactly; the
  standard error must be within 15% of a heteroskedasticity-robust reference
  and the CI must contain the true effect on the valid panels.
- p-values are accepted within ±0.05; `credible` must be right on every panel.

## Rules

- Polars for data handling (no pandas); numpy / scipy for the statistics.
