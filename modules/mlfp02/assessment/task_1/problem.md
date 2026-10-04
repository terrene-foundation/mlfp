# MLFP02 — Task 1: Bayesian Updating & Likelihood Estimation

**Weight**: 20 marks · **Outcomes**: 2.1 (Bayes, conjugate priors, choosing a distribution), 2.2 (MLE, MAP)
**Data**: `mlfp02/experiment_data.parquet` — the four-arm e-commerce experiment
(`user_id, experiment_group, metric_value, pre_metric_value, revenue, timestamp, segment, platform, country`).
`metric_value` is the user's order value in dollars during the experiment.

## Scenario

The growth team wants two things before the formal A/B read-out in Task 2.

1. **A belief about each arm's conversion rate** that they can update as data
   arrives. A user _converts_ when their order value is at least **$50**.
   The team's prior for any arm's conversion rate is a Beta distribution
   whose parameters they will hand you. They also want the probability that
   an arm's true conversion rate is higher than control's.
2. **A model of order value** for the finance forecast. Finance is choosing
   between a Gamma law (shape, scale) and a LogNormal law (μ, σ of the log
   order value) and wants the one the data supports better, judged by AIC.
   For new product lines with only a handful of orders, finance also wants a
   **MAP** Gamma fit that combines the few orders with last year's beliefs:
   - the prior density of the Gamma **shape** is the LogNormal density whose
     log has mean `ln 2` and standard deviation `0.5`;
   - the prior density of the Gamma **scale** is the LogNormal density whose
     log has mean `ln 20` and standard deviation `1.0`;
   - the two priors are independent; the MAP estimate maximises the
     posterior density over (shape, scale).

## What to submit

`starter.py` with these four functions implemented (signatures fixed):

| Function                                       | Returns                                                                                                                                                                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `conversion_posterior(orders, arm, prior)`     | dict: `alpha`, `beta` (posterior Beta parameters), `mean`, `ci_low`, `ci_high` (95% equal-tailed credible interval)                                                                                                                               |
| `prob_beats_control(orders, treatment, prior)` | float: posterior probability that `treatment`'s conversion rate exceeds `control`'s (both arms start from the same `prior`, independently)                                                                                                        |
| `fit_order_value(values)`                      | dict: `gamma_shape`, `gamma_scale`, `gamma_loglik`, `lognorm_mu`, `lognorm_sigma`, `lognorm_loglik`, `best_by_aic` (`"gamma"` or `"lognormal"`) — maximum-likelihood fits with no location shift; each `*_loglik` is the maximised log-likelihood |
| `map_gamma(values)`                            | dict: `shape`, `scale`, `log_posterior` (log-likelihood + log prior densities at your estimate, dropping no terms)                                                                                                                                |

`orders` is a polars DataFrame with the experiment schema (possibly a subset of
rows); `prior` is a `(a, b)` tuple; `values` is a 1-D numpy array of strictly
positive order values.

## Acceptance criteria

- The grader calls your functions on **data you have not seen** (fresh random
  subsets of the experiment, other priors, and order-value samples it
  generates itself). Results must be computed from the arguments — never from
  numbers you observed while developing.
- Posterior parameters and the credible interval must be exact (closed form).
- `prob_beats_control` must be within **±0.01** of the exact value, including
  on small subsets where the answer is far from 0 or 1.
- An MLE or MAP answer is accepted when the log-likelihood (log-posterior) at
  your parameters is within **0.01** of the true maximum, and the value you
  report equals the value at your parameters.
- Your AIC verdict must match the data on every sample the grader uses.

## Rules

- Polars for data handling (no pandas); numpy / scipy allowed for the
  mathematics. You may use any optimiser, but you are graded on whether you
  reached the optimum of the right objective.
- Develop against the real data via `shared.MLFPDataLoader` (see `starter.py`).
- Run `python starter.py` to try your functions on the real data.
