# MLFP04 — Task 1: Customer Segments and Mixture Models

**Weight**: 20 marks · **Outcomes**: 4.1 (clustering, choosing K, scaling decisions, interpreting segments), 4.2 (EM for Gaussian mixtures, soft assignments, local optima)
**Data**: loyalty-programme customer tables in the format below. `dev_customers.parquet` (shipped with the task) is one such table for development; the grader uses its own.

## Scenario

**Segmentation.** A Singapore retailer's loyalty team exports one row per
customer from its CRM and asks you to find the customer personas in it. Nobody
knows how many personas there are; it differs between the cohorts you will be
given. The CRM export looks like this:

| Column                | Meaning                                       |
| --------------------- | --------------------------------------------- |
| `customer_id`         | CRM identifier                                |
| `signup_channel`      | where the customer joined (app/web/store/...) |
| `recency_days`        | days since the last order                     |
| `orders_12m`          | orders in the last 12 months                  |
| `spend_12m_sgd`       | spend in the last 12 months, S$               |
| `tenure_months`       | months since joining                          |
| `avg_basket_sgd`      | average order value, S$                       |
| `pct_discount_orders` | share of orders that used a discount          |
| `app_sessions_30d`    | app sessions in the last 30 days              |

The columns are in very different units, and money and order counts are
heavily skewed. A small number of corporate bulk-buying accounts are mixed in
with ordinary shoppers. The team plans retention campaigns per persona, so
the first thing they want is to know which persona is drifting away (has gone
longest without ordering).

**Mixture model.** The data science lead wants a soft-assignment model they
can inspect line by line, so the Gaussian mixture must be fitted by your own
expectation-maximisation code rather than a library fit.

## What to submit

`starter.py` with two functions (signatures fixed):

| Function                       | Returns                                                                                                                                                                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `segment_customers(customers)` | dict: `labels` (one int segment id per row, in row order), `profiles` (polars DataFrame, one row per segment: `segment`, `n_customers`, and the **median** of each of the seven behaviour columns in original units), `at_risk_segment` (the id of the persona that has gone longest without ordering) |
| `fit_mixture(X, k, seed)`      | dict for a `k`-component full-covariance Gaussian mixture fitted to the numpy array `X` (n × d): `weights` (k,), `means` (k, d), `covariances` (k, d, d), `responsibilities` (n, k), `log_likelihood` (total over all rows, natural log)                                                               |

## Acceptance criteria

- The grader simulates three customer cohorts you have not seen, each with a
  secret number of planted personas (3 to 6) and different sizes.
- Your segments must recover the planted personas: adjusted Rand index of at
  least 0.90 on ordinary customers in **every** cohort, and the number of
  segments holding at least 3% of customers must equal the number of planted
  personas. A tiny extra segment of unusual accounts is acceptable.
- `profiles` must agree exactly with the medians and counts of your own
  labels; `at_risk_segment` must be the planted persona with the longest time
  since last order.
- The grader simulates three overlapping Gaussian mixtures (2 to 4
  dimensions, 2 to 4 components). Your returned parameters must be valid
  (weights positive and summing to 1, covariances symmetric
  positive-definite); your responsibilities and log-likelihood must be the
  ones your parameters imply; your log-likelihood per row must be within 0.01
  of the best of ten well-initialised maximum-likelihood fits; and one more EM
  iteration from your parameters must not improve the log-likelihood per row
  by more than 1e-4.
- Structural checks (format, consistency, framework use) earn marks only when
  the outcome checks in the same part pass.

## Rules

- Segmentation runs through kailash-ml `ClusteringEngine`.
- `fit_mixture` is your own EM in numpy / scipy: `sklearn.mixture` and the
  engine's mixture algorithm are not allowed there.
- Polars for data handling (no pandas). Your functions must work from their
  arguments alone: the grader never passes the same data twice.
