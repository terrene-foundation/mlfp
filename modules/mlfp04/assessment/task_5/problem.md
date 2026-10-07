# MLFP04 — Task 5: From Discovered Segments to a Neural Network

**Weight**: 25 marks · **Outcomes**: 4.8 (forward pass, backpropagation, stable loss, training a network with regularisation and early stopping), integrated with 4.1/4.3 (unsupervised features feeding a supervised model — "hidden layers as automated feature engineering")
**Data**: subscription-customer tables in the format below. `dev_churn.parquet` (shipped with the task) is one such table for development; the grader uses its own.

## Scenario

A subscription grocery service wants to predict which customers will cancel
next month. Each row has a `customer_id`, eight behaviour fields
(`visits_30d`, `avg_session_min`, `spend_90d_sgd`, `support_tickets_90d`,
`days_since_last_order`, `pct_promo_orders`, `delivery_delay_days`,
`app_rating_given`) and, for past customers, `churned` (1 = cancelled).

The analytics lead suspects churn is driven by which behavioural segment a
customer belongs to, and that the risky segments are not simply "high" or
"low" on any one field: a linear scorecard on the raw fields has done badly.
The project has three parts.

**A. Backpropagation you can audit.** The team's training library must be
checkable by hand. Implement the loss and gradients of a 3-layer network:

```text
H1 = relu(X W1 + b1)      W1: (d, h1)   b1: (h1,)
H2 = relu(H1 W2 + b2)     W2: (h1, h2)  b2: (h2,)
z  = H2 W3 + b3           W3: (h2, 1)   b3: (1,)
p  = sigmoid(z)
loss = mean binary cross-entropy of p against y  +  (l2 / 2) * (sum W1² + sum W2² + sum W3²)
```

Biases are not penalised. The loss must stay finite and accurate when the
network is confidently wrong (logits of ±60 or more).

**B. Discover the segments.** Without looking at `churned`, turn the
behaviour fields into features that expose the customer's segment.

**C. Predict churn.** Train a neural network on past customers and score
customers whose outcome is unknown, using what you discovered in B.

## What to submit

`starter.py` with three functions (signatures fixed):

| Function                                  | Returns                                                                                                                                                    |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `loss_and_gradients(params, X, y, l2)`    | `(loss, grads)`: `loss` a float; `grads` a dict with the same keys and shapes as `params` (`W1`, `b1`, `W2`, `b2`, `W3`, `b3`). `y` holds 0/1 labels      |
| `discover_features(customers)`            | polars DataFrame, one row per input row in the same order, numeric columns only; the input has no `churned` column                                         |
| `fit_and_predict(train, test)`            | numpy array of churn probabilities, one per `test` row in row order. `train` has `churned`; `test` does not                                                 |

## Acceptance criteria

- **A** — on random networks and data drawn by the grader, the loss must
  match an independent reference to relative 1e-6 and every gradient to
  relative 1e-5; this includes a network that is confidently wrong.
- **B** — on two secret populations, a plain logistic regression (5-fold
  cross-validated ROC-AUC, features standardised by the grader) must do at
  least 0.10 better on your features than on the raw fields.
- **C** — on the same populations, the grader holds out a third of the
  customers (shuffled, labels removed). Your held-out ROC-AUC must be at
  least 0.12 above a logistic regression on the raw fields, and within 0.04
  of the AUC of the true churn probabilities used to simulate the data.
- Rules checks earn marks only when an outcome check in the same part
  passes.

## Rules

- Part A is numpy only: `torch`, `jax`, `tensorflow` and `autograd` may not
  be imported anywhere in the file.
- Part B uses kailash-ml `ClusteringEngine` and/or `DimReductionEngine`.
- Part C's network is trained through kailash-ml `SklearnTrainable` or with
  your own `loss_and_gradients`.
- Polars for data handling (no pandas). Your functions must work from their
  arguments alone: the grader never passes the same data twice.
