# MLFP02 — Task 3: Regression, ANOVA & Logistic Inference

**Weight**: 25 marks · **Outcomes**: 2.5 (multiple regression, dummy coding, t/F inference, interactions), 2.6 (logistic regression, odds ratios, one-way ANOVA, Tukey HSD)
**Data**: `mlfp01/hdb_resale.parquet` (HDB resale transactions, raw) and `mlfp02/sg_credit_scoring.parquet` (100,000 borrowers).

## Scenario

**Valuation desk.** A valuer wants a transparent price model for HDB resale
flats: resale price explained by floor area, remaining lease, storey and flat
type, with `3 ROOM` as the reference flat type. They want to know which
effects are statistically distinguishable from zero, how well the model fits,
whether the price-per-square-metre slope differs by flat type, and how well
the model prices **later** sales it was not fitted on.

Facts about the raw file you must respect:

- `month` is the sale month (`YYYY-MM`). An HDB lease runs 99 years from
  `lease_commence_date`; remaining lease is measured at the sale year. The
  `remaining_lease` text column is unreliable — do not use it.
- `storey_range` looks like `"04 TO 06"`; some entries contain data-entry
  typos where the letter `O` was typed for the digit `0`. Storey is the
  midpoint of the range.
- Recording errors that must not influence the fit: resale prices outside
  S$100,000–S$2,000,000, and sales dated before the lease commenced.

**Pricing analyst.** Separately, the analyst wants to know whether **price
per square metre** differs across a set of flat types, and which pairs of flat
types differ once you account for making several comparisons at once.

**Credit risk.** The risk team wants a logistic model of `default` on
`credit_utilization, num_late_payments, previous_defaults, debt_to_income,
num_hard_inquiries`, with odds ratios expressed **per one standard deviation**
of each feature (standard deviation of the training data), and a verdict on
whether the two strongest drivers actually differ in strength.

## What to submit

`starter.py` with three functions (signatures fixed):

| Function                                     | Returns                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fit_price_model(transactions)`              | dict: `n_used` (rows used in the fit), `coefficients`, `std_errors`, `p_values` (dicts keyed `"intercept"`, `"floor_area_sqm"`, `"remaining_lease_years"`, `"storey_mid"`, and `"flat_type=<TYPE>"` for every non-reference type present), `r_squared`, `adj_r_squared`, `f_statistic`, `interaction_f`, `interaction_p` (test of adding floor-area × flat-type terms), `predict` (a function: raw rows in the same schema → numpy array of predicted prices, one per row) |
| `anova_flat_types(transactions, flat_types)` | dict: `f_stat`, `p_value`, `eta_squared`, `tukey` — keyed `"<A>                                                                                                                                                                                                                                                                                                                                                                                                            | <B>"`with the two type names in alphabetical order, each`{"p_adj": float, "significant": bool}` at the 5% family-wise level |
| `fit_default_model(train)`                   | dict: `odds_ratios`, `std_errors` (of the per-SD log-odds coefficients; both dicts keyed by the five feature names), `top_two` (the two features with the largest absolute effect), `top_two_p` (two-sided p-value that their effects are equal), `top_two_differ` (bool, 5% level), `predict_proba` (function: rows → default probabilities)                                                                                                                              |

## Acceptance criteria

- The grader fits your price model on a **secret raw sample** of transactions
  before a **secret cut-off month**, compares every coefficient, standard error
  and p-value with an independent classical-OLS reference on the same rows,
  and scores `predict` on later transactions you never see: held-out R² must
  be within 0.01 of the reference model's.
- ANOVA and Tukey results are checked on a secret, already-validated sample
  and a secret choice and order of flat types.
- The logistic model is checked against an independent maximum-likelihood fit
  (unpenalised) on a secret training sample; held-out AUC must be within 0.01
  of the reference.

## Rules

- Polars for data handling (no pandas); numpy / scipy for the statistics.
  You implement the fits yourself — `statsmodels` and scikit-learn models are
  not allowed in your submission.
- Develop on the real files via `shared.MLFPDataLoader` (see `starter.py`).
