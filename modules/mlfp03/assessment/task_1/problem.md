# MLFP03 — Task 1: Application-Time Model Inputs

**Weight**: 20 marks · **Dataset**: `mlfp02/sg_credit_scoring.parquet` (100,000 labelled loan applications, 36 columns, 12.9% default)
**Outcomes assessed**: domain-driven feature engineering (3.1), leakage detection and prevention (3.1), feature selection (3.1), split-first preprocessing with Kailash engines (3.1, 3.2)

## Scenario

A Singapore lender is rebuilding its default model. The modelling team will
take whatever **model inputs** you hand them and fit their own model on them,
so your job is the inputs: what goes in, how it is computed, and how it is
applied to applications that arrive later.

The risk team has already decided which labelled applications are held back
for the model's final sign-off. Those **hold-out** rows are in the history you
receive, with their outcomes, so that their inputs can be produced the same
way as everyone else's. Sign-off is only meaningful if nothing you build has
learned anything from them.

The credit committee has also asked for two affordability measures to be among
the inputs:

- **instalment burden** — how large the monthly instalment is relative to the
  applicant's monthly income;
- **savings cover** — how many months of instalments the applicant's savings
  would pay.

Nobody has told you whether every column in the file is something the bank
actually knows when an application arrives. Finding out is part of the task.

## Interface

```python
def build_model_inputs(history: pl.DataFrame, is_holdout: pl.Series) -> dict: ...
```

- `history` has the same 36 columns as the parquet file, including `default`.
- `is_holdout` is a Boolean series aligned with `history`'s rows; `True` marks
  a hold-out row.

Return a dict with three entries:

| Key         | Value                                                                                                                                                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `selected`  | list of the input column names you chose — at most **12**                                                                                                                                                                                 |
| `inputs`    | polars DataFrame with `customer_id` plus exactly the `selected` columns, in that order, with **one row per row of `history`** (training and hold-out rows alike)                                                                          |
| `transform` | a function that takes new applications (same columns as the file, **without** `default`) and returns a frame shaped like `inputs` for them. When an application arrives, any field recorded only after the loan's outcome is still empty. |

## Acceptance criteria

- Every input is numeric, with no missing or infinite values.
- **Hold-out rows teach you nothing.** Every choice and every fitted value
  (which inputs, fill values, scaling, any learned relationship) comes from
  the training rows only. The hold-out rows' inputs are exactly what
  `transform` produces for them.
- **Inputs exist at application time.** No input may depend on an identifier
  or on information recorded after the outcome.
- **Scoring does not learn.** An application's inputs do not depend on which
  other applications are scored with it.
- The inputs include the two affordability measures above.
- **The inputs keep the signal.** A plain logistic regression fitted on your
  training-row inputs must rank new applicants' default risk (ROC-AUC) almost
  as well as one fitted on every legitimate numeric field.
- The function is deterministic: the same training rows always give the same
  result.

## How you are graded (10 automated checks)

The grader calls your function on a **secret sample** of 10,000 labelled rows
with a **secret hold-out flag**, and scores `transform` on **new applications
you have never seen**, drawn from the same population. It never uses numbers
your code reports about itself.

| #   | Check                                                                                                                        |
| --- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1   | Output is well formed (keys, one row per application, ≤ 12 numeric inputs, no missing values) — every other check needs this |
| 2   | Changing only the applications' IDs leaves their inputs unchanged                                                            |
| 3   | Filling in the field recorded after the outcome leaves the inputs unchanged                                                  |
| 4   | Re-running with the **hold-out rows altered** (outcomes and values) leaves the selection unchanged                           |
| 5   | … and leaves every training row's inputs unchanged                                                                           |
| 6   | Hold-out rows in `inputs` equal `transform` applied to them                                                                  |
| 7   | An application's inputs are the same when it is scored inside a very different batch                                         |
| 8   | One input ranks applicants like the instalment burden                                                                        |
| 9   | One input ranks applicants like the savings cover                                                                            |
| 10  | The grader's logistic model on your inputs scores new applications within 0.015 ROC-AUC of its all-fields reference          |

Marks = 20 × checks passed / 10.

## Rules

- Polars for data handling (no pandas). Use the kailash-ml engines the module
  teaches where they fit — for example `FeatureEngineer` for ranking
  candidates and `PreprocessingPipeline` for imputation and scaling.
- `build_model_inputs` works only on its arguments: it must not load files
  itself and must not modify `history` in place.
- Develop on the real file via `shared.MLFPDataLoader` (see `starter.py`).
  Make your own hold-out flag to try your function.
