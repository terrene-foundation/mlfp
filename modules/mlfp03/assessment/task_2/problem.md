# MLFP03 — Task 2: Model Selection You Can Defend

**Weight**: 20 marks · **Dataset**: `mlfp02/sg_credit_scoring.parquet` (100,000 labelled loan applications, 36 columns, 12.9% default)
**Outcomes assessed**: the supervised model zoo (3.3), gradient boosting (3.4), bias–variance and regularisation (3.2), cross-validated comparison and honest performance estimates (3.2, 3.3), training through kailash-ml (3.7)

## Scenario

The lender's model-risk team will not sign off a default model because
"gradient boosting usually wins". They want a **comparison** of the model
families you know, run so that the numbers are comparable, a choice that
follows from that comparison, and an **estimate** of how the chosen model will
rank applicants it has never seen — an estimate they will check later against
real outcomes.

Your function will be run by the team on more than one book of applications.
Some are large. Some are small, and the credit bureau sometimes appends many
extra numeric attributes to every application whose usefulness nobody has
checked. Your procedure has to cope with both without anyone re-tuning it by
hand.

## Interface

```python
def select_and_fit(train: pl.DataFrame) -> dict: ...
```

`train` holds labelled applications with the file's columns, including
`default`. It may carry **extra numeric columns** that are not in the file;
treat them like any other field. When the fitted model is later used, each
application arrives with the same columns except `default`, and the field
recorded only after a loan's outcome is still empty.

Return a dict with four entries:

| Key             | Value                                                                                                                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cv_auc`        | dict mapping each model family you compared to its validated ROC-AUC. Family names come from: `logistic_regression`, `svm`, `knn`, `naive_bayes`, `decision_tree`, `random_forest`, `gradient_boosting` |
| `chosen`        | the family you chose (a key of `cv_auc`)                                                                                                                                                                |
| `estimated_auc` | your estimate of the chosen model's ROC-AUC on applications it has never seen                                                                                                                           |
| `predict_proba` | a function that takes a polars DataFrame of applications and returns a 1-D numpy array of default probabilities, one per row, in row order                                                              |

## Acceptance criteria

- The comparison covers **at least five** families, including
  `logistic_regression`, `random_forest` and `gradient_boosting`.
- Every family is scored on the **same** validation scheme, and the scores are
  out-of-sample: no family is scored on rows it was fitted on.
- The chosen family is the one the comparison favours.
- `estimated_auc` is honest: within **0.03** of the ROC-AUC the fitted model
  actually achieves on new applicants.
- The fitted model ranks new applicants well, both on a large book and on a
  small book with many uninformative extra attributes (see the grading table
  for the bar).
- A prediction depends only on the application it is for.
- Deterministic: the same `train` always gives the same result.

## How you are graded (8 automated checks)

The grader calls your function on a **secret sample** of the file, then
scores your model on **new applications you have never seen**, drawn from the
same population, against their real outcomes. It never uses numbers your code
reports about itself except to compare them with what it measures.

| #   | Check                                                                                                                                                                      |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Output is well formed (keys, finite scores, one probability in [0, 1] per application, not all equal) — every other check needs this                                       |
| 2   | The comparison covers at least five allowed families, including the three required ones                                                                                    |
| 3   | `chosen` is the best-validated family (within 0.005)                                                                                                                       |
| 4   | Every validated score is plausible as an out-of-sample result: none is above what even the true default probabilities achieve on unseen applicants (+0.03), none below 0.4 |
| 5   | `estimated_auc` is within 0.03 of the AUC measured on new applicants                                                                                                       |
| 6   | Shuffling and sub-setting the applications leaves each applicant's probability unchanged                                                                                   |
| 7   | **Large book** (4,000 rows): AUC on new applicants within 0.015 of the grader's own baseline model                                                                         |
| 8   | **Small book** (500 rows plus 40 uninformative extra attributes): AUC on new applicants within 0.03 of the grader's own regularised baseline model                         |

Marks = 20 × checks passed / 8.

## Rules

- Polars for data handling (no pandas). Fit the model you hand back through
  kailash-ml (`TrainingPipeline`, with `PreprocessingPipeline` for any
  imputation or scaling); state in a comment why you chose the validation
  scheme you used.
- `select_and_fit` works only on its argument: it must not load files itself,
  and it must finish within 10 minutes on a 4,000-row sample on a laptop.
- Develop on the real file via `shared.MLFPDataLoader` (see `starter.py`). The
  grader's new applicants are not in the file.
