# MLFP03 — Task 3: Decisions Priced in Dollars

**Weight**: 30 marks · **Dataset**: `mlfp02/sg_credit_scoring.parquet` (100,000 labelled loan applications, 36 columns, 12.9% default)
**Outcomes assessed**: class imbalance and cost-sensitive learning (3.5), probability calibration (3.5), metric choice and decision thresholds from business costs (3.5), honest evaluation (3.2), model registry and promotion (3.7, 3.8)

## Scenario

The lender's credit desk does not want a ranking; it wants **decisions**. Each
application is approved or declined, and the finance team has priced both
kinds of mistake:

- approving an applicant who then defaults costs the bank
  `costs["missed_default"]` dollars;
- declining an applicant who would have repaid costs the bank
  `costs["declined_good"]` dollars (the lost margin).

Correct decisions cost nothing. Finance re-prices these figures every quarter,
so your procedure must take them as inputs, not as constants.

The same model's probabilities also feed the bank's loss provisions, so a
predicted 8% must mean that about 8 in 100 such applicants default. Only
about one applicant in eight defaults, which makes both of these harder than
they look.

Finally, the decision model goes live through the bank's model registry: the
serving layer will load the **production** version and nothing else, so what
is registered there must be exactly what produced your numbers.

## Interface

```python
def build_decision_model(history: pl.DataFrame, costs: dict, registry_dir: str) -> dict: ...
```

- `history`: labelled applications with the file's columns, including `default`.
- `costs`: `{"missed_default": float, "declined_good": float}` in S$.
- `registry_dir`: an empty directory for the bank's registry. The registry
  is a kailash-ml `ModelRegistry` whose database is the SQLite file
  `registry_dir/registry.db` and whose artefacts are stored in
  `registry_dir/artifacts/` (a `LocalFileArtifactStore`).

New applications arrive with the file's columns except `default`; the field
recorded only after a loan's outcome is still empty.

Return a dict with four entries:

| Key             | Value                                                                                                                                                                              |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `predict_proba` | function: applications (polars DataFrame) → 1-D numpy array of default probabilities, in row order                                                                                 |
| `decide`        | function: applications → 1-D numpy **boolean** array, `True` = approve                                                                                                             |
| `expected_cost` | your estimate of the average cost per application (S$) that `decide` will incur on new applications                                                                                |
| `model_name`    | the registry name of your model. Its **production** version's artefact, unpickled, has a `predict_proba(applications)` that returns the same probabilities as your `predict_proba` |

## Acceptance criteria

- The probabilities rank applicants well and are **calibrated**: on average
  and across the whole range, a predicted probability matches the rate at
  which such applicants actually default.
- The decisions minimise the expected cost for the costs given — and still do
  when the costs are very different.
- `expected_cost` is an honest estimate (within 20% of what the decisions
  actually cost on new applicants).
- A prediction depends only on the application it is for.
- The production version in the registry reproduces your probabilities.
- Deterministic: the same inputs always give the same result.

## How you are graded (11 automated checks)

The grader calls your function on a **secret sample** of 8,000 rows, twice,
with **secret costs** each time. It scores everything on **new applications
you have never seen**, drawn from the same population. For those applicants
the grader knows how likely each one really was to default, so it can measure
calibration and the expected cost of your decisions precisely. Its references
are its own simple model fitted on the same sample.

| #   | Check                                                                                                                                 |
| --- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Output is well formed (keys, one probability in [0, 1] and one boolean per application, not all equal) — every other check needs this |
| 2   | Ranking: AUC on new applicants within 0.015 of the reference                                                                          |
| 3   | Calibrated on average: mean predicted probability within 0.01 of the true default rate                                                |
| 4   | Calibrated across the range: decile-binned gap between predicted and true probability ≤ 0.015                                         |
| 5   | Probability quality: Brier score within 0.0015 of the reference                                                                       |
| 6   | Decisions: expected cost per application within 3% of the reference decision rule, for the first set of costs                         |
| 7   | Decisions: the same, after a second call with very different costs                                                                    |
| 8   | `expected_cost` within 20% of the measured cost of your decisions                                                                     |
| 9   | Shuffling and sub-setting the applications leaves each applicant's probability unchanged                                              |
| 10  | The registry holds `model_name` at stage `production`                                                                                 |
| 11  | The production artefact's `predict_proba` reproduces your probabilities                                                               |

Marks = 30 × checks passed / 11.

## Rules

- Polars for data handling (no pandas). Train through kailash-ml
  (`TrainingPipeline`, with `PreprocessingPipeline` for imputation/scaling)
  and use `ModelRegistry` for registration and promotion. If you adjust
  probabilities after training, say in a comment which rows you used and why.
- `build_decision_model` must not load files itself, and must finish within
  5 minutes on a laptop. Anything you pickle into the registry must be
  defined at module level in your file.
- Develop on the real file via `shared.MLFPDataLoader` (see `starter.py`).
