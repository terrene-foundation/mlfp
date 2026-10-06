# MLFP03 — Task 4: The Release Review

**Weight**: 30 marks · **Dataset**: `mlfp02/sg_credit_scoring.parquet` (100,000 labelled loan applications, 36 columns, 12.9% default)
**Outcomes assessed**: per-applicant explanations that satisfy the Shapley properties (3.6), fairness measurement — disparate impact and equalised odds (3.6), production drift monitoring with sound statistical discipline (3.8)

## Scenario

Your decision model is up for release, and the credit committee's review has
three standing items. None of them is optional.

**Explanations.** Every scored applicant can ask _why_. The committee needs a
per-applicant breakdown of the score into one number per field, and it will
only accept a breakdown with three properties:

1. the numbers for an applicant add up to that applicant's score **minus the
   average score over the background book**;
2. a field the scoring function never reads gets **exactly zero**;
3. credit is divided among the fields by **Shapley's rule** — each field's
   share is its average marginal contribution, where a field that is "left
   out" of a coalition is replaced by values drawn from the background
   applicants (independently of the applicant's other fields).

The breakdown is in the score's own units (probability of default), per
applicant.

**Fairness.** Before any scorecard ships, the desk audits its **decisions**
across groups of applicants. For each group you report the approval share,
how it compares to the best-treated group (the _four-fifths rule_), and the
approval shares separately among applicants who repaid and among those who
defaulted (the two halves of _equalised odds_). Some applicants have no
recorded group; they are a group of their own and are never silently dropped.
The reviewer chooses which column to audit at review time — your report must
work for whichever column is named.

**Drift.** Once live, every incoming batch of applications is screened
against the reference book on **all of its numeric fields at once**, and the
monitor names the fields whose distribution has moved. A monitor that cries
wolf on stable batches gets switched off; one that misses a real shift gets
the desk into trouble. Testing dozens of fields on every batch will raise
spurious alarms unless your significance discipline accounts for how many
tests you are running — choose one that keeps a stable batch quiet while
still catching every genuinely shifted field.

## Interface

```python
def explain(predict_proba, background: pl.DataFrame, applications: pl.DataFrame) -> pl.DataFrame: ...
def fairness_report(audit: pl.DataFrame, group_column: str) -> pl.DataFrame: ...
def drift_alerts(reference: pl.DataFrame, batch: pl.DataFrame, features: list[str]) -> list[str]: ...
```

**`explain`** — `predict_proba` is the scoring function under review: it
takes a polars DataFrame with the same columns as `background` (minus the
`customer_id` identifier) and returns one probability per row. `background`
is the book the committee compares applicants against; `applications` are the
applicants to explain. Return a polars DataFrame with exactly one row per
application: the `customer_id` column plus one column per field, holding that
field's attribution for that applicant.

**`fairness_report`** — `audit` is a table of decided applications: the
`approved` decision (boolean), the realised `default` outcome (0/1), and the
column named by `group_column` (which may contain missing values). Return a
polars DataFrame with one row per group and exactly these columns:

| Column                  | Meaning                                                           |
| ----------------------- | ----------------------------------------------------------------- |
| `group`                 | the group label; missing membership is reported as `"unrecorded"` |
| `applicants`            | number of applicants in the group                                 |
| `approval_rate`         | share approved                                                    |
| `approval_ratio`        | `approval_rate` divided by the highest group's `approval_rate`    |
| `good_approval_rate`    | approval share among applicants who did **not** default           |
| `default_approval_rate` | approval share among applicants who defaulted                     |
| `passes_four_fifths`    | boolean: `approval_ratio >= 0.8`                                  |

**`drift_alerts`** — `reference` is the book the monitor was armed on,
`batch` is the latest incoming batch, and `features` lists every numeric
field to watch. Return the names of the fields whose distribution has moved,
as a list of strings (empty when nothing has moved).

## Acceptance criteria

- The explanations satisfy all three committee properties on scoring
  functions you have never seen — including ones with interactions,
  thresholds, and fields they ignore.
- The fairness report reproduces the desk's own arithmetic exactly, for
  whatever group column is named, with unrecorded membership surfaced as its
  own group.
- The drift monitor stays silent on batches drawn from the same population
  as the reference, and names exactly the shifted fields on a batch where
  some fields have moved.
- All three functions are deterministic and load no files themselves.

## How you are graded (11 automated checks)

The grader supplies everything your functions see, and none of it is the
development file:

- **`explain`** is handed two of the grader's **own** scoring functions
  (secret coefficients, an interaction, a step, and one field the function
  ignores) together with fresh background books and applicants. The grader
  computes the exact Shapley values itself and checks: your attributions add
  up; the ignored field gets nothing; your values match the exact ones for
  both functions.
- **`fairness_report`** is handed an audit table built from fresh
  applications — decisions and outcomes the grader made itself — once on a
  known column and once on an age band whose **column name changes every
  run** and whose membership is sometimes unrecorded. The grader recomputes
  every rate itself and compares groups, counts, rates, ratios, and flags.
- **`drift_alerts`** is handed a fresh reference and several fresh batches
  from the same population (you must raise nothing), then one batch in which
  **three secretly chosen fields** have been shifted (you must name exactly
  those three).

Marks = 30 × checks passed / 11.

## Rules

- Polars for data handling (no pandas). Monitor drift through the kailash-ml
  `DriftMonitor`. Anything you compute yourself must be computed from the
  arguments you are given — never from the development file, which the
  grader's inputs do not come from.
- Develop against the real file via `shared.MLFPDataLoader` (see
  `starter.py`); the grader's scoring functions, audit tables, references and
  batches are all drawn fresh and unseen.
- Each function must finish in seconds on a laptop for a few thousand
  applications.
