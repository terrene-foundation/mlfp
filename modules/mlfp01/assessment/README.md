# MLFP01 — End-of-Module Assessment: Data Pipelines & Visualisation

Five practical coding tasks on messy Singapore datasets. There is no multiple
choice.

The tasks state **goals, data and acceptance criteria**. They do not give the
steps. Finding what is wrong with the data, and deciding how to handle it, is
part of every task.

**Duration**: 3 hours · **Total**: 100 marks · **Open book**: documentation is
allowed; AI assistants are **not** allowed.

## Tasks

| Task | Marks | Dataset(s)                                                        | What it assesses                                                                                                                                                         | Spec lessons  |
| ---- | ----- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| 1    | 20    | `sg_taxi_trips.parquet`                                           | Data forensics against a data contract: parsing types, repair vs drop, label normalisation, identity rules, explicit nulls                                               | 1.1, 1.2, 1.8 |
| 2    | 20    | `hdb_resale.parquet` + `mrt_stations.parquet` + `schools.parquet` | Parsing hand-keyed strings, contradictory records, joins across inconsistently named keys, table grain, keeping every record                                             | 1.1–1.4       |
| 3    | 15    | `hdb_resale.parquet`                                              | Monthly trends: year-on-year change and rolling windows aligned to the **calendar** rather than to row offsets; excluding recording errors by rule                       | 1.3, 1.5      |
| 4    | 20    | `economic_indicators.csv`                                         | `DataExplorer` profiling, `AlertConfig`, cleaning that clears the alerts, and justifying the alerts that are accepted                                                    | 1.1, 1.7, 1.8 |
| 5    | 25    | clean taxi trips (supplied by the grader)                         | `PreprocessingPipeline` model inputs without leakage (booking-time fields, train-only fitting); chart choice and honest charts with `ModelVisualizer` / Plotly; findings | 1.3, 1.6, 1.8 |

Each task directory contains:

- `problem.md`: the scenario, the function interface, the acceptance
  criteria and the grading checklist;
- `starter.py`: data loading, the function signatures and a local runner. You
  complete it and submit it.

Instructors also hold `solution.py` (the reference) and `grader.py`. These are
not given to students.

## How grading works

Every grader computes its own ground truth from the raw data. **No number
that a submission reports about itself is trusted.** In particular:

- Results are compared row by row against expected values, matched on a key
  (`trip_id`, `row_id`, `(town, month)`, `(year, quarter)`). Row order never
  matters.
- Tasks 1–4 also run your function on an **unseen variant** of the data, with
  fresh IDs, other gaps, new recording errors and altered lookup tables. Code
  that remembers this year's rows, IDs or counts fails that check.
- Task 4 re-profiles your cleaned table with `DataExplorer`. It also
  recomputes your declared imputation method and tests your `AlertConfig` on
  frames the grader builds.
- Task 5 fits your preprocessing on training trips. It then transforms
  bookings you never see, and inspects the data inside each Plotly figure.

Marks are awarded per check: weight × (checks passed / checks). Some tasks
have **gates**, such as the required columns or one row per record. If a gate
fails, the task scores 0. A placeholder or hard-coded submission scores 0.

Run a grader (instructors):

```bash
python modules/mlfp01/assessment/task_1/grader.py path/to/submission.py
```

It prints a JSON report with each check, the marks, and diagnostic notes.

## How to work

1. Read `task_N/problem.md` in full.
2. **Inspect the data before you transform it.** Use `describe()`,
   `null_count()`, `value_counts()` and `DataExplorer`. The problems are not
   listed for you.
3. Implement the functions in `task_N/starter.py`. Run it to check your work
   on the real data.
4. Submit your completed `starter.py` files. Do not rename the functions or
   change their interfaces.

## Rules

- **Polars only.** No pandas (see the course framework-first standard).
- Load data with `shared.MLFPDataLoader` in the local runners. The graded
  functions themselves receive their data as arguments.
- Use the kailash-ml engines the tasks name: `DataExplorer` (Task 4), and
  `PreprocessingPipeline` and `ModelVisualizer` (Task 5). In a notebook, use
  the `shared.run_profile` / `run_compare` helpers rather than
  `asyncio.run(...)`; they work in scripts, Jupyter and Colab.
- All functions must be **deterministic**: no unseeded randomness. Each task
  should run in well under a minute on a laptop.
