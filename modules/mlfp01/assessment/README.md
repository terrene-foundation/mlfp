# MLFP01 — Assessment: Data Pipelines and Visualisation

Three practical coding tasks that verify you can load, transform, and clean real Singapore datasets using Polars and Kailash's `DataExplorer` engine.

## Tasks

| Task | Difficulty | Dataset               | Skills                                               |
| ---- | ---------- | --------------------- | ---------------------------------------------------- |
| 1    | Easy       | `sg_weather.csv`      | Polars group-by, aggregations, `with_columns`        |
| 2    | Medium     | `hdb_resale.parquet`  | Filtering, window functions, year-over-year deltas   |
| 3    | Hard       | `sg_taxi_trips.parquet` | End-to-end cleaning pipeline with `DataExplorer`   |

Each task directory contains:

- `problem.md` — problem statement, input/output spec, a visible test case
- `starter.py` — skeleton with `# TODO` markers for you to complete
- `solution.py` — instructor reference solution (study after your attempt)
- `grader.py` — automated grader

## How to run

```bash
cd modules/mlfp01/assessment/task_1
python grader.py starter.py    # grade your attempt
python grader.py solution.py   # verify the reference solution passes
```

The grader prints a JSON report:

```json
{
  "passed": true,
  "checks": {"returns_dataframe": true, "has_required_columns": true, ...},
  "total": 5,
  "max": 5
}
```

Exit code is `0` if all checks pass, `1` otherwise.

## Rubric

Each task is graded on a fixed set of automated checks (see `grader.py` for the exact list). Your score is `total / max`. A task is **passed** when all checks pass.

| Task    | Weight | Pass threshold |
| ------- | ------ | -------------- |
| Task 1  | 20%    | 100% of checks |
| Task 2  | 35%    | 100% of checks |
| Task 3  | 45%    | 100% of checks |

## Rules

- Use Polars only — no pandas
- Use `shared.MLFPDataLoader` to load data (works on both local and Colab)
- For task 3 you must use `DataExplorer` to profile before/after cleaning
- Solutions must run in under 60 seconds on a laptop
