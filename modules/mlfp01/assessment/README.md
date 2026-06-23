# MLFP01 — End-of-Module Assessment: Data Pipelines & Visualisation

Four practical coding tasks on real, messy Singapore datasets. There is no
multiple choice. Each task is auto-graded on **outcomes** — exact schemas,
re-derived ground truth, and strict data-quality invariants. A task passes only
when **every** automated check passes.

**Duration**: 3 hours · **Total**: 100 marks · **Open book** (documentation
allowed; AI assistants **not** allowed).

## Tasks

| Task | Weight | Difficulty | Dataset                      | Skills                                                          |
| ---- | ------ | ---------- | ---------------------------- | -------------------------------------------------------------- |
| 1    | 25     | Hard       | `sg_taxi_trips.parquet`      | Deterministic cleaning, payment normalisation, plausibility filters, dedup |
| 2    | 25     | Hard       | `hdb_resale.parquet`         | Messy-string parsing (OCR digits, dual-format lease), feature engineering |
| 3    | 20     | Hard       | `hdb_resale.parquet`         | Window functions — YoY, rolling averages, rank within group    |
| 4    | 30     | Hard       | `economic_indicators.csv`    | Multi-format parsing, type repair, `DataExplorer` profiling     |

Each task directory contains:

- `problem.md` — scenario, exact `solve()` contract, visible sanity checks,
  grading checklist, and rules.
- `starter.py` — the skeleton you complete and submit.

## How to work

1. Open `task_N/problem.md` and read the full specification.
2. Implement `solve()` in `task_N/starter.py`.
3. **Read the data before you transform it** — the messiness (OCR errors,
   mixed date formats, text-typed numbers, impossible values) is the point.
4. When done, **submit your completed `starter.py` files to the assessment
   portal** for grading. Do not rename `solve()` or change its return contract.

## Rules

- **Polars only** — no pandas (see the course framework-first standard).
- Load data via `shared.MLFPDataLoader` (works in VS Code and on Colab).
- Use kailash-ml engines where the task calls for them (Task 4: `DataExplorer`).
- All solutions must be **deterministic** (no unseeded randomness) and run in
  well under the time budget on a laptop.
