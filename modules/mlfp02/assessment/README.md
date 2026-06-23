# MLFP02 — End-of-Module Assessment: Statistics for Machine Learning

Four practical coding tasks on real datasets — an e-commerce experiment log, a
Singapore credit-scoring book, and a multi-table ICU record set. There is no
multiple choice. Each task is auto-graded on **outcomes** — exact answer keys,
independently re-derived ground truth, and strict numeric tolerances. A task
passes only when **every** automated check passes.

**Duration**: 3 hours · **Total**: 100 marks · **Open book** (documentation
allowed; AI assistants **not** allowed).

## Tasks

| Task | Weight | Difficulty | Dataset                     | Skills                                                                 |
| ---- | ------ | ---------- | --------------------------- | --------------------------------------------------------------------- |
| 1    | 20     | Hard       | `experiment_data.parquet`   | Conditional probability, Bayes inversion, SRM chi-square, base-rate fallacy, Beta-Binomial posterior |
| 2    | 25     | Hard       | `experiment_data.parquet`   | Welch t-test, seeded bootstrap CI, CUPED variance reduction, Bonferroni vs Benjamini-Hochberg |
| 3    | 25     | Hard       | `sg_credit_scoring.parquet` | Closed-form OLS with full inference, partial F-test, logistic MLE & odds ratios |
| 4    | 30     | Hard       | five `icu_*.parquet` tables | Multi-table joins, messy-string parsing, point-in-time feature table, imputation policy |

Each task directory contains:

- `problem.md` — scenario, exact `solve()` contract, visible sanity checks,
  grading checklist, and rules.
- `starter.py` — the skeleton you complete and submit.

## How to work

1. Open `task_N/problem.md` and read the full specification.
2. Implement `solve()` in `task_N/starter.py`.
3. **Reason before you compute** — the difficulty is in the statistics
   (which population, which covariate, which correction), not in the syntax.
   Follow each documented contract exactly: keys, definitions, and the fixed
   bootstrap seed / resample order.
4. When done, **submit your completed `starter.py` files to the assessment
   portal** for grading. Do not rename `solve()` or change its return contract.

## Rules

- **Polars only** for data wrangling — no pandas (see the course framework-first
  standard). `numpy` / `scipy.stats` are allowed for the statistical computation.
- Load data via `shared.MLFPDataLoader` (works in VS Code and on Colab).
- All solutions must be **deterministic** — the only randomness anywhere is the
  Task 2 bootstrap, which is fixed by `np.random.default_rng(2024)`.
- Every task runs in well under the time budget on a laptop.
