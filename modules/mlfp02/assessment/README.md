# MLFP02 — End-of-Module Assessment: Statistics for Machine Learning

Five coding tasks on the module's real datasets: the four-arm e-commerce
experiment log, HDB resale transactions, the Singapore credit-scoring book,
and simulated policy panels. There is no multiple choice and no
fill-in-the-blank. Each task states a business goal, the data, the
constraints and the acceptance criteria. How you get there is up to you.

**Duration**: 3 hours · **Total**: 100 marks · **Open book** (documentation allowed; AI assistants **not** allowed).

## Tasks

| Task                                          | Marks | Data                                              | Learning outcomes assessed                                                                                                                                                                         | Kailash                                         |
| --------------------------------------------- | ----- | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1 — Bayesian updating & likelihood estimation | 20    | `experiment_data.parquet`, order values           | 2.1 Bayes, Beta-Binomial posterior, P(B > A); choosing a distribution by AIC · 2.2 MLE (Gamma, LogNormal), MAP with stated priors                                                                  | —                                               |
| 2 — Experiment read-out                       | 25    | `experiment_data.parquet`                         | 2.3 bootstrap CI, permutation test, Bonferroni vs BH · 2.4 SRM against the **designed** 40/35/15/10 split, power / sample size, ship decision gated by SRM · 2.7 CUPED with a pre-period covariate | `ExperimentTracker`                             |
| 3 — Regression, ANOVA & logistic inference    | 25    | `hdb_resale.parquet`, `sg_credit_scoring.parquet` | 2.5 OLS with dummy coding, t/F inference, interaction test, out-of-time prediction, data validation · 2.6 one-way ANOVA + Tukey HSD, logistic MLE, odds ratios per SD, Wald comparison of effects  | —                                               |
| 4 — Difference-in-differences                 | 15    | simulated policy panels                           | 2.7 ATT, robust SE, parallel-trends test, placebo test, credibility verdict                                                                                                                        | —                                               |
| 5 — Point-in-time features                    | 15    | `hdb_resale.parquet`                              | 2.8 trailing-window features, no look-ahead, FeatureStore materialise / as-of retrieval / serving                                                                                                  | `FeatureStore`, `FeatureSchema`, `FeatureGroup` |

Each `task_N/` directory gives students:

- `problem.md` — scenario, function contracts, acceptance criteria, rules;
- `starter.py` — data loading and the function signatures to implement.

`solution.py` and `grader.py` are for instructors only and are withheld from
students. `grading_harness.py` is the graders' shared plumbing.

## How it is graded

Run a grader from the repository root:

```bash
python modules/mlfp02/assessment/task_N/grader.py path/to/submission.py
python modules/mlfp02/assessment/task_N/grader.py path/to/submission.py --seed 123   # replay a run
```

- Graders call the student's functions on **grader-held data**: per-run
  secret subsamples, secret time cut-offs with held-out rows, synthetic data
  with planted ground truth (true effects, faulty arms, trend gaps), and
  stores the grader writes itself. References are recomputed in the grader,
  independently of the solution. Hard-coded numbers, constant predictors,
  echoed inputs and code that ignores its arguments therefore fail.
- Marks = task weight × checks passed / checks run. The JSON output lists
  each check, a diagnostic note for every failed check, and the seed.
- Optimisation and resampling answers are graded by optimality or within
  Monte-Carlo tolerances, so any correct method passes.
- Typical run time is 15–60 s per task. Tasks 2 and 5 start a Kailash store.

## Rules for students

- Polars for data handling (no pandas); numpy / scipy for the mathematics.
  Task 3 bans `statsmodels` / scikit-learn models — you implement the fits.
- Load data through `shared.MLFPDataLoader` (works locally and on Colab).
- Your functions must compute everything from their arguments: the grader
  never passes the same data twice.
