# MLFP04 — End-of-Module Assessment: Unsupervised ML and the Bridge to Neural Networks

Five coding tasks covering all eight Module 4 lessons. There is no multiple
choice and no fill-in-the-blank. Each task states a business goal, the data,
the constraints and the acceptance criteria. How you get there is up to you:
choosing features and scaling, choosing K, choosing and combining detectors,
cleaning text and deciding what makes a topic good are part of the work.

**Duration**: 3 hours · **Total**: 100 marks · **Open book** (documentation allowed; AI assistants **not** allowed).

## Tasks

| Task                                             | Marks | Data                                                              | Learning outcomes assessed                                                                                                                                     | Kailash                                                       |
| ------------------------------------------------ | ----- | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1 — Customer segments and mixture models         | 20    | loyalty CRM exports (simulated, planted personas)                 | 4.1 feature and scaling decisions, choosing K, recovering and profiling segments, naming the at-risk segment · 4.2 EM from scratch, local optima, convergence  | `ClusteringEngine`                                            |
| 2 — Reduction, embeddings and anomaly screening  | 20    | `mlfp02/sg_credit_scoring.parquet` (real) with injected anomalies | 4.3 PCA variance and loadings on a common scale, neighbour-preserving 2-D maps · 4.4 three anomaly kinds, combining detectors, the LOF masking trap            | `DimReductionEngine`, `AnomalyDetectionEngine`                |
| 3 — Baskets and recommendations                  | 20    | till exports and rating histories (simulated, messy)              | 4.5 itemsets, support / confidence / lift, complete rule sets · 4.7 matrix factorisation vs bias baseline, personalised ranking, re-ratings, new users         | —                                                             |
| 4 — Topics from news text                        | 15    | `mlfp05/ag_news.parquet` (real, raw text)                         | 4.6 cleaning, TF-IDF, NMF topics, coherent and distinct keywords that describe their documents                                                                 | `DimReductionEngine` (NMF)                                    |
| 5 — From discovered segments to a neural network | 25    | subscription customers (simulated, planted segments)              | 4.8 forward / backward pass, stable cross-entropy, training a network · integration: unsupervised features feeding a supervised network (the module's project) | `ClusteringEngine` / `DimReductionEngine`, `SklearnTrainable` |

Each `task_N/` directory gives students:

- `problem.md` — scenario, function contracts, acceptance criteria, rules;
- `starter.py` — data loading and the function signatures to implement;
- a development data file where the task uses simulated data
  (`dev_customers.parquet`, `dev_baskets.parquet` + `dev_ratings.parquet`,
  `dev_churn.parquet`).

`solution.py` and `grader.py` are for instructors only and are withheld from
students, as are `grading_harness.py` (the graders' shared plumbing) and the
`_*.py` data builders. Rebuild a development file with
`python task_N/_<builder>.py` from the repository root.

## How it is graded

Run a grader from the repository root:

```bash
python modules/mlfp04/assessment/task_N/grader.py path/to/submission.py
python modules/mlfp04/assessment/task_N/grader.py path/to/submission.py --seed 123   # replay a run
```

- Graders call the student's functions on **grader-held data** drawn with a
  fresh secret seed: new cohorts with a secret number of planted personas,
  new Gaussian mixtures, secret samples of real credit applications with
  shuffled columns and injected anomalies of known kinds, new till exports
  with secret thresholds, held-back ratings and brand-new users, secret
  AG News batches with hidden section labels, held-out customers whose
  labels never reach the submission. References are recomputed by the
  grader (brute-force rule sets, PCA, best-of-ten mixture fits, NPMI,
  finite-difference-verified gradients, true churn probabilities).
  Hard-coded numbers, constant predictors, echoed inputs and code that
  ignores its arguments fail.
- Marks = task weight × checks passed / checks run. Format and rules checks
  (e.g. "uses `ClusteringEngine`") earn marks only when an outcome check in
  the same part passes, so a well-formed stub scores 0.
- The JSON output lists each check, a diagnostic note for every failed check,
  and the seed.
- Typical run time is 20–90 s per task on a laptop (Task 2 includes a t-SNE
  fit; Task 5 trains networks on two populations).

Reference results (grader on `solution.py`, a placeholder stub and a
plausible-but-wrong submission per task, recorded when the tasks were built):

| Task | Reference | Stub | Plausible but wrong                                                            |
| ---- | --------- | ---- | ------------------------------------------------------------------------------ |
| 1    | 20 / 20   | 0    | 13.3 — z-scores raw units incl. the id column; EM with one start, 5 iterations |
| 2    | 20 / 20   | 0    | 0 — PCA on raw units, PCA map, Isolation Forest only                           |
| 3    | 20 / 20   | 0    | 0 — raw till names; user + item averages as recommender                        |
| 4    | 15 / 15   | 0    | 0 — raw counts, no cleaning or stop words, frequency keywords                  |
| 5    | 25 / 25   | 0    | 7.5 — log(sigmoid) loss; PCA-only features; logistic regression                |

## Rules for students

- Polars for data handling (no pandas); numpy / scipy for the mathematics.
- Use the Kailash engine each task names; Task 1's EM and Task 5's
  backpropagation are your own numpy code.
- Load course data through `shared.MLFPDataLoader`.
- Your functions must compute everything from their arguments: the grader
  never passes the same data twice.
