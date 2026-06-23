# MLFP04 — End-of-Module Assessment

**Module 4: Unsupervised Machine Learning & Advanced Techniques**

Total: **100 marks** · Four tasks, **25 marks each** · Suggested time: **3 hours**
· **Open-book, no AI assistants**

This assessment is auto-graded. Each task is a self-contained folder with a
`problem.md` (the brief) and a `starter.py` (the file you complete and submit).
Every task is framework-first: the core machine-learning step runs through a
**kailash-ml engine** — raw `sklearn` / `torch` shortcuts are blocked by the
brief, and the graders score genuine outcomes (recovered structure, ranking
quality, accuracy), not surface code.

## Tasks

| Task   | Topic                                        | kailash-ml engine                               | Marks |
| ------ | -------------------------------------------- | ----------------------------------------------- | ----- |
| task_1 | Customer segmentation by clustering          | `ClusteringEngine`                              | 25    |
| task_2 | Dimensionality reduction & anomaly detection | `DimReductionEngine` + `AnomalyDetectionEngine` | 25    |
| task_3 | NLP topic discovery with NMF                 | `DimReductionEngine` (NMF)                      | 25    |
| task_4 | Neural network foundations                   | `SklearnTrainable` (MLP)                        | 25    |

Each task is graded by 10 strict automated checks; all 10 must pass for full
marks. The checks test recovered structure against planted ground truth the
brief never gives you (e.g. adjusted Rand index vs hidden cluster labels,
ROC-AUC vs hidden anomaly flags, topic purity vs the real domains, and accuracy
that an independent linear baseline provably cannot reach).

## Datasets

- **task_1, task_2, task_4** — deterministic synthetic data generated inside the
  task from a fixed seed (a planted-persona cohort, a low-rank sensor matrix
  with injected outliers, and concentric circles). No file or download needed.
- **task_3** — **real** Singapore-domain Q&A text from
  `data/mlfp04/sg_domain_qa.parquet`, loaded via `shared.MLFPDataLoader` (the
  four most distinct domains: finance, food, geography, transport).

## How to work

1. Read `task_N/problem.md` — it states the scenario, the exact output contract
   (`solve()` return keys and types), the visible sanity checks, and the rules.
2. Complete the `solve()` function in `task_N/starter.py`. Follow the numbered
   TODOs. Do not change the data-generation helpers, seeds, or sort orders — the
   grader regenerates the exact same data.
3. Keep everything **Polars** (no pandas) and route the ML step through the
   named kailash-ml engine.
4. **Submit your completed `starter.py` for each task to the portal.**

## Rules

- Open-book: course notes, kailash-ml docs, and your own Module 4 exercises are
  allowed.
- **No AI assistants.**
- Polars only — no pandas.
- No hardcoded secrets or model names.
- Solutions must be deterministic (keep the given seeds).

> The reference solutions and automated graders are withheld. Submit your
> completed starter files; they are scored on the portal.
