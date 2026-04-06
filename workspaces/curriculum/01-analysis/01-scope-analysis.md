# ASCENT Scope Analysis — v4 Curriculum Implementation

## Objective

Complete implementation of the ASCENT 10-module ML engineering curriculum, including all exercises, textbook tutorials, datasets, assessments, and lecture materials.

## Current State (Session 5 Start)

| Deliverable          | Target | Completed             | Gap |
| -------------------- | ------ | --------------------- | --- |
| Python tutorials     | 83     | 83                    | 0   |
| Rust tutorials       | ~80    | 7 → 14 (in progress)  | ~66 |
| Exercise solutions   | 48     | 34 → 44 (in progress) | 4   |
| Exercise local (.py) | 48     | 34 → stripping        | 14+ |
| Notebooks (Jupyter)  | 48     | 34                    | 14+ |
| Notebooks (Colab)    | 48     | 34                    | 14+ |
| Datasets             | 10+    | 0 → 1 (in progress)   | 9+  |
| Lecture decks        | 48     | 0                     | 48  |
| Quizzes              | 48     | 0                     | 48  |
| shared.run_profile   | 1      | 1                     | 0   |
| uv.lock              | 1      | 0                     | 1   |

## Exercise Mapping (v4 Curriculum)

### New Exercises (16 total)

- M1: 1.1 (Python basics), 1.2 (filter), 1.3 (functions), 1.6 (visualization)
- M2: 2.2 (MLE), 2.4 (bootstrap)
- M3: 3.1 (bias-variance), 3.6 (DataFlow)
- M4: 4.2 (EM/GMMs), 4.3 (PCA/UMAP)
- M5: 5.5 (MCP), 5.7 (rewrite multi-agent), 5.8 (Nexus deployment)
- M6: 6.4 (advanced alignment), 6.7 (agent governance)

### Moved/Renumbered (18 total)

- M1 ex_2 → M2 ex_1 (Bayesian)
- M1 ex_4 → M2 ex_3 (Hypothesis testing)
- M2 ex_1→ex_7, ex_2→ex_8, ex_3→ex_5, ex_4→ex_6
- M3 ex_1→ex_2, ex_2→ex_3, ex_3→ex_4, ex_4→ex_5, ex_5→ex_7, ex_6→ex_8
- M4 ex_2→ex_4, ex_3→ex_5, ex_4→ex_6, ex_5→ex_7, ex_6→ex_8
- M5 ex_5→ex_6, ex_6→ex_7
- M6 ex_3→ex_5, ex_4→ex_6, ex_5→ex_3, ex_6→ex_8

## Risk Assessment

| Risk                               | Likelihood | Impact | Mitigation                                     |
| ---------------------------------- | ---------- | ------ | ---------------------------------------------- |
| SDK packages not on PyPI           | Certain    | Medium | Tutorials validate against source, not runtime |
| Rust kailash-rs incomplete         | High       | Medium | Stub traits where needed, note PARITY          |
| Dataset realism                    | Medium     | Low    | Use real Singapore statistical ranges          |
| Notebook conversion errors         | Low        | Low    | py_to_notebook.py is tested                    |
| Exercise scaffolding too easy/hard | Medium     | Medium | Red team will catch this                       |

## Execution Strategy

Parallel autonomous agent execution across 8+ streams:

1. Exercise solutions (4 module-group agents)
2. Exercise stripping (2 agents, started after solutions)
3. Rust tutorials (1 agent)
4. Infrastructure (1 agent)
5. Dataset generation (1 agent)
6. Notebook conversion (after stripping)
7. Red team validation (after all content)
8. Decks + quizzes (parallel with above)
