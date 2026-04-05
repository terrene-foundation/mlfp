# ASCENT Implementation Plan — v4 Curriculum

## Execution Summary

All deliverables being produced in parallel autonomous agent streams.

## Deliverable Status

### Phase 1: Content Foundation (Complete)

- [x] 48/48 exercise solutions across 6 modules
- [x] v4 curriculum alignment (renumbering, moves, adaptations)
- [x] Infrastructure: shared.run_profile, sg_weather.csv, uv.lock
- [x] ASCENTDataLoader updated for local data/ directory
- [x] 11/11 datasets generated

### Phase 2: Exercise Packaging (In Progress)

- [ ] 48/48 local exercise files (stripped from solutions)
- [ ] 48/48 Jupyter notebooks (converted from local)
- [ ] 48/48 Colab notebooks (converted from local)

### Phase 3: Supplementary Materials (In Progress)

- [ ] 48 lecture decks (Marp markdown)
- [ ] 6 quiz files (~24-32 questions each)
- [ ] Rust tutorials for packages 01-08

### Phase 4: Validation

- [ ] Red team all 83 Python tutorials against SDK source
- [ ] Red team all exercise solutions
- [ ] Verify three-format consistency
- [ ] Check no hardcoded keys/secrets

## Module Mapping (v4)

| v4 Exercise   | Source                                       | Status                 |
| ------------- | -------------------------------------------- | ---------------------- |
| M1 ex_1 (1.1) | NEW: Python basics + weather                 | Written                |
| M1 ex_2 (1.2) | NEW: Filter HDB                              | Written                |
| M1 ex_3 (1.3) | NEW: Functions + aggregation                 | Written                |
| M1 ex_4 (1.4) | ADAPTED from old M1 ex_1 (joins)             | Adapted                |
| M1 ex_5 (1.5) | ADAPTED from old M1 ex_1 (windows)           | Adapted                |
| M1 ex_6 (1.6) | NEW: Visualization                           | Written                |
| M1 ex_7 (1.7) | ADAPTED from old M1 ex_3 (profiling)         | Adapted                |
| M1 ex_8 (1.8) | ADAPTED from old M1 ex_5 (cleaning)          | Adapted                |
| M2 ex_1 (2.1) | MOVED from M1 ex_2 (Bayesian)                | Moved + header updated |
| M2 ex_2 (2.2) | NEW: MLE/MAP                                 | Written                |
| M2 ex_3 (2.3) | MOVED from M1 ex_4 (Hypothesis)              | Moved + header updated |
| M2 ex_4 (2.4) | NEW: Bootstrap                               | Written                |
| M2 ex_5 (2.5) | EXISTING M2 ex_3 (CUPED)                     | Renumbered             |
| M2 ex_6 (2.6) | EXISTING M2 ex_4 (DiD)                       | Renumbered             |
| M2 ex_7 (2.7) | EXISTING M2 ex_1 (Feature eng)               | Renumbered             |
| M2 ex_8 (2.8) | EXISTING M2 ex_2 (Feature store)             | Renumbered             |
| M3 ex_1 (3.1) | NEW: Bias-variance                           | Written                |
| M3 ex_2 (3.2) | EXISTING M3 ex_1 (Grad boost)                | Renumbered             |
| M3 ex_3 (3.3) | EXISTING M3 ex_2 (Imbalance)                 | Renumbered             |
| M3 ex_4 (3.4) | EXISTING M3 ex_3 (SHAP + LIME)               | Adapted                |
| M3 ex_5 (3.5) | EXISTING M3 ex_4 (Workflow)                  | Renumbered             |
| M3 ex_6 (3.6) | NEW: DataFlow persistence                    | Written                |
| M3 ex_7 (3.7) | EXISTING M3 ex_5 (Registry/HPO)              | Renumbered             |
| M3 ex_8 (3.8) | EXISTING M3 ex_6 (Production)                | Renumbered             |
| M4 ex_1 (4.1) | EXISTING M4 ex_1 (Clustering)                | Kept                   |
| M4 ex_2 (4.2) | NEW: EM/GMMs                                 | Written                |
| M4 ex_3 (4.3) | NEW: PCA/UMAP                                | Written                |
| M4 ex_4 (4.4) | EXISTING M4 ex_2 (Anomaly)                   | Adapted                |
| M4 ex_5 (4.5) | EXISTING M4 ex_3 (NLP)                       | Adapted                |
| M4 ex_6 (4.6) | EXISTING M4 ex_4 (Drift)                     | Renumbered             |
| M4 ex_7 (4.7) | EXISTING M4 ex_5 (DL)                        | Adapted                |
| M4 ex_8 (4.8) | EXISTING M4 ex_6 (Capstone)                  | Adapted                |
| M5 ex_1 (5.1) | EXISTING (Delegate)                          | Kept                   |
| M5 ex_2 (5.2) | EXISTING (CoT)                               | Kept                   |
| M5 ex_3 (5.3) | EXISTING (ReAct)                             | Kept                   |
| M5 ex_4 (5.4) | EXISTING (RAG)                               | Kept                   |
| M5 ex_5 (5.5) | NEW: MCP server                              | Written                |
| M5 ex_6 (5.6) | EXISTING M5 ex_5 (ML agents)                 | Renumbered             |
| M5 ex_7 (5.7) | REWRITTEN (multi-agent with formal patterns) | Rewritten              |
| M5 ex_8 (5.8) | NEW: Nexus deployment                        | Written                |
| M6 ex_1 (6.1) | EXISTING (SFT)                               | Kept                   |
| M6 ex_2 (6.2) | EXISTING (DPO)                               | Kept                   |
| M6 ex_3 (6.3) | EXISTING M6 ex_5 (RL)                        | Renumbered             |
| M6 ex_4 (6.4) | NEW: Advanced alignment/merging              | Written                |
| M6 ex_5 (6.5) | EXISTING M6 ex_3 (PACT)                      | Renumbered             |
| M6 ex_6 (6.6) | EXISTING M6 ex_4 (Governed)                  | Renumbered             |
| M6 ex_7 (6.7) | NEW: Agent governance at scale               | Written                |
| M6 ex_8 (6.8) | EXISTING M6 ex_6 (Capstone)                  | Adapted                |
