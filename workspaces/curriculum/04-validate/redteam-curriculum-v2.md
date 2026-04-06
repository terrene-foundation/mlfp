# Red Team Report: Expanded Curriculum v2 (48 Lessons)

**Reviewed by**: 3 panels (M1-M2, M3-M4, M5-M6), each with beginner + expert + trainer perspectives
**Date**: 2026-04-05

## Critical Finding: The Curriculum and Exercises Describe Different Courses

The expanded curriculum v2 reorganized modules to match official Terrene Open Academy titles:

- M5 curriculum = "Deep Learning and Vision"
- M6 curriculum = "Language Models and Agentic Workflows"

But the existing exercises were built for the OLD structure:

- M5 exercises = Kaizen agents (Delegate, CoT, ReAct, RAG, ML agents, A2A)
- M6 exercises = Alignment + Governance + RL + Capstone

**The exercises are more mature than the curriculum. The curriculum must be updated to match the exercises, then gaps filled.**

## Consolidated Verdict Table (All 48 Lessons)

### Module 1: Data Pipelines and Visualisation Mastery with Python

| Lesson | Verdict    | Issue                                                                    | Fix                                                                             |
| ------ | ---------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| 1.1    | CRITICAL   | Teaching Python variables via WorkflowBuilder is backwards for beginners | Start with plain Polars + Kailash engine calls, not raw workflow infrastructure |
| 1.2    | NEEDS WORK | Functions + full Node architecture too dense                             | Focus on Python functions; defer Node subclassing                               |
| 1.3    | NEEDS WORK | Decorators too early (week 3 for zero-Python)                            | Move decorators/custom nodes to M3 (when classes are solid)                     |
| 1.4    | GOOD       | Remove lazy evaluation to 1.5                                            | Minor tweak                                                                     |
| 1.5    | GOOD       | Strong lesson, real data at scale                                        | Add lazy evaluation here                                                        |
| 1.6    | GOOD       | Defer ML-specific chart types to M3                                      | Focus on EDA charts only                                                        |
| 1.7    | CRITICAL   | async/await unrealistic 6 lessons after learning variables               | Provide async as scaffolding; students configure + interpret, not write async   |
| 1.8    | NEEDS WORK | ConnectionManager premature                                              | Remove; focus on PreprocessingPipeline capstone                                 |

### Module 2: Statistical Mastery for ML and AI Success

| Lesson | Verdict    | Issue                                             | Fix                                                    |
| ------ | ---------- | ------------------------------------------------- | ------------------------------------------------------ |
| 2.1    | NEEDS WORK | Exponential family too abstract to start          | Start with concrete distributions + conjugate examples |
| 2.2    | GOOD       | Add MLE failure cases                             | Minor addition                                         |
| 2.3    | GOOD       | Add Bonferroni vs BH-FDR guidance                 | Minor addition                                         |
| 2.4    | GOOD       | Strong lesson                                     | —                                                      |
| 2.5    | GOOD       | Add multi-armed bandits from brief                | Minor addition                                         |
| 2.6    | NEEDS WORK | Too much scope (DiD + propensity + Rubin + Pearl) | Focus on DiD + propensity; defer Pearl DAGs to reading |
| 2.7    | GOOD       | Add leakage detection exercise                    | Good enhancement                                       |
| 2.8    | GOOD       | Add project menu (3-4 options)                    | Good enhancement                                       |

### Module 3: Supervised ML for Building and Deploying Models

| Lesson | Verdict    | Issue                                                        | Fix                              |
| ------ | ---------- | ------------------------------------------------------------ | -------------------------------- |
| 3.1    | NEEDS WORK | Content overload; interop elevated too high                  | Demote interop to lab setup      |
| 3.2    | GOOD       | Minor: LightGBM early stopping deprecation                   | Update API                       |
| 3.3    | GOOD       | Solid                                                        | —                                |
| 3.4    | NEEDS WORK | Promises LIME/PDP/ALE; delivers only SHAP                    | Add LIME or adjust curriculum    |
| 3.5    | CRITICAL   | Placement disrupts ML flow; solution lacks advanced patterns | Move to end of M1 or start of M3 |
| 3.6    | NEEDS WORK | No standalone exercise; async/ORM jump unscaffolded          | Create dedicated exercise        |
| 3.7    | GOOD       | Comprehensive                                                | —                                |
| 3.8    | GOOD       | Strong capstone                                              | —                                |

### Module 4: Unsupervised ML and Advanced Techniques

| Lesson | Verdict    | Issue                                                | Fix                                |
| ------ | ---------- | ---------------------------------------------------- | ---------------------------------- |
| 4.1    | GOOD       | Minor: optional dep handling                         | —                                  |
| 4.2    | NEEDS WORK | Full EM promised; only 15 lines of GMM               | Create dedicated EM exercise       |
| 4.3    | NEEDS WORK | No PCA exercise, shallow UMAP/t-SNE                  | Add PCA reconstruction exercise    |
| 4.4    | GOOD       | EnsembleEngine.blend() not called                    | Use actual SDK call                |
| 4.5    | GOOD       | Foundation representations skipped                   | Add TF-IDF warmup                  |
| 4.6    | GOOD       | Strong governance framing                            | —                                  |
| 4.7    | NEEDS WORK | Synthetic noise instead of real data                 | Use structured data or real images |
| 4.8    | CRITICAL   | Wrong import; no auth implemented; placement problem | Fix import; move Nexus auth to M5  |

### Module 5: Deep Learning / LLMs and Agents

| Lesson | Verdict    | Issue                                             | Fix                                                   |
| ------ | ---------- | ------------------------------------------------- | ----------------------------------------------------- |
| 5.1    | NEEDS WORK | Backprop 4hrs ambitious                           | Scope to 2-layer net + one grad descent loop          |
| 5.2    | NEEDS WORK | First PyTorch + ResNet too steep                  | Split or add PyTorch primer                           |
| 5.3    | GOOD       | Practical, well-scoped                            | —                                                     |
| 5.4    | NEEDS WORK | Flash Attention too deep for in-class             | Demote to conceptual; focus on implementing attention |
| 5.5    | NEEDS WORK | 5 topics that are each a full lecture             | Scope to minimal transformer block                    |
| 5.6    | GOOD       | Good consolidation                                | —                                                     |
| 5.7    | CRITICAL   | MCP servers have nothing to do with deep learning | Move to M4 or M6                                      |
| 5.8    | GOOD       | Strong capstone                                   | —                                                     |

### Module 6: Language Models and Agentic Workflows

| Lesson | Verdict    | Issue                                         | Fix                                           |
| ------ | ---------- | --------------------------------------------- | --------------------------------------------- |
| 6.1    | NEEDS WORK | LLM theory + full Kaizen core in one lesson   | Split or pre-read theory                      |
| 6.2    | CRITICAL   | 11 agents in 4 hours impossible               | Teach 5 core; rest as homework                |
| 6.3    | GOOD       | RAG well-scoped                               | —                                             |
| 6.4    | CRITICAL   | 9 patterns absurd for one session             | Teach 3-4 production patterns; reference rest |
| 6.5    | NEEDS WORK | 6 ML agents + journey system overloaded       | Focus on 4-agent pipeline                     |
| 6.6    | CRITICAL   | LoRA+QLoRA+SFT+DPO+GRPO+merging = full course | Split into SFT+LoRA and DPO+eval sessions     |
| 6.7    | NEEDS WORK | Regulatory + PACT SDK + MCP + CLI too wide    | Pre-read regulatory; in-class PACT only       |
| 6.8    | CRITICAL   | RL + capstone in one lesson                   | Separate RL; capstone is pure integration     |

## Summary Statistics

| Verdict    | Count | Percentage |
| ---------- | ----- | ---------- |
| GOOD       | 18    | 37.5%      |
| NEEDS WORK | 18    | 37.5%      |
| CRITICAL   | 12    | 25.0%      |

## Top 5 Structural Actions Required

1. **Reconcile curriculum with exercises** — update curriculum to match what exercises teach, then fill gaps
2. **Resolve beginner vs professional tension** — user says start from zero; official programme says "intermediate"; exercises assume professional
3. **Create 14 missing exercises** (2 per module) to reach 48
4. **Move content between modules** — MCP from M5, Nexus auth from M4, workflow orchestration earlier
5. **Split overloaded lessons** — M6.2 (agents), M6.4 (patterns), M6.6 (alignment) each need 2 sessions

## Recommended Curriculum v3 Actions

The curriculum needs a v3 revision that:

1. Accepts the user's direction (start from basics, end at masters)
2. Matches the official Terrene Open Academy module titles
3. Aligns with the existing 34 exercises (which are solid and reviewed)
4. Adds 14 new exercises for expanded lessons
5. Moves misplaced content (statistics from M1 → M2, workflow orchestration → M1, MCP → M4)
6. Splits the 6 CRITICAL overloaded lessons into realistic 4-hour sessions
