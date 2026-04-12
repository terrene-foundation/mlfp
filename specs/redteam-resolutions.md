# Red-Team Resolutions

All issues identified during the red-team and completeness audit, with their resolutions applied in v2.

Source: mlfp-curriculum-v2.md, Part III.

## Resolutions (v2)

| Issue | Resolution in v2 |
|---|---|
| C1: M3.3 overloaded (5 model families) | Split into M3.3 (model zoo) + M3.4 (gradient boosting) |
| C2: SVM/KNN/NB missing | Added to M3.3 |
| C3: Dropout/batch norm missing | Added to M4.8 (DL Training Toolkit) |
| C4: Prompt engineering missing | Added to M6.1 |
| C5: M6.2 attempts 10 techniques | Restructured: 2 deep (LoRA, Adapters) + 8 survey |
| H1-H4: DL training mechanics scattered | Consolidated in M4.8 Training Toolkit section |
| H5: Feature selection missing | Added to M3.1 |
| H6: GRPO missing | Added to M6.3 alongside DPO |
| H7: Model merging missing | Added to M6.2 |
| H8: LLM eval missing | Added to M6.3 |
| H9: M2.6 logistic + ANOVA too dense | Logistic prioritised, ANOVA scoped to one-way only |
| H10: M5.8 RL no transition | Added explicit bridge segment |
| M4.7->4.8 bridge implicit | Made explicit opening segment in M4.8 |
| M5.1 autoencoders 9 variants | Scoped to 4 deep + 5 survey |
| M5 consolidation missing | Added comparison segment after M5.4 (Transformers) |

## Version History

| Version | Date | Changes |
|---|---|---|
| v1 | 2026-04-09 | Initial spec based on user's 6-module structure |
| v2 | 2026-04-09 | Red-team + completeness audit incorporation. Added: learning objectives per lesson, key formulas, assessment criteria, DL training toolkit, prompt engineering, GRPO, model merging, SVM/KNN/NB, feature selection, evaluation benchmarks. Split M3.3. Restructured M6.2. |
