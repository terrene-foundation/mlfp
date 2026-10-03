# mlfp03 — handoffs from the exercise shard (S1, merged 340f3bb6)

## Content corrections the teaching material must follow
- Credit data: `customer_id` and the planted leak `future_default_indicator` (r≈0.96 with default) are DROPPED from every credit loader (shared/mlfp03 `CREDIT_NON_FEATURE_COLUMNS`); ex_4 now teaches students how the leak is found (leak screen). Every credit example in deck/textbook/lessons/notes must drop them and may describe the leak screen. 33 features remain.
- ex_3 churn: the label-defining recency column (`days_since_last_order`) is removed from features.
- Lesson 3.2: learning curves read the standard way (large train/val gap = HIGH VARIANCE).
- Lesson 3.4: tuning/early stopping on a validation split; test used once.
- Lesson 3.5: focal loss is real focal loss; calibration fitted on a held-out 20% via `TrainingPipeline.calibrate` (not cv=5); threshold tuned out-of-fold.
- Lesson 3.6: model trained with `TrainingPipeline`; SHAP additivity in log-odds; fairness audited on race, gender, age (four-fifths: race 0.94, gender 0.96 pass; age band 0.02 fails — routed to human/risk-committee sign-off).
- Lesson 3.7: workflows use real nodes — `PythonCodeNode`, `SwitchNode`; errors are no longer swallowed.
- Lesson 3.8: DriftMonitor needs its OWN SQLite file (sharing the registry DB fails: "no column named id"); DataFlow DriftCheck CRUD; KS threshold 0.001 (Bonferroni-tightened); decision threshold 0.130 = c_FP/(c_FP+c_FN); rollback = archived → staging → production (there is no `registry.promote`; use `promote_model`); model-card fairness numbers are measured; readiness = 11 computed gates + human sign-off; conformal guarantee is marginal and needs exchangeability.
- Deck/lesson 3.8 Module 4 preview must match Module 4's exercises.
- No real organisation names credited with invented figures.

## Integration (S7)
- Regenerate Colab notebooks for every changed file (all of mlfp03).
- Execute: ex_1/01–05, ex_2/01–05, ex_3/01–06, ex_4/01–04, ex_5/01–05, ex_7/01–05, ex_8/01–05 (ex_8 in order 01→05; ex_8/02 heavy). ex_6/01–05 and ex_8/03–05 already ran locally and pass.

## Deferred spec gaps (S6)
- ex_1: temporal features, polynomial/interaction features, forward/backward selection, correlation-threshold filter, FeatureEngineer.
- ex_5: stacking/blending; regression metrics (R², MAE, RMSE, MAPE); log loss in the metrics taxonomy.
- ex_6: KernelSHAP demo; ModelExplainer (its additivity fails on this model — investigate).

## Not changed (justified)
- ex_5.1–5.4 call LightGBM directly: TrainingPipeline has no sample-weight / init-score hook.

## Warnings to triage in S7 (zero-tolerance)
- LightGBM 4.x + sklearn "X does not have valid feature names" (TrainingPipeline fits on unnamed numpy → features `Column_N`).
- SHAP: LightGBM output is now a list.
- DataFlow `close_async` leaves pool tasks pending at exit.
- kailash-ml registry `_kml_drift_reports` table conflicts with DriftMonitor's.
