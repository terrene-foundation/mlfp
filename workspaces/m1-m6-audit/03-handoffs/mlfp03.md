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

---
# Additions from the lesson-slides shard (S2b, merged 79d75ae3)
- Lesson 3.1 worked target = `long_stay` (ICU stay > median), not mortality; leak list updated.
- Lesson 3.2 worked example = noisy sine (MSEs recomputed), not HDB.
- Lesson 3.3 churn leaderboard = measured 5-fold CV on leak-free data + "always churn" baseline.
- Lesson 3.5: FN S$10,000 / FP S$1,500 → t* = 0.130; weighted booster → calibrate on held-out 20% of train → t* on calibrated probs (recalibration removes the weight shift — not double counting). PR-AUC rule of thumb: imbalance under 20% (speaker-notes.md:640 says 5% — align to 20%).
- Lesson 3.6: with unequal base rates any two fairness criteria conflict (except a perfect predictor); regulator claims softened to non-binding FEAT principles; SHAP waterfall in log-odds, illustrative.
- Lesson 3.8: registry stages staging/shadow/production/archived; drift monitor own DB; KS 0.001; t* 0.130 on the model card; rollback via promote_model; 11 gates + human sign-off; Module 4 completes the Foundation Certificate.
- Lesson notes + textbook pages still describe the old 3.1 target and old 3.3 leaderboard → S3/S4.
- Snippet checker: raw `<=` inside HTML code is stripped as a tag → write `&lt;=` (HTML-correct anyway).

---
# Additions from the deck shard (S2a, merged 8f6e256e)
- Deck assessment slide = 4 auto-graded tasks (20/25/25/30) — S5 redesign must update it if marks change.
- "Always churn" baseline: accuracy 0.745, F1 0.854, AUC 0.5. Stacking/blending via EnsembleEngine ≈ 0.79 AUC.
- TrainingPipeline accepts split_strategy="stratified_kfold" but scores only the first fold — say so.
- Brier score is NOT a pure calibration measure. Nested-CV fits = 5×(5K+1). No Free Lunch = Wolpert (1996).
- Owner questions: spec still says "Quiz + ML pipeline project" and HDB-based 3.1/3.2 exercises (intent, not stack).
- X-PP: shared/mlfp03 ex_2.load_credit_data, ex_3.build_train_test_split, ex_7.prepare_credit_frames still fit preprocessing on all rows before splitting — fix in the cross-cutting integration shard.
- Upstream: EnsembleEngine.stack/blend crash unless base models are pre-fitted; TrainingPipeline silently skips average_precision; LocalRuntime.execute() without a context manager → DeprecationWarning.

---
# Additions from the textbook shard (S3a, merged 80e1f1e3)
- OWNER DECISION: ICU data — only 198 of 416,526 vital-sign readings (4 admissions) fall in the first 24h, so ex_1's
  first-24h feature selection ranks noise (test AUC ≈ 0.50; best MI barely above a shuffled-target baseline). Options:
  regenerate the ICU dataset with dense early vitals, move the prediction cutoff, or reframe ex_1. Textbook teaches the finding honestly.
- Lesson pages: 3.1 → ICU long_stay + the window-report finding; 3.6 textbook uses an UNWEIGHTED LightGBM at t* = 0.130:
  DI ≥ 0.97 race/gender, 0.26 ages 21–34 — and cites ex_6's 0.94/0.96 (class-weighted model) — keep consistent;
  3.8 model card = 9 sections (Mitchell et al.) — lesson page says 7, change it.
- Leak screen: leak column alone AUC 0.99; leaky model 0.994. scale_pos_weight = n_neg/n_pos. C(53,3) = 23,426.
- PythonCodeNode sandbox refuses polars / lightgbm / shared imports (textbook says so).
- Upstream: LocalRuntime logs _record_execution_metrics error under skip_branches; HyperparameterSearch sampler unseeded
  (best params vary); TrainingPipeline F1 is support-weighted.

---
# Additions from the assessment shard (S5, merged e3dc1a50)
- Deck + spec assessment slide: "4 auto-graded coding tasks, 100 marks (20/20/30/30): leakage-free features;
  model zoo; evaluation/imbalance/interpretability; production registry/drift/deploy" (P5).
- Graders score against grader-held truth; reference passes all four (20/20/30/30); stubs and constant-SHAP
  plausible-wrong submissions score 0.
