# mlfp02 — handoffs from the exercise shard (S1, merged ec982293)

## Content corrections the teaching material must follow
- SRM: the experiment was designed 40/35/15/10 (control/treatment_a/treatment_b/variant_c) — test SRM against the DESIGNED allocation, never 50/50; variant_c is the mis-allocated arm; analyses use control vs treatment_a (passes its own SRM, p≈0.62). A failed SRM forces "DO NOT SHIP".
- Conversion = an order of at least $50 (≈19.1% control vs 24.8% treatment) — not `metric_value > 0` (~99%).
- Profile likelihood re-fits sigma at each mean value (differs from Wald).
- Tukey HSD = studentized range (cross-checked vs scipy.stats.tukey_hsd), not Bonferroni pairwise t-tests.
- Valuing a single flat needs a PREDICTION interval, not the CI of the mean.
- Peeking: teach the simulated ≈25% false-positive rate, not the "independent peeks" 64%.
- CUPED uses only pre-treatment covariates.
- ex_4/03 cashback tier LOSES ≈$140/customer/year across the CI (old "clearly profitable" conclusion reversed).
- AIC → prediction; BIC → recovering the true model.
- FeatureStore: use `kailash_ml.features.FeatureSchema` (the top-level `kailash_ml.FeatureSchema` is rejected by FeatureStore); new cooling-measures panel helper in shared/mlfp02.
- ex_8 validates data (removes 3,536 sentinel rows; R² 0.22 → 0.85).
- Real companies/regulators with invented figures anonymised; figures illustrative.

## Integration (S7)
- Regenerate all mlfp02 Colab notebooks.
- Execute every solution except ex_1/03. Heavy: ex_3/01, ex_3/04 (10K resamples × 354K rows); ex_8/01–03 (feature store, minutes each).

## Open questions for review
- ex_1/ex_2 loaders keep sentinel prices ($10, $9M) → price SD ≈ $420K; ex_1/01 uses percentile intervals instead. Decide whether ex_1/ex_2 should clean them (as ex_8 does) or keep them deliberately.
- ex_6/02 re-optimised threshold 0.24 vs theoretical break-even 0.017 — printed explanation is "miscalibration"; verify.

## Deferred spec gaps (S6)
- 2.1–2.3: Poisson; Exponential fit + AIC; LLN demo; parametric bootstrap; one-sample & one-tailed tests; likelihood curve (ex_1/03).
- 2.5–2.8: log(price) model; k-fold CV; lat/lon features; multinomial logit; employee-attrition data (ex_6); FeatureEngineer generate/select; DiD placebo test; capstone options + logistic model; real cooling-measure data; TrainingPipeline.
- ExperimentTracker in the Train phase of ex_1–3, 5, 6.

## Upstream (kailash-ml) notes
- Top-level `kailash_ml.FeatureSchema` rejected by FeatureStore; re-registering a schema name as version 2 fails.

---
# Additions from the slides shard (S2, merged 3edfaa86)
- Lesson 2.6 worked example = HDB resale above-median target (as ex_6): OR ≈ 54 per SD of floor area, accuracy 0.81, AUC 0.92. Tukey = studentized range.
- Lesson 2.7: CUPED ρ = 0.7 → 49%; on the real experiment data (control vs treatment_a, pre-period covariate) the reduction is only ≈4% — say so. DiD on ex_7.4's simulated panel: estimate −23,208 vs true −20,000. SRM vs the designed split.
- Lesson 2.8: capstone = HDB resale valuation (wine option dropped — no dataset); 3,536 impossible rows removed, R² 0.828, floor area +9,096 per sqm.
- Lesson 2.4: ≈11,554 per arm (not 5,800).
- Textbook/notes pages 2.4–2.8 still carry old APIs, missing datasets, 51% CUPED, ~5,800/arm → S3/S4 must fix. speaker-notes.md regenerate (D4). index.html: old ICU card for 2.6 and wrong dataset table → integration.
- Owner question: spec says "End of Module Assessment: Quiz + mini-project" but the real assessment is auto-graded coding tasks (S5 will redesign; reconcile spec then).
- Lesson 2.4 slides use a "BOGO" case while the deck opens with the hawker-centre case (numbers now correct; framing differs).

---
# Additions from the lesson-textbook shard (S3b, merged 8add0d7f)
- BUG (integration): shared/mlfp02/ex_8.py FEATURE_STORE_URL = "sqlite:///mlfp02_ex8_features.db" (relative) → FeatureStore.materialize fails "unable to open database file" (reproduced twice). Exercise 8 + lesson 2.8 slides call create_feature_store() with that default → make the default an ABSOLUTE path (e.g. under OUTPUT_DIR resolved), then re-verify.
- HDB file has impossible leases (negative remaining-lease age); lesson 2.5 keeps them to match the slides (textbook says so) — S6/owner may decide to clean.
- Shared numbers (all match merged slides): 11,554/arm; 19.1% vs 24.8%; SRM p = 0.624; ρ = 0.21 → 4.4% CUPED reduction; ATT −23,208; area OR 54; acc 0.81; AUC 0.92; R² 0.828; +9,096/sqm; 3,536 rows removed.
- Textbook numbers (S3a, merged 6d83b0b7): posterior 846,770 ± 2,260; SRM χ² 13,889 all arms / control-vs-treatment_a p = 0.62; CUPED 4.4%; DiD −23,208 (pre-trend p = 0.56; broken panel p = 0.024); OR 54.4, acc 0.815, AUC 0.92; regression R² 0.860 (CV 0.860), out-of-time R² 0.831; power sizes 1,004 and 5,139; ROI $5,100/week ~10-week payback.

---
# Additions from the assessment shard (S5, merged 2479f959)
- Deck assessment slide + spec end line: "5 auto-graded coding tasks, 100 marks (20/25/25/15/15)" (P5).
- BUG shared/mlfp02/ex_8.py: storey regex misreads letter-O typos ("1O TO 12" → 1, "28 TO 3O" → 3) — fix in integration.
- shared/mlfp02/ex_8.py EXPERIMENT_STORE_URL is still a relative sqlite URL — make absolute (X-DF hygiene).
- ex_8 town medians average monthly medians; Task 5 uses median over all sales in the window (deliberate) — lesson text must not equate them.
- Upstream kailash-ml 2.2.2: FeatureStore.materialize fails on nulls even with nullable=True; ~12 rows/s writes; first
  ExperimentTracker.start_run ~18 s; process hangs at exit without close().
- New instructor-only grading_harness.py; starters use `raise NotImplementedError` as student scaffolds (zero-tolerance
  hook may need an exemption for assessment starters).
