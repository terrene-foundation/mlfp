# mlfp01 — handoffs from the exercise shard (S1, merged 56bb0eb1)

For the slides (S2), textbook (S3) and notes (S4) shards, and integration (S7).

## Changed helper API (shared/run_profile.py) — update every call site
- `run_profile(df, alert_config=None)` → DataProfile (was taught as `run_profile(explorer, df, ...)`)
- `run_compare(df_a, df_b)` → dict
- `run_report(df, title=...)` → HTML (new; wraps DataExplorer.to_html)
- All work inside Jupyter/Colab (no bare asyncio.run).
Call sites: mlfp01 deck.html, lessons/07/slides.html, lessons/08/slides.html + textbook.html; mlfp02 lessons/08/slides.html + textbook.html.

## Content corrections the teaching material must follow
- Weekday: Polars `dt.weekday()` is 1–7, Monday = 1; weekend is `>= 6`. textbook.md ~3314 and ~3528.
- "Distance to MRT": the MRT table's distance column is spacing between neighbouring stations; HDB data has no flat coordinates. ex_4 now uses town-level features — station count (MRT access), typical station spacing, station centroid, and haversine distance to the CBD (the feature that tracks price, r ≈ −0.55). Replace `distance_to_mrt_km` / `nearest_mrt` / walkability framing in textbook.md, deck.html, lessons/04/slides.html + textbook.html; specs/module-1.md 1.4 wording.
- ex_8 capstone now works on the real taxi schema (timestamps parsed, duration derived, `fare_sgd` target, planted defects: swapped coords, negative fares, passengers < 1, 2027 trips, 15 payment spellings, duplicate trip_ids) and compares original vs cleaned. Lesson 1.8 walkthroughs should follow it.
- README.md: Exercise 8 title = "Data Pipelines and End-to-End Project".
- Synthetic datasets are described as synthetic/illustrative, not attributed to agencies.

## Integration (S7)
- Regenerate all 8 mlfp01 Colab notebooks; the generator must inline shared.run_profile (run_profile, run_compare, run_report) — ex_8 imports them.
- Execute all 8 solutions (ex_1–ex_8) on the fleet.

## Deferred spec gaps (S6)
- ex_5 seasonality task; ex_8 REST/API extraction; ex_1 the three describe() questions; ex_4 an outer join; ex_7 async vs sync-wrapper structure (ex_7 teaches async on purpose — decide in S6).

## Upstream note
- `ModelVisualizer()` emits an ExperimentalWarning from kailash-ml (not fixable in course code).

---
# Additions from the slides shard (S2, merged 1db350ae)

For the textbook (S3) and notes (S4) shards — follow the corrected slides:
- PreprocessingPipeline: `setup()` fits on ALL rows before splitting (see cross-cutting.md X-PP) — teach "hold out test rows first". ex_8 Task 8 comment and the Lesson 1.8 textbook claim setup() learns from train only: WRONG, correct it.
- Lesson 1.1: describe() min/max on the month column are alphabetical, not coolest/hottest month; weather values recomputed from the real 12-row file.
- "HDB Flash Crash" opening is labelled illustrative; real anomalies in the data: 107 sales at S$10, 144 at S$9M, median S$849,126.
- Lesson 1.4: `station_count` (and CBD distance) instead of distance-to-MRT; joins upper-case the town key and collapse MRT to one row per town (naive case-fixed join explodes to 186,997 rows).
- Lesson 1.5: build a complete town × month calendar before shift(12)/rolling; synthetic data has NO seasonality — teach testing the hypothesis.
- Lesson 1.6: area explains ≈22% of price (r = 0.47), not 40–50%; Plotly takes Polars directly (no pandas).
- Lesson 1.7: rebuilt on sg_cpi.csv (as ex_7): three date formats in one column; nulls at exactly 5% don't fire (rule is > 0.05); "8 alert types" are the real eight; outliers are a per-column statistic, not an alert.
- Lesson 1.8: taxi demand is flat by hour (1,917–2,054 trips each hour) — no invented rush-hour peaks.
- speaker-notes.md: regenerate from the deck's notes (most were rewritten).
- Open (S6): spec 1.1 says weather CSV ~1K rows; the file has 12 — ship a daily dataset or change the spec.
