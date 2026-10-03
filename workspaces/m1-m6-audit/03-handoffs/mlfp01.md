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
