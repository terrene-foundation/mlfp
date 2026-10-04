# MLFP01 — Task 4: Profile, Clean and Justify with DataExplorer

**Weight**: 20 marks · **Dataset**: `data/mlfp01/economic_indicators.csv` (401 rows, 8 columns)
**Outcomes assessed**: automated profiling with `DataExplorer`, alert
configuration and justified cleaning decisions (1.7); parsing and type repair
(1.1); null handling (1.8)

## Scenario

An economics team keeps Singapore's macro indicators in one file that mixes
monthly and quarterly rows. They want a clean **quarterly** table, and they want
the data-quality work to be auditable. Every problem that the kailash-ml
`DataExplorer` flags must either be fixed or be explicitly accepted with a
reason.

Use `DataExplorer` to discover the problems. The `shared` helpers
`run_profile(df, alert_config=None)` and `run_compare(df_a, df_b)` run it
synchronously in a script, in Jupyter and in Colab. The task does not list the
problems for you.

## Interface

```python
def audit_indicators(raw: pl.DataFrame) -> dict: ...
```

`raw` has the same columns and dtypes as the CSV. The "raw quarterly slice" is
the rows of `raw` whose `period_type` is `"quarterly"`. The function works only
on the frame it is given. It will also be run on an **unseen variant** of the
file, with rows in another order, other gaps and other repeated records.

It returns a dict with exactly these keys:

| Key                 | Value                                                                                                                                                                                                                                                             |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `raw_alerts`        | `list[str]`: every alert `DataExplorer` (default thresholds) raises on the raw quarterly slice, each encoded as described below                                                                                                                                   |
| `cleaned`           | `pl.DataFrame`: the clean quarterly table (see acceptance criteria)                                                                                                                                                                                               |
| `imputation`        | `str`: the method you used to fill the missing indicator values. One of `"median"`, `"mean"` (one statistic of the column over the cleaned quarters), `"interpolate"` (linear in time between the neighbouring quarters) or `"forward_fill"` (last known quarter) |
| `null_alert_config` | an `AlertConfig` under which a column with **even one** missing value raises a `high_nulls` alert, whatever the table size, and a complete column raises none                                                                                                     |
| `accepted_alerts`   | `dict[str, str]`: for **every** alert `DataExplorer` (default thresholds) still raises on your `cleaned` frame, the encoded alert mapped to your reason for accepting it. No other entries                                                                        |
| `quality_delta`     | `dict`: `{"rows_removed": int, "nulls_filled": int}`, the rows removed and the missing values filled between the raw quarterly slice and `cleaned`                                                                                                                |

**Alert encoding.** An alert on one column is `"type:column"` (for example
`"constant:period_type"`). An alert on a pair of columns is `"type:a,b"`, with
the two names in alphabetical order. A whole-table alert is just `"type"`.

## Acceptance criteria for `cleaned`

- Exactly **one row per quarter** covered by the data. The same quarter must not
  appear twice, however it was written.
- Columns: `period_year` (Int), `period_quarter` (Int, 1–4),
  `gdp_growth_pct`, `unemployment_rate`, `inflation_rate`,
  `trade_balance_sgd_bn`, `property_price_index` (Float) and
  `tourist_arrivals` (Int64, the true integer count).
- No missing values. Gaps are filled with the method you declare in
  `imputation`, applied in time order. The grader recomputes your declared
  method and compares.
- Values that were present in the raw data are unchanged.
- No `high_nulls` or `duplicates` alert remains. Any other alert that remains
  must appear in `accepted_alerts` with a reason of at least one sentence.

Choose the imputation method that suits quarterly economic time series. In the
docstring of `audit_indicators`, say in two or three sentences why you chose it.
The docstring is read when your work is reviewed.

## How you are graded

| #   | Check                                                                         |
| --- | ----------------------------------------------------------------------------- |
| G1  | Gate: returns the dict contract, `cleaned` has every required column          |
| 1   | `raw_alerts` equals the alerts the grader gets by profiling the raw slice     |
| 2   | `cleaned` has exactly one row per quarter                                     |
| 3   | `tourist_arrivals` is Int64 with the correct counts                           |
| 4   | Values that were present are unchanged                                        |
| 5   | Gaps filled exactly as your declared `imputation` method would fill them      |
| 6   | `null_alert_config` fires on a single missing value in a large table          |
| 7   | `accepted_alerts` matches the grader's own re-profile of your `cleaned` frame |
| 8   | `quality_delta` is correct                                                    |
| 9   | Unseen variant of the file: the cleaned table is correct                      |

Marks = 20 × (checks 1–9 passed / 9), and 0 if the gate fails.

## Rules

- **Polars only** for data. No pandas. Profile with the kailash-ml
  `DataExplorer`, either directly or through `shared.run_profile` and
  `shared.run_compare`.
- Deterministic. Run `starter.py` to try your function before submitting.
