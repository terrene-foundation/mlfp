# MLFP01 — Task 3: Town Price Trends with Correct Time Alignment

**Weight**: 15 marks · **Dataset**: `data/mlfp01/hdb_resale.parquet` (50,150 rows, 2015-01 to 2024-12)
**Outcomes assessed**: grouped aggregation (1.3), window functions, year-on-year
change and rolling statistics with correct time alignment (1.5)

## Scenario

A housing analyst wants a monthly price-trend table for every HDB town. The
table must compare like with like. A year-on-year change has to compare a
month with the **same calendar month one year earlier**, not with "the row 12
places up". A rolling average has to cover **calendar months**, not a number of
rows.

The analyst also warns you that the price column contains recording errors:
prices that no real HDB resale could have. They must not enter any statistic.
Finding them, and deciding how to recognise them by rule, is up to you.

## Interface

```python
def town_trends(hdb: pl.DataFrame) -> pl.DataFrame: ...
```

`hdb` has the same columns and dtypes as the parquet file. The function works
only on the frame it is given. It will also be run on an **unseen variant** of
the data, with different months missing for some towns and new recording
errors. It must apply rules and must not remember this file.

## Required output (acceptance criteria)

One row per **(town, calendar month) that has at least one genuine sale**.
Columns (any order; extras allowed):

| Column                | Type  | Meaning                                                                                                                        |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| `town`                | Str   | as in the input                                                                                                                |
| `month`               | Date  | first day of the calendar month                                                                                                |
| `n_sales`             | Int   | number of genuine sales in that town and month                                                                                 |
| `median_price`        | Float | median genuine resale price                                                                                                    |
| `yoy_pct`             | Float | % change of `median_price` against the same town in the same calendar month one year earlier; null if that month has no row    |
| `rolling_3m_avg`      | Float | mean of the town's `median_price` over the three calendar months ending with this month, using only the months that have a row |
| `price_rank_in_month` | Int   | rank of `median_price` among the towns that have a row in that month: 1 = most expensive; tied towns share the lower rank      |

## How you are graded

The grader computes the expected table itself and matches rows by
`(town, month)`. Row order does not matter.

| #   | Check                                                                   |
| --- | ----------------------------------------------------------------------- |
| G1  | Gate: a DataFrame with every required column, `month` as a Date           |
| G2  | Gate: exactly the expected (town, month) rows, each once                  |
| 1   | `n_sales` correct (recording errors excluded)                             |
| 2   | `median_price` correct                                                    |
| 3   | `yoy_pct` correct, including its nulls                                    |
| 4   | `rolling_3m_avg` correct                                                  |
| 5   | `price_rank_in_month` correct                                             |
| 6   | Unseen variant (other gaps, new recording errors): every column correct   |

Marks = 15 × (checks 1–6 passed / 6), and 0 if a gate fails.

## Rules

- **Polars only.** No pandas. Use Polars expressions and window functions; no
  Python loops over rows.
- Deterministic. Run `starter.py` to try your function before submitting.
