# MLFP01 — Task 2: HDB Year-over-Year Price Changes in Ang Mo Kio

**Difficulty**: Medium
**Dataset**: `data/mlfp01/hdb_resale.parquet` (50K rows of HDB resale transactions 2015-2024)
**Time**: 30-45 minutes

## Problem

The HDB resale dataset contains every resale transaction for HDB flats in Singapore. Relevant columns:

- `month` (String) — transaction month in `"YYYY-MM"` format
- `town` (String) — HDB town name (e.g., `"ANG MO KIO"`, `"BEDOK"`)
- `flat_type` (String) — e.g., `"3 ROOM"`, `"4 ROOM"`, `"5 ROOM"`
- `resale_price` (Int64) — sale price in SGD

Implement a function `solve()` that computes, for **ANG MO KIO only**, the **year-over-year (YoY) median resale price change by flat type**. Specifically:

1. Filter to rows where `town == "ANG MO KIO"`.
2. Derive a `year` column (Int64) from the first 4 characters of `month`.
3. For each `(year, flat_type)`, compute the **median** `resale_price`.
4. For each `flat_type`, sort by `year`, and add a `prev_year_median` column using a **lag window** (the previous year's median for the same flat type).
5. Compute `yoy_pct_change = 100 * (median_price - prev_year_median) / prev_year_median`.
6. Drop rows where `prev_year_median` is null (the earliest year for each flat type).
7. Return the final DataFrame sorted by `flat_type`, then `year` ascending.

## Input / Output

```python
def solve() -> pl.DataFrame:
    ...
```

**Returns**: a Polars DataFrame with **exactly these 5 columns in this order**:

| Column             | Type    | Meaning                                      |
| ------------------ | ------- | -------------------------------------------- |
| `flat_type`        | String  | HDB flat type                                |
| `year`             | Int64   | Transaction year                             |
| `median_price`     | Float64 | Median resale price for that year + type     |
| `prev_year_median` | Float64 | Previous year's median (lag-1 within flat_type) |
| `yoy_pct_change`   | Float64 | `100 * (median - prev) / prev`               |

Rows where `prev_year_median` is null must be dropped.

## Visible test case

After running your solution:

- `result` has column names exactly `["flat_type", "year", "median_price", "prev_year_median", "yoy_pct_change"]`
- Every value in `prev_year_median` is not null
- For `flat_type == "4 ROOM"`, `year == 2016`, `prev_year_median` equals the 2015 median
- `result.height` is roughly `(num_flat_types * 9)` (9 = years 2016-2024)

## Hints

- Use `pl.col("month").str.slice(0, 4).cast(pl.Int64)` to extract the year.
- Use `.over("flat_type")` on a sorted window to compute lags within each flat type.
- `pl.col("median_price").shift(1).over("flat_type")` gives lag-1 within group, but you must sort the DataFrame by `(flat_type, year)` first.

## Grading

The grader calls `solve()` and checks:

1. Return type is a Polars DataFrame
2. Column names match the spec exactly
3. No nulls in `prev_year_median`
4. `yoy_pct_change` for (`4 ROOM`, 2016) matches the ground truth (recomputed independently)
5. `yoy_pct_change` for (`5 ROOM`, 2020) matches the ground truth
6. Row count equals the expected value
