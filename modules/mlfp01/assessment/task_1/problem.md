# MLFP01 — Task 1: Monthly Weather Statistics

**Difficulty**: Easy
**Dataset**: `data/mlfp01/sg_weather.csv` (12 rows, 3 columns — one row per calendar month)
**Time**: 15-25 minutes

## Problem

The Singapore weather dataset contains one row per calendar month, with columns:

- `month` (String) — calendar month name (e.g., `"January"`, `"February"`)
- `mean_temperature_c` (Float64) — mean temperature in Celsius for that month
- `total_rainfall_mm` (Int64) — total rainfall in millimetres for that month

Implement a function `solve()` that returns a Polars `DataFrame` with the following **exact schema**, sorted by `month` in its original calendar order:

| Column                   | Type    | How to compute                                        |
| ------------------------ | ------- | ----------------------------------------------------- |
| `month`                  | String  | Passed through                                        |
| `mean_temperature_c`     | Float64 | Passed through                                        |
| `total_rainfall_mm`      | Int64   | Passed through                                        |
| `temp_deviation_c`       | Float64 | `mean_temperature_c - annual_mean_temperature`        |
| `rainfall_vs_mean_pct`   | Float64 | `100 * (total_rainfall_mm - annual_mean_rainfall) / annual_mean_rainfall` |
| `is_wet_month`           | Boolean | `True` if `total_rainfall_mm` > annual mean rainfall |

`annual_mean_temperature` and `annual_mean_rainfall` are computed over all 12 rows.

## Input / Output

```python
def solve() -> pl.DataFrame:
    ...
```

**Returns**: a Polars DataFrame with exactly 12 rows and exactly 6 columns named above.

## Visible test case

After running your solution:

- `result.shape == (12, 6)`
- `result.columns == ["month", "mean_temperature_c", "total_rainfall_mm", "temp_deviation_c", "rainfall_vs_mean_pct", "is_wet_month"]`
- `result["is_wet_month"].sum()` is between 4 and 8 (about half the months are above average)
- The row where `month == "January"` has `temp_deviation_c` close to `-0.93` (within 0.1)

## Grading

The grader calls `solve()` and checks:

1. Return type is a Polars DataFrame
2. Row count is 12
3. Column names match the spec exactly
4. `temp_deviation_c` sums to approximately 0 (within 0.01)
5. `is_wet_month` count matches a hidden expected value
6. Specific row values match the hidden ground truth (January deviation, wettest month rainfall)
