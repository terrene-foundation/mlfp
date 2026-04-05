---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.5: Window Functions and Trends

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Apply window functions using `over()` for group-aware calculations
- Compute rolling averages with `rolling_mean()`
- Use `shift()` to compare values across time periods
- Understand lazy frames and how they optimise computation

---

## Recap: Lesson 1.4

- Dictionaries store key-value pairs for lookups
- `if`/`elif`/`else` and `when().then().otherwise()` handle conditional logic
- `join()` combines tables: `inner` for matches only, `left` for all primary data
- Multi-key joins match on multiple columns simultaneously

---

## What Are Window Functions?

Window functions compute values **within groups** without collapsing rows.

```
group_by().agg()     → one row per group (collapse)
over()               → one row per original row (preserve)
```

Think of it as: "calculate per group, but stamp the result back onto every row."

---

## Window Functions with `over()`

```python
# Average price PER TOWN, stamped onto every row
df_windowed = df.with_columns(
    pl.col("price").mean().over("town").alias("town_avg_price")
)
```

```
┌──────────┬────────┬────────────────┐
│ town     ┆ price  ┆ town_avg_price │
│ TAMPINES ┆ 480000 ┆ 465000         │
│ TAMPINES ┆ 450000 ┆ 465000         │  ← same avg for group
│ BEDOK    ┆ 380000 ┆ 395000         │
│ BEDOK    ┆ 410000 ┆ 395000         │
└──────────┴────────┴────────────────┘
```

---

## Deviation from Group Mean

```python
df_analysis = df.with_columns(
    pl.col("price").mean().over("town").alias("town_avg"),
).with_columns(
    (pl.col("price") - pl.col("town_avg")).alias("price_deviation"),
    (
        (pl.col("price") - pl.col("town_avg"))
        / pl.col("town_avg") * 100
    ).alias("pct_deviation"),
)
```

"How much does each flat deviate from its town average?"

---

## Ranking Within Groups

```python
df_ranked = df.with_columns(
    pl.col("price")
      .rank(descending=True)
      .over("town")
      .alias("price_rank_in_town")
)

# Top 3 most expensive flats per town
top3 = df_ranked.filter(pl.col("price_rank_in_town") <= 3)
```

`rank().over()` ranks rows within each group independently.

---

## Time Series Data

HDB prices have a temporal dimension. Sorting by date unlocks trend analysis.

```python
df_sorted = df.sort("transaction_date")

print(df_sorted.select("transaction_date", "price").head())
```

```
┌──────────────────┬────────┐
│ transaction_date ┆ price  │
│ 2020-01          ┆ 350000 │
│ 2020-01          ┆ 420000 │
│ 2020-02          ┆ 365000 │
└──────────────────┴────────┘
```

---

## Rolling Averages: `rolling_mean()`

A rolling average smooths out noise to reveal trends.

```python
monthly_avg = (
    df
    .group_by("transaction_date")
    .agg(pl.col("price").mean().alias("avg_price"))
    .sort("transaction_date")
)

smoothed = monthly_avg.with_columns(
    pl.col("avg_price")
      .rolling_mean(window_size=3)
      .alias("rolling_3m_avg")
)
```

A 3-month rolling average: each point is the mean of the current and 2 prior months.

---

## Visualising Rolling Averages (Concept)

```
Price ($)
  |
  |        ╱╲    ╱╲
  |   ╱╲  ╱  ╲  ╱  ╲     ← Raw monthly average
  |  ╱  ╲╱    ╲╱    ╲
  | ╱                 ╲
  |╱                   ╲
  |
  |     ──────────────     ← Rolling 3-month average
  |   ╱                ╲      (smoother trend)
  |──╱                  ╲──
  └─────────────────────────→ Time
```

Rolling averages reduce volatility while preserving direction.

---

## Temporal Shifts: `shift()`

`shift()` moves values forward or backward in time.

```python
df_shifted = monthly_avg.with_columns(
    pl.col("avg_price").shift(1).alias("prev_month_price"),
    pl.col("avg_price").shift(-1).alias("next_month_price"),
)

# Month-over-month change
df_mom = df_shifted.with_columns(
    (pl.col("avg_price") - pl.col("prev_month_price")).alias("mom_change"),
)
```

`shift(1)` = previous row's value; `shift(-1)` = next row's value.

---

## Percentage Change

```python
df_pct = monthly_avg.with_columns(
    pl.col("avg_price").shift(1).alias("prev"),
).with_columns(
    ((pl.col("avg_price") - pl.col("prev")) / pl.col("prev") * 100)
      .alias("pct_change")
)
```

"How much did the average price change compared to last month?"

---

## Lazy Frames: Optimised Computation

```python
# Eager (immediate execution)
result = df.filter(...).with_columns(...).sort(...)

# Lazy (deferred, optimised execution)
result = (
    df.lazy()
    .filter(...)
    .with_columns(...)
    .sort(...)
    .collect()     # ← triggers execution
)
```

Lazy mode lets Polars **optimise** the entire pipeline before running it.

---

## Why Lazy Frames Are Faster

```
Eager:  filter → with_columns → sort → filter
        (each step materialises a full DataFrame)

Lazy:   filter → with_columns → sort → filter
        (Polars combines filters, skips unused columns,
         parallelises operations, then executes once)
```

The query optimiser can:

- Push filters down (process less data)
- Eliminate unused columns
- Parallelise independent operations

---

## Lazy Frame Example

```python
result = (
    df.lazy()
    .filter(pl.col("town") == "TAMPINES")
    .with_columns(
        pl.col("price").mean().over("flat_type").alias("type_avg"),
    )
    .with_columns(
        (pl.col("price") - pl.col("type_avg")).alias("deviation"),
    )
    .sort("deviation", descending=True)
    .head(10)
    .collect()
)
```

Call `.lazy()` at the start, `.collect()` at the end.

---

## When to Use Lazy Frames

| Scenario                       | Use                          |
| ------------------------------ | ---------------------------- |
| Quick exploration (small data) | Eager -- simpler, immediate  |
| Production pipelines           | Lazy -- optimised, faster    |
| Large datasets (>1M rows)      | Lazy -- memory efficient     |
| Chaining many operations       | Lazy -- optimiser shines     |
| Debugging intermediate steps   | Eager -- `print()` each step |

---

## Exercise Preview

**Exercise 1.5: HDB Price Trend Analyser**

You will:

1. Compute town-level window statistics with `over()`
2. Calculate rolling 3-month and 6-month averages
3. Use `shift()` to measure month-over-month price changes
4. Build a lazy frame pipeline for the full analysis

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                             |
| -------------------------------------- | ----------------------------------------------- |
| Forgetting `.collect()` on lazy frames | Result is a `LazyFrame`, not a `DataFrame`      |
| Unsorted data for rolling functions    | Always `sort()` by time before `rolling_mean()` |
| Wrong `shift()` direction              | `shift(1)` = previous; `shift(-1)` = next       |
| `over()` vs `group_by()` confusion     | `over()` preserves rows; `group_by()` collapses |

---

## Summary

- `over()` computes per-group values without collapsing rows
- `rolling_mean()` smooths time series to reveal trends
- `shift()` enables period-over-period comparisons
- Lazy frames optimise multi-step pipelines automatically
- Sort by time before any temporal operation

---

## Next Lesson

**Lesson 1.6: Data Visualization**

We will learn:

- Creating interactive charts with Plotly
- Using `ModelVisualizer` from Kailash ML
- Choosing the right chart for the right question
