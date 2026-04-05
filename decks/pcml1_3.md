---
marp: true
theme: default
paginate: true
header: "ASCENT — Professional Certificate in Machine Learning"
footer: "© 2026 Terrene Foundation | Terrene Open Academy"
---

# Lesson 1.3: Functions and Aggregation

### Module 1: Data Pipelines and Visualisation

---

## Learning Objectives

By the end of this lesson, you will be able to:

- Define reusable functions with `def` and `return`
- Use `for` loops to iterate over collections
- Aggregate data with `group_by()` and `agg()`
- Combine multiple aggregation expressions in a single call

---

## Recap: Lesson 1.2

- Boolean expressions evaluate to True/False
- `pl.col()` references columns in Polars expressions
- `filter()` selects rows; `select()` and `with_columns()` shape columns
- `sort()` orders data; operations chain together

---

## Why Functions?

Without functions, you repeat code:

```python
mean_ang = df.filter(pl.col("town") == "ANG MO KIO")["price"].mean()
mean_bed = df.filter(pl.col("town") == "BEDOK")["price"].mean()
mean_tam = df.filter(pl.col("town") == "TAMPINES")["price"].mean()
```

With a function, you write it once:

```python
def mean_price(df, town):
    return df.filter(pl.col("town") == town)["price"].mean()
```

---

## Defining Functions: `def`

```python
def greet(name):
    return f"Welcome to ASCENT, {name}!"

message = greet("Alice")
print(message)  # Welcome to ASCENT, Alice!
```

Structure:

- `def` keyword, function name, parameters in parentheses
- Indented body (4 spaces)
- `return` sends a value back to the caller

---

## Functions with Multiple Parameters

```python
def price_summary(df, town, flat_type):
    subset = df.filter(
        (pl.col("town") == town) & (pl.col("flat_type") == flat_type)
    )
    return {
        "count": len(subset),
        "mean": subset["price"].mean(),
        "median": subset["price"].median(),
    }

result = price_summary(df, "TAMPINES", "4 ROOM")
print(result)
```

---

## Default Parameters

```python
def top_flats(df, n=5, descending=True):
    return df.sort("price", descending=descending).head(n)

# Use defaults
top_flats(df)              # top 5, most expensive first

# Override defaults
top_flats(df, n=10)        # top 10, most expensive first
top_flats(df, descending=False)  # top 5, cheapest first
```

Defaults make functions flexible without requiring every argument.

---

## `for` Loops

```python
towns = ["TAMPINES", "BEDOK", "WOODLANDS"]

for town in towns:
    count = len(df.filter(pl.col("town") == town))
    print(f"{town}: {count:,} transactions")
```

Output:

```
TAMPINES: 1,234 transactions
BEDOK: 987 transactions
WOODLANDS: 1,456 transactions
```

---

## `for` Loops with `range()`

```python
# Print numbers 0 to 4
for i in range(5):
    print(i)

# Loop through rows by index
for i in range(min(5, len(df))):
    row = df.row(i)
    print(row)
```

`range(n)` generates numbers from 0 to n-1.

---

## List Comprehensions (Preview)

A compact way to build lists from loops:

```python
# Traditional loop
means = []
for town in towns:
    m = df.filter(pl.col("town") == town)["price"].mean()
    means.append(m)

# List comprehension — same result, one line
means = [
    df.filter(pl.col("town") == town)["price"].mean()
    for town in towns
]
```

---

## Aggregation: The Power of `group_by()`

```python
town_stats = df.group_by("town").agg(
    pl.col("price").mean().alias("avg_price"),
)
print(town_stats.sort("avg_price", descending=True))
```

`group_by()` splits data into groups, then `agg()` computes summaries per group.

```
┌───────────┬────────────┐
│ town      ┆ avg_price  │
│ ---       ┆ ---        │
│ str       ┆ f64        │
╞═══════════╪════════════╡
│ BUKIT TI..┆ 650000.0   │
│ CENTRAL ..┆ 580000.0   │
│ ...       ┆ ...        │
└───────────┴────────────┘
```

---

## Multiple Aggregations

```python
town_stats = df.group_by("town").agg(
    pl.col("price").mean().alias("avg_price"),
    pl.col("price").median().alias("median_price"),
    pl.col("price").std().alias("std_price"),
    pl.col("price").count().alias("num_transactions"),
    pl.col("floor_area").mean().alias("avg_area"),
)
```

Pass multiple expressions to `agg()` for a comprehensive summary.

---

## Grouping by Multiple Columns

```python
detailed = df.group_by("town", "flat_type").agg(
    pl.col("price").mean().alias("avg_price"),
    pl.col("price").count().alias("count"),
)

print(detailed.sort("avg_price", descending=True).head(10))
```

Grouping by two columns gives statistics for each combination (e.g., "TAMPINES + 4 ROOM").

---

## Common Aggregation Functions

| Expression                        | Description              |
| --------------------------------- | ------------------------ |
| `pl.col("x").mean()`              | Average                  |
| `pl.col("x").median()`            | Median (50th percentile) |
| `pl.col("x").std()`               | Standard deviation       |
| `pl.col("x").min()` / `.max()`    | Minimum / Maximum        |
| `pl.col("x").sum()`               | Total                    |
| `pl.col("x").count()`             | Number of values         |
| `pl.col("x").n_unique()`          | Distinct values          |
| `pl.col("x").first()` / `.last()` | First / Last value       |

---

## Combining Functions and Aggregation

```python
def town_report(df, town):
    stats = (
        df
        .filter(pl.col("town") == town)
        .group_by("flat_type")
        .agg(
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").count().alias("count"),
        )
        .sort("avg_price", descending=True)
    )
    print(f"\n=== {town} ===")
    print(stats)

for town in ["TAMPINES", "BEDOK", "WOODLANDS"]:
    town_report(df, town)
```

---

## Exercise Preview

**Exercise 1.3: Town Comparison Report**

You will:

1. Write functions to compute price statistics per town
2. Use `group_by()` and `agg()` for multi-level summaries
3. Loop through towns to generate a formatted comparison report
4. Identify the most and least expensive flat types per town

Scaffolding level: **Heavy (~70% code provided)**

---

## Common Pitfalls

| Mistake                                | Fix                                            |
| -------------------------------------- | ---------------------------------------------- |
| Forgetting `return` in a function      | Function returns `None` by default             |
| Wrong indentation                      | Python uses indentation for blocks, not braces |
| `group_by()` without `agg()`           | Must call `agg()` to get results               |
| Forgetting `alias()` on aggregations   | Columns get auto-named (e.g., `price_mean`)    |
| Modifying a list while looping over it | Create a new list instead                      |

---

## Summary

- `def` creates reusable functions; `return` sends values back
- `for` loops iterate over collections; `range()` generates number sequences
- `group_by().agg()` is the core Polars aggregation pattern
- Multiple expressions in `agg()` produce multi-column summaries
- Functions + aggregation = reusable, composable analysis

---

## Next Lesson

**Lesson 1.4: Joins and Multi-Table Data**

We will learn:

- Python dictionaries for structured data
- `if`/`else` conditional logic
- Joining DataFrames with `join()` (left and inner joins)
