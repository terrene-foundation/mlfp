# MLFP01 — Task 2: HDB Feature Table with Town Enrichment

**Weight**: 20 marks · **Datasets**: `data/mlfp01/hdb_resale.parquet` (50,150 rows),
`data/mlfp_assessment/mrt_stations.parquet`, `data/mlfp_assessment/schools.parquet`
**Outcomes assessed**: types and string parsing (1.1), derived columns (1.2),
functions and aggregation (1.3), joins and key reconciliation (1.4)

## Scenario

A valuation team wants one model-ready feature row for every HDB resale record.
Some of the features are locked inside text columns that were typed in by hand
from paper records. Others come from two lookup tables, the MRT station list and
the school list. Those tables were produced by other agencies and were not
designed to join to the resale data.

Inspect all three tables before you write any transformation. Nobody has listed
the problems for you.

## Interface

```python
def engineer_features(
    hdb: pl.DataFrame, mrt: pl.DataFrame, schools: pl.DataFrame
) -> pl.DataFrame: ...
```

The three arguments have the same columns and dtypes as the three files. The
function works only on the frames it is given: it must not load files itself.
It will also be run on **unseen variants** of the three tables, with different
rows, stations and schools. It must therefore derive everything from its
inputs. Do not hard-code per-town numbers.

## Required output (acceptance criteria)

Exactly **one row per input resale record**. No record may be dropped, and no
join may duplicate a record. Columns (any order; extras allowed):

| Column                                                | Type     | Meaning                                                                                                                                                                            |
| ----------------------------------------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `row_id`                                              | Int      | position of the record in `hdb` (0-based), so each output row can be traced to its source row                                                                                      |
| `town`, `flat_type`, `floor_area_sqm`, `resale_price` | as input | passed through unchanged                                                                                                                                                           |
| `sale_year`                                           | Int      | calendar year of the sale                                                                                                                                                          |
| `storey_midpoint`                                     | Float    | middle floor of the flat's storey band (band 07–09 → 8.0). Every record has a readable band                                                                                        |
| `flat_type_rooms`                                     | Int      | number of rooms: 2 ROOM → 2 … 5 ROOM → 5; EXECUTIVE counts as 6; MULTI-GENERATION as 7                                                                                             |
| `flat_age_years`                                      | Int      | age of the flat in the year of sale, in whole years. A flat cannot be sold before its lease commenced; for such a contradictory record the age is unknown (null)                   |
| `remaining_lease_years`                               | Float    | remaining lease in years, with months as a fraction of a year. Where it was not recorded, use the statutory 99-year lease minus `flat_age_years`; null only if that is unknown too |
| `price_per_sqm`                                       | Float    | resale price per square metre                                                                                                                                                      |
| `mrt_station_count`                                   | Int      | number of distinct MRT stations in the flat's town. One station served by several lines is one station. A town with no station in the table has 0                                  |
| `school_count`                                        | Int      | number of schools in the flat's town. A town with no school in the table has 0                                                                                                     |

Town names are not written identically across the three tables. Match a town
whenever the lookup table covers it, even if it is spelled or styled
differently.

## How you are graded

The grader computes the expected value of every feature for every record. It
matches your rows to source records by `row_id`, so row order does not matter.

| #   | Check                                                                     |
| --- | ------------------------------------------------------------------------- |
| G1  | Gate: a DataFrame with every required column                              |
| G2  | Gate: exactly one row per input record (`row_id` covers 0…n−1 once each)      |
| G3  | Gate: pass-through columns unchanged                                           |
| 1   | `sale_year` and `price_per_sqm` correct for every record                       |
| 2   | `storey_midpoint` correct for every record                                     |
| 3   | `flat_type_rooms` correct for every record                                     |
| 4   | `flat_age_years` correct, including the unknown ages                           |
| 5   | `remaining_lease_years` correct, including the imputed and unknown leases      |
| 6   | `mrt_station_count` correct for every record                                   |
| 7   | `school_count` correct for every record                                        |
| 8   | Unseen variant of the three tables: every feature correct                      |

Marks = 20 × (checks 1–8 passed / 8), and 0 if a gate fails.

## Rules

- **Polars only.** No pandas. Deterministic.
- Run `starter.py` to try your function on the real tables before submitting.
