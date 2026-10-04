# MLFP01 — Task 1: Taxi Trip Data Forensics

**Weight**: 20 marks · **Dataset**: `data/mlfp01/sg_taxi_trips.parquet` (50,000 raw rows, 12 columns)
**Outcomes assessed**: data types and parsing (1.1), filtering and derived columns
(1.2), null handling and deterministic cleaning (1.8)

## Scenario

A ride-hailing operator merged the trip logs of three dispatch systems and
extracted the result on **1 January 2025**. The analytics team cannot use the
log until it meets the data contract below.

Nobody has listed what is wrong with the log. Finding the problems is part of
the task: inspect the data, decide what each problem is, and write a cleaning
function that enforces the contract.

Your function will be run on the full log **and on other extracts from the
same dispatch systems that you have not seen**. It must enforce the contract
by rule. A function that remembers particular trip IDs, row positions or
counts from this file will fail on the unseen extracts.

## Interface

```python
def clean_trips(raw: pl.DataFrame) -> pl.DataFrame: ...
```

`raw` has the same 12 columns and dtypes as the parquet file. `clean_trips`
works only on the frame it is given: it must not load files itself and must not
modify `raw` in place.

## The data contract (acceptance criteria)

**Output columns.** One row per usable trip, with at least these 14 columns
(any order; extra columns are allowed):

- the 12 source columns, with `pickup_datetime` and `dropoff_datetime` as
  Polars `Datetime`
- `trip_duration_min` (Float): minutes from pickup to dropoff
- `avg_speed_kmh` (Float): `distance_km` divided by the duration in hours

**A usable trip** is a record that describes a trip that could really have
happened. All of the following must hold:

1. it was picked up before the log was extracted (before 2025-01-01 00:00:00);
2. its fare is positive and it carried at least one passenger;
3. its pickup point lies inside Singapore — latitude 1.15 to 1.47 and longitude
   103.60 to 104.05, both inclusive;
4. its average speed is between 2 and 120 km/h, inclusive.

Where a recorded value is wrong but the correct value is unambiguous, **repair
it**. Do not discard a trip you can repair.

**Identity.** A `trip_id` identifies exactly one usable trip. Remove the records
that are not usable trips first. If two or more of the remaining records share
a `trip_id`, you cannot tell which one owns it, so keep none of them.

**Values.**

- `payment_type` holds exactly one of the four methods the operator accepts,
  `Card`, `Cash`, `NETS` or `Grab`, whatever spelling the dispatch system used.
- A missing tip means no tip was paid (`0.0`). A missing pickup or dropoff zone
  is recorded as `"Unknown"`.
- Every value that was already correct in the raw record is passed through
  unchanged.

## How you are graded (10 automated checks)

The grader computes its own expected result from the raw data. It never uses
numbers your code reports about itself.

| #   | Check                                                                    |
| --- | ------------------------------------------------------------------------ |
| 1   | All 14 required columns are present, with Datetime timestamps            |
| 2   | Full log: the set of kept `trip_id`s matches exactly, with no duplicates |
| 3   | Full log: `payment_type` is correct for every kept trip                  |
| 4   | Full log: repaired values are correct for every kept trip                |
| 5   | Full log: missing tips and zones are filled as the contract says         |
| 6   | Full log: `trip_duration_min` and `avg_speed_kmh` are correct            |
| 7   | Full log: the untouched values are unchanged                             |
| 8   | Unseen extract: the set of kept `trip_id`s matches exactly               |
| 9   | Unseen extract: `payment_type` and the repaired values are correct       |
| 10  | `raw` is not modified by your function                                   |

Marks = 20 × (checks passed / 10). A check that cannot run because the output
is unusable counts as failed.

## Rules

- **Polars only.** No pandas.
- Deterministic: no random sampling, and the same input always gives the same
  output.
- Self-check before you submit: run `starter.py`. It loads the log, calls your
  function and prints a summary. Every number you quote in your own notes
  should come from running it.
