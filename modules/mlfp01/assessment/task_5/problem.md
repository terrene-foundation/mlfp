# MLFP01 — Task 5: From Clean Trips to Model Inputs and Charts

**Weight**: 25 marks · **Data**: clean taxi trips (the Task 1 data contract), supplied by the grader
**Outcomes assessed**: ETL and `PreprocessingPipeline` (1.8); chart selection
and honest charts with `ModelVisualizer` / Plotly (1.6); interpreting results
(1.3, 1.6)

## Scenario

The taxi operator wants two things from the clean trip table.

- **Part A: model inputs.** A fare model will quote a price **at booking
  time**. Prepare the inputs for it with the kailash-ml `PreprocessingPipeline`.
- **Part B: a chart pack.** Management asked four questions. Answer each with
  one chart, plus two short findings.

Your functions receive frames that the grader prepares. They do not depend on
your Task 1 answer. To try them locally, clean the log with your own Task 1
function.

## Part A: interface and acceptance criteria

```python
def fit_preprocessor(train: pl.DataFrame) -> object: ...
def prepare_bookings(fitted: object, bookings: pl.DataFrame) -> pl.DataFrame: ...
```

- `train` holds clean **completed** trips. It has every column of the Task 1
  contract, including the fare (`fare_sgd`), which is the model's target.
- `bookings` holds **new bookings** that the grader kept back. You never see
  them. They contain only what is known when a trip is booked: `trip_id`,
  `pickup_datetime` (Datetime), `pickup_zone`, `dropoff_zone`, `distance_km`
  (the planned route), `passengers`, `payment_type`, `pickup_latitude` and
  `pickup_longitude`. They do **not** contain the fare.
- `fit_preprocessor` learns every preprocessing rule from `train`, and from
  nothing else. `prepare_bookings` applies those learned rules to `bookings`.

`prepare_bookings` must return:

1. exactly one row per booking, all columns numeric, no missing values, with
   the trip distance still represented;
2. no feature built from an identifier;
3. a feature that represents the **time of day** of the booking;
4. numeric features scaled with statistics learned **only from the training
   trips**.

Use `PreprocessingPipeline` for the imputation, encoding and scaling. Deciding
which columns are legitimate inputs, and what to derive first, is your job.

## Part B: interface and acceptance criteria

```python
def make_charts(trips: pl.DataFrame) -> dict: ...
```

`trips` is the clean trip table. Return
`{"figures": {...}, "findings": {...}}`, where `figures` maps each key below to
**one Plotly figure with one trace**. `ModelVisualizer` returns Plotly figures;
`plotly.express` accepts Polars frames directly.

| Key                 | Management's question                                                                          |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| `fare_distribution` | How are fares spread out? Is there a long tail of expensive trips? (every trip's fare)         |
| `fare_vs_distance`  | How does the fare change as the trip gets longer? (every trip; distance on the x-axis)         |
| `zone_median_fare`  | Which pickup zones have the highest **median** fare? (every known pickup zone, one value each) |
| `monthly_trips`     | How has the number of trips per month changed over the period? (every month)                   |

Every chart must use a chart type that suits its question and must plot the
real data. It must also be honest and readable:

- both axes carry a title;
- categories are ordered by their value;
- a value axis that encodes length starts at zero;
- time runs left to right.

`findings` must hold:

- `top_median_fare_zone`: the known pickup zone with the highest median fare;
- `busiest_month`: the month with the most trips, written as `"YYYY-MM"`.

## How you are graded

The grader cleans the log itself, splits the clean trips into training trips
and held-back bookings, calls your functions, and inspects what they return:
the model inputs, and the data inside each figure.

| #   | Check                                                                                 |
| --- | ------------------------------------------------------------------------------------- |
| 1   | A: one complete numeric row per held-back booking, with distance represented          |
| 2   | A: no identifier-derived feature                                                      |
| 3   | A: time of day represented                                                            |
| 4   | A: scaling learned from the training trips only                                       |
| 5   | B: `fare_distribution`: a suitable chart type showing every fare                      |
| 6   | B: `fare_vs_distance`: a suitable chart type showing every trip                       |
| 7   | B: `zone_median_fare`: a suitable chart type, correct medians, ordered, zero baseline |
| 8   | B: `monthly_trips`: a suitable chart type, chronological, correct counts              |
| 9   | B: both findings correct                                                              |

Marks = 25 × (checks passed / 9).

## Rules

- **Polars only.** No pandas. Use `PreprocessingPipeline` for Part A and
  `ModelVisualizer` and/or `plotly.express` for Part B.
- Deterministic. Run `starter.py` to try your functions before submitting.
