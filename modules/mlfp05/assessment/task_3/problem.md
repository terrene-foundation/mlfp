# MLFP05 — Task 3: Sequence Forecasting on Real Market Data

**Difficulty**: Hard
**Weight**: 45%
**Dataset**: Straits Times Index (^STI) daily closes via `yfinance`
**Time**: 60-90 minutes
**Target**: Walk-forward val MSE strictly below a naive baseline

## Problem

You are given ~14 years of real daily OHLCV bars for the Straits Times Index
(fetched by `yfinance` on first run and cached to
`data/mlfp05/sti/sti_close.parquet`). The grader constructs a strict
**walk-forward split**: the first 80% of trading days are training data, the
last 20% are validation. You must build a recurrent model that predicts the
next day's z-score-normalised close from a 10-day window of features, trained
**only** on the training portion.

The model is graded on two things: (1) the architectural choice — your model
must be recurrent (LSTM or GRU) with gradient clipping, not a pure MLP, and
(2) the outcome — your walk-forward validation MSE must be strictly below a
naive "tomorrow equals today" baseline computed on the same split.

This replicates the real-world sequence-modelling trap: any model that simply
copies the last observed value passes the smell test on a short split, so the
grader requires you to **beat** the copy-last baseline on held-out data.

## Required interface

```python
import torch
import polars as pl
from typing import Callable

def solve() -> tuple[torch.nn.Module, Callable[[pl.DataFrame], torch.Tensor]]:
    """
    Returns:
        model: a trained torch.nn.Module in eval mode. MUST contain at least
               one nn.LSTM or nn.GRU module in its children.
        predict: a callable(df: pl.DataFrame) -> torch.Tensor
                 where df has the same schema yfinance returns (columns at
                 least ["Date", "Close", "High", "Low", "Volume"], sorted
                 ascending by Date). Returns a 1-D float tensor of length
                 (len(df) - 10), one z-score-normalised prediction per
                 10-day rolling window, lined up so that the t-th prediction
                 corresponds to day t+10.
    """
```

Normalisation note: both training and prediction MUST z-score features using
statistics computed **only on the first 80% of the input frame** (the same
training slice the grader will use). Using future statistics is a data leak
and the grader will detect it by feeding a longer frame.

## Visible sanity check

- `model` contains at least one `nn.LSTM` or `nn.GRU` child
- `predict(sti_df)` returns a 1-D float `torch.Tensor` of length `len(df) - 10`
- On the hidden walk-forward split, the student's model validation MSE is
  strictly less than the "copy-last" naive baseline MSE

## Grading

1. Load `data/mlfp05/sti/sti_close.parquet` (fetched by `ex_3.py` earlier).
2. Call `student.solve()`.
3. Check the model contains an LSTM or GRU child module.
4. Check `predict(full_df)` returns a 1-D tensor of the expected length.
5. Compute the walk-forward 80/20 split on the full frame.
6. Compute the **naive baseline** MSE: predict normalised close at t+10
   equals the normalised close at t+9 (the last value in the window).
7. Compute the **student** MSE: run `predict(full_df)` and take only the
   indices that fall in the val slice, MSE against the true normalised closes.
8. Pass when student MSE < naive baseline MSE (strictly).
9. Check that gradient clipping is applied during training (grep the source
   for `clip_grad_norm_` or `clip_grad_value_`).

Budget: 6 minutes wall-clock on a 2024 laptop CPU.

## Hints

- Window size 10, features `["Close", "High", "Low", "Volume"]`, target =
  next-day normalised close
- `nn.LSTM(input_size=4, hidden_size=32, num_layers=1, batch_first=True)`
- Train for ~20 epochs with Adam(lr=1e-3), clip grads at 1.0
- Use the last hidden state (`h_n[-1]`) fed into a `nn.Linear(32, 1)` head
- Naive baseline: `pred_t = normalised_close[t+9]` — a single-line numpy op
- Don't use pandas — polars only
