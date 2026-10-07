# MLFP05 — Task 3: One-Step-Ahead Load Forecast That Beats "Tomorrow = Today"

**Weight**: 25 marks · **Data**: synthetic generator provided in `starter.py`
· **Outcomes assessed**: recurrent architectures for sequence forecasting (5.3)

## Scenario

A grid operator forecasts electricity load one step ahead. Their fallback
forecast is **"tomorrow = today"** (the naive last-value forecast). Your job
is a recurrent model whose one-step-ahead error is genuinely lower — on load
series you have never seen.

The load series comes from a documented generator (a damped AR(2) process
with a daily seasonal component and noise). You train on one long series
from the generator; the grader regenerates **fresh series from the same
process with fresh secret seeds** and measures your returned model's error
against a naive forecast the grader computes itself. No number your code
reports about itself is read. An untrained recurrent net loses to the naive
forecast and fails.

## Interface

```python
def solve() -> dict: ...
```

Returns a dict with at least:

- `"model"`: your trained recurrent model, a `torch.nn.Module`;
- `"window"`: the look-back window length your model consumes (an int ≥ 8).

**Model contract**: input `(N, window, 1)` float32 (the last `window` load
readings); output `(N,)` or `(N, 1)` — the forecast for the next reading.

`starter.py` provides `make_series(seed)` (the generator) and
`make_windows(series, window)` (sliding-window pairs). Your architecture,
window length, and training recipe are yours to choose.

## Acceptance criteria (what the grader measures)

The grader draws two fresh series (fresh seeds), builds the windows itself,
runs **your returned model**, and computes the naive last-value forecast on
the same windows.

| #   | Check                                                                         |
| --- | ----------------------------------------------------------------------------- |
| 1   | A `torch.nn.Module` and an int `window` ≥ 8 are returned (gate)               |
| 2   | The model honours the input/output contract on grader-built windows (gate)    |
| 3   | A recurrent layer (`nn.GRU` / `nn.LSTM` / `nn.RNN`) fires in the forward pass |
| 4   | Eval-mode forward is deterministic                                            |
| 5   | Fresh series 1: your MSE ≤ 0.97 × grader's naive-forecast MSE                 |
| 6   | Fresh series 2: same bar, different series                                    |
| 7   | Your MSE ≤ a mean-value forecaster's MSE (series 1)                           |
| 8   | Your forecasts vary (not a constant)                                          |

Marks = 25 × (non-gate checks passed / 6). If a gate fails, the task scores 0.

## Rules

- Raw PyTorch; CPU only; fix your seeds; `solve()` finishes in ~3 minutes.
- The model must be recurrent (the point of the task). A plain feed-forward
  network that happens to beat naive fails check 3.
- Self-check: run `starter.py`; it prints your model's MSE versus naive on
  a development series.
