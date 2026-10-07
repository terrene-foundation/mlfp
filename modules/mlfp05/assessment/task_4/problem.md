# MLFP05 — Task 4: Ship the Postcode Reader as an ONNX Artefact

**Weight**: 25 marks · **Data**: the 8×8 digits dataset bundled with
scikit-learn · **Outcomes assessed**: training a deployable classifier and
exporting it through the kailash-ml OnnxBridge (5.2, 5.7)

## Scenario

The mail-sorting machines do not run Python training frameworks — they run
ONNX Runtime. Your job is end to end: **train** a small classifier on the
flattened digits, **export** it to a portable `.onnx` artefact with the
kailash-ml `OnnxBridge`, and hand back both the PyTorch model and the
artefact.

The grader acts as the machine fleet: it loads your `.onnx` with
`onnxruntime` itself, feeds it **grader-held mail** (its own split of the
digits, fresh secret seed), and checks the artefact's answers two ways —
numerical parity with your PyTorch model, and accuracy against grader-held
labels. No number your code reports about itself is read. An untrained model
exports fine and fails the accuracy floor; an artefact that always answers
"3" fails the variety check.

## Interface

```python
def solve() -> dict: ...
```

Returns a dict with at least:

- `"model"`: the trained `torch.nn.Module` (eval-ready);
- `"onnx_path"`: `pathlib.Path` to the exported `.onnx` artefact;
- `"export_result"`: the object `OnnxBridge().export(...)` returned.

**Serving contract** (what the machines expect):

- one input, float32, shape `(batch, 64)` — the 8×8 image flattened,
  intensities scaled to `[0, 1]`;
- one output, float32, shape `(batch, 10)` — class logits.

`starter.py` provides `load_digits_flat()` (documented training split,
flattened, scaled). Architecture and recipe are yours — a compact MLP is
plenty for this data.

## Acceptance criteria (what the grader measures)

| #   | Check                                                                                                       |
| --- | ----------------------------------------------------------------------------------------------------------- |
| 1   | Dict with a `torch.nn.Module` and a path is returned (gate)                                                 |
| 2   | `export_result.success` is true **and** the `.onnx` file exists (gate)                                      |
| 3   | `onnxruntime.InferenceSession` loads the artefact; input is float32 `(batch, 64)`                           |
| 4   | Parity: max abs difference between torch and ONNX logits ≤ 1e-4 over five grader-built batches (sizes 1–64) |
| 5   | Artefact accuracy on the grader's held-out split ≥ 0.88                                                     |
| 6   | Artefact accuracy beats the grader's majority-class baseline                                                |
| 7   | Artefact's predicted classes on held-out mail span at least 5 classes                                       |
| 8   | The artefact is deterministic (same input twice → identical logits)                                         |

Marks = 25 × (non-gate checks passed / 6). If a gate fails, the task scores 0.

## Rules

- Export via `OnnxBridge().export(model, "torch", output_path=..., sample_input=...)`
  — `framework` is the string `"torch"`, and `sample_input` is a tensor with
  the serving shape `(1, 64)`. `export` returns a result object; it does not
  raise on failure, so read `.success`.
- Raw PyTorch for the model; CPU only; no pretrained weights or downloads.
- Fix your seeds; `solve()` finishes in ~3 minutes on a laptop CPU.
- Self-check: run `starter.py`; it loads your artefact with onnxruntime and
  prints parity and accuracy on your own validation slice.
