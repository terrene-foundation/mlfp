# MLFP05 — Task 2: Transfer Learning + ONNX Deployment

**Difficulty**: Medium
**Weight**: 35%
**Dataset**: CIFAR-10 (50k train / 10k test, 32×32 colour, 10 classes)
**Time**: 45-75 minutes
**Target**: test accuracy ≥ 0.55 AND PyTorch/ONNX prediction parity on 100 images

## Problem

You will:

1. Load a pre-trained `torchvision.models.resnet18` with ImageNet weights.
2. **Freeze** the backbone and train a new classifier head on a 2000-image
   CIFAR-10 training subset with ImageNet-normalised inputs.
3. Export the model to **ONNX**.
4. Implement a `predict` callable that runs **ONNX Runtime** (not PyTorch) for
   inference so the grader can check PyTorch/ONNX parity.

This mirrors the ship path a real ML engineer takes: train in PyTorch, serve
through an ONNX runtime so the production system does not need torch.

## Required interface

```python
from pathlib import Path
import torch
from typing import Callable

def solve(onnx_path: Path) -> tuple[torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Train the model, export to `onnx_path`, then return:
        model:   the trained torch.nn.Module (in eval mode)
        predict: a callable(images: torch.Tensor) -> torch.Tensor
                 where images has shape (N, 3, 32, 32) in [0, 1] and the
                 returned tensor has shape (N,) with int64 class labels.
                 predict() MUST run the ONNX model via onnxruntime, not
                 the torch.nn.Module.
    """
```

The grader passes `onnx_path = task_dir / "student.onnx"` and will delete the
file before every run.

## Visible sanity check

- `onnx_path` exists and is ≤ 60 MB after `solve()`
- The model is `resnet18`-shaped (i.e. its state dict contains a key
  ending in `layer4.1.conv2.weight`) — this is how we verify you actually
  loaded a pretrained backbone rather than shipping a one-line linear model
- The student's `predict` on 100 test images matches the direct PyTorch
  predictions from the same returned `model` on at least 95/100 images
- Test accuracy on a hidden 1000-image slice ≥ 0.55

## Grading

1. Call `solve(onnx_path)`.
2. Check `onnx_path` exists, size ≤ 60 MB.
3. Check the returned model's state dict contains a `resnet18` key.
4. Build a 1000-image held-out test slice using
   `torchvision.datasets.CIFAR10(train=False)` with ImageNet normalisation.
5. Run the student's `predict()` — this is the ONNX path.
6. Run the model directly via `model(...).argmax(1)` — this is the PyTorch path.
7. Check: PyTorch/ONNX agreement on 100 images ≥ 0.95.
8. Check: ONNX test accuracy ≥ 0.55.

Budget: 6 minutes wall-clock on a 2024 laptop CPU.

## Hints

- `torchvision.models.resnet18(weights="IMAGENET1K_V1")` loads the backbone
- Freeze: `for p in backbone.parameters(): p.requires_grad_(False)`
- Replace `backbone.fc` with `nn.Linear(512, 10)` for 10 CIFAR-10 classes
- ImageNet normalisation means subtracting `[0.485, 0.456, 0.406]` and
  dividing by `[0.229, 0.224, 0.225]` per channel
- Export with `torch.onnx.export(model, dummy, onnx_path, input_names=["input"],
  output_names=["logits"], opset_version=17, dynamic_axes={"input": {0: "N"}})`
- Use `onnxruntime.InferenceSession(str(onnx_path))` to load and run inference
- Because you're training only the head (≈5k params), 1 epoch is enough to
  clear 55% — the backbone already knows ImageNet features
