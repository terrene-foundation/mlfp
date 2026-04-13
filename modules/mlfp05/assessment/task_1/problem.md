# MLFP05 — Task 1: Fashion-MNIST Image Classifier

**Difficulty**: Easy
**Weight**: 20%
**Dataset**: Fashion-MNIST (60k train / 10k test, 28×28 grayscale, 10 classes)
**Time**: 20-40 minutes
**Target**: ≥ 0.85 test accuracy on a hidden 5000-image test split

## Problem

Train a small CNN image classifier on Fashion-MNIST and return a trained model
plus a `predict(images)` callable. The grader will call your `solve()` and then
run `predict` on a held-out test split that you never see.

Fashion-MNIST is downloaded by `torchvision` and cached under
`data/mlfp05/fashion_mnist/`. Use a fixed seed so the grader can reproduce.

## Required interface

```python
import torch

def solve() -> tuple[torch.nn.Module, callable]:
    """
    Returns:
        model: a trained torch.nn.Module in eval() mode
        predict: a callable predict(images: torch.Tensor) -> torch.Tensor
                 where images has shape (N, 1, 28, 28) in [0, 1] and the
                 returned tensor has shape (N,) with int64 class labels
                 in 0..9
    """
    ...
```

## Visible sanity check

After `solve()`:

- `model` is an instance of `torch.nn.Module`
- `model.training is False` (i.e. `model.eval()` was called)
- The total number of parameters is between 5,000 and 5,000,000
  (disallows returning a linear model with no hidden capacity, and also
  disallows a 100M-parameter model that won't fit the 6-minute budget)
- `predict(torch.zeros(2, 1, 28, 28))` returns a `torch.Tensor` of shape `(2,)`
  with `dtype == torch.int64` and values in `0..9`

## Grading

The grader:

1. Calls `solve()` and validates the return type / shape / dtype
2. Loads Fashion-MNIST **test** split via `torchvision.datasets.FashionMNIST`
3. Runs `predict` on 5000 test images
4. Computes top-1 accuracy
5. Passes when accuracy ≥ 0.85

Budget: 6 minutes wall-clock on a 2024 laptop CPU.

## Hints

- A conv-pool-conv-pool-fc architecture trains in ~1 minute on CPU and reaches
  ~0.89 test accuracy in 3 epochs
- Use `torch.optim.Adam` with `lr=1e-3`
- `torch.nn.CrossEntropyLoss` on raw logits (no softmax in the head)
- Fix seeds before model creation: `torch.manual_seed(42); np.random.seed(42)`
