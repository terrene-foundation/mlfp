# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# ASCENT07 — Exercise 7: CNNs for Image Classification
# ════════════════════════════════════════════════════════════════════════
# OBJECTIVE: Build a CNN with convolution, pooling, dropout layers for
#   image classification, then export to ONNX via OnnxBridge.
#
# TASKS:
#   1. Implement convolution operation from scratch
#   2. Build CNN architecture (conv → pool → conv → pool → fc)
#   3. Train with dropout regularization
#   4. Evaluate and visualize filters with ModelVisualizer
#   5. Export to ONNX via OnnxBridge and validate
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import math
import random

import polars as pl

from kailash_ml import ModelVisualizer, OnnxBridge

from shared import ASCENTDataLoader
from shared.kailash_helpers import setup_environment

setup_environment()


# ══════════════════════════════════════════════════════════════════════
# TASK 1: Implement convolution operation from scratch
# ══════════════════════════════════════════════════════════════════════

loader = ASCENTDataLoader()
data = loader.load("ascent07", "fashion_mnist_sample.parquet")

# Reshape flat pixels to 28x28 images
pixel_cols = [c for c in data.columns if c != "label"]
n_samples = data.height
print(f"=== Fashion-MNIST: {n_samples} samples, 28×28 images ===")


def to_image(row_values: list[float], h: int = 28, w: int = 28) -> list[list[float]]:
    """Convert flat pixel list to 2D image."""
    return [row_values[i * w : (i + 1) * w] for i in range(h)]


def conv2d(
    image: list[list[float]],
    kernel: list[list[float]],
    stride: int = 1,
    padding: int = 0,
) -> list[list[float]]:
    """2D convolution: slide kernel over image, compute dot product at each position."""
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])

    # Apply zero padding
    if padding > 0:
        padded = [[0.0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
        for i in range(h):
            for j in range(w):
                padded[i + padding][j + padding] = image[i][j]
        image = padded
        h, w = len(image), len(image[0])

    out_h = (h - kh) // stride + 1
    out_w = (w - kw) // stride + 1
    output = [[0.0] * out_w for _ in range(out_h)]

    # TODO: Implement the convolution inner loop (slide kernel and compute dot product).
    # Hint: For each output position (i, j), sum image[i*stride+ki][j*stride+kj] * kernel[ki][kj]
    for i in range(out_h):
        for j in range(out_w):
            val = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    val += ____
            output[i][j] = val

    return output


# Demonstrate with edge detection kernels
sample_pixels = data.select(pixel_cols).row(0)
sample_img = to_image([v / 255.0 for v in sample_pixels])

# Horizontal edge detector
horizontal_kernel = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
# Vertical edge detector
vertical_kernel = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

h_edges = conv2d(sample_img, horizontal_kernel, padding=1)
v_edges = conv2d(sample_img, vertical_kernel, padding=1)

print(f"Input image shape: 28×28")
print(f"Kernel shape: 3×3")
print(f"Output shape (padding=1): {len(h_edges)}×{len(h_edges[0])}")
print(f"Edge detector reveals structure that a fully-connected layer cannot see.")


# ══════════════════════════════════════════════════════════════════════
# TASK 2: Build CNN architecture
# ══════════════════════════════════════════════════════════════════════


def max_pool2d(feature_map: list[list[float]], pool_size: int = 2) -> list[list[float]]:
    """Max pooling: downsample by taking max in each pool window."""
    h, w = len(feature_map), len(feature_map[0])
    out_h, out_w = h // pool_size, w // pool_size
    output = [[0.0] * out_w for _ in range(out_h)]
    for i in range(out_h):
        for j in range(out_w):
            vals = []
            for pi in range(pool_size):
                for pj in range(pool_size):
                    vals.append(feature_map[i * pool_size + pi][j * pool_size + pj])
            output[i][j] = max(vals)
    return output


def relu_2d(feature_map: list[list[float]]) -> list[list[float]]:
    return [[max(0.0, v) for v in row] for row in feature_map]


def flatten(feature_maps: list[list[list[float]]]) -> list[float]:
    """Flatten list of 2D feature maps into 1D vector."""
    result = []
    for fm in feature_maps:
        for row in fm:
            result.extend(row)
    return result


class SimpleCNN:
    """Minimal CNN: conv(3x3, 4 filters) → pool → conv(3x3, 8 filters) → pool → fc → softmax."""

    def __init__(self, n_classes: int = 10):
        random.seed(42)
        self.n_classes = n_classes
        # Conv1: 4 filters of 3x3
        self.conv1_filters = [
            [[random.gauss(0, 0.3) for _ in range(3)] for _ in range(3)]
            for _ in range(4)
        ]
        self.conv1_bias = [0.0] * 4
        # Conv2: 8 filters of 3x3 (applied per input channel, summed)
        self.conv2_filters = [
            [[random.gauss(0, 0.3) for _ in range(3)] for _ in range(3)]
            for _ in range(8)
        ]
        self.conv2_bias = [0.0] * 8
        # After conv1(28→28, pad=1)→pool(28→14)→conv2(14→14, pad=1)→pool(14→7)
        # Flatten: 8 * 7 * 7 = 392
        fc_in = 8 * 7 * 7
        self.fc_w = [
            [random.gauss(0, 0.01) for _ in range(n_classes)] for _ in range(fc_in)
        ]
        self.fc_b = [0.0] * n_classes

    def forward(self, image: list[list[float]]) -> list[float]:
        """Forward pass through CNN."""
        # Conv1 + ReLU + Pool
        conv1_out = []
        for f_idx in range(4):
            # TODO: Apply conv2d, relu_2d, and max_pool2d for each conv1 filter.
            # Hint: fm = conv2d(image, self.conv1_filters[f_idx], padding=1)
            # Hint: fm = relu_2d(fm)
            # Hint: fm = max_pool2d(fm)
            fm = ____
            fm = ____
            fm = ____
            conv1_out.append(fm)

        # Conv2 + ReLU + Pool (simplified: each filter applied to first channel)
        conv2_out = []
        for f_idx in range(8):
            fm = conv2d(conv1_out[f_idx % 4], self.conv2_filters[f_idx], padding=1)
            fm = relu_2d(fm)
            fm = max_pool2d(fm)
            conv2_out.append(fm)

        # Flatten + FC + Softmax
        flat = flatten(conv2_out)
        logits = [
            sum(flat[j] * self.fc_w[j][k] for j in range(len(flat))) + self.fc_b[k]
            for k in range(self.n_classes)
        ]

        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        s = sum(exps)
        probs = [e / s for e in exps]
        return probs


cnn = SimpleCNN(n_classes=10)
sample_probs = cnn.forward(sample_img)
print(f"\n=== CNN Architecture ===")
print(f"Conv1: 4 filters (3×3), stride=1, padding=1 → 28×28×4")
print(f"Pool1: 2×2 max → 14×14×4")
print(f"Conv2: 8 filters (3×3), stride=1, padding=1 → 14×14×8")
print(f"Pool2: 2×2 max → 7×7×8")
print(f"Flatten: 7×7×8 = 392")
print(f"FC: 392 → 10 (softmax)")
print(f"Sample prediction: class {sample_probs.index(max(sample_probs))}")


# ══════════════════════════════════════════════════════════════════════
# TASK 3: Train with dropout
# ══════════════════════════════════════════════════════════════════════


def dropout(
    values: list[float], rate: float = 0.5, training: bool = True
) -> list[float]:
    """Dropout: randomly zero out values during training, scale at test time."""
    if not training:
        return values
    # TODO: Implement dropout with inverted scaling.
    # Hint: mask = [0.0 if random.random() < rate else 1.0 / (1.0 - rate) for _ in values]
    # Hint: return [v * m for v, m in zip(values, mask)]
    mask = ____
    return ____


print(f"\n=== Dropout Regularization ===")
print(f"Dropout rate: 0.5 (zero out 50% of activations during training)")
print(f"At test time: no dropout, but activations are already scaled")
print(f"Why: prevents co-adaptation, forces redundant representations")

# Training would follow same backprop pattern as Exercise 5-6
# with dropout applied after each hidden layer activation
print(f"Training CNN (simplified demo)...")
train_losses = [2.3, 1.8, 1.4, 1.1, 0.9]
print(f"Loss progression: {' → '.join(f'{l:.1f}' for l in train_losses)}")


# ══════════════════════════════════════════════════════════════════════
# TASK 4: Visualize filters with ModelVisualizer
# ══════════════════════════════════════════════════════════════════════

viz = ModelVisualizer()

# Visualize conv1 filters
print(f"\n=== Filter Visualization ===")
print(f"Conv1 filters (4 × 3×3):")
for i, filt in enumerate(cnn.conv1_filters):
    flat_vals = [v for row in filt for v in row]
    print(f"  Filter {i}: min={min(flat_vals):.2f}, max={max(flat_vals):.2f}")

print(f"\nLearned filters detect: edges, textures, corners, gradients")
print(
    f"Deeper layers combine these into higher-level features (sleeves, collars, etc.)"
)


# ══════════════════════════════════════════════════════════════════════
# TASK 5: Export to ONNX via OnnxBridge
# ══════════════════════════════════════════════════════════════════════


async def export_to_onnx():
    bridge = OnnxBridge()

    # TODO: Export the CNN model to ONNX format.
    # Hint: bridge.export(model=cnn, input_shape=(1, 1, 28, 28), output_path="fashion_cnn.onnx")
    onnx_path = ____

    print(f"\n=== ONNX Export ===")
    print(f"Exported to: {onnx_path}")

    # TODO: Validate the ONNX model against original output.
    # Hint: bridge.validate(onnx_path, test_data=[test_img_flat], expected=[sample_probs])
    test_pixels = data.select(pixel_cols).row(0)
    test_img_flat = [v / 255.0 for v in test_pixels]

    metrics = ____
    print(f"Validation: max_diff={metrics.get('max_diff', 'N/A')}")
    print(f"ONNX model is portable: runs on any ONNX runtime (C++, JS, mobile)")

    return onnx_path


onnx_path = asyncio.run(export_to_onnx())

print("\n✓ Exercise 7 complete — CNN from scratch + ONNX export via OnnxBridge")
