# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP04 — Exercise 8.5: Regularisation, Clipping, Early Stop, ONNX Export
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Contrast baseline vs dropout vs batch-norm vs both
#   - Apply gradient clipping to prevent exploding gradients
#   - Implement early stopping on validation loss
#   - Export the trained CNN to ONNX via kailash-ml OnnxBridge
#
# PREREQUISITES: 04_optimisers_schedulers.py
#
# ESTIMATED TIME: ~45 min
#
# TASKS:
#   1. Theory — why training tricks compound
#   2. Build — four regularisation variants + EarlyStopping class
#   3. Train — full loop with clipping, scheduler, early stop
#   4. Visualise — val loss across variants + final gradient norms
#   5. Apply — edge deployment: shipping the CNN without a Python runtime
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import onnxruntime as ort
from kailash_ml import OnnxBridge

from shared.mlfp04.ex_8 import (
    N_CLASSES,
    OUTPUT_DIR,
    SG_HOSPITAL_CLASSES,
    TriageCNN,
    build_sg_loaders,
    device,
    eval_cnn,
    eval_cnn_auc,
    train_cnn_one_epoch,
    viz,
)

print("\n" + "=" * 70)
print("  Regularisation, Clipping, Early Stopping, and ONNX Export")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════
# THEORY — The regularisation stack
# ════════════════════════════════════════════════════════════════════════
# Regularisation is anything that limits what the network can memorise:
#   Dropout     — zero a random subset of activations at train time,
#                 forcing the network to distribute representation.
#   BatchNorm   — normalise activations per batch, which stabilises
#                 the optimisation landscape and acts as a mild
#                 regulariser by injecting batch noise.
#   Weight decay — shrink weights towards zero every step (AdamW).
#   Early stop  — stop training when val loss plateaus, using the
#                 training-time window itself as a regulariser.
#   Grad clip   — cap the gradient norm so a single pathological
#                 batch cannot knock the weights to infinity.
#
# Used together, they compound: dropout prevents co-adaptation, BN
# stabilises the landscape so clipping rarely fires, and early stopping
# picks the best checkpoint of the ones those tricks produced.

train_loader, test_loader, X_test_np, y_test_np = build_sg_loaders()
criterion = nn.BCEWithLogitsLoss()

# ════════════════════════════════════════════════════════════════════════
# TASK 2 — BUILD: regularisation variants and EarlyStopping
# ════════════════════════════════════════════════════════════════════════


def build_variant(use_dropout: bool, use_bn: bool) -> nn.Module:
    """A smaller variant of TriageCNN with configurable regularisation."""
    layers: list[nn.Module] = [nn.Conv2d(1, 32, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(32))
    layers += [nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1)]
    if use_bn:
        layers.append(nn.BatchNorm2d(64))
    layers += [
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(4),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 64),
        nn.ReLU(),
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.3))
    layers.append(nn.Linear(64, N_CLASSES))
    return nn.Sequential(*layers).to(device)


class EarlyStopping:
    """Monitor validation loss and stop when it plateaus."""

    def __init__(self, patience: int = 4, min_delta: float = 1e-3) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — TRAIN: regularisation grid + full pipeline with clipping + ES
# ════════════════════════════════════════════════════════════════════════
print("\n--- Regularisation grid (6 epochs each) ---")
variant_labels = [
    (False, False, "Baseline"),
    (True, False, "Dropout"),
    (False, True, "BatchNorm"),
    (True, True, "Dropout+BN"),
]

variant_val_histories: dict[str, list[float]] = {}
for use_do, use_bn, label in variant_labels:
    net = build_variant(use_do, use_bn)
    opt_ = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    val_curve: list[float] = []
    for _ in range(6):
        train_cnn_one_epoch(net, train_loader, opt_, criterion)
        val_curve.append(eval_cnn(net, test_loader, criterion))
    variant_val_histories[label] = val_curve
    print(f"  {label:<12}: val {val_curve[0]:.4f} -> {val_curve[-1]:.4f}")

# ── Checkpoint A ───────────────────────────────────────────────────────
assert (
    variant_val_histories["Dropout+BN"][-1]
    <= variant_val_histories["Baseline"][-1] + 0.2
), "Task 3: Dropout+BN should not be dramatically worse than baseline"

# Full pipeline: TriageCNN + AdamW + cosine + clip + early stop
print("\n--- Full pipeline: TriageCNN + AdamW + cosine + clip + early stop ---")
model = TriageCNN(n_classes=N_CLASSES, dropout_rate=0.3).to(device)
optimiser = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

n_epochs = 12
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimiser, T_max=n_epochs, eta_min=1e-6
)
stopper = EarlyStopping(patience=4, min_delta=1e-3)

history = {"train_loss": [], "val_loss": [], "lr": [], "grad_norm": []}
stopped_epoch = n_epochs
for epoch in range(n_epochs):
    train_loss, grad = train_cnn_one_epoch(
        model, train_loader, optimiser, criterion, clip_value=1.0
    )
    val_loss = eval_cnn(model, test_loader, criterion)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["lr"].append(scheduler.get_last_lr()[0])
    history["grad_norm"].append(grad)
    scheduler.step()
    print(
        f"  Epoch {epoch + 1:>2}/{n_epochs}: "
        f"train={train_loss:.4f} val={val_loss:.4f} "
        f"lr={history['lr'][-1]:.6f} grad={grad:.3f}"
    )
    if stopper.step(val_loss):
        stopped_epoch = epoch + 1
        print(f"  -> early stop triggered at epoch {stopped_epoch}")
        break

# ── Checkpoint B ───────────────────────────────────────────────────────
assert stopper.best_loss < math.inf, "Task 3: early stopping must record a best loss"
assert all(
    g > 0 for g in history["grad_norm"]
), "Task 3: gradient norms must be positive"
print("\n[ok] Checkpoint passed — full training pipeline complete\n")

# ════════════════════════════════════════════════════════════════════════
# TASK 4 — VISUALISE the regularisation grid and the final run
# ════════════════════════════════════════════════════════════════════════
fig_variants = viz.training_history(variant_val_histories, x_label="Epoch")
fig_variants.update_layout(title="Regularisation Variants — Validation BCE")
var_path = OUTPUT_DIR / "05_variants_val.html"
fig_variants.write_html(var_path)
print(f"[viz] Variants: {var_path}")

fig_full = viz.training_history(
    {
        "Train BCE": history["train_loss"],
        "Val BCE": history["val_loss"],
        "Grad Norm": history["grad_norm"],
    },
    x_label="Epoch",
)
fig_full.update_layout(title="Full Pipeline — Loss and Gradient Norm")
full_path = OUTPUT_DIR / "05_full_pipeline.html"
fig_full.write_html(full_path)
print(f"[viz] Full pipeline: {full_path}")

# ── (C) Dropout effect: side-by-side final val loss comparison ────────
import plotly.graph_objects as go

variant_names = list(variant_val_histories.keys())
variant_final = [variant_val_histories[v][-1] for v in variant_names]
variant_initial = [variant_val_histories[v][0] for v in variant_names]
colors = ["#FECB52", "#636EFA", "#00CC96", "#AB63FA"]


def _rgba(hex_color: str, alpha: float) -> str:
    """plotly 6.x rejects 8-digit #RRGGBBAA hex; emit rgba() instead."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


fig_dropout = go.Figure()
fig_dropout.add_trace(
    go.Bar(
        x=variant_names,
        y=variant_initial,
        name="Epoch 1 Val Loss",
        marker_color=[_rgba(c, 0.53) for c in colors],
        text=[f"{v:.4f}" for v in variant_initial],
        textposition="outside",
    )
)
fig_dropout.add_trace(
    go.Bar(
        x=variant_names,
        y=variant_final,
        name="Epoch 6 Val Loss",
        marker_color=colors,
        text=[f"{v:.4f}" for v in variant_final],
        textposition="outside",
    )
)
fig_dropout.update_layout(
    title="Regularisation Effect: Initial vs Final Validation Loss",
    xaxis_title="Variant",
    yaxis_title="Validation BCE Loss",
    barmode="group",
)
dropout_path = OUTPUT_DIR / "05_dropout_comparison.html"
fig_dropout.write_html(str(dropout_path))
print(f"[viz] Dropout comparison: {dropout_path}")

# ── (D) Train vs val loss with early stopping marker ─────────────────
epochs_run = list(range(1, len(history["train_loss"]) + 1))
fig_es = go.Figure()
fig_es.add_trace(
    go.Scatter(
        x=epochs_run,
        y=history["train_loss"],
        mode="lines+markers",
        name="Train Loss",
        marker_color="#636EFA",
    )
)
fig_es.add_trace(
    go.Scatter(
        x=epochs_run,
        y=history["val_loss"],
        mode="lines+markers",
        name="Val Loss",
        marker_color="#EF553B",
    )
)
if stopped_epoch < n_epochs:
    fig_es.add_vline(
        x=stopped_epoch,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Early Stop (epoch {stopped_epoch})",
        annotation_position="top left",
    )
# Mark the best val loss epoch
best_val_epoch = int(np.argmin(history["val_loss"])) + 1
best_val_loss = min(history["val_loss"])
fig_es.add_trace(
    go.Scatter(
        x=[best_val_epoch],
        y=[best_val_loss],
        mode="markers",
        marker=dict(size=14, color="green", symbol="star"),
        name=f"Best Val (epoch {best_val_epoch})",
    )
)
fig_es.update_layout(
    title="Training vs Validation Loss with Early Stopping",
    xaxis_title="Epoch",
    yaxis_title="BCE Loss",
)
es_path = OUTPUT_DIR / "05_early_stopping.html"
fig_es.write_html(str(es_path))
print(f"[viz] Early stopping: {es_path}")

# INTERPRETATION (computed from this run):
best_variant = min(variant_val_histories, key=lambda v: variant_val_histories[v][-1])
print("\nRegularisation grid — final validation BCE:")
for label in variant_val_histories:
    print(f"  {label:<12} {variant_val_histories[label][-1]:.4f}")
print(f"  Lowest final val loss this run: {best_variant}")
clip_epochs = sum(g > 1.0 for g in history["grad_norm"])
print(
    f"Full run: mean pre-clip gradient norm exceeded the clip threshold (1.0) "
    f"in {clip_epochs} of {len(history['grad_norm'])} epochs."
)
if clip_epochs:
    print(
        "  -> In those epochs the clipper rescaled at least the typical step "
        "back to norm 1.0, capping how far one batch can move the weights."
    )
else:
    print(
        "  -> The average step never needed clipping here; clipping is "
        "insurance against rare spikes, not something that fires constantly."
    )
final_aucs = eval_cnn_auc(model, X_test_np, y_test_np)
print(
    f"Best val loss {best_val_loss:.4f} at epoch {best_val_epoch}; "
    f"test macro AUC of the final model {final_aucs['macro']:.3f} "
    f"(per class: "
    + ", ".join(f"{n} {final_aucs[n]:.2f}" for n in SG_HOSPITAL_CLASSES)
    + ")"
)

# ════════════════════════════════════════════════════════════════════════
# TASK 5 — APPLY: Edge Deployment Without a Python Runtime
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (hypothetical): a Singapore team needs to run this triage
# model on small edge devices with limited RAM and no Python runtime.
# Shipping the PyTorch training stack (Python + torch, typically well
# over a gigabyte installed) is not an option. ONNX solves this: the
# model is serialised as a portable graph that ONNX Runtime — a far
# smaller native runtime with C/C++ APIs — can execute on many kinds of
# hardware.
#
# kailash-ml's OnnxBridge wraps the conversion: it checks compatibility
# up-front, exports torch modules (via torch.onnx.export with a dynamic
# batch dimension), and reports the outcome as a result object instead of
# raising, so a pipeline can decide what to do when export fails.

print("--- ONNX export via kailash-ml OnnxBridge ---")
# Step 1: Check compatibility before trying to export.
bridge = OnnxBridge()
compat = bridge.check_compatibility(model, framework="torch")
print(f"Compatible:  {compat.compatible}")
print(f"Confidence:  {compat.confidence}")

# Step 2: Export. torch exports trace the forward pass, so OnnxBridge
# needs a concrete sample input with the right shape.
onnx_output = OUTPUT_DIR / "triage_cnn.onnx"
model.eval()
sample_input = torch.from_numpy(X_test_np[:1]).to(device)
export_result = bridge.export(
    model, framework="torch", output_path=onnx_output, sample_input=sample_input
)
print(f"Export status: {export_result.onnx_status}")
if not export_result.success:
    raise RuntimeError(f"ONNX export failed: {export_result.error_message}")
model.eval()  # the trace runs in inference mode; keep it there for the check
# Recent torch exporters may store the weights in a side-car
# "<name>.onnx.data" file next to the graph — ship both files together.
onnx_files = [onnx_output] + sorted(OUTPUT_DIR.glob(onnx_output.name + ".data"))
total_kb = sum(f.stat().st_size for f in onnx_files) / 1024
print(f"Exported:    {', '.join(f.name for f in onnx_files)}")
print(f"Model size:  {total_kb:.1f} KB (graph + weights)")

# Step 3: Validate the exported graph against the PyTorch original with
# ONNX Runtime. A max-diff within tolerance means the edge runtime will
# produce the same predictions as the training runtime.
session = ort.InferenceSession(str(onnx_output))
input_name = session.get_inputs()[0].name
sample = X_test_np[:10]
with torch.no_grad():
    torch_out = model(torch.from_numpy(sample).to(device)).cpu().numpy()
onnx_out = session.run(None, {input_name: sample})[0]
max_diff = float(np.max(np.abs(torch_out - onnx_out)))
print(f"Max diff (torch vs onnx): {max_diff:.8f}")

# ── Checkpoint C ───────────────────────────────────────────────────────
assert compat.compatible, "Task 5: TriageCNN must be ONNX-compatible"
assert export_result.success, "Task 5: OnnxBridge export must succeed"
assert onnx_output.exists(), "Task 5: ONNX file must be written"
assert (
    max_diff < 1e-4
), f"Task 5: ONNX output must match PyTorch within 1e-4, got {max_diff}"
print("\n[ok] Checkpoint passed — model exported to ONNX and validated\n")

# BUSINESS IMPACT (illustrative): what you measured above is the part you
# can rely on — the exported file size and that ONNX Runtime reproduces
# the PyTorch outputs within 1e-4. The cost saving on a real fleet depends
# on the device, the runtime build and the batch size; benchmark latency
# and memory on the target hardware before quoting a number.
#
# LIMITATION: Not every PyTorch op has an ONNX equivalent. Dynamic
# control flow, custom autograd functions, and some advanced layers
# require rewriting for export. OnnxBridge reports compatibility up-front
# so you find this out before you ship, and export() returns a failed
# result (not an exception) when tracing breaks.

# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  [x] Compared four regularisation variants on the same training budget
  [x] Built an EarlyStopping class with patience + min_delta
  [x] Ran a full TriageCNN pipeline with cosine decay, gradient clipping,
      and early stopping wired together
  [x] Exported the trained model to ONNX via kailash-ml OnnxBridge and
      validated the output against the PyTorch original
  [x] Planned an (illustrative) edge deployment that needs no Python runtime

  KEY INSIGHT: Regularisation tricks compound. You rarely need all of
  them — pick two that address the specific failure mode of your task
  (overfitting, exploding gradients, unstable training) and ship the
  model via ONNX.

  This completes Exercise 8 and Module 4. Module 5 takes these building
  blocks and applies them to real Fashion-MNIST, CIFAR-10, and transfer
  learning with pretrained backbones.
"""
)

# THE USML BRIDGE — COMPLETE
# ════════════════════════════════════════════════════════════════════════
# Clustering / DR / association rules / matrix factorisation / neural
# hidden layers — the bridge from "unsupervised ML without labels" to
# "deep learning with labels" is now a continuous gradient. Hidden
# layers learn representations automatically; everything upstream of
# them was you choosing which representation to construct by hand.
