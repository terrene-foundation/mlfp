# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 7, Part 2: Transfer Learning with ResNet-18
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this section, you will be able to:
#   - Load a pre-trained ResNet-18 (ImageNet weights) and freeze its
#     convolutional backbone
#   - Replace the final classifier head for CIFAR-10 (10 classes)
#   - Compare transfer learning vs from-scratch on the same dataset
#   - Visualise layer activations (pretrained vs random init) and
#     use Grad-CAM to see what the model "looks at"
#   - Apply transfer learning to a real medical imaging scenario
#
# PREREQUISITES: Part 1 (from-scratch baseline), M5/ex_2 (CNNs).
# ESTIMATED TIME: ~30 min
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from shared.mlfp05.ex_7 import (
    BATCH_SIZE,
    CLASS_NAMES,
    EPOCHS,
    INPUT_SIZE,
    N_CLASSES,
    OUTPUT_DIR,
    count_params,
    create_visualizer,
    device,
    extract_features,
    compute_tsne,
    cluster_quality,
    init_engines,
    load_cifar10,
    plot_tsne,
    register_model,
    save_training_plots,
    train_model,
)


# ════════════════════════════════════════════════════════════════════════
# THEORY — Feature Reuse: The Doctor Who Specialises
# ════════════════════════════════════════════════════════════════════════
# Think of a doctor who studied general medicine for 7 years, then
# specialises in dermatology for 1 year. The general training (anatomy,
# physiology, diagnosis methodology) transfers directly — the doctor
# doesn't re-learn how to read a biopsy from scratch.
#
# Transfer learning works the same way. A ResNet-18 trained on ImageNet
# (1.28M images, 1000 classes) has already learned:
#   - Layer 1-2: Edge detectors, colour gradients (universally useful)
#   - Layer 3-4: Texture patterns, simple shapes (mostly transferable)
#   - Layer 5+:  Object parts, task-specific combinations (less transferable)
#
# We FREEZE these learned features and only train a new classification
# head for our specific task (CIFAR-10's 10 classes). This means:
#   - ~5,000 trainable parameters instead of ~11 million
#   - Training completes in minutes instead of hours
#   - Works well even with small datasets (because features are pre-learned)
#
# What transfers well:
#   - Low-level features (edges, textures) — almost always useful
#   - Mid-level features (patterns, shapes) — usually useful
#
# What doesn't transfer:
#   - Task-specific combinations (ImageNet "cat ears" ≠ CIFAR "cat face")
#   - Domain-specific patterns (medical images ≠ natural photos)
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PART 2: Transfer Learning with Frozen ResNet-18")
print("=" * 70)


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load data and engines
# ════════════════════════════════════════════════════════════════════════

train_set, val_set, train_loader, val_loader = load_cifar10()
conn, tracker, exp_name, registry, has_registry = init_engines()

print(f"\n  Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Build the transfer-learning ResNet-18
# ════════════════════════════════════════════════════════════════════════
# torchvision.models.resnet18(weights=IMAGENET1K_V1) loads the ImageNet
# checkpoint (~44 MB). We freeze all convolutional layers and replace
# the final fc with a fresh 10-class head.


def build_transfer_resnet(
    n_classes: int = N_CLASSES,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build a ResNet-18 with frozen backbone and fresh classifier head."""
    # No fallback to random weights: if the ImageNet download fails this
    # raises, because a random frozen backbone would make every "transfer"
    # result in this exercise meaningless.
    # TODO: Load the ImageNet-1K ResNet-18 weights and build the model
    # Hint: torchvision.models exposes a ResNet18_Weights enum and a
    #   resnet18(weights=...) constructor
    weights = ____
    model = ____
    print(f"  Loaded pre-trained ResNet-18 (weights={weights})")

    if freeze_backbone:
        for p in model.parameters():
            ____  # TODO: stop gradients flowing into this parameter

    # TODO: Replace the 1000-class ImageNet head with a fresh n_classes head
    in_features = model.fc.in_features
    model.fc = ____
    return model


transfer_model = build_transfer_resnet()
n_trainable = count_params(transfer_model, trainable_only=True)
n_total = count_params(transfer_model)
print(
    f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)"
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert n_trainable < n_total, "Backbone should be frozen (fewer trainable than total)"
assert n_trainable < 15000, f"Only fc head should be trainable, got {n_trainable:,}"
# INTERPRETATION: We're training only ~5K parameters (the new fc head)
# while keeping ~11M parameters frozen. The frozen backbone provides
# powerful feature extraction learned from 1.28M ImageNet images.
print("--- Checkpoint 1 passed --- transfer model built\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Build the from-scratch CNN for comparison
# ════════════════════════════════════════════════════════════════════════


def build_scratch_cnn(n_classes: int = N_CLASSES) -> nn.Module:
    """Baseline: same architecture as Part 1."""
    # TODO: Return the same small CNN you built in Part 1
    #   (three conv-BN-ReLU blocks, 32 -> 64 -> 128 channels, global
    #   average pooling, dropout 0.3, linear classifier)
    return ____


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — Train both models and compare
# ════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  TRAINING: Transfer Learning (frozen ResNet-18 + new head)")
print("=" * 70)
transfer_losses, transfer_accs, transfer_train_accs = train_model(
    transfer_model,
    "resnet18_transfer",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

print("\n" + "=" * 70)
print("  TRAINING: From-Scratch CNN baseline")
print("=" * 70)
scratch_model = build_scratch_cnn()
scratch_losses, scratch_accs, scratch_train_accs = train_model(
    scratch_model,
    "cnn_from_scratch",
    tracker,
    exp_name,
    train_loader,
    val_loader,
    epochs=EPOCHS,
)

best_transfer = max(transfer_accs)
best_scratch = max(scratch_accs)

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert best_transfer > 0.50, (
    f"Transfer val accuracy {best_transfer:.3f} below 0.50 -- check ImageNet "
    "normalisation, input resize, or epoch count"
)
assert len(transfer_losses) == EPOCHS, "Should have per-epoch losses"
assert len(scratch_losses) == EPOCHS, "Should have per-epoch losses"
# INTERPRETATION: Transfer learning leverages features already learned
# on ImageNet's 1.28M images. Even though we only train a linear head,
# the frozen ResNet-18 backbone provides powerful feature extraction
# that the from-scratch CNN cannot match in the same number of epochs.
print(f"\n  Transfer best val_acc: {best_transfer:.3f}")
print(f"  Scratch  best val_acc: {best_scratch:.3f}")
print(f"  Advantage: {best_transfer - best_scratch:+.3f}")
print("\n--- Checkpoint 2 passed --- both models trained and compared\n")

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — five instruments before Visualise
# ══════════════════════════════════════════════════════════════════
# kailash-ml's run_diagnostic_checkpoint runs a few real forward/backward
# passes (no optimiser step) with gradient, activation and dead-neuron
# hooks attached, and replays the real per-epoch training losses. It
# RETURNS the findings; print_prescription_pad prints them. The pass
# puts the model in train mode, which updates BatchNorm running
# statistics, so we diagnose a COPY and leave the trained model intact.
import copy

from kailash_ml.diagnostics import run_diagnostic_checkpoint

from shared.mlfp05.diagnostics import print_prescription_pad
from shared.mlfp05.ex_7 import classifier_diag_loss

print("\n── Diagnostic Report (Transfer ResNet-18 (ImageNet pretrained, frozen backbone)) ──")
diag, findings = run_diagnostic_checkpoint(
    copy.deepcopy(transfer_model),
    train_loader,
    classifier_diag_loss,
    title="Transfer ResNet-18 (ImageNet pretrained, frozen backbone)",
    n_batches=8,
    train_losses=transfer_losses,
    show=False,
)
print_prescription_pad(findings, "Transfer ResNet-18 (ImageNet pretrained, frozen backbone)")
# HOW TO READ THE PRESCRIPTION PAD FOR THIS MODEL:
#  Gradient flow — only the new fc head is trainable; the frozen backbone
#     has no parameter gradients by design, so judge this reading by the
#     fc layer. "Exploding" on fc means the head's updates are large
#     relative to its weights (lower the learning rate); "vanishing" means
#     the head has stopped learning.
#  Dead neurons — the ReLUs belong to the frozen ImageNet backbone. A high
#     dead fraction means many pretrained features are silent on CIFAR-10;
#     that is a domain-gap signal (unfreeze the last stage, or use the
#     adapters of Part 4), not a training bug.
#  Loss trend — read from the real per-epoch losses of the head.
#  If any reading is UNKNOWN, the library could not compute it from this
#  run; the message says why.


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — Register models in ModelRegistry
# ════════════════════════════════════════════════════════════════════════

if has_registry:
    # TODO: Register both models with the shared register_model helper
    #   ("cifar10_resnet18_transfer" and "cifar10_cnn_scratch"), passing
    #   each model's best val accuracy and final training loss
    transfer_ver = ____
    scratch_ver = ____

    # Promote the better model
    async def _promote():
        if max(transfer_accs) >= max(scratch_accs):
            # TODO: Promote the transfer model's registered version to the
            #   "production" stage, with a reason quoting both accuracies
            # Hint: registry.promote_model is a coroutine (await it)
            ____
            print("  Promoted: cifar10_resnet18_transfer -> production")
        else:
            await registry.promote_model(
                name="cifar10_cnn_scratch",
                version=scratch_ver.version,
                target_stage="production",
                reason=(
                    f"Scratch model outperforms transfer: "
                    f"val_acc={max(scratch_accs):.4f} vs {max(transfer_accs):.4f}"
                ),
            )
            print("  Promoted: cifar10_cnn_scratch -> production")

    asyncio.run(_promote())
else:
    print("  Note: ModelRegistry not available. Skipping registration.")

# ── Checkpoint 3 ─────────────────────────────────────────────────────
# INTERPRETATION: The ModelRegistry gives every model a version, metrics,
# and an audit trail. Promoting a model to production records the exact
# comparison that justified the decision.
print("\n--- Checkpoint 3 passed --- models registered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Visualise: Layer activations and Grad-CAM
# ════════════════════════════════════════════════════════════════════════
# Two powerful visualisations reveal HOW transfer learning works:
#
# 1. LAYER ACTIVATION COMPARISON — We compare activations from the
#    pre-trained ResNet (structured, edge-aware) vs a randomly
#    initialised ResNet (noisy, meaningless).
#
# 2. GRAD-CAM — Gradient-weighted Class Activation Mapping shows which
#    regions of the input image the model focuses on for its prediction.
#    This is the "visual explanation" of what the model learned.

print("-- Layer activation comparison --")

# Build a randomly-initialised ResNet for comparison
random_resnet = torchvision.models.resnet18(weights=None)
random_resnet.eval()
random_resnet.to(device)

# Get a sample image
sample_batch_x, sample_batch_y = next(iter(val_loader))
sample_img = sample_batch_x[0:1].to(device)

# Hook into conv1 to capture first-layer activations
pretrained_acts = []
random_acts = []


def hook_pretrained(m, inp, out):
    pretrained_acts.append(out.detach().cpu())


def hook_random(m, inp, out):
    random_acts.append(out.detach().cpu())


# TODO: Attach the two hooks to each model's first conv layer, run the
#   sample image through both models (no gradients, eval mode), then
#   remove the hooks
# Hint: nn.Module.register_forward_hook returns a handle with .remove()
____

pretrained_act = pretrained_acts[0][0]  # (64, H, W)
random_act = random_acts[0][0]

# Visualise first 8 activation maps from each
n_show = 8
fig_acts = go.Figure()

for i in range(n_show):
    # Pretrained activations
    act_pre = pretrained_act[i].numpy()
    fig_acts.add_trace(
        go.Heatmap(
            z=np.flipud(act_pre),
            colorscale="Viridis",
            showscale=False,
            name=f"Pretrained filter {i}",
            visible=(i == 0),
        )
    )

for i in range(n_show):
    # Random activations
    act_rand = random_act[i].numpy()
    fig_acts.add_trace(
        go.Heatmap(
            z=np.flipud(act_rand),
            colorscale="Viridis",
            showscale=False,
            name=f"Random filter {i}",
            visible=False,
        )
    )

# Dropdown to switch between pretrained and random
fig_acts.update_layout(
    title="Layer 1 Activations: Pre-trained (structured) vs Random (noise)",
    template="plotly_white",
    updatemenus=[
        {
            "buttons": [
                {
                    "label": f"Pretrained #{i}",
                    "method": "update",
                    "args": [{"visible": [j == i for j in range(2 * n_show)]}],
                }
                for i in range(n_show)
            ]
            + [
                {
                    "label": f"Random #{i}",
                    "method": "update",
                    "args": [{"visible": [j == n_show + i for j in range(2 * n_show)]}],
                }
                for i in range(n_show)
            ],
            "direction": "down",
            "showactive": True,
        }
    ],
)
acts_path = OUTPUT_DIR / "02_activation_comparison.html"
fig_acts.write_html(str(acts_path))
print(f"  Saved: {acts_path}")

# -- Grad-CAM visualisation --
print("-- Computing Grad-CAM heatmaps --")

transfer_model.eval()
sample_img_grad = sample_img.clone().requires_grad_(True)

# Forward pass through the model
gradcam_activations = []
gradcam_gradients = []


def fwd_hook(m, inp, out):
    gradcam_activations.append(out)


def bwd_hook(m, inp, out):
    gradcam_gradients.append(out[0])


fwd_handle = transfer_model.layer4.register_forward_hook(fwd_hook)
bwd_handle = transfer_model.layer4.register_full_backward_hook(bwd_hook)

logits = transfer_model(sample_img_grad)
pred_class = logits.argmax(dim=-1).item()

# Backward pass for the predicted class
transfer_model.zero_grad()
logits[0, pred_class].backward()

fwd_handle.remove()
bwd_handle.remove()

# Compute Grad-CAM: weighted combination of activation maps
grads = gradcam_gradients[0]  # (1, C, H, W)
acts = gradcam_activations[0]  # (1, C, H, W)
# TODO: Grad-CAM
#   1. Channel weights = gradients averaged over the spatial dims
#   2. CAM = weighted sum of the activation maps over channels
#   3. Keep only positive evidence (ReLU)
#   4. Take the single (H, W) map as a numpy array, min-max normalised
weights = ____
cam = ____
cam = ____
cam = ____

# Normalise to [0, 1]
cam = ____

fig_gradcam = go.Figure()
fig_gradcam.add_trace(
    go.Heatmap(
        z=np.flipud(cam),
        colorscale="Jet",
        showscale=True,
        colorbar=dict(title="Attention"),
    )
)
fig_gradcam.update_layout(
    title=(
        f"Grad-CAM: Where the model looks for class "
        f"'{CLASS_NAMES[pred_class]}' (true: '{CLASS_NAMES[sample_batch_y[0]]}')"
    ),
    template="plotly_white",
    width=500,
    height=500,
)
gradcam_path = OUTPUT_DIR / "02_gradcam.html"
fig_gradcam.write_html(str(gradcam_path))
print(f"  Saved: {gradcam_path}")
print(f"  Predicted: {CLASS_NAMES[pred_class]}, True: {CLASS_NAMES[sample_batch_y[0]]}")

# -- t-SNE comparison --
print("\n-- t-SNE: Transfer vs Scratch feature spaces --")
transfer_feats, transfer_labels = extract_features(transfer_model, val_loader)
scratch_feats, scratch_labels = extract_features(scratch_model, val_loader)

coords_transfer = compute_tsne(transfer_feats)
coords_scratch = compute_tsne(scratch_feats)

cq_transfer = cluster_quality(coords_transfer, transfer_labels)
cq_scratch = cluster_quality(coords_scratch, scratch_labels)

print(f"  Cluster quality (lower = better):")
print(f"    Transfer: {cq_transfer:.4f}")
print(f"    Scratch:  {cq_scratch:.4f}")

plot_tsne(
    coords_transfer,
    transfer_labels,
    "t-SNE: Transfer Learning Features (ResNet-18)",
    OUTPUT_DIR / "02_transfer_tsne.html",
)

# Training curves comparison
viz = create_visualizer()
save_training_plots(
    viz,
    {
        "transfer loss": transfer_losses,
        "transfer val_acc": transfer_accs,
        "scratch loss": scratch_losses,
        "scratch val_acc": scratch_accs,
    },
    OUTPUT_DIR / "02_training_comparison.html",
)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert cam.shape[0] > 0, "Grad-CAM should produce a spatial heatmap"
assert transfer_feats.shape[0] > 0, "Should have extracted transfer features"
# INTERPRETATION: The Grad-CAM heatmap shows the model focuses on the
# object in the image, not the background. Pre-trained features give
# structured attention patterns; scratch-trained features attend more
# diffusely. The t-SNE shows tighter class clusters for transfer
# learning because ImageNet pre-training teaches general visual
# category separation.
print("\n--- Checkpoint 4 passed --- visualisations complete\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Apply: Medical Image Classification at a Dermatology Clinic
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (illustrative): A public dermatology clinic in Singapore wants
# to build an AI system to classify skin conditions from dermatology
# images. It has only 2,000 labelled images across 10 condition types.
# We have no dermatology data here, so 2,000 CIFAR-10 images stand in
# for that small labelled set.
#
# Training a CNN from scratch on 2,000 images would give poor results.
# Transfer learning from ImageNet provides a strong foundation: the
# low-level features (edges, textures, colour patterns) transfer well
# to medical images, even though ImageNet doesn't contain dermatology
# photos.

print("\n" + "=" * 70)
print("  APPLY: Medical Image Classification — Dermatology Clinic (illustrative)")
print("=" * 70)

# Simulate the medical scenario: only 2,000 labelled images
from torch.utils.data import Subset, DataLoader as DL

rng = np.random.default_rng(42)
n_medical = 2000
# TODO: Sample n_medical distinct training indices, wrap them in a
#   Subset and a shuffled loader (same pattern as Part 1's startup)
indices = ____
medical_subset = ____
medical_loader = ____

# Transfer model on 2K images
medical_transfer = build_transfer_resnet()
# TODO: Train with train_model (run name "medical_transfer_2k", lr=1e-3)
medical_t_losses, medical_t_accs, _ = ____

# From-scratch model on 2K images
medical_scratch = build_scratch_cnn()
# TODO: Train with train_model (run name "medical_scratch_2k", lr=1e-3)
medical_s_losses, medical_s_accs, _ = ____

best_medical_transfer = ____  # TODO: best val accuracy of each run
best_medical_scratch = ____

print(f"\n  === Dermatology Clinic Scenario (2,000 images) ===")
print(f"  {'Method':<25} {'Val Accuracy':>15} {'Trainable Params':>18}")
print("  " + "-" * 60)
print(
    f"  {'Transfer (ResNet-18)':<25} "
    f"{best_medical_transfer:>15.1%} "
    f"{count_params(medical_transfer, trainable_only=True):>18,}"
)
print(
    f"  {'From scratch':<25} "
    f"{best_medical_scratch:>15.1%} "
    f"{count_params(medical_scratch, trainable_only=True):>18,}"
)
print(f"  {'Advantage':<25} {best_medical_transfer - best_medical_scratch:>+15.1%}")
print()
LABEL_COST = 5.00  # S$ per image for dermatologist review — illustrative
medical_advantage = best_medical_transfer - best_medical_scratch
print(f"  COST-BENEFIT ANALYSIS (illustrative label cost):")
print(f"  Labelling cost per image (dermatologist review): ~S${LABEL_COST:.2f}")
print(f"  Current labelled images: {n_medical:,} (S${n_medical * LABEL_COST:,.0f})")
print(
    f"  Every extra 1,000 labels the scratch model would need to close the "
    f"{medical_advantage:+.1%} gap costs S${1000 * LABEL_COST:,.0f}"
)
print(f"  (Part 3 measures how scratch accuracy grows with more labels.)")
print(f"  Transfer learning: the pre-trained weights are free and open")

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert (
    best_medical_transfer > 0.15
), f"Transfer with 2K data should beat random (acc={best_medical_transfer:.3f})"
# INTERPRETATION: With only 2,000 images, the transfer model starts from
# features that already encode edges, textures, and shapes — it only
# needs to learn which combinations indicate each class. Read the
# printed advantage: a large positive gap means fewer misclassified
# patients for the same labelling budget; a small or negative gap
# would mean the ImageNet features transfer poorly to this domain.
print(
    f"  Transfer advantage at 2,000 images: {medical_advantage:+.1%} "
    f"({'transfer wins' if medical_advantage > 0 else 'transfer did NOT win in this run'})"
)
print("\n--- Checkpoint 5 passed --- medical scenario complete\n")


# ════════════════════════════════════════════════════════════════════════
# CLEANUP
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 2 COMPLETE — What You've Learned")
print("=" * 70)
print(
    f"""
  [x] Loaded pre-trained ResNet-18 and froze the backbone
  [x] Trained only the classification head ({n_trainable:,} params vs {n_total:,} total)
  [x] Transfer vs scratch on full data: {best_transfer:.1%} vs {best_scratch:.1%}
  [x] Visualised layer activations: pre-trained (structured) vs random (noise)
  [x] Computed Grad-CAM heatmaps showing model attention
  [x] t-SNE cluster quality: transfer {cq_transfer:.3f} vs scratch {cq_scratch:.3f}
  [x] Medical scenario: {best_medical_transfer:.1%} transfer vs {best_medical_scratch:.1%} scratch with 2K images

  KEY INSIGHT: Transfer learning reuses visual features learned from
  1.28M ImageNet images. Like a doctor who studied general medicine
  before specialising, the model starts with deep knowledge instead of
  a blank slate. This is especially powerful when labelled data is
  scarce and expensive (medical images, industrial defects).

  NEXT: Part 3 quantifies data efficiency — how many labelled images
  do you actually need with transfer learning? This answers the VP's
  question: "How much do we need to spend on labelling?"
"""
)
