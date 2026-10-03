# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5.1: Vanilla GAN (The Minimax Game)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - The adversarial minimax game: Generator vs Discriminator
#   - Why GANs work (Nash equilibrium intuition for professionals)
#   - Build and train an MLP-based GAN on full MNIST (60K images)
#   - Diagnose training dynamics: when is D "winning" vs healthy balance
#   - Visualise generated digits, training progression, and loss dynamics
#   - Apply synthetic data generation to a Singapore insurer's data
#     scarcity problem — and learn why synthetic data is NOT private
#     by default
#
# PREREQUISITES: M5/ex_1 (autoencoders — generative model foundations)
# ESTIMATED TIME: ~45 min
# DATASET: MNIST — 60,000 real 28x28 grayscale digits
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from shared.mlfp05.ex_5 import (
    LATENT_DIM,
    OUTPUT_DIR,
    Generator,
    Discriminator,
    init_environment,
    load_mnist,
    setup_engines,
    close_engines,
    plot_image_grid,
    plot_latent_interpolation,
    plot_training_progression,
    plot_loss_curves,
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: The Minimax Game
# ════════════════════════════════════════════════════════════════════════
# Imagine a counterfeiter (Generator) trying to produce fake banknotes
# that fool a detective (Discriminator). The counterfeiter never sees
# real banknotes — they only learn from the detective's feedback:
# "this one looks fake because the watermark is wrong."
#
# Over time:
#   - The counterfeiter improves their forgeries
#   - The detective gets better at spotting fakes
#   - Eventually they reach a standoff (Nash equilibrium) where the
#     detective can't tell real from fake — that's a trained GAN.
#
# Mathematically, this is a minimax game:
#   min_G max_D [ E[log D(x)] + E[log(1 - D(G(z)))] ]
#
# D wants to maximise: score real images high, fake images low
# G wants to minimise: make D score fake images high
#
# When D is perfectly confused (outputs 0.5 for everything),
# D_loss converges to ln(4) ≈ 1.386.
#
# MODE COLLAPSE: The biggest GAN failure mode. The counterfeiter finds
# ONE type of banknote that fools the detective and keeps making only
# that one. In MNIST terms: the generator only produces 1s because
# they're the simplest digit. We'll measure this with mode coverage.
# ════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  PHASE 1 — THEORY: The Adversarial Minimax Game")
print("=" * 70)
print(
    """
  GAN = Generator + Discriminator in a zero-sum game.

  Generator (counterfeiter):
    - Input: random noise z ~ N(0, 1)
    - Output: fake image that should look real
    - Goal: fool the discriminator

  Discriminator (detective):
    - Input: real OR fake image
    - Output: probability that the image is real
    - Goal: correctly classify real vs fake

  Training alternates:
    1. Train D on a batch of real + fake images
    2. Train G to make D output "real" for fake images

  Nash equilibrium: D outputs 0.5 for everything (can't tell the difference)

  KEY RISK — Mode collapse:
    G discovers one "easy" output (e.g., only digit 1) and stops exploring.
    We detect this by checking whether all 10 digit classes are generated.
"""
)

# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: Generator + Discriminator
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 — BUILD: Generator and Discriminator Networks")
print("=" * 70)

device = init_environment()
X_real, y_real, real_loader = load_mnist(device)
conn, tracker, exp_name, registry = setup_engines()

# Verify architectures
G_test = Generator().to(device)
D_test = Discriminator().to(device)
z_test = torch.randn(4, LATENT_DIM, device=device)

print(f"\n  Generator: {sum(p.numel() for p in G_test.parameters()):,} parameters")
print(f"    Input:  z ~ N(0, 1), dim={LATENT_DIM}")
print(f"    Output: {tuple(G_test(z_test).shape)} image (28x28)")
print(f"\n  Discriminator: {sum(p.numel() for p in D_test.parameters()):,} parameters")
print(f"    Input:  28x28 image")
print(f"    Output: {tuple(D_test(G_test(z_test)).shape)} scalar logit")

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert G_test(z_test).shape == (4, 1, 28, 28), "Generator output shape wrong"
assert D_test(G_test(z_test)).shape == (4, 1), "Discriminator output shape wrong"
del G_test, D_test, z_test
print("\n--- Checkpoint 1 passed --- G and D architectures verified\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Vanilla GAN with Binary Cross-Entropy
# ════════════════════════════════════════════════════════════════════════
# L_D = -E[log D(x)] - E[log(1 - D(G(z)))]    (discriminator loss)
# L_G = -E[log D(G(z))]                         (generator loss — non-saturating)
#
# Why not the textbook minimax G loss, min E[log(1 - D(G(z)))]? Early in
# training D rejects fakes confidently (D(G(z)) ≈ 0), and log(1 - D) is
# flat there — its gradient SATURATES and G learns almost nothing. The
# non-saturating version maximises log D(G(z)) instead: same fixed point,
# but its gradient is LARGEST exactly when D rejects the fakes. In BCE
# terms it is simply "score the fakes against the REAL label".
print("\n" + "=" * 70)
print("  PHASE 3 — TRAIN: Vanilla GAN (BCEWithLogitsLoss)")
print("=" * 70)

EPOCHS = 15
LR = 2e-4


async def train_vanilla_gan(epochs: int = EPOCHS, lr: float = LR):
    """Train a vanilla GAN with BCE loss, logging to ExperimentTracker."""
    G = Generator().to(device)
    D = Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    g_losses, d_losses = [], []
    epoch_snapshots = {}

    # Capture initial state (random noise output)
    epoch_snapshots[0] = copy.deepcopy(G.state_dict())

    async with tracker.track(experiment=exp_name, run_name="vanilla_gan") as run:
        await run.log_params(
            {
                "architecture": "Vanilla_GAN_MLP",
                "latent_dim": str(LATENT_DIM),
                "lr": str(lr),
                "epochs": str(epochs),
                "batch_size": "128",
                "loss_type": "BCEWithLogitsLoss",
                "optimizer": "Adam(0.5,0.999)",
            }
        )

        for epoch in range(epochs):
            eg, ed = [], []
            for (real_batch,) in real_loader:
                bs = real_batch.size(0)

                # ── Train Discriminator ──────────────────────────────
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake = G(z).detach()
                loss_d = bce(D(real_batch), torch.ones(bs, 1, device=device)) + bce(
                    D(fake), torch.zeros(bs, 1, device=device)
                )
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # ── Train Generator ──────────────────────────────────
                z = torch.randn(bs, LATENT_DIM, device=device)
                loss_g = bce(D(G(z)), torch.ones(bs, 1, device=device))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                eg.append(loss_g.item())
                ed.append(loss_d.item())

            avg_g, avg_d = float(np.mean(eg)), float(np.mean(ed))
            g_losses.append(avg_g)
            d_losses.append(avg_d)
            await run.log_metrics({"g_loss": avg_g, "d_loss": avg_d}, step=epoch + 1)
            print(
                f"  [Vanilla GAN] epoch {epoch+1:2d}/{epochs}  "
                f"D={avg_d:.3f}  G={avg_g:.3f}"
            )

            # Capture snapshots for progression visualisation
            if (epoch + 1) in {1, 5, 10, 15}:
                epoch_snapshots[epoch + 1] = copy.deepcopy(G.state_dict())

        await run.log_metrics(
            {"final_g_loss": g_losses[-1], "final_d_loss": d_losses[-1]}
        )

    return G, D, g_losses, d_losses, epoch_snapshots


print("\n  Training vanilla GAN on full MNIST (60K images)...")
G_gan, D_gan, gan_g_losses, gan_d_losses, gan_snapshots = asyncio.run(
    train_vanilla_gan()
)

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Vanilla GAN (track G + D separately)
# ══════════════════════════════════════════════════════════════════
# GANs have TWO networks training in an adversarial loop. We run the
# Prescription Pad on BOTH the Generator and the Discriminator so we
# can see which side is "winning" and which side is starving for
# signal. Each loss closure re-runs the REAL objective from training
# (no weights are updated), and `train_losses` replays the per-epoch
# losses captured above.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad

diag_bce = nn.BCEWithLogitsLoss()


def _g_loss(m, batch):
    # Generator objective used in training: non-saturating BCE, with the
    # trained discriminator D_gan scoring fresh fakes against "real".
    bs = batch[0].size(0)
    z = torch.randn(bs, LATENT_DIM, device=device)
    return diag_bce(D_gan(m(z)), torch.ones(bs, 1, device=device))


def _d_loss(m, batch):
    # Discriminator objective used in training: real -> 1, fake -> 0.
    bs = batch[0].size(0)
    with torch.no_grad():
        fake = G_gan(torch.randn(bs, LATENT_DIM, device=device))
    return diag_bce(m(batch[0]), torch.ones(bs, 1, device=device)) + diag_bce(
        m(fake), torch.zeros(bs, 1, device=device)
    )


g_diag, g_findings = run_diagnostic_checkpoint(
    G_gan,
    real_loader,
    _g_loss,
    title="Vanilla GAN — Generator",
    n_batches=6,
    train_losses=gan_g_losses,
    show=False,
)
print_prescription_pad(g_findings, "Vanilla GAN — Generator")

d_diag, d_findings = run_diagnostic_checkpoint(
    D_gan,
    real_loader,
    _d_loss,
    title="Vanilla GAN — Discriminator",
    n_batches=6,
    train_losses=gan_d_losses,
    show=False,
)
print_prescription_pad(d_findings, "Vanilla GAN — Discriminator")

# HOW TO READ YOUR TWO PRESCRIPTION PADS (your readings depend on your run):
#
#  GRADIENT FLOW — compare G's pad with D's. If D's gradients are healthy
#     but G's are tiny (vanishing), D is dominating: G is being told
#     "everything you make is fake" without being told how to improve.
#     >> Try: reduce D's learning rate relative to G's, add label
#        smoothing (real target 0.9 instead of 1.0), or switch to
#        WGAN-GP (ex_5/02).
#
#  DEAD NEURONS / SATURATION — the generator ends in Tanh. A high
#     saturated fraction there means many pixels are pinned at pure
#     black/white. Combined with a gallery full of near-identical digits,
#     that is the signature of MODE COLLAPSE (count the distinct digits
#     in the 4A gallery — a healthy run shows all ten).
#
#  LOSS TREND — GAN losses are NOT supposed to fall monotonically. G and
#     D trade wins, so the curves oscillate, and G's loss often RISES as
#     D gets stronger. A WARNING here is not automatically a failure.
#     The failure signature is D's loss flat-lining near 0 (D has won
#     permanently); D's loss hovering near ln(4) ≈ 1.386 means D cannot
#     tell real from fake — the equilibrium you want.
#
#  CONNECT TO ex_5/02: run WGAN-GP and compare its pads with these. The
#  claim to test is "the Wasserstein critic gives G a more informative
#  gradient" — check whether your readings actually support it.
# ══════════════════════════════════════════════════════════════════

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(gan_g_losses) == EPOCHS, f"Expected {EPOCHS} epochs, got {len(gan_g_losses)}"
# INTERPRETATION: In a healthy GAN, D loss hovers around ln(4) ~ 1.386,
# meaning D is about 50% accurate (can't tell real from fake). If D loss
# drops toward 0, D has "won" — it separates real from fake perfectly —
# and the feedback G receives stops being informative ("everything you
# make is fake"), so training stalls or collapses.
print("\n--- Checkpoint 2 passed --- vanilla GAN trained\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: Generated Gallery, Loss Curves, Progression
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 4 — VISUALISE: What Has the Generator Learned?")
print("=" * 70)

# 4A: Generated image gallery (8x8 grid)
print("\n  4A: Gallery of 64 generated digits")
G_gan.eval()
with torch.no_grad():
    z_gallery = torch.randn(64, LATENT_DIM, device=device)
    gallery_images = G_gan(z_gallery)

fig_gallery = plot_image_grid(
    gallery_images,
    title="Vanilla GAN — Generated Digits (8x8 Grid)",
    save_path=str(OUTPUT_DIR / "ex_5_01_vanilla_gallery.png"),
)
plt.show()

# 4B: Training progression (epoch 0 → 1 → 5 → 10 → 15)
print("\n  4B: Training progression — from noise to digits")
G_progression = Generator().to(device)
fig_progression = plot_training_progression(
    G_progression,
    device,
    gan_snapshots,
    title="Vanilla GAN — Training Progression (Epoch 0 → 15)",
    save_path=str(OUTPUT_DIR / "ex_5_01_vanilla_progression.png"),
)
plt.show()
# Reload the final state after progression visualisation
G_gan.load_state_dict(gan_snapshots[max(gan_snapshots.keys())])
G_gan.eval()

# 4C: G vs D loss dynamics
print("\n  4C: Generator vs Discriminator loss curves")
fig_losses = plot_loss_curves(
    gan_g_losses,
    gan_d_losses,
    title="Vanilla GAN — Training Dynamics",
    save_path=str(OUTPUT_DIR / "ex_5_01_vanilla_losses.png"),
)
plt.show()

# 4D: Latent space interpolation
print("\n  4D: Latent space interpolation — smooth transitions between digits")
fig_interp = plot_latent_interpolation(
    G_gan,
    device,
    title="Vanilla GAN — Latent Interpolation",
    save_path=str(OUTPUT_DIR / "ex_5_01_vanilla_interpolation.png"),
)
plt.show()

# ── Checkpoint 3 ─────────────────────────────────────────────────────
import os

assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_01_vanilla_gallery.png")
), "Gallery image should exist"
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_01_vanilla_progression.png")
), "Progression image should exist"
# INTERPRETATION: The gallery shows whether the generator produces
# diverse, recognisable digits. The progression shows HOW it learns:
# epoch 0 is pure noise, early epochs show blurry shapes, later epochs
# show distinct digits. If all generated images look the same, that's
# mode collapse — the generator found one "easy" output.
print("\n--- Checkpoint 3 passed --- vanilla GAN visualisations complete\n")

# Also save training curves with ModelVisualizer (HTML interactive)
from kailash_ml import ModelVisualizer

viz = ModelVisualizer()
fig_html = viz.training_history(
    metrics={"GAN G loss": gan_g_losses, "GAN D loss": gan_d_losses},
    x_label="Epoch",
    y_label="Loss",
)
fig_html.write_html(str(OUTPUT_DIR / "ex_5_01_vanilla_training.html"))
print("  Interactive training curves saved to ex_5_01_vanilla_training.html")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: Synthetic Data for a Singapore Insurer
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 5 — APPLY: Synthetic Data for Insurance (Data Scarcity)")
print("=" * 70)
print(
    """
  BUSINESS SCENARIO (illustrative):
  You are a data scientist at a Singapore life insurer. Your team wants
  a fraud-detection model but has only 2,000 labelled claim records —
  too few for robust ML — and every new data-access request goes
  through a lengthy Personal Data Protection Act (PDPA) review.

  IDEA: Train a generative model on the real records and sample
  synthetic records with the same statistical properties, to
  supplement the real data while you develop the model.

  WHAT SYNTHETIC DATA DOES NOT GIVE YOU FOR FREE:
  - Privacy. A GAN is trained on real records and can memorise and
    regenerate them. Synthetic data is NOT automatically non-personal
    data: it needs formal guarantees (e.g. differentially-private
    training) plus memorisation / membership-inference testing, and
    the PDPA call belongs to your data-protection officer.
  - New information. Samples from a model fitted on 2,000 records
    cannot carry more signal than those records. Whether they improve
    the fraud model is an empirical question you answer on a held-out
    set of REAL claims.

  WHAT WE CAN CHECK HERE: whether synthetic records match real ones
  statistically.
"""
)

# Proxy for the insurance scenario, using MNIST:
# Real policyholder "profiles" = real MNIST digits (limited sample)
# Synthetic profiles = GAN-generated digits
# Caveat: G_gan was trained on all 60K digits, not on the 2,000-record
# sample, so this comparison is more favourable than the real scenario.

print("\n  Simulating the insurance data scenario with MNIST as proxy...")

# Step 1: Take a small "real" sample (simulating limited insurance data)
rng = np.random.default_rng(42)
small_sample_idx = rng.choice(len(X_real), 2000, replace=False)
X_small_real = X_real[small_sample_idx]
y_small_real = y_real[small_sample_idx]

print(f"  'Real' policyholder records: {len(X_small_real)}")

# Step 2: Generate synthetic data to supplement
G_gan.eval()
with torch.no_grad():
    z_synthetic = torch.randn(18000, LATENT_DIM, device=device)
    X_synthetic = G_gan(z_synthetic)

print(f"  Synthetic records generated: {len(X_synthetic)}")
print(f"  Combined dataset: {len(X_small_real) + len(X_synthetic)} records")

# Step 3: Compare real vs synthetic pixel distributions
X_real_flat = X_small_real.view(-1).cpu().numpy()
X_synth_flat = X_synthetic.view(-1).cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "Real vs Synthetic Distribution Comparison\n"
    "(Insurance Policyholder Profile Proxy)",
    fontsize=14,
    fontweight="bold",
)

# Pixel intensity distribution
axes[0].hist(X_real_flat, bins=50, alpha=0.6, label="Real (2K records)", density=True)
axes[0].hist(X_synth_flat, bins=50, alpha=0.6, label="Synthetic (18K)", density=True)
axes[0].set_xlabel("Feature Value", fontsize=12)
axes[0].set_ylabel("Density", fontsize=12)
axes[0].set_title("Feature Distribution Overlap", fontsize=13)
axes[0].legend(fontsize=11)

# Mean feature values per sample
real_means = X_small_real.view(len(X_small_real), -1).mean(dim=1).cpu().numpy()
synth_means = X_synthetic.view(len(X_synthetic), -1).mean(dim=1).cpu().numpy()
axes[1].hist(real_means, bins=30, alpha=0.6, label="Real", density=True)
axes[1].hist(synth_means, bins=30, alpha=0.6, label="Synthetic", density=True)
axes[1].set_xlabel("Mean Feature Value per Record", fontsize=12)
axes[1].set_ylabel("Density", fontsize=12)
axes[1].set_title("Record-Level Distribution", fontsize=13)
axes[1].legend(fontsize=11)

# Variance per sample
real_vars = X_small_real.view(len(X_small_real), -1).var(dim=1).cpu().numpy()
synth_vars = X_synthetic.view(len(X_synthetic), -1).var(dim=1).cpu().numpy()
axes[2].hist(real_vars, bins=30, alpha=0.6, label="Real", density=True)
axes[2].hist(synth_vars, bins=30, alpha=0.6, label="Synthetic", density=True)
axes[2].set_xlabel("Feature Variance per Record", fontsize=12)
axes[2].set_ylabel("Density", fontsize=12)
axes[2].set_title("Record Diversity", fontsize=13)
axes[2].legend(fontsize=11)

plt.tight_layout()
fig.savefig(
    str(OUTPUT_DIR / "ex_5_01_vanilla_real_vs_synthetic.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.show()
print("  Distribution comparison saved")

# Step 4: Stakeholder-ready summary
print("\n  ┌────────────────────────────────────────────────────────────┐")
print("  │  STAKEHOLDER SUMMARY: Synthetic Data Quality Assessment   │")
print("  ├────────────────────────────────────────────────────────────┤")
print(f"  │  Real records available:       {len(X_small_real):>8,}                  │")
print(f"  │  Synthetic records generated:  {len(X_synthetic):>8,}                  │")
print(
    f"  │  Combined training set:        {len(X_small_real)+len(X_synthetic):>8,}                  │"
)
print(
    f"  │  Data augmentation factor:     {(len(X_small_real)+len(X_synthetic))/len(X_small_real):.1f}x                       │"
)
print("  │                                                            │")
real_mean_val = float(np.mean(real_means))
synth_mean_val = float(np.mean(synth_means))
mean_diff_pct = abs(real_mean_val - synth_mean_val) / (abs(real_mean_val) + 1e-8) * 100
print(f"  │  Mean feature difference:      {mean_diff_pct:>7.1f}%                  │")
print("  │  Privacy / PDPA:               NOT ASSESSED (needs a      │")
print("  │                                memorisation test + DPO)   │")
stats_status = "STATISTICALLY CLOSE" if mean_diff_pct < 5.0 else "DISTRIBUTION GAP"
print(f"  │  Statistical match (<5% diff): {stats_status:<27} │")
print("  │  Next step: does it help a model scored on REAL hold-out?  │")
print("  └────────────────────────────────────────────────────────────┘")

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_01_vanilla_real_vs_synthetic.png")
), "Distribution comparison should exist"
assert len(X_synthetic) == 18000, "Should generate 18K synthetic records"
print("\n--- Checkpoint 4 passed --- insurance application complete\n")


# ════════════════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════════════════
asyncio.run(close_engines(conn))


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  VANILLA GAN FUNDAMENTALS:
  [x] The minimax game: Generator (counterfeiter) vs Discriminator (detective)
  [x] Binary cross-entropy loss for adversarial training
  [x] Nash equilibrium: D loss converging to ln(4) ~ 1.386
  [x] Mode collapse risk: generator producing only "easy" outputs

  VISUAL INTUITION:
  [x] Generated digit gallery (8x8 grid) — can you read the digits?
  [x] Training progression: noise → blurry shapes → recognisable digits
  [x] G vs D loss curves — healthy balance vs D domination
  [x] Latent interpolation — smooth transitions prove learned manifold

  REAL-WORLD APPLICATION:
  [x] Synthetic data to supplement scarce training records
  [x] Statistical validation: real vs synthetic distribution comparison
  [x] Why synthetic data is not private by default, and what must be
      tested before anyone treats it as non-personal data

  KEY INSIGHT:
  Vanilla GANs work but are UNSTABLE. The original minimax G loss,
  log(1 - D(G(z))), saturates when D confidently rejects fakes; that is
  why we trained G with the non-saturating loss -log D(G(z)). But the
  game still minimises the JS divergence, and when the real and
  generated distributions barely overlap JS is stuck at its maximum
  (log 2) — it says "different" without saying "how far". D can win,
  G chases a moving target, and modes collapse.

  Next: Exercise 5.2 — WGAN-GP solves this instability with
  Wasserstein distance (smooth gradients even when distributions
  don't overlap) and gradient penalty (replaces weight clipping).
"""
)
