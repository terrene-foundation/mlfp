# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5.3: GAN Evaluation and Model Registry
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Why "looks good" is not a valid evaluation metric for GANs
#   - FID (Frechet Inception Distance) — the standard automated metric,
#     and why its numbers depend on the feature extractor
#   - Mode coverage analysis — detecting hidden mode collapse
#   - Shannon entropy as a diversity measure
#   - A nearest-neighbour novelty check for memorised training images
#   - Register trained generators in ModelRegistry with quality metrics
#   - Build a quality assurance pipeline for synthetic data production
#   - Apply: QA validation for the insurance company's synthetic data
#     pipeline before deploying to production fraud detection models
#
# PREREQUISITES: ex_5/01_vanilla_gan.py, ex_5/02_wgan_gp.py
# ESTIMATED TIME: ~40 min
# DATASET: MNIST — 60,000 real 28x28 grayscale digits
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from shared.mlfp05.ex_5 import (
    LATENT_DIM,
    OUTPUT_DIR,
    Generator,
    Discriminator,
    LeNetFeatureExtractor,
    init_environment,
    load_mnist,
    load_mnist_test,
    setup_engines,
    close_engines,
    train_feature_extractor,
    compute_fid,
    mode_coverage,
    plot_image_grid,
    plot_latent_interpolation,
    plot_loss_curves,
    register_generator,
)
from kailash_ml import ModelVisualizer


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why GAN Evaluation Is Hard
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 1 — THEORY: The GAN Evaluation Problem")
print("=" * 70)
print(
    """
  THE PROBLEM:
  Unlike classifiers (accuracy, F1) or regressors (MSE, R²), GANs have
  no single ground truth to compare against. A generator that produces
  perfect images of the digit "7" — and ONLY "7" — has flawless
  individual images but is useless for any practical application.

  "LOOKS GOOD" IS DANGEROUS:
  Human visual inspection doesn't scale, is subjective, and misses
  subtle distribution mismatches. A GAN producing sharp but repetitive
  images fools the eye while silently failing on diversity.

  THREE EVALUATION DIMENSIONS:

  1. DISTRIBUTION MATCH (fidelity + diversity together):
     Does the cloud of generated images look like the cloud of real
     images? FID compares the two DISTRIBUTIONS — it is not a
     per-image score. Metric: FID (lower = better)

  2. DIVERSITY (mode coverage):
     Does the generator cover the full range of real data?
     Metric: mode coverage count + Shannon entropy

  3. NOVELTY (not memorising):
     Is the generator creating NEW images, not copying training data?
     Metric: nearest-neighbour distance to the training set, compared
     with the same distance for real images it never saw

  FID — FRECHET INCEPTION DISTANCE:

  FID treats both real and generated images as points in a learned
  feature space (from a pre-trained classifier). It then fits a
  multivariate Gaussian to each set and measures the distance between
  the two Gaussians:

    FID = ||mu_r - mu_g||^2 + Tr(Sig_r + Sig_g - 2*sqrt(Sig_r @ Sig_g))

  Intuition for professionals:
  - FID = 0: the two feature distributions are identical
  - FID has NO universal scale: it depends on the feature extractor.
    Published thresholds ("FID < 10 is excellent") are for 2048-d
    Inception features on natural images. We use a 64-d LeNet trained
    on MNIST (Inception expects 299x299 colour images), so our numbers
    are only comparable with other numbers from THIS extractor.
  - To make a number meaningful, compare it with a FLOOR: the FID
    between two sets of REAL digits. No generator can beat sampling
    noise, so "how many times the floor" is the readable figure.

  MODE COLLAPSE DETECTION:

  Mode collapse is the GAN equivalent of a factory that produces only
  one product. Shannon entropy quantifies diversity:
  - max entropy = log2(10) = 3.32 (uniform across all 10 digit classes)
  - entropy = 0: generator produces only one class (total collapse)

  FID does penalise missing modes, but it squeezes fidelity and
  diversity into ONE number: a middling FID cannot tell you whether
  every digit is slightly blurry or three digits are missing entirely.
  Coverage and entropy answer that second question directly.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: FID Computation Pipeline + Feature Extractor
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 2 — BUILD: FID Pipeline and Feature Extractor")
print("=" * 70)

device = init_environment()
X_real, y_real, real_loader = load_mnist(device)
conn, tracker, exp_name, registry = setup_engines()

# Train the feature extractor for FID computation
fid_ext = train_feature_extractor(X_real, y_real, device, epochs=5)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
fid_ext.eval()
with torch.no_grad():
    _test_acc = (
        (fid_ext((X_real[:1000] + 1) / 2).argmax(-1) == y_real[:1000]).float().mean()
    )
assert _test_acc > 0.8, f"Feature extractor accuracy too low: {_test_acc:.3f}"
print(f"  Feature extractor test accuracy: {_test_acc:.1%}")
print("\n--- Checkpoint 1 passed --- feature extractor trained\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: Both GAN Variants for Comparative Evaluation
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 3 — TRAIN: Vanilla GAN + WGAN-GP for Comparison")
print("=" * 70)

import torch.nn as nn


# TODO: Train vanilla GAN (15 epochs) with BCE loss
# Hint: Same pattern as ex_5/01 — Generator, Discriminator, Adam(0.5, 0.999),
#       BCEWithLogitsLoss, alternate D and G training each batch
print("\n  Training Vanilla GAN (15 epochs)...")
G_gan = Generator().to(device)
D_gan = Discriminator().to(device)
opt_g_gan = torch.optim.Adam(G_gan.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d_gan = torch.optim.Adam(D_gan.parameters(), lr=2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()
gan_g_losses, gan_d_losses = [], []

for epoch in range(15):
    eg, ed = [], []
    for (real_batch,) in real_loader:
        bs = real_batch.size(0)
        z = torch.randn(bs, LATENT_DIM, device=device)
        fake = G_gan(z).detach()
        # TODO: D loss = BCE on real (target=1) + BCE on fake (target=0)
        # Hint: the ex_5/01 discriminator loss, with D_gan.
        loss_d = ____
        opt_d_gan.zero_grad()
        loss_d.backward()
        opt_d_gan.step()
        z = torch.randn(bs, LATENT_DIM, device=device)
        # TODO: G loss = fool D by labelling fakes as real
        # Hint: the ex_5/01 non-saturating generator loss, with G_gan/D_gan.
        loss_g = ____
        opt_g_gan.zero_grad()
        loss_g.backward()
        opt_g_gan.step()
        eg.append(loss_g.item())
        ed.append(loss_d.item())
    gan_g_losses.append(float(np.mean(eg)))
    gan_d_losses.append(float(np.mean(ed)))
    print(
        f"  [Vanilla] epoch {epoch+1:2d}/15  "
        f"D={gan_d_losses[-1]:.3f}  G={gan_g_losses[-1]:.3f}"
    )


# Gradient penalty for WGAN-GP — the same function you wrote in ex_5/02;
# only its final line is left for you here.
def gradient_penalty(D, real, fake):
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = D(interp)
    grad = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # TODO: Return gradient penalty = mean of (||grad||_2 - 1)^2
    # Hint: per-example norm over ALL pixels, as in ex_5/02.
    return ____


# TODO: Train WGAN-GP (20 epochs) with critic training
# Hint: 5 critic steps per G step, Adam(0.5, 0.9), lr=1e-4, lambda = 10
print("\n  Training WGAN-GP (20 epochs)...")
G_wgan = Generator().to(device)
D_wgan = Discriminator().to(device)
opt_g_wgan = torch.optim.Adam(G_wgan.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_d_wgan = torch.optim.Adam(D_wgan.parameters(), lr=1e-4, betas=(0.5, 0.9))
wgan_g_losses, wgan_d_losses = [], []

for epoch in range(20):
    eg, ed = [], []
    for (real_batch,) in real_loader:
        bs = real_batch.size(0)
        for _ in range(5):
            z = torch.randn(bs, LATENT_DIM, device=device)
            fake = G_wgan(z).detach()
            gp = gradient_penalty(D_wgan, real_batch, fake)
            # TODO: Wasserstein critic loss + gradient penalty
            # Hint: the ex_5/02 critic loss, with D_wgan and lambda = 10.
            loss_d = ____
            opt_d_wgan.zero_grad()
            loss_d.backward()
            opt_d_wgan.step()
        z = torch.randn(bs, LATENT_DIM, device=device)
        # TODO: G loss — maximise critic score on fakes
        # Hint: the ex_5/02 generator loss, with G_wgan/D_wgan.
        loss_g = ____
        opt_g_wgan.zero_grad()
        loss_g.backward()
        opt_g_wgan.step()
        eg.append(loss_g.item())
        ed.append(loss_d.item())
    wgan_g_losses.append(float(np.mean(eg)))
    wgan_d_losses.append(float(np.mean(ed)))
    print(
        f"  [WGAN-GP] epoch {epoch+1:2d}/20  "
        f"critic={wgan_d_losses[-1]:.3f}  G={wgan_g_losses[-1]:.3f}"
    )

# ── Checkpoint 2 ─────────────────────────────────────────────────────
assert len(gan_g_losses) == 15, "Vanilla GAN should train 15 epochs"
assert len(wgan_g_losses) == 20, "WGAN-GP should train 20 epochs"
print("\n--- Checkpoint 2 passed --- both generators trained\n")


# ════════════════════════════════════════════════════════════════════════
# Compute FID scores
# ════════════════════════════════════════════════════════════════════════
print("\n  Computing FID scores...")

N_FID = 10000
G_gan.eval()
G_wgan.eval()

with torch.no_grad():
    # TODO: Generate fake images from both generators (N_FID each), scale to [0, 1]
    # Hint: generators output [-1, 1]; the extractor was trained on [0, 1].
    gan_fake_01 = ____
    wgan_fake_01 = ____

rng = np.random.default_rng(42)
real_sub = (X_real[rng.choice(len(X_real), N_FID, replace=False)] + 1) / 2

# TODO: Compute FID for both generators using the trained feature extractor
# Hint: shared.mlfp05.ex_5.compute_fid(extractor, real, generated) — the
#       real side is the [0, 1] subsample `real_sub`.
fid_gan = ____
fid_wgan = ____

# Real-vs-real FID floor: two sets of REAL digits (training vs the test
# split no generator has seen). No generator can beat sampling noise.
X_test, y_test = load_mnist_test(device)
X_test_01 = (X_test + 1) / 2
fid_floor = compute_fid(fid_ext, real_sub, X_test_01)

print(f"\n  FID Scores (lower = better; LeNet-64 feature space):")
print(f"    Real vs real (floor): {fid_floor:.3g}")
print(f"    Vanilla GAN: {fid_gan:.2f}  ({fid_gan / fid_floor:.1f}x floor)")
print(f"    WGAN-GP:     {fid_wgan:.2f}  ({fid_wgan / fid_floor:.1f}x floor)")

# TODO: Compute mode coverage for both generators
# Hint: shared.mlfp05.ex_5.mode_coverage — the LeNet extractor doubles as
#       the digit classifier.
cov_gan, dist_gan, ent_gan = ____
cov_wgan, dist_wgan, ent_wgan = ____

print(f"\n  Mode Coverage (all 10 digit classes):")
print(f"    Vanilla GAN: {cov_gan}/10 classes, entropy={ent_gan:.2f}/3.32")
print(f"    Distribution: {dist_gan}")
print(f"    WGAN-GP:     {cov_wgan}/10 classes, entropy={ent_wgan:.2f}/3.32")
print(f"    Distribution: {dist_wgan}")


# ── Novelty check: is either generator copying its training images? ──
# For each image, find the distance (in the extractor's feature space) to
# its NEAREST training image. Baseline: the same distance for real test
# digits, which the generators never saw. A memorising generator sits much
# CLOSER to the training set than unseen real digits do (ratio << 1).
N_NOV = 2000


def _features(images_01: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat(
            [
                fid_ext.extract_features(images_01[i : i + 5000])
                for i in range(0, len(images_01), 5000)
            ]
        )


train_feats = _features((X_real + 1) / 2)


def nn_distance(images_01: torch.Tensor) -> torch.Tensor:
    """Distance from each image to its nearest training image (feature space)."""
    q = _features(images_01)
    return torch.cat(
        [
            torch.cdist(q[i : i + 250], train_feats).min(dim=1).values
            for i in range(0, len(q), 250)
        ]
    )


d_unseen = nn_distance(X_test_01[:N_NOV]).median().item()
nov_gan = nn_distance(gan_fake_01[:N_NOV]).median().item() / d_unseen
nov_wgan = nn_distance(wgan_fake_01[:N_NOV]).median().item() / d_unseen

print(f"\n  Novelty (median NN distance to training set / unseen real digits'):")
print(f"    Vanilla GAN: {nov_gan:.2f}")
print(f"    WGAN-GP:     {nov_wgan:.2f}")
print("    ~1.0 = as novel as unseen real digits; well below 1 = near-copies")


# Log to ExperimentTracker
async def _log_evaluation():
    async with tracker.track(experiment=exp_name, run_name="fid_evaluation") as run:
        await run.log_param("n_generated", str(N_FID))
        await run.log_metrics(
            {
                "fid_vanilla_gan": fid_gan,
                "fid_wgan_gp": fid_wgan,
                "coverage_vanilla": float(cov_gan),
                "coverage_wgan": float(cov_wgan),
                "entropy_vanilla": ent_gan,
                "entropy_wgan": ent_wgan,
                "fid_real_floor": fid_floor,
                "novelty_vanilla": nov_gan,
                "novelty_wgan": nov_wgan,
            }
        )


asyncio.run(_log_evaluation())

# ── Checkpoint 3 ─────────────────────────────────────────────────────
# FID is mathematically non-negative, but the numerical estimate (sample
# covariances + eigenvalues of Sigma_r @ Sigma_g) can land tiny-negative
# from floating-point error. Allow a small tolerance.
assert (
    fid_gan >= -1e-3 and fid_wgan >= -1e-3
), f"FID expected ~0+; got fid_gan={fid_gan:.6f}, fid_wgan={fid_wgan:.6f}"
assert fid_floor >= -1e-3, f"Real-vs-real FID should be ~0+; got {fid_floor:.6f}"
assert nov_gan > 0 and nov_wgan > 0, "Novelty ratios must be positive"
assert 0 <= ent_gan <= np.log2(10) + 0.01, "Entropy out of range"
assert cov_gan >= 1 and cov_wgan >= 1, "Must produce at least 1 class"
# INTERPRETATION: FID = 0 means identical feature distributions. Read
# each FID against the real-vs-real floor printed above, not against
# published Inception-FID numbers — this extractor has its own scale.
# WGAN-GP often achieves better coverage because the critic still gives
# an informative gradient when the distributions barely overlap — but
# check YOUR numbers: one training run is one sample.
print("\n--- Checkpoint 3 passed --- FID and mode coverage computed\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: FID Comparison, Mode Coverage, Latent Walk
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 4 — VISUALISE: Evaluation Metrics Dashboard")
print("=" * 70)

# 4A: FID score comparison bar chart
print("\n  4A: FID score comparison")
# TODO: Create bar chart comparing FID scores of both generators
# Hint: one bar per generator; the real-vs-real floor is the reference line.
fig_fid, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    ["Vanilla GAN", "WGAN-GP"],
    [fid_gan, fid_wgan],
    color=["#e74c3c", "#2ecc71"],
    width=0.5,
    edgecolor="black",
    linewidth=1.2,
)
ax.set_ylabel("FID Score (lower = better)", fontsize=13)
ax.set_title(
    "Frechet Distance Comparison (LeNet-64 features)\n"
    f"(computed on {N_FID:,} generated vs {N_FID:,} real images)",
    fontsize=14,
    fontweight="bold",
)
# TODO: Add value labels on bars
# Hint: Axes.text, centred on each bar (x + width/2), just above its height.
for bar, val in zip(bars, [fid_gan, fid_wgan]):
    ____  # TODO: Add value label text on each bar
ax.axhline(
    y=fid_floor,
    color="blue",
    linestyle="--",
    alpha=0.6,
    label=f"Real-vs-real floor ({fid_floor:.3g})",
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
plt.tight_layout()
fig_fid.savefig(
    str(OUTPUT_DIR / "ex_5_03_fid_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# 4B: Mode coverage matrix — heatmap of generated digit classes
print("\n  4B: Mode coverage matrix (all 10 digit classes)")
# TODO: Create side-by-side bar charts showing per-digit class distribution
# Hint: For each GAN, compute counts per class 0-9, convert to percentages,
#       bar chart with green (present) / red (missing) colours
fig_mode, axes = plt.subplots(1, 2, figsize=(16, 5))
fig_mode.suptitle(
    "Mode Coverage: Which Digits Does Each GAN Generate?",
    fontsize=14,
    fontweight="bold",
)

for idx, (name, dist, ent, cov) in enumerate(
    [
        ("Vanilla GAN", dist_gan, ent_gan, cov_gan),
        ("WGAN-GP", dist_wgan, ent_wgan, cov_wgan),
    ]
):
    counts = [dist.get(c, 0) for c in range(10)]
    total = sum(counts)
    pcts = [c / total * 100 if total > 0 else 0 for c in counts]
    colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in counts]

    bars = axes[idx].bar(
        range(10), pcts, color=colors, edgecolor="black", linewidth=0.8
    )
    axes[idx].set_xlabel("Digit Class", fontsize=12)
    axes[idx].set_ylabel("% of Generated Images", fontsize=12)
    axes[idx].set_title(
        f"{name}\n{cov}/10 classes, entropy={ent:.2f}/3.32", fontsize=13
    )
    axes[idx].set_xticks(range(10))
    axes[idx].axhline(
        y=10, color="blue", linestyle="--", alpha=0.5, label="Ideal uniform (10%)"
    )
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.2, axis="y")

    # TODO: Add percentage labels on each bar
    # Hint: same Axes.text pattern as the 4A value labels; skip empty bars.
    ____

plt.tight_layout()
fig_mode.savefig(
    str(OUTPUT_DIR / "ex_5_03_mode_coverage.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# 4C: Latent space interpolation walk — smooth transitions
print("\n  4C: Latent space interpolation (WGAN-GP)")
fig_walk = plot_latent_interpolation(
    G_wgan,
    device,
    n_steps=12,
    n_rows=6,
    title="WGAN-GP Latent Walk — Smooth Digit Transitions",
    save_path=str(OUTPUT_DIR / "ex_5_03_latent_walk.png"),
)
plt.show()

# 4D: Combined training dynamics
print("\n  4D: Combined training dynamics comparison")
viz = ModelVisualizer()
fig_all = viz.training_history(
    metrics={
        "Vanilla G": gan_g_losses,
        "Vanilla D": gan_d_losses,
        "WGAN G": wgan_g_losses,
        "WGAN Critic": wgan_d_losses,
    },
    x_label="Epoch",
    y_label="Loss",
)
fig_all.write_html(str(OUTPUT_DIR / "ex_5_03_combined_training.html"))
print("  Interactive combined training curves saved")

# 4E: Comprehensive evaluation dashboard
print("\n  4E: Evaluation dashboard")
# TODO: Create a 2x2 dashboard with FID comparison, entropy comparison,
#       G loss curves, and D/Critic loss curves
# Hint: fig_dash, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_dash, axes = plt.subplots(2, 2, figsize=(16, 12))
fig_dash.suptitle(
    "GAN Evaluation Dashboard — Vanilla GAN vs WGAN-GP",
    fontsize=16,
    fontweight="bold",
)

# TODO: Top-left — FID bar chart
# Hint: same chart as 4A, drawn on the top-left Axes, with value labels.
____

# TODO: Top-right — Entropy bar chart with max entropy line
# Hint: the reference line is the maximum entropy for 10 equally likely
#       classes.
____

# TODO: Bottom-left — G loss curves for both generators
# Hint: the two runs have different epoch counts (15 vs 20) — build each
#       x-axis from its own loss list.
____

# TODO: Bottom-right — D/Critic loss curves for both generators
# Hint: same as bottom-left, with the D / critic loss lists. Remember the
#       two losses are on different scales and the critic's is about -W.
____

plt.tight_layout()
fig_dash.savefig(
    str(OUTPUT_DIR / "ex_5_03_evaluation_dashboard.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_03_fid_comparison.png")
), "FID comparison should exist"
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_03_evaluation_dashboard.png")
), "Dashboard should exist"
print("\n--- Checkpoint 4 passed --- evaluation visualisations complete\n")


# ════════════════════════════════════════════════════════════════════════
# Register generators in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
print("\n  Registering generators in ModelRegistry...")
ver_gan = register_generator(
    registry, "vanilla_gan_generator", G_gan, fid_gan, cov_gan, ent_gan
)
ver_wgan = register_generator(
    registry, "wgan_gp_generator", G_wgan, fid_wgan, cov_wgan, ent_wgan
)

# ── Checkpoint 5 ─────────────────────────────────────────────────────
if registry is not None:
    assert ver_gan is not None, "Vanilla GAN should be registered"
    assert ver_wgan is not None, "WGAN-GP should be registered"
print("\n--- Checkpoint 5 passed --- generators registered\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: QA Pipeline for Insurance Synthetic Data
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PHASE 5 — APPLY: Synthetic Data QA for Insurance Production")
print("=" * 70)
print(
    """
  BUSINESS SCENARIO (illustrative):
  You are the ML engineering lead at the Singapore insurer from
  Exercise 5.1. Your team has trained a WGAN-GP to generate synthetic
  policyholder profiles for fraud detection model training.

  Before deploying synthetic data to production, the Chief Risk Officer
  (CRO) asks: "How do you KNOW the synthetic data is good enough?
  What if the generator produces biased profiles that make the fraud
  model miss certain claim types?"

  THE QA GATE THIS SCRIPT RUNS:
  1. Distribution match: FID within a set multiple of the real-vs-real
     floor (the multiple is a POLICY choice — calibrate it per extractor)
  2. Mode coverage: at least 8 of 10 categories generated
  3. Diversity: Shannon entropy of the generated categories
  4. Balance: no category below 3% of generated samples
  5. Novelty: generated samples must not sit much closer to the
     training set than unseen real samples do (near-copy check)

  GATES THIS SCRIPT DOES NOT RUN (still required before production):
  - Downstream validation: a fraud model trained with the synthetic
    data, scored on REAL held-out claims, against a real-data baseline
  - Privacy review: the novelty check catches near-copies only; it is
    not a privacy guarantee (that needs DP training and membership-
    inference testing)

  Synthetic data that fails any gate is rejected and the generator is
  retrained or tuned.
"""
)

# Step 1: Define QA thresholds (policy choices, not universal constants)
FID_FLOOR_MULTIPLE = 10.0  # FID at most 10x the real-vs-real floor
FID_THRESHOLD = FID_FLOOR_MULTIPLE * fid_floor
MIN_MODE_COVERAGE = 8  # At least 8/10 classes
MIN_ENTROPY = 2.5  # Minimum diversity (max is 3.32)
MIN_CLASS_PCT = 3.0  # No class below 3% of total
MIN_NOVELTY_RATIO = 0.8  # NN distance at least 80% of unseen real digits'

print("  QA Thresholds:")
print(f"    FID score:        < {FID_THRESHOLD:.3g} ({FID_FLOOR_MULTIPLE:.0f}x floor)")
print(f"    Mode coverage:    >= {MIN_MODE_COVERAGE}/10 classes")
print(f"    Shannon entropy:  >= {MIN_ENTROPY}/3.32")
print(f"    Min class share:  >= {MIN_CLASS_PCT}%")
print(f"    Novelty ratio:    >= {MIN_NOVELTY_RATIO}")

# Step 2: Run QA on both generators
print("\n  Running QA pipeline on both generators...")

# TODO: Evaluate both generators against QA thresholds
# Hint: For each generator, check fid < threshold, coverage >= min,
#       entropy >= min, and min class percentage >= threshold
qa_results = {}
for name, G, fid, cov, dist, ent, nov in [
    ("Vanilla GAN", G_gan, fid_gan, cov_gan, dist_gan, ent_gan, nov_gan),
    ("WGAN-GP", G_wgan, fid_wgan, cov_wgan, dist_wgan, ent_wgan, nov_wgan),
]:
    # Smallest share over ALL 10 classes — a class never generated counts
    # as 0%, not as "absent from the dict".
    total = sum(dist.values())
    min_pct = min(dist.get(c, 0) for c in range(10)) / total * 100 if total else 0.0

    # TODO: Check each QA criterion
    # Hint: one boolean per threshold defined in Step 1. Mind the
    #       direction: FID is lower-is-better, the rest higher-is-better.
    fid_pass = ____
    cov_pass = ____
    ent_pass = ____
    pct_pass = ____
    nov_pass = nov >= MIN_NOVELTY_RATIO

    all_pass = fid_pass and cov_pass and ent_pass and pct_pass and nov_pass

    qa_results[name] = {
        "fid": fid,
        "fid_pass": fid_pass,
        "coverage": cov,
        "cov_pass": cov_pass,
        "entropy": ent,
        "ent_pass": ent_pass,
        "min_class_pct": min_pct,
        "pct_pass": pct_pass,
        "novelty": nov,
        "nov_pass": nov_pass,
        "overall": all_pass,
    }

# Step 3: QA results visualisation
# TODO: Create normalised bar chart showing % of QA threshold met
# Hint: For FID (lower=better), normalise as threshold/value * 100
#       For others (higher=better), normalise as value/threshold * 100
fig_qa, ax = plt.subplots(figsize=(14, 7))
fig_qa.suptitle(
    "Synthetic Data QA Pipeline Results\n"
    "Production Gate for Insurance Fraud Model Training",
    fontsize=14,
    fontweight="bold",
)

categories = [
    f"FID Score\n(< {FID_THRESHOLD:.3g})",
    "Mode Coverage\n(>= 8/10)",
    "Entropy\n(>= 2.5)",
    "Min Class %\n(>= 3%)",
    "Novelty Ratio\n(>= 0.8)",
]
vanilla_scores = [
    qa_results["Vanilla GAN"]["fid"],
    qa_results["Vanilla GAN"]["coverage"],
    qa_results["Vanilla GAN"]["entropy"],
    qa_results["Vanilla GAN"]["min_class_pct"],
    qa_results["Vanilla GAN"]["novelty"],
]
wgan_scores = [
    qa_results["WGAN-GP"]["fid"],
    qa_results["WGAN-GP"]["coverage"],
    qa_results["WGAN-GP"]["entropy"],
    qa_results["WGAN-GP"]["min_class_pct"],
    qa_results["WGAN-GP"]["novelty"],
]
thresholds = [
    FID_THRESHOLD,
    MIN_MODE_COVERAGE,
    MIN_ENTROPY,
    MIN_CLASS_PCT,
    MIN_NOVELTY_RATIO,
]

x = np.arange(len(categories))
width = 0.3

# TODO: Normalise scores to percentage of threshold met
# Hint: 100% = exactly at the threshold. Invert the ratio for FID
#       (lower is better), guard against dividing by zero, and cap at
#       150% so one huge value does not flatten the chart (ylim is 160).
vanilla_norm = []
wgan_norm = []
for v, w, t, cat in zip(vanilla_scores, wgan_scores, thresholds, categories):
    if "FID" in cat:
        ____  # TODO: Append normalised FID (lower=better)
        ____
    else:
        ____  # TODO: Append normalised coverage/entropy/pct (higher=better)
        ____

bars1 = ax.bar(
    x - width / 2, vanilla_norm, width, label="Vanilla GAN", color="#e74c3c", alpha=0.8
)
bars2 = ax.bar(
    x + width / 2, wgan_norm, width, label="WGAN-GP", color="#2ecc71", alpha=0.8
)
ax.axhline(
    y=100, color="black", linestyle="--", linewidth=2, label="QA Threshold (100%)"
)
ax.set_ylabel("% of QA Threshold Met", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, 160)

# TODO: Add PASS/FAIL indicators on each bar
# Hint: For each bar pair, check the pass/fail from qa_results,
#       add green "PASS" or red "FAIL" text above the bar
____

plt.tight_layout()
fig_qa.savefig(
    str(OUTPUT_DIR / "ex_5_03_qa_pipeline.png"), dpi=150, bbox_inches="tight"
)
plt.show()

# Step 4: Stakeholder-ready QA report
van_overall = qa_results["Vanilla GAN"]["overall"]
wgan_overall = qa_results["WGAN-GP"]["overall"]

print("\n  ┌────────────────────────────────────────────────────────────────┐")
print("  │  SYNTHETIC DATA QA REPORT — Insurance Fraud Model Pipeline    │")
print("  ├────────────────────────────────────────────────────────────────┤")
print("  │                                                                │")
print(f"  │  {'Metric':<22} {'Vanilla GAN':>13} {'WGAN-GP':>13} {'Threshold':>12} │")
print(f"  │  {'─'*60}  │")
print(
    f"  │  {'FID Score':<22} {fid_gan:>13.1f} {fid_wgan:>13.1f} "
    f"{'< ' + format(FID_THRESHOLD, '.3g'):>12} │"
)
print(
    f"  │  {'Mode Coverage':<22} {cov_gan:>12}/10 {cov_wgan:>12}/10 {'>= 8/10':>12} │"
)
print(
    f"  │  {'Shannon Entropy':<22} {ent_gan:>13.2f} {ent_wgan:>13.2f} {'>= 2.50':>12} │"
)
van_min = qa_results["Vanilla GAN"]["min_class_pct"]
wgan_min = qa_results["WGAN-GP"]["min_class_pct"]
print(
    f"  │  {'Min Class Share':<22} {van_min:>12.1f}% {wgan_min:>12.1f}% {'>= 3.0%':>12} │"
)
print(
    f"  │  {'Novelty Ratio':<22} {nov_gan:>13.2f} {nov_wgan:>13.2f} {'>= 0.80':>12} │"
)
print("  │                                                                │")
van_status = "APPROVED" if van_overall else "REJECTED"
wgan_status = "APPROVED" if wgan_overall else "REJECTED"
print(f"  │  {'OVERALL STATUS':<22} {van_status:>13} {wgan_status:>13}              │")
print("  │                                                                │")
better = "WGAN-GP" if fid_wgan < fid_gan else "Vanilla GAN"
print(f"  │  Lower FID: {better:<50} │")
print(f"  │  Best FID score: {min(fid_gan, fid_wgan):<44.1f} │")
print("  │                                                                │")
approved = [n for n, r in qa_results.items() if r["overall"]]
if approved:
    decision = f"{', '.join(approved)} passed this gate"
    follow_up = "Next: downstream test on REAL claims + privacy review"
else:
    decision = "No generator passed — retrain or tune before reuse"
    follow_up = "See the FAIL bars above for which gate blocked each one"
print(f"  │  DECISION: {decision:<51} │")
print(f"  │  {follow_up:<61} │")
print("  └────────────────────────────────────────────────────────────────┘")

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert os.path.exists(
    str(OUTPUT_DIR / "ex_5_03_qa_pipeline.png")
), "QA pipeline chart should exist"
print("\n--- Checkpoint 6 passed --- QA pipeline complete\n")


# ════════════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  EXPERIMENT SUMMARY")
print("=" * 70)
print(f"\n  Experiment: {exp_name}")
print(f"  Dataset: MNIST (60,000 images), latent_dim={LATENT_DIM}")
print(f"\n  {'Metric':<25} {'Vanilla GAN':>14} {'WGAN-GP':>14}")
print(f"  {'-'*53}")
print(f"  {'Epochs':<25} {'15':>14} {'20':>14}")
print(f"  {'Final G loss':<25} {gan_g_losses[-1]:>14.3f} {wgan_g_losses[-1]:>14.3f}")
print(
    f"  {'Final D/Critic loss':<25} {gan_d_losses[-1]:>14.3f} {wgan_d_losses[-1]:>14.3f}"
)
print(f"  {'FID score':<25} {fid_gan:>14.2f} {fid_wgan:>14.2f}")
print(f"  {'Mode coverage':<25} {cov_gan:>13}/10 {cov_wgan:>13}/10")
print(f"  {'Class entropy':<25} {ent_gan:>14.2f} {ent_wgan:>14.2f}")
print(f"  {'Novelty ratio':<25} {nov_gan:>14.2f} {nov_wgan:>14.2f}")
print(f"\n  Best generator by FID: {better}")


# ════════════════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════════════════
asyncio.run(close_engines(conn))


# ════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — WGAN-GP generator, one kailash-ml call
# ════════════════════════════════════════════════════════════════════════
# FID, coverage and novelty judge the OUTPUT. The Prescription Pad judges
# the network: run_diagnostic_checkpoint instruments the trained WGAN-GP
# generator, replays a few batches of its real training objective (score
# fakes with the trained critic; no weights are updated) and returns the
# gradient-flow, dead-neuron and loss-trend readings.
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad


def _wgan_g_loss(m, batch):
    z = torch.randn(batch[0].size(0), LATENT_DIM, device=device)
    return -D_wgan(m(z)).mean()


diag, findings = run_diagnostic_checkpoint(
    G_wgan,
    real_loader,
    _wgan_g_loss,
    title="WGAN-GP generator",
    n_batches=8,
    train_losses=wgan_g_losses,
    show=False,
)
print_prescription_pad(findings, "WGAN-GP generator")
# HOW TO READ IT: put this pad beside the QA report. Healthy gradients
# with a FAILED coverage gate means the network trains fine but has
# collapsed onto a few modes — a data/objective problem, not a plumbing
# one. Vanishing gradients point at the critic (under-trained, or GP not
# applied). For a GAN, a loss-trend WARNING is not by itself a failure:
# G's loss moves as the critic gets stronger.


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  GAN EVALUATION METRICS:
  [x] FID (Frechet Inception Distance): the standard automated GAN
      metric. Measures distributional distance in a learned feature
      space — only comparable within the same extractor, so read it
      against a real-vs-real floor.
  [x] Mode coverage: counting how many classes the generator produces.
      A sharp but repetitive GAN fails this test.
  [x] Shannon entropy: quantifying generation diversity.
      Max = log2(10) = 3.32 for MNIST (uniform across all digits).
  [x] Minimum class share: no category can be underrepresented
      (prevents hidden bias in synthetic datasets).
  [x] Novelty ratio: a near-copy check against the training set
      (necessary for privacy, nowhere near sufficient).

  MODEL REGISTRY:
  [x] Registered both generators with FID + coverage + entropy metrics
  [x] Registered metrics let you compare generator versions side by side

  QA PIPELINE FOR PRODUCTION:
  [x] Defined quantitative thresholds (FID vs floor, coverage >= 8/10, ...)
  [x] Automated evaluation — no "looks good to me" subjectivity
  [x] Stakeholder-ready report whose decision is computed, not asserted
  [x] Named the gates this script does NOT run (downstream test, privacy)

  REAL-WORLD APPLICATION:
  [x] Insurance synthetic data QA: evidence before production
  [x] CRO-ready quality report with pass/fail per metric
  [x] The business risk of biased or memorised synthetic data

  KEY INSIGHTS:
  - FID alone is not enough: it blends fidelity and diversity into one
    number, so it cannot say WHICH of the two is failing
  - Mode coverage alone is not enough: a generator can cover all modes
    but produce blurry, low-quality images
  - You need BOTH distribution match (FID) AND diversity (coverage +
    entropy) — and a novelty check before anyone calls the data safe
  - Re-run the same gate on every new generator version; the registry
    keeps the metrics side by side so regressions are visible

  WHEN TO USE WHICH GAN:
  - Vanilla GAN: quick prototyping, simple datasets, no stability needs
  - WGAN-GP: production use, medical/financial data, stability required
  - Both need the same evaluation pipeline — the QA doesn't change,
    only the generator architecture does.

  GAN vs VAE (M5 Exercise 1) vs Diffusion:
  - GANs:      Sharp images, hard to train, fast sampling
  - VAEs:      Blurry but stable, continuous latent space, fast
  - Diffusion: Sharp + stable, best quality, SLOW sampling
"""
)
