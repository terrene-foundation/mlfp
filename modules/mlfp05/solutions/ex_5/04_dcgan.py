# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 5.4 — DCGAN: Convolutional GAN for MNIST
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain WHY convolutional GANs dominate image generation: weight
#     sharing matches the spatial structure of images (a "3" is a "3"
#     wherever it sits in the frame)
#   - Build the DCGAN pair (Radford et al. 2016): ConvTranspose2d
#     generator with BatchNorm+ReLU+Tanh; strided-conv discriminator
#     with LeakyReLU — no pooling layers anywhere
#   - Apply the DCGAN training recipe: Adam(2e-4, betas=(0.5, 0.999)),
#     BatchNorm in both nets, no bias into BatchNorm layers
#   - Measure quality honestly with FID (LeNet feature space) and mode
#     coverage — never "the grid looks nice" alone
#   - Compare against the MLP GAN of 01_vanilla_gan.py through
#     ExperimentTracker receipts
#   - Apply to synthetic defect imagery for a manufacturing lab
#
# PREREQUISITES: M5/ex_5/01_vanilla_gan.py (GAN training loop, FID,
#   mode coverage). The MLP GAN there borrows the DCGAN RECIPE; this
#   file implements the DCGAN ARCHITECTURE.
# ESTIMATED TIME: ~35 min
#
# DATASET: MNIST (60K, scaled to [-1, 1]). Trains on CPU: 6 epochs at
#   batch 128 — enough for recognisable digits; say so honestly.
#
# PHASES:
#   1. THEORY  — Why convolutions; the DCGAN guidelines
#   2. BUILD   — ConvGenerator + ConvDiscriminator
#   3. TRAIN   — adversarial loop, tracked with ExperimentTracker
#   4. VISUALISE — sample grid, progression, FID + mode coverage
#   5. APPLY   — synthetic defect imagery for a manufacturing lab
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import copy

import numpy as np
import torch
import torch.nn as nn

from shared.mlfp05.ex_5 import (
    LATENT_DIM,
    OUTPUT_DIR,
    close_engines,
    compute_fid,
    init_environment,
    load_mnist,
    load_mnist_test,
    mode_coverage,
    plot_image_grid,
    plot_latent_interpolation,
    plot_loss_curves,
    plot_training_progression,
    register_generator,
    setup_engines,
    train_feature_extractor,
)

device = init_environment()


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Why Convolutions, and the DCGAN Guidelines
# ════════════════════════════════════════════════════════════════════════
# The MLP GAN of 01_vanilla_gan.py treats a 28x28 digit as a flat
# 784-vector. It works — but it must learn from scratch that pixel
# (i, j) is related to pixel (i, j+1). Every digit position, every
# stroke orientation, is a separate pattern for the MLP.
#
# CONVOLUTION matches the data's structure by construction:
#   - WEIGHT SHARING: one filter detects a curve ANYWHERE in the frame
#   - LOCALITY: a filter reads a 4x4 patch, not all 784 pixels at once
#   - TRANSLATION STRUCTURE: shifting the digit shifts the feature map
#
# THE DCGAN GUIDELINES (Radford, Metz & Chintala, 2016) — empirical
# rules that turned GAN training from alchemy into engineering:
#
#   1. Replace pooling with STRIDED convolutions (D) and
#      FRACTIONALLY-STRIDED convolutions (ConvTranspose, G). The network
#      learns its own up/downsampling instead of a fixed max/avg rule.
#   2. BatchNorm in BOTH networks (except G's output layer and D's input
#      layer) — stabilises the adversarial tug-of-war.
#   3. ReLU in G, LeakyReLU(0.2) in D. Leaky keeps gradients flowing
#      even when D rejects a region confidently.
#   4. Adam with lr=2e-4 and betas=(0.5, 0.999) — the lower first moment
#      damps oscillation in the two-player game.
#   5. No bias on layers feeding BatchNorm: BN subtracts the batch mean,
#      so such a bias is cancelled and receives zero gradient.

print("=" * 70)
print("  PHASE 1 — THEORY: convolutional GANs and the DCGAN guidelines")
print("=" * 70)
print(
    """
  MLP GAN: 784 flat inputs, every spatial relationship relearned.
  DCGAN:   weight sharing + locality match the structure of images.

  Radford et al. (2016) guidelines:
    1. Strided conv (D) / ConvTranspose (G) instead of pooling
    2. BatchNorm in both nets (not G output / D input)
    3. ReLU in G, LeakyReLU(0.2) in D
    4. Adam lr=2e-4, betas=(0.5, 0.999)
    5. No bias into BatchNorm layers (BN cancels it, zero gradient)
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: ConvGenerator + ConvDiscriminator
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: the convolutional pair")
print("=" * 70)

DCGAN_EPOCHS = 6
DCGAN_LR = 2e-4
DCGAN_BETAS = (0.5, 0.999)


class ConvGenerator(nn.Module):
    """DCGAN generator: z -> ConvTranspose stack -> (1, 28, 28) in [-1, 1].

    Pure convolutional path (no Linear layers):
        z (B, latent, 1, 1)
        -> ConvTranspose(latent->256, k7, s1, p0)  -> BN -> ReLU  [7x7]
        -> ConvTranspose(256->128,  k4, s2, p1)    -> BN -> ReLU  [14x14]
        -> ConvTranspose(128->1,    k4, s2, p1)    -> Tanh        [28x28]
    No bias on ConvTranspose layers feeding BatchNorm (guideline 5).
    """

    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Accept the flat (B, latent) layout the shared helpers produce
        return self.net(z.view(-1, self.latent_dim, 1, 1))


class ConvDiscriminator(nn.Module):
    """DCGAN discriminator: (1, 28, 28) -> scalar logit (no sigmoid).

        Conv(1->64,   k4, s2, p1) -> LeakyReLU        [14x14]  (no BN on input)
        Conv(64->128, k4, s2, p1) -> BN -> LeakyReLU  [7x7]
        Flatten -> Linear(128*7*7 -> 1)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


# ── Checkpoint 1: architecture shapes and parameter counts ────────────
_g = ConvGenerator().to(device)
_d = ConvDiscriminator().to(device)
_z = torch.randn(4, LATENT_DIM, device=device)
with torch.no_grad():
    _fake = _g(_z)
    _logit = _d(_fake)
assert _fake.shape == (4, 1, 28, 28), f"G output {_fake.shape} != (4, 1, 28, 28)"
assert float(_fake.min()) >= -1.0 and float(_fake.max()) <= 1.0, (
    "Tanh output must stay in [-1, 1]"
)
assert _logit.shape == (4, 1), f"D output {_logit.shape} != (4, 1)"

from shared.mlfp05.ex_5 import Discriminator as MLPDiscriminator
from shared.mlfp05.ex_5 import Generator as MLPGenerator

n_g_conv = sum(p.numel() for p in _g.parameters())
n_d_conv = sum(p.numel() for p in _d.parameters())
n_g_mlp = sum(p.numel() for p in MLPGenerator().parameters())
n_d_mlp = sum(p.numel() for p in MLPDiscriminator().parameters())
print(f"\n  Generator:     {n_g_conv:>10,} params (MLP GAN: {n_g_mlp:,})")
print(f"  Discriminator: {n_d_conv:>10,} params (MLP GAN: {n_d_mlp:,})")
print("  Spatial path: z -> 7x7x256 -> 14x14x128 -> 28x28x1 (all learned)")
print("\n--- Checkpoint 1 passed --- DCGAN architecture verified\n")
del _g, _d, _z, _fake, _logit


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: the adversarial loop, DCGAN recipe
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"  PHASE 3 — TRAIN: DCGAN on MNIST ({DCGAN_EPOCHS} epochs, CPU budget)")
print("=" * 70)

X_real, y_real, real_loader = load_mnist(device)
conn, tracker, exp_name, registry = setup_engines()


async def train_dcgan() -> tuple[nn.Module, nn.Module, list[float], list[float], dict]:
    """Adversarial training with the DCGAN recipe, tracked honestly."""
    G = ConvGenerator().to(device)
    D = ConvDiscriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=DCGAN_LR, betas=DCGAN_BETAS)
    opt_d = torch.optim.Adam(D.parameters(), lr=DCGAN_LR, betas=DCGAN_BETAS)
    bce = nn.BCEWithLogitsLoss()
    g_losses, d_losses = [], []
    snapshots = {0: copy.deepcopy(G.state_dict())}

    async with tracker.track(experiment=exp_name, run_name="dcgan_conv") as run:
        await run.log_params(
            {
                "architecture": "DCGAN_conv",
                "latent_dim": str(LATENT_DIM),
                "lr": str(DCGAN_LR),
                "betas": str(DCGAN_BETAS),
                "epochs": str(DCGAN_EPOCHS),
                "batch_size": "128",
                "loss": "BCEWithLogits (non-saturating G)",
            }
        )
        for epoch in range(DCGAN_EPOCHS):
            eg, ed = [], []
            for (real_batch,) in real_loader:
                bs = real_batch.size(0)

                # Train D: real -> 1, fake -> 0
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake = G(z).detach()
                loss_d = bce(D(real_batch), torch.ones(bs, 1, device=device)) + bce(
                    D(fake), torch.zeros(bs, 1, device=device)
                )
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

                # Train G (non-saturating): make D call fakes "real"
                z = torch.randn(bs, LATENT_DIM, device=device)
                loss_g = bce(D(G(z)), torch.ones(bs, 1, device=device))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()

                eg.append(loss_g.item())
                ed.append(loss_d.item())

            g_losses.append(float(np.mean(eg)))
            d_losses.append(float(np.mean(ed)))
            await run.log_metrics(
                {"g_loss": g_losses[-1], "d_loss": d_losses[-1]}, step=epoch + 1
            )
            print(
                f"  [DCGAN] epoch {epoch+1}/{DCGAN_EPOCHS}  "
                f"D={d_losses[-1]:.3f}  G={g_losses[-1]:.3f}"
            )
            if (epoch + 1) in {1, 2, 4, DCGAN_EPOCHS}:
                snapshots[epoch + 1] = copy.deepcopy(G.state_dict())
        await run.log_metrics(
            {"final_g_loss": g_losses[-1], "final_d_loss": d_losses[-1]}
        )
    return G, D, g_losses, d_losses, snapshots


G_conv, D_conv, g_losses, d_losses, snapshots = asyncio.run(train_dcgan())

# ══════════════════════════════════════════════════════════════════
# DIAGNOSTIC CHECKPOINT — Prescription Pad on both networks
# ══════════════════════════════════════════════════════════════════
from kailash_ml.diagnostics import run_diagnostic_checkpoint
from shared.mlfp05.diagnostics import print_prescription_pad

_diag_bce = nn.BCEWithLogitsLoss()


def _g_loss(m, batch):
    bs = batch[0].size(0)
    z = torch.randn(bs, LATENT_DIM, device=device)
    return _diag_bce(D_conv(m(z)), torch.ones(bs, 1, device=device))


def _d_loss(m, batch):
    bs = batch[0].size(0)
    with torch.no_grad():
        fake = G_conv(torch.randn(bs, LATENT_DIM, device=device))
    return _diag_bce(m(batch[0]), torch.ones(bs, 1, device=device)) + _diag_bce(
        m(fake), torch.zeros(bs, 1, device=device)
    )


_, g_findings = run_diagnostic_checkpoint(
    G_conv, real_loader, _g_loss, title="DCGAN — Generator",
    n_batches=6, train_losses=g_losses, show=False,
)
print_prescription_pad(g_findings, "DCGAN — Generator")

_, d_findings = run_diagnostic_checkpoint(
    D_conv, real_loader, _d_loss, title="DCGAN — Discriminator",
    n_batches=6, train_losses=d_losses, show=False,
)
print_prescription_pad(d_findings, "DCGAN — Discriminator")

# ── Checkpoint 2: training converged, game stayed alive ───────────────
assert len(g_losses) == DCGAN_EPOCHS
assert g_losses[-1] < g_losses[0], "G loss should decrease as fakes improve"
assert d_losses[-1] > 0.05, (
    f"D loss collapsed to {d_losses[-1]:.4f} — D died, G gets no gradient"
)
print("\n--- Checkpoint 2 passed --- DCGAN trained, adversarial game alive\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: samples, progression, FID + mode coverage
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: recognisable digits, measured")
print("=" * 70)

# (a) Sample grid from the final generator
G_conv.eval()
with torch.no_grad():
    final_samples = G_conv(torch.randn(64, LATENT_DIM, device=device))
plot_image_grid(
    final_samples,
    title=f"DCGAN samples after {DCGAN_EPOCHS} epochs",
    save_path=str(OUTPUT_DIR / "04_dcgan_samples.png"),
)

# (b) Progression: same fixed z through training snapshots
plot_training_progression(
    G_conv, device, snapshots,
    title="DCGAN training progression (fixed latent vectors)",
    save_path=str(OUTPUT_DIR / "04_dcgan_progression.png"),
)

# (c) Loss dynamics
plot_loss_curves(
    g_losses, d_losses, title="DCGAN training dynamics",
    save_path=str(OUTPUT_DIR / "04_dcgan_losses.png"),
)

# (d) Latent interpolation — evidence of a continuous manifold
plot_latent_interpolation(
    G_conv, device,
    title="DCGAN latent interpolation",
    save_path=str(OUTPUT_DIR / "04_dcgan_interpolation.png"),
)

# (e) Honest quality metrics: FID in LeNet feature space + mode coverage.
# Train the shared feature extractor on MNIST (same protocol as 03).
extractor = train_feature_extractor(X_real, y_real, device)
X_test, y_test = load_mnist_test(device)

with torch.no_grad():
    eval_samples = G_conv(torch.randn(2000, LATENT_DIM, device=device))
    # The trivial baseline every FID number needs: uniform pixel noise in
    # the same [-1, 1] range. "FID 115" means nothing until you know the
    # noise floor — the same discipline as perplexity-vs-vocab in ex_3/07.
    noise_samples = torch.rand(2000, 1, 28, 28, device=device) * 2.0 - 1.0
fid = compute_fid(extractor, X_test[:2000], eval_samples)
fid_noise = compute_fid(extractor, X_test[:2000], noise_samples)
n_classes_covered, per_class, entropy = mode_coverage(G_conv, extractor, device)

print(f"\n  QUALITY METRICS (measured, this run):")
print(f"    FID (LeNet feature space, 2K samples vs 2K test): {fid:.2f}")
print(f"    FID reference — uniform pixel noise:              {fid_noise:.2f}")
print(f"    Mode coverage: {n_classes_covered}/10 digit classes generated")
print(f"    Class entropy: {entropy:.2f} bits (uniform = 3.32)")
print(
    "    Compare against your 01_vanilla_gan.py run — same extractor "
    "protocol, MLP architecture. The receipts are both in ExperimentTracker."
)

register_generator(registry, "dcgan_generator", G_conv, fid, n_classes_covered,
                   entropy)

# ── Checkpoint 3: quality floors (conservative, honest) ───────────────
import os

assert n_classes_covered >= 6, (
    f"Only {n_classes_covered}/10 digit classes generated — mode collapse"
)
assert entropy > 2.0, f"Class entropy {entropy:.2f} too low — low diversity"
assert np.isfinite(fid) and fid >= 0.0, "FID must be a finite distance"
assert fid < 0.5 * fid_noise, (
    f"FID {fid:.2f} should be far below the noise reference {fid_noise:.2f} — "
    "a trained conv GAN is much closer to real digits than pixel noise is"
)
for artefact in (
    "04_dcgan_samples.png",
    "04_dcgan_progression.png",
    "04_dcgan_losses.png",
    "04_dcgan_interpolation.png",
):
    assert os.path.exists(OUTPUT_DIR / artefact), f"Missing: {artefact}"
print("\n--- Checkpoint 3 passed --- visual proof + honest metrics\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: synthetic defect imagery for a manufacturing lab
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a precision-manufacturing lab
# qualifies visual-inspection models. Real defect photos are rare (the
# line is designed NOT to produce defects), so the lab augments training
# data with synthetic defect images.
#
# WHY THE DCGAN ARCHITECTURE MATTERS HERE:
#   Defects are SPATIAL: a scratch is a scratch anywhere on the wafer.
#   An MLP generator wastes capacity relearning "edge" at every pixel
#   offset; a convolutional generator shares that filter everywhere.
#   With the same rare-defect budget, the conv net spends its parameters
#   on VARIETY of defects, not on re-deriving locality.
#
# THE HONEST LIMITS (what we tell the quality manager):
#   - FID is computed in a LeNet feature space trained on the SAME domain
#     (MNIST here). It is comparable across our generators, NOT to FID
#     numbers published with InceptionV3 on natural images.
#   - Mode coverage answers "do we generate all defect types?" — the
#     question that matters for qualification data. A beautiful grid of
#     3 defect types out of 10 is a failed augmentation program.
#   - Synthetic data AUGMENTS real defects; it does not replace the
#     final real-defect validation gate.

print("=" * 70)
print("  PHASE 5 — APPLY: synthetic defect imagery, honestly scoped")
print("=" * 70)
print(
    f"""
  SYNTHETIC DATA PROGRAM — QUALIFICATION REPORT (measured, this run):

    Generator:              DCGAN (convolutional, {n_g_conv:,} params)
    Training:               {DCGAN_EPOCHS} epochs on 60K reference images
    FID (LeNet space):      {fid:.2f}  (lower = closer to real distribution)
    FID noise reference:    {fid_noise:.2f}  (uniform pixel noise)
    Mode coverage:          {n_classes_covered}/10 classes
    Class entropy:          {entropy:.2f} bits (uniform = 3.32)

  DECISION RULES THE LAB CAN DEFEND:
    1. Coverage gate: augmentation proceeds only if all classes appear
       ({n_classes_covered}/10 {'meets' if n_classes_covered == 10 else 'does not fully meet'}
       the 10/10 bar this run).
    2. Fidelity gate: FID tracked in the registry; a regression on
       retraining blocks promotion of the new generator.
    3. Real-data gate: final model qualification ALWAYS uses held-out
       real defects. Synthetic data trains; reality validates.

  STAKEHOLDER-READY OUTPUT:
    "The convolutional generator covers {n_classes_covered} of 10 defect
    classes with class entropy {entropy:.2f} bits and FID {fid:.2f} in
    our domain's feature space. Because defects are spatial patterns,
    the convolutional architecture spends its capacity on defect variety
    rather than relearning locality — which is what the MLP generator
    had to do. Synthetic images augment the training set; the
    qualification gate remains real held-out defects."
"""
)

# ── Checkpoint 4: application metrics are the measured ones ───────────
assert 0 < n_classes_covered <= 10
assert fid >= 0.0, "FID is a distance — cannot be negative"
print("--- Checkpoint 4 passed --- synthetic-data application demonstrated\n")

asyncio.run(close_engines(conn))


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Convolutions match image structure: weight sharing + locality —
      a stroke detector is learned ONCE and applied everywhere
  [x] DCGAN guidelines (Radford 2016): strided/ConvTranspose instead of
      pooling, BatchNorm in both nets, ReLU(G)/LeakyReLU(D),
      Adam(2e-4, betas=(0.5, 0.999)), no bias into BatchNorm

  BUILD + TRAIN:
  [x] ConvGenerator: z -> 7x7x256 -> 14x14x128 -> 28x28x1, pure
      ConvTranspose ({n_g_conv:,} params vs MLP's {n_g_mlp:,})
  [x] ConvDiscriminator: strided convs to a scalar logit
  [x] {DCGAN_EPOCHS}-epoch adversarial loop, tracked in ExperimentTracker;
      diagnostic Prescription Pad run on BOTH networks

  VISUALISE (the proof):
  [x] Sample grid + fixed-z progression across epochs
  [x] Latent interpolation (continuous manifold evidence)
  [x] Honest metrics: FID {fid:.2f} against a noise reference of
      {fid_noise:.2f} (LeNet space), coverage {n_classes_covered}/10,
      entropy {entropy:.2f} bits

  APPLY:
  [x] Synthetic defect imagery program with three defensible gates:
      coverage, fidelity (registry-tracked FID), and a real-data final
      qualification gate
  [x] Stated limit: synthetic data trains; reality validates

  KEY INSIGHT: "DCGAN" is two things — an ARCHITECTURE (convolutions)
  and a RECIPE (the five guidelines). The MLP GAN in 01 already used the
  recipe; this file showed the architecture. When a GAN underperforms,
  diagnose which half is missing before reaching for a bigger model —
  and let FID + mode coverage, not a pretty grid, deliver the verdict.
"""
)
