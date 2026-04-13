# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 — Exercise 5: Generative Models (DCGAN + WGAN-GP on MNIST)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   - Build Generator and Discriminator networks for image generation
#   - Train a DCGAN-style GAN with binary cross-entropy (the min-max game)
#   - Implement a Wasserstein critic with gradient penalty (WGAN-GP)
#   - Compute FID (Frechet Inception Distance) — the standard GAN metric
#   - Diagnose mode collapse via 10-class coverage and Shannon entropy
#   - Track G/D losses, hyperparams, and FID with ExperimentTracker
#   - Register trained generators in ModelRegistry with quality metrics
#   - Visualise training dynamics with ModelVisualizer
#
# PREREQUISITES: M5/ex_1 (autoencoders/VAE), M5/ex_2 (CNN training loops).
# ESTIMATED TIME: ~120-150 min
# DATASET: MNIST — 60,000 real 28x28 grayscale digits (full, no subsampling)
#
# TASKS:
#   1. Load full MNIST (60K), set up ExperimentTracker and ModelRegistry
#   2. Build Generator and Discriminator (MLP for CPU compatibility)
#   3. Train vanilla GAN with BCEWithLogitsLoss, log to ExperimentTracker
#   4. Implement gradient penalty and train WGAN-GP, log to tracker
#   5. Compute FID score for both generators
#   6. Mode coverage diagnostic — verify all 10 digit classes generated
#   7. Register generators in ModelRegistry with FID and coverage metrics
#   8. Visualise G vs D loss curves, compare training dynamics
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision

from kailash.db import ConnectionManager
from kailash_ml import ModelVisualizer
from kailash_ml.engines.experiment_tracker import ExperimentTracker
from kailash_ml.engines.model_registry import ModelRegistry
from kailash_ml.types import MetricSpec

from shared.kailash_helpers import get_device, setup_environment

setup_environment()
torch.manual_seed(42)
np.random.seed(42)
device = get_device()
print(f"Using device: {device}")


# ════════════════════════════════════════════════════════════════════════
# TASK 1 — Load FULL MNIST (60K) and set up kailash-ml engines
# ════════════════════════════════════════════════════════════════════════
# GANs need the full dataset — with too few images the discriminator
# memorises digits and gives the generator no useful gradient signal.

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "mlfp05" / "mnist"
DATA_DIR.mkdir(parents=True, exist_ok=True)

train_set = torchvision.datasets.MNIST(
    root=str(DATA_DIR),
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

X_real = torch.stack([train_set[i][0] for i in range(len(train_set))])
y_real = torch.tensor(
    [train_set[i][1] for i in range(len(train_set))], dtype=torch.long
)
X_real = (X_real * 2.0 - 1.0).to(device)  # Scale to [-1, 1] for tanh generator

print(
    f"MNIST: {len(X_real)} digits, shape {tuple(X_real.shape[1:])}, "
    f"pixel range [{X_real.min():.2f}, {X_real.max():.2f}]"
)
class_dist = ", ".join(f"{c}={int((y_real == c).sum())}" for c in range(10))
print(f"  class distribution: {class_dist}")


# TODO: Implement setup_engines — ConnectionManager, ExperimentTracker,
#       create_experiment (name="m5_gans"), and ModelRegistry
async def setup_engines():
    # TODO: Create ConnectionManager("sqlite:///mlfp05_gans.db") and call initialize()
    # Hint: conn = ConnectionManager("sqlite:///mlfp05_gans.db"); await conn.initialize()
    conn = ____
    await ____
    # TODO: Create ExperimentTracker and create experiment named "m5_gans"
    # Hint: tracker = ExperimentTracker(conn)
    #       exp_name = await tracker.create_experiment(name="m5_gans", description=...)
    tracker = ____
    exp_name = await tracker.create_experiment(name=____, description=____)
    try:
        # TODO: Create ModelRegistry(conn)
        # Hint: registry = ModelRegistry(conn)
        registry = ____
        has_registry = True
    except Exception as e:
        registry, has_registry = None, False
        print(f"  Note: ModelRegistry setup skipped ({e})")
    return conn, tracker, exp_name, registry, has_registry


conn, tracker, exp_name, registry, has_registry = asyncio.run(setup_engines())

real_loader = DataLoader(
    TensorDataset(X_real), batch_size=128, shuffle=True, drop_last=True
)

# ── Checkpoint 1 ─────────────────────────────────────────────────────
assert X_real.shape[0] == 60000, f"Expected 60K MNIST, got {X_real.shape[0]}"
assert tracker is not None, "ExperimentTracker should be initialised"
# INTERPRETATION: The full 60K set provides enough diversity for the
# generator to learn the manifold of handwritten digits across all 10
# classes. The ExperimentTracker logs G/D losses and FID scores so we
# can compare vanilla GAN vs WGAN-GP quantitatively.
print("\n--- Checkpoint 1 passed --- data loaded and engines initialised\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 2 — Generator and Discriminator (MLP architecture)
# ════════════════════════════════════════════════════════════════════════
# MLP rather than full DCGAN convolutions so the exercise runs on CPU.
LATENT_DIM = 64
IMG_DIM = 28 * 28


class Generator(nn.Module):
    """z -> 784-d -> (1, 28, 28). BatchNorm + Tanh (DCGAN best practice)."""

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 256):
        super().__init__()
        # TODO: Build self.net as nn.Sequential:
        #   Linear(latent_dim, hidden) -> BatchNorm1d(hidden) -> LeakyReLU(0.2)
        #   -> Linear(hidden, hidden*2) -> BatchNorm1d(hidden*2) -> LeakyReLU(0.2)
        #   -> Linear(hidden*2, IMG_DIM) -> Tanh()
        # Hint: output is flattened pixels; forward() reshapes to (B, 1, 28, 28)
        self.net = ____

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)


class Discriminator(nn.Module):
    """28x28 -> scalar logit. Dropout prevents D from overfitting to real images."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        # TODO: Build self.net as nn.Sequential:
        #   Flatten() -> Linear(IMG_DIM, hidden*2) -> LeakyReLU(0.2) -> Dropout(0.3)
        #   -> Linear(hidden*2, hidden) -> LeakyReLU(0.2) -> Dropout(0.3) -> Linear(hidden, 1)
        # Hint: no Sigmoid — BCEWithLogitsLoss handles the sigmoid internally
        self.net = ____

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Checkpoint 2 ─────────────────────────────────────────────────────
_tg, _td = Generator().to(device), Discriminator().to(device)
_tz = torch.randn(4, LATENT_DIM, device=device)
assert _tg(_tz).shape == (4, 1, 28, 28), "Generator output shape wrong"
assert _td(_tg(_tz)).shape == (4, 1), "Discriminator output shape wrong"
del _tg, _td, _tz
# INTERPRETATION: G maps noise z (64 dims) to a 28x28 image. D maps any
# image to a scalar "realness" score. The adversarial game: G maximises
# D's score on fakes, D minimises it.
print("\n--- Checkpoint 2 passed --- G and D architectures verified\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 3 — Train a vanilla GAN with binary cross-entropy
# ════════════════════════════════════════════════════════════════════════
# L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
# L_G = -E[log D(G(z))]
async def train_vanilla_gan_async(epochs: int = 15, lr: float = 2e-4):
    G, D = Generator().to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    # TODO: Define BCEWithLogitsLoss for adversarial training
    # Hint: bce = nn.BCEWithLogitsLoss()
    bce = ____
    g_losses, d_losses = [], []

    async with tracker.run(experiment_name=exp_name, run_name="vanilla_gan") as ctx:
        await ctx.log_params(
            {
                "architecture": ____,  # Hint: "Vanilla_GAN_MLP"
                "latent_dim": ____,  # Hint: str(LATENT_DIM)
                "lr": ____,  # Hint: str(lr)
                "epochs": ____,  # Hint: str(epochs)
                "batch_size": ____,  # Hint: "128"
                "loss_type": ____,  # Hint: "BCEWithLogitsLoss"
                "optimizer": ____,  # Hint: "Adam(0.5,0.999)"
            }
        )

        for epoch in range(epochs):
            eg, ed = [], []
            for (real_batch,) in real_loader:
                bs = real_batch.size(0)
                # ── Discriminator ─────────────────────────────────────────
                z = torch.randn(bs, LATENT_DIM, device=device)
                fake = G(z).detach()
                # TODO: D loss = bce on real (target=1) + bce on fake (target=0)
                # Hint: loss_d = bce(D(real_batch), torch.ones(bs,1,device=device))
                #              + bce(D(fake),       torch.zeros(bs,1,device=device))
                loss_d = ____
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                # ── Generator ─────────────────────────────────────────────
                z = torch.randn(bs, LATENT_DIM, device=device)
                # TODO: G loss = fool D by labelling fakes as real (target=1)
                # Hint: loss_g = bce(D(G(z)), torch.ones(bs, 1, device=device))
                loss_g = ____
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                eg.append(loss_g.item())
                ed.append(loss_d.item())

            avg_g, avg_d = float(np.mean(eg)), float(np.mean(ed))
            g_losses.append(avg_g)
            d_losses.append(avg_d)
            await ctx.log_metrics(
                {____: avg_g, ____: avg_d},  # Hint: "g_loss", "d_loss"
                step=epoch + 1,
            )
            print(f"  [GAN] epoch {epoch+1:2d}/{epochs}  D={avg_d:.3f}  G={avg_g:.3f}")

        await ctx.log_metrics(
            {
                ____: g_losses[-1],
                ____: d_losses[-1],
            }  # Hint: "final_g_loss", "final_d_loss"
        )

    return G, g_losses, d_losses


def train_vanilla_gan(epochs: int = 15, lr: float = 2e-4):
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_vanilla_gan_async(epochs, lr))


# ════════════════════════════════════════════════════════════════════════
# TASK 4 — WGAN with Gradient Penalty
# ════════════════════════════════════════════════════════════════════════
# L_critic = E[D(fake)] - E[D(real)] + lambda * GP
# L_G      = -E[D(fake)]
# GP (Gulrajani 2017): interpolate real+fake, penalise ||grad D||_2 != 1
def gradient_penalty(
    D: nn.Module, real: torch.Tensor, fake: torch.Tensor
) -> torch.Tensor:
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
    # TODO: Gradient penalty = mean of (||grad||_2 - 1)^2
    # Hint: ((grad.reshape(batch, -1).norm(2, dim=1) - 1) ** 2).mean()
    return ____


async def train_wgan_gp_async(
    epochs: int = 20, lr: float = 1e-4, n_critic: int = 5, lam: float = 10.0
):
    G, D = Generator().to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
    g_losses, d_losses = [], []

    async with tracker.run(experiment_name=exp_name, run_name="wgan_gp") as ctx:
        await ctx.log_params(
            {
                "architecture": ____,  # Hint: "WGAN_GP_MLP"
                "latent_dim": ____,  # Hint: str(LATENT_DIM)
                "lr": ____,  # Hint: str(lr)
                "epochs": ____,  # Hint: str(epochs)
                "batch_size": ____,  # Hint: "128"
                "loss_type": ____,  # Hint: "Wasserstein+GP"
                "n_critic": ____,  # Hint: str(n_critic)
                "gp_lambda": ____,  # Hint: str(lam)
                "optimizer": ____,  # Hint: "Adam(0.5,0.9)"
            }
        )

        for epoch in range(epochs):
            eg, ed = [], []
            for (real_batch,) in real_loader:
                bs = real_batch.size(0)
                # ── Critic (n_critic steps per G step) ────────────────────
                for _ in range(n_critic):
                    z = torch.randn(bs, LATENT_DIM, device=device)
                    fake = G(z).detach()
                    gp = gradient_penalty(D, real_batch, fake)
                    # TODO: Wasserstein critic loss + gradient penalty
                    # Hint: loss_d = D(fake).mean() - D(real_batch).mean() + lam * gp
                    loss_d = ____
                    opt_d.zero_grad()
                    loss_d.backward()
                    opt_d.step()
                # ── Generator ─────────────────────────────────────────────
                z = torch.randn(bs, LATENT_DIM, device=device)
                # TODO: G loss = minimise -D(G(z)) (maximise D score on fakes)
                # Hint: loss_g = -D(G(z)).mean()
                loss_g = ____
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                eg.append(loss_g.item())
                ed.append(loss_d.item())

            avg_g, avg_d = float(np.mean(eg)), float(np.mean(ed))
            g_losses.append(avg_g)
            d_losses.append(avg_d)
            await ctx.log_metrics(
                {____: avg_g, ____: avg_d},  # Hint: "g_loss", "d_loss"
                step=epoch + 1,
            )
            print(
                f"  [WGAN-GP] epoch {epoch+1:2d}/{epochs}  D={avg_d:.3f}  G={avg_g:.3f}"
            )

        await ctx.log_metrics(
            {
                ____: g_losses[-1],
                ____: d_losses[-1],
            }  # Hint: "final_g_loss", "final_d_loss"
        )

    return G, g_losses, d_losses


def train_wgan_gp(
    epochs: int = 20, lr: float = 1e-4, n_critic: int = 5, lam: float = 10.0
):
    """Sync wrapper — one asyncio.run per training call."""
    return asyncio.run(train_wgan_gp_async(epochs, lr, n_critic, lam))


print("\n== Training Vanilla GAN on MNIST (60K) ==")
G_gan, gan_g_losses, gan_d_losses = train_vanilla_gan(epochs=15)

# ── Checkpoint 3 ─────────────────────────────────────────────────────
assert len(gan_g_losses) == 15, f"Expected 15 epochs, got {len(gan_g_losses)}"
# INTERPRETATION: In a well-trained GAN, D loss hovers around ln(4) ~= 1.386
# (D is ~50% accurate). If D loss drops to 0, D "wins" and G gets no gradient.
print("\n--- Checkpoint 3 passed --- vanilla GAN trained\n")

print("\n== Training WGAN-GP on MNIST (60K) ==")
G_wgan, wgan_g_losses, wgan_d_losses = train_wgan_gp(epochs=20)

# ── Checkpoint 4 ─────────────────────────────────────────────────────
assert len(wgan_g_losses) == 20, f"Expected 20 epochs, got {len(wgan_g_losses)}"
# INTERPRETATION: WGAN critic loss approximates Wasserstein distance —
# unlike BCE, it's a meaningful quality metric (lower = closer distributions).
print("\n--- Checkpoint 4 passed --- WGAN-GP trained\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 5 — FID Score (Frechet Inception Distance)
# ════════════════════════════════════════════════════════════════════════
# FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))
# Lower FID = closer to real distribution = better generator.


class LeNetFeatureExtractor(nn.Module):
    """CNN for FID features. Returns 64-dim vectors (like Inception's pool3)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def compute_fid(
    extractor: nn.Module, real: torch.Tensor, generated: torch.Tensor
) -> float:
    """FID between real and generated using eigendecomposition (no scipy)."""
    extractor.eval()

    def _extract(images):
        feats = []
        with torch.no_grad():
            for i in range(0, len(images), 512):
                feats.append(
                    extractor.extract_features(images[i : i + 512]).cpu().numpy()
                )
        return np.concatenate(feats)

    rf, gf = _extract(real), _extract(generated)
    mu_r, mu_g = rf.mean(0), gf.mean(0)
    sig_r, sig_g = np.cov(rf, rowvar=False), np.cov(gf, rowvar=False)

    diff = mu_r - mu_g
    product = sig_r @ sig_g
    eigvals, eigvecs = np.linalg.eigh(product)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_prod = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    return float(diff @ diff + np.trace(sig_r + sig_g - 2 * sqrt_prod))


print("\n== Training feature extractor (for FID + mode coverage) ==")
fid_ext = LeNetFeatureExtractor().to(device)
fid_opt = torch.optim.Adam(fid_ext.parameters(), lr=1e-3)
X_01 = (X_real + 1.0) / 2.0  # [0, 1] for classifier
y_dev = y_real.to(device)

for epoch in range(5):
    losses = []
    for xb, yb in DataLoader(TensorDataset(X_01, y_dev), batch_size=256, shuffle=True):
        loss = F.cross_entropy(fid_ext(xb), yb)
        fid_opt.zero_grad()
        loss.backward()
        fid_opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        acc = (fid_ext(X_01[:10000]).argmax(-1) == y_dev[:10000]).float().mean()
    print(f"  epoch {epoch+1}/5  loss={np.mean(losses):.3f}  acc={acc:.3f}")
fid_ext.eval()

print("\n== Computing FID scores ==")
N_FID = 10000
G_gan.eval()
G_wgan.eval()
with torch.no_grad():
    gan_fake_01 = (G_gan(torch.randn(N_FID, LATENT_DIM, device=device)) + 1) / 2
    wgan_fake_01 = (G_wgan(torch.randn(N_FID, LATENT_DIM, device=device)) + 1) / 2

rng = np.random.default_rng(42)
real_sub = X_01[rng.choice(len(X_01), N_FID, replace=False)]

fid_gan = compute_fid(fid_ext, real_sub, gan_fake_01)
fid_wgan = compute_fid(fid_ext, real_sub, wgan_fake_01)
print(f"  Vanilla GAN FID: {fid_gan:.2f}")
print(f"  WGAN-GP    FID:  {fid_wgan:.2f}  (lower = better)")


async def _log_fid_async():
    async with tracker.run(experiment_name=exp_name, run_name="fid_evaluation") as ctx:
        await ctx.log_param(____, str(N_FID))  # Hint: "n_generated"
        await ctx.log_metrics(
            {____: fid_gan, ____: fid_wgan}
        )  # Hint: "fid_vanilla_gan", "fid_wgan_gp"


asyncio.run(_log_fid_async())

# ── Checkpoint 5 ─────────────────────────────────────────────────────
assert fid_gan >= 0 and fid_wgan >= 0, "FID must be non-negative"
# INTERPRETATION: FID = 0 means identical distributions. Typical MNIST
# GAN FID after a few epochs: 10-100. Production papers target FID < 10.
print("\n--- Checkpoint 5 passed --- FID scores computed\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 6 — Mode coverage diagnostic
# ════════════════════════════════════════════════════════════════════════
# Mode collapse: G finds a few "easy" digits (1s, 7s) and stops exploring.
def mode_coverage(G, classifier, n=5000):
    """Returns (n_classes, per_class_counts, shannon_entropy)."""
    G.eval()
    classifier.eval()
    with torch.no_grad():
        fake_01 = (G(torch.randn(n, LATENT_DIM, device=device)) + 1) / 2
        preds = classifier(fake_01).argmax(-1).cpu().numpy()
    unique, counts = np.unique(preds, return_counts=True)
    probs = counts / counts.sum()
    # TODO: Shannon entropy: -sum(p * log2(p + epsilon)) for numerical stability
    # Hint: entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
    entropy = ____
    return int(len(unique)), {int(k): int(v) for k, v in zip(unique, counts)}, entropy


cov_gan, dist_gan, ent_gan = mode_coverage(G_gan, fid_ext)
cov_wgan, dist_wgan, ent_wgan = mode_coverage(G_wgan, fid_ext)

print(f"Vanilla GAN: {cov_gan}/10 classes, entropy={ent_gan:.2f}/3.32")
print(f"  {dist_gan}")
print(f"WGAN-GP:     {cov_wgan}/10 classes, entropy={ent_wgan:.2f}/3.32")
print(f"  {dist_wgan}")


async def _log_mode_coverage_async():
    async with tracker.run(experiment_name=exp_name, run_name="mode_coverage") as ctx:
        await ctx.log_metrics(
            {
                ____: float(cov_gan),  # Hint: "gan_coverage"
                ____: ent_gan,  # Hint: "gan_entropy"
                ____: float(cov_wgan),  # Hint: "wgan_coverage"
                ____: ent_wgan,  # Hint: "wgan_entropy"
            }
        )


asyncio.run(_log_mode_coverage_async())

# ── Checkpoint 6 ─────────────────────────────────────────────────────
assert cov_gan >= 1 and cov_wgan >= 1, "Should produce at least 1 class"
assert 0 <= ent_gan <= np.log2(10) + 0.01, "Entropy out of range"
# INTERPRETATION: WGAN-GP typically achieves better coverage because
# the Wasserstein distance provides gradients even when distributions
# don't overlap (vanilla GAN's JS divergence gives zero gradient when
# D perfectly separates real from fake).
print("\n--- Checkpoint 6 passed --- mode coverage measured\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 7 — Register generators in ModelRegistry
# ════════════════════════════════════════════════════════════════════════
async def register_generators():
    if not has_registry:
        print("  ModelRegistry not available -- skipping")
        return {}
    versions = {}
    for name, model, fid, cov, ent in [
        ("dcgan_generator", G_gan, fid_gan, cov_gan, ent_gan),
        ("wgan_gp_generator", G_wgan, fid_wgan, cov_wgan, ent_wgan),
    ]:
        # TODO: Serialize model.state_dict() and register with FID + coverage metrics
        # Hint: artifact=pickle.dumps(model.state_dict())
        ver = await registry.register_model(
            name=f"m5_{name}",
            artifact=____,
            metrics=[
                MetricSpec(name="fid_score", value=fid),
                MetricSpec(name="mode_coverage", value=float(cov)),
                MetricSpec(name="class_entropy", value=ent),
            ],
        )
        versions[name] = ver
        print(f"  Registered {name}: v={ver.version}, FID={fid:.2f}, coverage={cov}/10")
    return versions


model_versions = asyncio.run(register_generators())

# ── Checkpoint 7 ─────────────────────────────────────────────────────
if has_registry:
    assert len(model_versions) == 2, "Should register both generators"
# INTERPRETATION: The registry stores artifacts alongside FID and coverage.
# To decide whether a new training run improved quality, compare FID against
# the previous version — lower FID = closer to real = promote to serving.
print("\n--- Checkpoint 7 passed --- generators registered\n")


# ════════════════════════════════════════════════════════════════════════
# TASK 8 — Visualise training dynamics
# ════════════════════════════════════════════════════════════════════════
viz = ModelVisualizer()

# TODO: Plot vanilla GAN G and D loss curves with viz.training_history
# Hint: metrics={"GAN G loss": gan_g_losses, "GAN D loss": gan_d_losses}
fig_gan = viz.training_history(
    metrics=____,
    x_label="Epoch",
    y_label="Loss",
)
fig_gan.write_html("ex_5_vanilla_gan_training.html")

# TODO: Plot WGAN-GP G and D loss curves
# Hint: metrics={"WGAN-GP G loss": wgan_g_losses, "WGAN-GP D loss": wgan_d_losses}
fig_wgan = viz.training_history(
    metrics=____,
    x_label="Epoch",
    y_label="Loss",
)
fig_wgan.write_html("ex_5_wgan_gp_training.html")

fig_all = viz.training_history(
    metrics={
        "GAN G": gan_g_losses,
        "GAN D": gan_d_losses,
        "WGAN G": wgan_g_losses,
        "WGAN D": wgan_d_losses,
    },
    x_label="Epoch",
    y_label="Loss",
)
fig_all.write_html("ex_5_combined_training.html")
print("Training curves saved to ex_5_*.html")

# ── Checkpoint 8 ─────────────────────────────────────────────────────
import os

assert os.path.exists("ex_5_combined_training.html"), "Combined HTML should exist"
print("\n--- Checkpoint 8 passed --- visualisations saved\n")


print("\n" + "=" * 70)
print("  EXPERIMENT SUMMARY")
print("=" * 70)
print(f"\n  Experiment: {exp_name}")
print(f"  Dataset: MNIST (60,000 images), latent_dim={LATENT_DIM}")
print(f"\n  {'Metric':<25} {'Vanilla GAN':>14} {'WGAN-GP':>14}")
print(f"  {'-'*53}")
print(f"  {'Epochs':<25} {'15':>14} {'20':>14}")
print(f"  {'Final G loss':<25} {gan_g_losses[-1]:>14.3f} {wgan_g_losses[-1]:>14.3f}")
print(f"  {'Final D loss':<25} {gan_d_losses[-1]:>14.3f} {wgan_d_losses[-1]:>14.3f}")
print(f"  {'FID score':<25} {fid_gan:>14.2f} {fid_wgan:>14.2f}")
print(f"  {'Mode coverage':<25} {cov_gan:>13}/10 {cov_wgan:>13}/10")
print(f"  {'Class entropy':<25} {ent_gan:>14.2f} {ent_wgan:>14.2f}")

better = "WGAN-GP" if fid_wgan < fid_gan else "Vanilla GAN"
print(f"\n  Best generator by FID: {better}")

asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    """
  GAN ARCHITECTURES:
  [x] Built Generator (z -> image) and Discriminator (image -> score)
  [x] Trained vanilla GAN with BCEWithLogitsLoss (Goodfellow 2014 min-max)
  [x] Implemented WGAN-GP with Gulrajani 2017 gradient penalty — GP
      replaces weight clipping which caused capacity underuse

  EVALUATION AND DIAGNOSTICS:
  [x] Computed FID (Frechet Inception Distance) — the standard metric.
      Lower FID = closer to real distribution = better generator
  [x] Measured mode coverage — healthy GAN covers all 10 digit classes
  [x] Used Shannon entropy to quantify generation diversity

  ML ENGINEERING:
  [x] Trained on FULL MNIST (60K images, no subsampling)
  [x] Tracked G/D losses, hyperparams, and FID with ExperimentTracker
  [x] Registered generators in ModelRegistry with FID + coverage metrics
  [x] Visualised training dynamics with ModelVisualizer

  KEY INSIGHTS:
  - WGAN-GP provides gradients even when D perfectly separates real/fake
    (vanilla GAN's JS divergence gives zero gradient in this case)
  - FID is the standard evaluation metric — loss curves alone don't
    tell you generation quality
  - Mode coverage detects collapse that FID might miss (a generator
    producing perfect 1s has low FID on that mode but zero diversity)

  WHEN TO USE WHICH: GANs = sharp images, hard to train. VAEs = blurry
  but stable. Diffusion = sharp + stable but slow sampling.

  Next: Exercise 6 — Graph Neural Networks (GCNs) for structured data...
"""
)
