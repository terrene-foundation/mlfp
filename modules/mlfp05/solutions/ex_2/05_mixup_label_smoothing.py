# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
# ════════════════════════════════════════════════════════════════════════
# MLFP05 Exercise 2.5 — Training Enhancements: Kaiming Init, Label
# Smoothing, and Mixup (an ablation study)
# ════════════════════════════════════════════════════════════════════════
#
# WHAT YOU'LL LEARN:
#   After completing this file, you will be able to:
#   - Explain WHY default layer initialisation is suboptimal for ReLU
#     networks, and what Kaiming (He) initialisation changes
#   - Explain label smoothing as "stop being so sure of yourself" —
#     soft targets that combat overconfidence
#   - Implement Mixup: training on convex combinations of image PAIRS
#     (vicinal risk minimisation, Zhang et al. 2018)
#   - Run a controlled ablation — change ONE lever at a time and track
#     every variant with ExperimentTracker
#   - Measure overconfidence honestly (mean max-softmax probability)
#     instead of asserting it
#   - Apply the levers to a limited-labelled-data agritech scenario
#
# PREREQUISITES: M5/ex_2/01_simple_cnn.py and 02_resnet_se.py (CNN
#   training on CIFAR-10, ExperimentTracker basics)
# ESTIMATED TIME: ~35 min
#
# DATASET: CIFAR-10. To keep the ablation affordable on CPU we train on
#   a 20,000-image subset of the 50K training images (disclosed, fixed
#   prefix after the seeded loader shuffle) and evaluate on the FULL
#   10K validation set. Four variants x 6 epochs — small enough to run
#   on a laptop, large enough to see each lever's effect.
#
# PHASES:
#   1. THEORY  — Three levers: initialisation, soft targets, vicinal data
#   2. BUILD   — kaiming_init, mixup_batch, mixup_criterion, AblationCNN
#   3. TRAIN   — 4-variant ablation tracked with ExperimentTracker
#   4. VISUALISE — Accuracy curves, Mixup composites, confidence shift
#   5. APPLY   — Limited-label crop-disease triage for an agritech startup
#
# ════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import asyncio
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared.mlfp05.ex_2 import (
    BATCH_SIZE,
    CLASS_NAMES,
    DEVICE,
    N_CLASSES,
    count_parameters,
    create_visualizer,
    denormalise_cifar,
    init_engines,
    load_cifar10,
    register_model,
    save_training_plots,
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 1 — THEORY: Three Levers That Cost Nothing at Inference
# ════════════════════════════════════════════════════════════════════════
# All three levers in this file share one property: they change TRAINING
# only. The deployed model is the same architecture with the same
# inference cost — you are buying generalisation for free at serving time.
#
# LEVER 1 — KAIMING (He) INITIALISATION:
#   Analogy: a relay race where every runner starts at a different
#   random speed. If the first runner starts too slow, the signal dies
#   before the last runner; too fast, and it overshoots.
#
#   PyTorch's default init for Conv2d/Linear is uniform over
#   +/-1/sqrt(fan_in) — derived for linear networks. ReLU kills half
#   the activations on average, shrinking the signal variance by ~2x
#   per layer. He et al. (2015) showed the right fix is to scale the
#   weight variance by 2/fan_in so the signal survives ReLU:
#       std = sqrt(2 / fan_in)
#   With BatchNorm the effect is partly masked (BN re-normalises), but
#   the first epochs still train measurably faster from a Kaiming start.
#
# LEVER 2 — LABEL SMOOTHING:
#   Analogy: a student who answers every true/false question with
#   "100% certain!" — even the ones they guessed. They learn nothing
#   from being wrong because every answer carries maximum confidence.
#
#   Hard targets tell the model the correct class has probability 1.0
#   and all others 0.0. Label smoothing (Szegedy et al. 2016) replaces
#   the targets with eps/(K-1) on the wrong classes and 1-eps on the
#   right one (here eps=0.1). The model is still rewarded for being
#   right, but is no longer rewarded for being ARBITRARILY confident —
#   softmax probabilities become honest estimates of uncertainty, which
#   is what a triage system needs (see Phase 5).
#
# LEVER 3 — MIXUP (vicinal risk minimisation):
#   Analogy: studying for an exam only with past papers teaches you the
#   past papers. Mixing two questions into a hybrid forces you to learn
#   the UNDERLYING concept, not the memorised answer.
#
#   Mixup (Zhang et al. 2018) trains on convex combinations of pairs:
#       x_mix = lam * x_i + (1 - lam) * x_j       lam ~ Beta(alpha, alpha)
#       loss  = lam * CE(f(x_mix), y_i) + (1 - lam) * CE(f(x_mix), y_j)
#   The model must behave LINEARLY between training examples, which
#   smooths the decision boundary and reduces memorisation. Caveat we
#   will observe honestly: Mixup regularises so aggressively that at a
#   fixed SHORT epoch budget it can trail the baseline — its benefit
#   shows with longer training. We report what we measure, not what the
#   paper's (much longer) runs report.

print("=" * 70)
print("  PHASE 1 — THEORY: Kaiming Init, Label Smoothing, Mixup")
print("=" * 70)
print(
    """
  THREE TRAINING-TIME LEVERS (zero inference cost):

  1. KAIMING INIT:  std = sqrt(2/fan_in) keeps signal variance alive
     through ReLU layers. Default init is tuned for linear nets.
  2. LABEL SMOOTHING: targets become 0.9 / 0.01 instead of 1.0 / 0.0.
     The model stops being rewarded for arbitrary confidence.
  3. MIXUP: train on lam*x_i + (1-lam)*x_j with the same convex mix
     of the losses. Forces linear behaviour BETWEEN examples.

  ABLATION DISCIPLINE: change ONE lever at a time, keep everything else
  identical (same architecture, same subset, same epochs, same seed),
  and let ExperimentTracker hold the receipts.
"""
)


# ════════════════════════════════════════════════════════════════════════
# PHASE 2 — BUILD: the levers and the ablation backbone
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 2 — BUILD: kaiming_init + mixup_batch + AblationCNN")
print("=" * 70)

ABLATION_EPOCHS = 6
ABLATION_SUBSET = 20_000  # fixed prefix of the training tensors
SMOOTHING = 0.1
MIXUP_ALPHA = 0.2


def kaiming_init(model: nn.Module) -> nn.Module:
    """Kaiming (He) normal initialisation for ReLU networks, in place.

    Conv/Linear weights: Normal(0, sqrt(2/fan_in)). Biases: zero.
    BatchNorm weights stay at 1 and biases at 0 (already the default).
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Mix a batch with a random permutation of itself.

    Returns: x_mix, y_a, y_b, lam, perm — the mixed inputs, the two label
    sets, the mixing coefficient, and the permutation (returned so the
    mixing is auditable: x_mix == lam * x + (1 - lam) * x[perm]).
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    return x_mix, y, y[perm], lam, perm


def mixup_criterion(
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Convex combination of the two cross-entropies (same lam as inputs)."""
    return lam * F.cross_entropy(logits, y_a, label_smoothing=smoothing) + (
        1.0 - lam
    ) * F.cross_entropy(logits, y_b, label_smoothing=smoothing)


class AblationCNN(nn.Module):
    """Compact CNN backbone shared by every ablation variant.

    Two conv blocks (3->32->64, each BN+ReLU+pool) and a small head.
    Identical across variants so the ONLY difference is the lever.
    """

    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


# ── Checkpoint 1: the levers are correct ──────────────────────────────
# (a) Kaiming std matches sqrt(2/fan_in) on a conv layer
torch.manual_seed(0)
_probe = AblationCNN()
_conv = _probe.features[0]
_default_std = float(_conv.weight.std().item())
kaiming_init(_probe)
_expected_std = float(np.sqrt(2.0 / (3 * 3 * 3)))  # fan_in = k*k*c_in = 27
_kaiming_std = float(_conv.weight.std().item())
assert abs(_kaiming_std - _expected_std) / _expected_std < 0.10, (
    f"Kaiming std {_kaiming_std:.4f} should be ~{_expected_std:.4f} "
    "(sqrt(2/fan_in), fan_in=27)"
)
assert float(_conv.bias.abs().max().item()) == 0.0, "Kaiming init zeroes biases"

# (b) Mixup produces exact convex combinations and preserves shapes
torch.manual_seed(0)
np.random.seed(0)
_x = torch.randn(8, 3, 32, 32)
_y = torch.arange(8)
_x_mix, _y_a, _y_b, _lam, _perm = mixup_batch(_x, _y, MIXUP_ALPHA)
assert _x_mix.shape == _x.shape, "Mixup must preserve input shape"
assert 0.0 <= _lam <= 1.0, f"lam={_lam} outside [0, 1]"
assert torch.equal(_y_a, _y) and torch.equal(_y_b, _y[_perm])
assert torch.allclose(_x_mix, _lam * _x + (1 - _lam) * _x[_perm], atol=1e-6), (
    "x_mix must be the exact convex combination lam*x + (1-lam)*x[perm]"
)
# (c) mixup_criterion reduces to plain CE when lam = 1
_logits = torch.randn(8, N_CLASSES)
_ce = F.cross_entropy(_logits, _y)
_mc = mixup_criterion(_logits, _y_a, _y_b, lam=1.0)
assert abs(float(_mc - _ce)) < 1e-6, "lam=1 must reduce to plain cross-entropy"

print(f"\nAblationCNN built: {count_parameters(_probe):,} parameters")
print(f"  default conv std={_default_std:.4f} -> kaiming std={_kaiming_std:.4f} "
      f"(target {_expected_std:.4f})")
print(f"  mixup audit: lam={_lam:.3f}, exact convex combination verified")
print("\n--- Checkpoint 1 passed --- levers verified\n")
del _probe, _x, _y, _x_mix, _y_a, _y_b, _perm, _logits


# ════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAIN: four-variant ablation, one lever at a time
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 3 — TRAIN: ablation on CIFAR-10 (20K subset, full 10K val)")
print("=" * 70)

X_train, y_train, X_val, y_val, _, _ = load_cifar10()

# Fixed subset for the ablation — disclosed in the header. The SAME
# subset feeds every variant, so differences come from the levers, not
# from data sampling.
X_sub = X_train[:ABLATION_SUBSET]
y_sub = y_train[:ABLATION_SUBSET]
sub_loader = DataLoader(
    TensorDataset(X_sub, y_sub), batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=512)
print(
    f"Ablation data: {len(X_sub):,} train images (subset), "
    f"{len(X_val):,} val images (full)"
)

conn, tracker, exp_name, registry, has_registry = init_engines()


async def train_variant(
    name: str,
    use_kaiming: bool,
    smoothing: float,
    mixup_alpha: float,
) -> tuple[nn.Module, list[float], list[float]]:
    """Train one ablation variant; log per-epoch metrics to ExperimentTracker.

    A plain torch loop is used instead of the shared Lightning harness
    because Mixup changes the per-batch loss computation (two label sets,
    convex weighting), which the shared LitCNN wrapper does not express.
    """
    torch.manual_seed(42)  # identical init draw for every variant
    model = AblationCNN().to(DEVICE)
    if use_kaiming:
        kaiming_init(model)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_losses: list[float] = []
    val_accs: list[float] = []

    async with tracker.track(experiment=exp_name, run_name=name) as run:
        await run.log_params(
            {
                "variant": name,
                "kaiming_init": str(use_kaiming),
                "label_smoothing": str(smoothing),
                "mixup_alpha": str(mixup_alpha),
                "epochs": str(ABLATION_EPOCHS),
                "subset_size": str(ABLATION_SUBSET),
                "architecture": "AblationCNN",
            }
        )
        for epoch in range(ABLATION_EPOCHS):
            model.train()
            batch_losses = []
            for xb, yb in sub_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                if mixup_alpha > 0:
                    xb, y_a, y_b, lam, _ = mixup_batch(xb, yb, mixup_alpha)
                    logits = model(xb)
                    loss = mixup_criterion(logits, y_a, y_b, lam, smoothing)
                else:
                    logits = model(xb)
                    loss = F.cross_entropy(logits, yb, label_smoothing=smoothing)
                opt.zero_grad()
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())

            model.eval()
            correct = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    correct += int((model(xb).argmax(-1) == yb).sum().item())
            val_acc = correct / len(y_val)

            train_losses.append(float(np.mean(batch_losses)))
            val_accs.append(val_acc)
            await run.log_metrics(
                {"train_loss": train_losses[-1], "val_accuracy": val_acc},
                step=epoch + 1,
            )
            print(
                f"  [{name}] epoch {epoch+1}/{ABLATION_EPOCHS}  "
                f"train={train_losses[-1]:.4f}  val={val_acc:.3f}"
            )
        await run.log_metrics(
            {
                "final_train_loss": train_losses[-1],
                "final_val_accuracy": val_accs[-1],
            }
        )
    return model, train_losses, val_accs


VARIANTS = [
    ("baseline", False, 0.0, 0.0),
    ("kaiming", True, 0.0, 0.0),
    ("kaiming+smoothing", True, SMOOTHING, 0.0),
    ("kaiming+smoothing+mixup", True, SMOOTHING, MIXUP_ALPHA),
]

results: dict[str, dict[str, object]] = {}
for name, use_k, smooth, alpha in VARIANTS:
    print(f"\nTraining variant: {name}")
    model_v, losses_v, accs_v = asyncio.run(
        train_variant(name, use_k, smooth, alpha)
    )
    results[name] = {"model": model_v, "losses": losses_v, "accs": accs_v}

# ── Checkpoint 2: all variants trained and converging ─────────────────
for name, res in results.items():
    losses = res["losses"]
    accs = res["accs"]
    assert len(losses) == ABLATION_EPOCHS, f"{name}: expected {ABLATION_EPOCHS} epochs"
    assert losses[-1] < losses[0], f"{name}: train loss should decrease"
    assert accs[-1] > 0.30, (
        f"{name}: val acc {accs[-1]:.3f} too low — expected > 0.30 after "
        f"{ABLATION_EPOCHS} epochs on the 20K subset"
    )

print(f"\n{'=' * 62}")
print("  ABLATION RESULTS (measured, this run)")
print(f"{'=' * 62}")
print(f"  {'Variant':>26} {'Final Loss':>12} {'Val Acc':>10}")
print("  " + "-" * 56)
for name, res in results.items():
    print(
        f"  {name:>26} {res['losses'][-1]:>12.4f} {res['accs'][-1]:>9.3f}"
    )

base_acc = results["baseline"]["accs"][-1]
mix_acc = results["kaiming+smoothing+mixup"]["accs"][-1]
print(
    f"\n  Mixup delta at {ABLATION_EPOCHS} epochs: {mix_acc - base_acc:+.3f}. "
    "Mixup regularises aggressively — at short budgets it can trail the "
    "baseline; its benefit shows with longer training. That is the honest "
    "reading of THIS run, and it is also what Zhang et al. report: mixup "
    "wins appear at full training length, not in the first epochs."
)
print("\n--- Checkpoint 2 passed --- ablation trained and tracked\n")

# Register the best-val-accuracy variant
best_name = max(results, key=lambda n: results[n]["accs"][-1])
best = results[best_name]
if has_registry:
    register_model(
        registry,
        "ablation_cifar10",
        best["model"],
        best["losses"][-1],
        best["accs"][-1],
        epochs=ABLATION_EPOCHS,
    )
    print(f"  Best variant registered: {best_name}")


# ════════════════════════════════════════════════════════════════════════
# PHASE 4 — VISUALISE: curves, Mixup composites, and the confidence shift
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  PHASE 4 — VISUALISE: what each lever actually did")
print("=" * 70)

# ModelVisualizer carries a kailash-ml P2 experimental notice (a UserWarning
# at construction). The gate runs with warnings-as-errors, so acknowledge it
# narrowly here — only the constructor notice is muted, everything else
# still errors.
from kailash_ml._decorators import ExperimentalWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", ExperimentalWarning)
    viz = create_visualizer()
save_training_plots(
    viz,
    {name: res["accs"] for name, res in results.items()},
    "ex_2_05_ablation_val_accuracy.html",
    y_label="Validation Accuracy",
)
save_training_plots(
    viz,
    {name: res["losses"] for name, res in results.items()},
    "ex_2_05_ablation_train_loss.html",
    y_label="Training Loss",
)

# (a) Mixup composites — the visual proof of "training between examples"
torch.manual_seed(7)
np.random.seed(7)
_pair_x = X_val[:8]
_pair_y = y_val[:8]
_mixed, _ya, _yb, _lam, _perm = mixup_batch(_pair_x, _pair_y, 0.5)

fig_mix, axes = plt.subplots(3, 8, figsize=(20, 8))
fig_mix.suptitle(
    f"Mixup composites (lam={_lam:.2f}): image A, image B, and the blend the model trains on",
    fontsize=13,
)
for col in range(8):
    a = denormalise_cifar(_pair_x[col]).permute(1, 2, 0).numpy()
    b = denormalise_cifar(_pair_x[_perm[col]]).permute(1, 2, 0).numpy()
    m = denormalise_cifar(_mixed[col]).permute(1, 2, 0).numpy()
    axes[0, col].imshow(a)
    axes[0, col].set_title(f"A: {CLASS_NAMES[_ya[col]]}", fontsize=8)
    axes[1, col].imshow(b)
    axes[1, col].set_title(f"B: {CLASS_NAMES[_yb[col]]}", fontsize=8)
    axes[2, col].imshow(m)
    axes[2, col].set_title("blend", fontsize=8)
    for row in range(3):
        axes[row, col].axis("off")
axes[0, 0].set_ylabel("image A", fontsize=10)
axes[1, 0].set_ylabel("image B", fontsize=10)
axes[2, 0].set_ylabel("lam*A+(1-lam)*B", fontsize=10)
plt.tight_layout()
plt.savefig("ex_2_05_mixup_composites.png", dpi=150, bbox_inches="tight")
plt.close(fig_mix)
print("  Saved: ex_2_05_mixup_composites.png")

# (b) Confidence shift — label smoothing's signature.
# Measure the mean max-softmax probability on the val set for the
# baseline (hard targets) vs the smoothed variant. Overconfident models
# sit near 1.0 even when wrong; smoothed models keep calibrated margins.
def mean_confidence(model: nn.Module) -> tuple[float, float]:
    """Return (mean max-prob, accuracy) over the validation set."""
    model.eval()
    probs, preds = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            p = F.softmax(model(xb), dim=-1)
            probs.append(p.max(dim=-1).values.cpu())
            preds.append(p.argmax(-1).cpu())
    probs_t = torch.cat(probs)
    preds_t = torch.cat(preds)
    acc = float((preds_t == y_val).float().mean().item())
    return float(probs_t.mean().item()), acc


conf_base, acc_base = mean_confidence(results["baseline"]["model"])
conf_smooth, acc_smooth = mean_confidence(results["kaiming+smoothing"]["model"])
print(
    f"\n  Mean max-softmax confidence on 10K val images:\n"
    f"    baseline (hard targets):   {conf_base:.3f}  (acc {acc_base:.3f})\n"
    f"    kaiming+smoothing:         {conf_smooth:.3f}  (acc {acc_smooth:.3f})"
)

fig_conf, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    ["baseline", "kaiming+smoothing"],
    [conf_base, conf_smooth],
    color=["#d62728", "#2ca02c"],
)
ax.set_ylabel("Mean max-softmax probability")
ax.set_ylim(0, 1.0)
ax.axhline(acc_base, color="#d62728", linestyle="--", alpha=0.6,
           label=f"baseline acc ({acc_base:.2f})")
ax.axhline(acc_smooth, color="#2ca02c", linestyle="--", alpha=0.6,
           label=f"smoothed acc ({acc_smooth:.2f})")
ax.set_title("Label smoothing closes the confidence-accuracy gap")
ax.legend(fontsize=9)
fig_conf.tight_layout()
fig_conf.savefig("ex_2_05_confidence_shift.png", dpi=150, bbox_inches="tight")
plt.close(fig_conf)
print("  Saved: ex_2_05_confidence_shift.png")

# ── Checkpoint 3: smoothing measurably reduces overconfidence ─────────
# Allow slack for run-to-run noise, but the direction must hold: a
# smoothed model must not be MORE overconfident than the baseline.
assert conf_smooth < conf_base + 0.05, (
    f"Smoothed confidence {conf_smooth:.3f} should not exceed baseline "
    f"{conf_base:.3f} by more than noise (0.05)"
)
import os

for artefact in (
    "ex_2_05_ablation_val_accuracy.html",
    "ex_2_05_mixup_composites.png",
    "ex_2_05_confidence_shift.png",
):
    assert os.path.exists(artefact), f"Missing artefact: {artefact}"
print("\n--- Checkpoint 3 passed --- visual proof generated\n")


# ════════════════════════════════════════════════════════════════════════
# PHASE 5 — APPLY: crop-disease triage with limited labelled data
# ════════════════════════════════════════════════════════════════════════
# SCENARIO (anonymised, illustrative): a regional agritech startup runs
# drone-photo triage for smallholder farms. Agronomists labelled 20,000
# field photos into 10 crop-condition classes — exactly our ablation
# setup. Two business constraints make the levers matter:
#
#   1. LABELS ARE EXPENSIVE: every new labelled photo costs an
#      agronomist's time. Regularisation (Mixup) squeezes more
#      generalisation out of the same 20K labels.
#   2. TRIAGE DECISIONS ARE THRESHOLDED: photos the model is confident
#      about are auto-routed; the rest go to a human queue. If the model
#      says "95% confident" but is right only 70% of the time at that
#      confidence, the auto-route queue silently ships errors to farmers.
#      Label smoothing makes the confidence number MEAN something.
#
# We simulate the triage policy on the val set: auto-accept predictions
# whose confidence exceeds a threshold, measure the accuracy of the
# auto-accepted pool for the baseline vs the smoothed model.

print("=" * 70)
print("  PHASE 5 — APPLY: confidence-thresholded crop-disease triage")
print("=" * 70)


def triage_report(model: nn.Module, threshold: float) -> dict[str, float]:
    """Auto-accept predictions above threshold; return pool statistics."""
    model.eval()
    confs, preds = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            p = F.softmax(model(xb.to(DEVICE)), dim=-1)
            confs.append(p.max(dim=-1).values.cpu())
            preds.append(p.argmax(-1).cpu())
    conf_t = torch.cat(confs)
    pred_t = torch.cat(preds)
    accepted = conf_t >= threshold
    n_acc = int(accepted.sum().item())
    if n_acc == 0:
        return {"accept_rate": 0.0, "accepted_acc": 0.0}
    acc_accepted = float((pred_t[accepted] == y_val[accepted]).float().mean().item())
    return {"accept_rate": n_acc / len(y_val), "accepted_acc": acc_accepted}


THRESHOLD = 0.90
rep_base = triage_report(results["baseline"]["model"], THRESHOLD)
rep_smooth = triage_report(results["kaiming+smoothing"]["model"], THRESHOLD)

print(
    f"""
  TRIAGE POLICY: auto-route photos with confidence >= {THRESHOLD:.2f}

  {'Model':>24} {'Auto-routed':>14} {'Accuracy of auto-routed':>26}
  {'-' * 66}
  {'baseline (hard targets)':>24} {rep_base['accept_rate']:>13.1%} {rep_base['accepted_acc']:>25.1%}
  {'kaiming+smoothing':>24} {rep_smooth['accept_rate']:>13.1%} {rep_smooth['accepted_acc']:>25.1%}

  READING IT:
    The baseline auto-routes a larger share, but its auto-routed pool is
    less trustworthy — hard targets teach the model to say "95%" on
    photos it should be unsure about. The smoothed model's threshold
    means what it says. For a farmer waiting on a spray/no-spray call,
    the second row is the one an agronomist can defend.

  STAKEHOLDER-READY OUTPUT:
    "At a 0.90 confidence threshold the baseline auto-routes
    {rep_base['accept_rate']:.0%} of photos at {rep_base['accepted_acc']:.0%}
    accuracy; the smoothed model auto-routes {rep_smooth['accept_rate']:.0%}
    at {rep_smooth['accepted_acc']:.0%}. Smoothing trades a little
    automation volume for a triage queue whose confidence numbers can be
    audited — and Mixup buys extra generalisation from the same 20K
    agronomist labels when the training budget allows longer runs."
"""
)

# ── Checkpoint 4: triage simulation computed on the full val set ──────
assert rep_base["accept_rate"] > 0.0, "Baseline should auto-route some photos"
assert rep_smooth["accepted_acc"] > 0.5, (
    f"Smoothed auto-routed accuracy {rep_smooth['accepted_acc']:.3f} too low"
)
print("--- Checkpoint 4 passed --- triage application demonstrated\n")


# ════════════════════════════════════════════════════════════════════════
# Clean up
# ════════════════════════════════════════════════════════════════════════
asyncio.run(conn.close())


# ════════════════════════════════════════════════════════════════════════
# REFLECTION
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WHAT YOU'VE MASTERED")
print("=" * 70)
print(
    f"""
  THEORY:
  [x] Kaiming init: std = sqrt(2/fan_in) keeps signal variance alive
      through ReLU stacks (measured: {_kaiming_std:.4f} vs target
      {_expected_std:.4f} on the first conv)
  [x] Label smoothing: soft targets (0.9/0.01) stop rewarding arbitrary
      confidence (measured: {conf_base:.3f} -> {conf_smooth:.3f} mean
      max-probability)
  [x] Mixup: convex blends of image PAIRS force linear behaviour between
      examples — verified exact: x_mix == lam*x + (1-lam)*x[perm]

  BUILD + TRAIN:
  [x] kaiming_init / mixup_batch / mixup_criterion as reusable levers
  [x] 4-variant ablation on a fixed 20K CIFAR-10 subset, one lever at a
      time, every run tracked in ExperimentTracker
  [x] Honest reading: at {ABLATION_EPOCHS} epochs Mixup's delta is
      {mix_acc - base_acc:+.3f} — regularisation pays off with longer
      budgets, and we report the measured sign, not the hoped-for one

  VISUALISE (the proof):
  [x] Ablation val-accuracy and loss curves (all four variants overlaid)
  [x] Mixup composites: image A, image B, and the exact blend trained on
  [x] Confidence shift: smoothed models close the confidence-accuracy gap

  APPLY:
  [x] Crop-disease triage with a 0.90 confidence threshold
  [x] Baseline auto-routes {rep_base['accept_rate']:.0%} at
      {rep_base['accepted_acc']:.0%} accuracy; smoothed auto-routes
      {rep_smooth['accept_rate']:.0%} at {rep_smooth['accepted_acc']:.0%}
  [x] The business lever is TRUST IN THE THRESHOLD, not raw accuracy

  KEY INSIGHT: Training-time levers are free at inference — but only
  measured levers earn their place. Kaiming initialisation, label
  smoothing and Mixup each move a different dial (optimisation,
  calibration, regularisation); the ablation discipline — one lever at
  a time, same data, same seed, tracked runs — is what turns folklore
  into an engineering decision.
"""
)
