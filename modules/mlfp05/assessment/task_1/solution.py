# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""
MLFP05 — Assessment Task 1: Autoencoder Anomaly Detection (REFERENCE SOLUTION)

Undercomplete AE trained on healthy-only telemetry; reconstruction error is the
anomaly score. Deterministic, CPU-only, < 15s.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

INPUT_DIM = 12
SEED = 7


def make_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic synthetic sensor telemetry — identical to starter."""
    rng = np.random.default_rng(SEED)
    basis = rng.normal(size=(3, INPUT_DIM))

    def healthy(n: int) -> np.ndarray:
        z = rng.normal(size=(n, 3))
        return (z @ basis + 0.15 * rng.normal(size=(n, INPUT_DIM))).astype(np.float32)

    def anomaly(n: int) -> np.ndarray:
        return (2.5 * rng.normal(size=(n, INPUT_DIM))).astype(np.float32)

    X_train = healthy(800)
    n_test_healthy, n_test_anom = 320, 80
    X_test = np.vstack([healthy(n_test_healthy), anomaly(n_test_anom)])
    y_test = np.concatenate(
        [np.zeros(n_test_healthy, dtype=int), np.ones(n_test_anom, dtype=int)]
    )
    perm = rng.permutation(len(y_test))
    return X_train, X_test[perm], y_test[perm]


def solve() -> dict:
    torch.manual_seed(SEED)
    X_train, X_test, y_test = make_dataset()

    latent_dim = 4  # undercomplete: 4 < 12 (healthy manifold is rank-3)

    class AE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(INPUT_DIM, 16),
                nn.ReLU(),
                nn.Linear(16, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 16),
                nn.ReLU(),
                nn.Linear(16, INPUT_DIM),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = AE()

    train_tensor = torch.tensor(X_train)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=64, shuffle=True)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _epoch in range(40):
        for (batch,) in loader:
            recon = model(batch)
            loss = F.mse_loss(recon, batch)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test)
        recon = model(test_tensor)
        scores = ((test_tensor - recon) ** 2).mean(dim=1).cpu().numpy()

    return {
        "model": model,
        "scores": scores,
        "y_test": y_test,
        "input_dim": INPUT_DIM,
        "latent_dim": latent_dim,
    }


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    out = solve()
    auc = roc_auc_score(out["y_test"], out["scores"])
    s, yt = out["scores"], out["y_test"]
    sep = s[yt == 1].mean() / max(s[yt == 0].mean(), 1e-9)
    print(f"latent_dim={out['latent_dim']}  AUC={auc:.3f}  separation={sep:.2f}")
