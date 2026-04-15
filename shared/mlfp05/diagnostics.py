# Copyright 2026 Terrene Foundation
# SPDX-License-Identifier: Apache-2.0
"""DL Diagnostics Toolkit — clinical instruments for deep-network training.

Wraps PyTorch forward/backward hooks into a student-friendly API that
surfaces the four failure modes professionals must recognise:

    1. Stethoscope  — loss-curve shape (under/over-fit, instability)
    2. Blood Test   — gradient flow per layer (vanishing / exploding)
    3. X-Ray        — activation statistics & dead neurons
    4. Prescription — automated diagnosis with actionable suggestions

Typical use inside an exercise::

    from shared.mlfp05.diagnostics import DLDiagnostics

    with DLDiagnostics(model) as diag:
        diag.track_gradients()
        diag.track_activations()
        diag.track_dead_neurons()
        for epoch in range(epochs):
            for batch in dataloader:
                loss = train_step(model, batch)
                diag.record_batch(loss=loss.item(),
                                  lr=optimizer.param_groups[0]["lr"])
            diag.record_epoch(val_loss=evaluate(model, val_loader))
        diag.plot_training_dashboard().show()
        diag.report()

All DataFrames are Polars. All plots are Plotly. No matplotlib, no pandas.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
import torch
import torch.nn as nn
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader

from shared.kailash_helpers import get_device

logger = logging.getLogger(__name__)

# Module names whose outputs are "dead-neuron sensitive" — we track fraction
# of zero outputs for these specifically.
_DEAD_NEURON_SENSITIVE: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.ELU,
    nn.SiLU,
)

# Module names whose outputs we monitor for activation statistics.
_ACTIVATION_MONITORED: tuple[type[nn.Module], ...] = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.ELU,
    nn.SiLU,
    nn.Tanh,
    nn.Sigmoid,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
)


@dataclass
class _HookHandles:
    """Container for registered hook handles so we can detach cleanly."""

    gradient: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    activation: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    dead_neuron: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)
    grad_cam: list[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def all(self) -> list[torch.utils.hooks.RemovableHandle]:
        return self.gradient + self.activation + self.dead_neuron + self.grad_cam


class DLDiagnostics:
    """Clinical instrumentation for PyTorch training loops.

    Collects per-batch time series of gradient norms, activation statistics,
    dead-neuron fractions, and losses; exposes Plotly visualisations and an
    automated report that surfaces overfitting, vanishing gradients, and
    pathological dead-ReLU layers.

    Args:
        model: The ``nn.Module`` to instrument. The model is NOT modified;
            only forward/backward hooks are attached.
        dead_neuron_threshold: Fraction of zero outputs above which a layer
            is flagged as "dead" in :meth:`report`. Defaults to ``0.5``.
        window: Number of recent batches used for dead-neuron statistics.
            Defaults to ``64``.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        >>> with DLDiagnostics(model) as diag:
        ...     diag.track_gradients()
        ...     diag.track_activations()
        ...     # ... training loop ...
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        dead_neuron_threshold: float = 0.5,
        window: int = 64,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"DLDiagnostics requires an nn.Module; got {type(model).__name__}"
            )
        if not 0.0 < dead_neuron_threshold < 1.0:
            raise ValueError("dead_neuron_threshold must be in (0, 1)")
        if window < 1:
            raise ValueError("window must be >= 1")

        self.model = model
        self.device = get_device()
        self.dead_neuron_threshold = dead_neuron_threshold
        self.window = window

        # Time series storage — lists of dicts, converted to Polars on demand.
        self._grad_log: list[dict[str, Any]] = []
        self._act_log: list[dict[str, Any]] = []
        self._dead_log: list[dict[str, Any]] = []
        self._batch_log: list[dict[str, Any]] = []
        self._epoch_log: list[dict[str, Any]] = []

        # Running per-layer firing masks for dead-neuron detection.
        # Key: layer name -> tensor of firing counts per neuron (1D).
        self._firing_counts: dict[str, torch.Tensor] = {}
        self._firing_samples: dict[str, int] = {}

        # Counters bound to hook closures so they share scope.
        self._batch_idx = 0
        self._epoch_idx = 0

        self._handles = _HookHandles()
        self._tracking = {"gradients": False, "activations": False, "dead": False}

        # Grad-CAM capture buffers (populated on demand).
        self._gradcam_activation: torch.Tensor | None = None
        self._gradcam_gradient: torch.Tensor | None = None

        logger.info(
            "dldiagnostics.init",
            extra={
                "model_class": type(model).__name__,
                "device": str(self.device),
                "window": window,
            },
        )

    # ── Context-manager support ────────────────────────────────────────────

    def __enter__(self) -> DLDiagnostics:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        self.detach()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.detach()
        except Exception:
            # Finalizers MUST NOT raise. Silent cleanup is the documented
            # contract for __del__ in rules/patterns.md.
            pass

    # ── Hook registration ──────────────────────────────────────────────────

    def track_gradients(self) -> DLDiagnostics:
        """Register backward hooks on every trainable parameter.

        Records the L2 norm of each parameter's gradient at every backward
        pass, keyed by parameter name.

        Returns:
            ``self`` for chaining.
        """
        if self._tracking["gradients"]:
            return self
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            handle = param.register_hook(self._make_grad_hook(name))
            self._handles.gradient.append(handle)
        self._tracking["gradients"] = True
        logger.info(
            "dldiagnostics.track_gradients",
            extra={"hooks_registered": len(self._handles.gradient)},
        )
        return self

    def track_activations(self) -> DLDiagnostics:
        """Register forward hooks on monitored submodules.

        Records mean/std/min/max/dead-fraction of activations per layer
        at every forward pass.

        Returns:
            ``self`` for chaining.
        """
        if self._tracking["activations"]:
            return self
        for name, module in self.model.named_modules():
            if name == "" or not isinstance(module, _ACTIVATION_MONITORED):
                continue
            handle = module.register_forward_hook(self._make_act_hook(name))
            self._handles.activation.append(handle)
        self._tracking["activations"] = True
        logger.info(
            "dldiagnostics.track_activations",
            extra={"hooks_registered": len(self._handles.activation)},
        )
        return self

    def track_dead_neurons(self) -> DLDiagnostics:
        """Register forward hooks on ReLU-family layers to track dead neurons.

        A "neuron" here is a channel (Conv) or output unit (Linear). The
        rolling fraction of batches where that neuron output zero is tracked.

        Returns:
            ``self`` for chaining.
        """
        if self._tracking["dead"]:
            return self
        for name, module in self.model.named_modules():
            if name == "" or not isinstance(module, _DEAD_NEURON_SENSITIVE):
                continue
            handle = module.register_forward_hook(self._make_dead_hook(name))
            self._handles.dead_neuron.append(handle)
        self._tracking["dead"] = True
        logger.info(
            "dldiagnostics.track_dead_neurons",
            extra={"hooks_registered": len(self._handles.dead_neuron)},
        )
        return self

    def detach(self) -> None:
        """Remove ALL registered hooks and release references.

        Safe to call multiple times. Invoked automatically on context exit.
        """
        for handle in self._handles.all():
            try:
                handle.remove()
            except Exception:
                # Hook removal failures are benign (module may already be
                # gone). See rules/zero-tolerance.md Rule 3 carve-out for
                # cleanup paths.
                pass
        self._handles = _HookHandles()
        self._tracking = {k: False for k in self._tracking}
        self._gradcam_activation = None
        self._gradcam_gradient = None

    # ── Recording ─────────────────────────────────────────────────────────

    def record_batch(self, *, loss: float, lr: float | None = None) -> None:
        """Record per-batch scalar training signals.

        Args:
            loss: Training loss for the batch (post-backward).
            lr: Current learning rate (optional; read from optimizer).
        """
        if not math.isfinite(loss):
            logger.warning(
                "dldiagnostics.record_batch.nonfinite_loss",
                extra={"loss": str(loss), "batch": self._batch_idx},
            )
        self._batch_log.append(
            {
                "batch": self._batch_idx,
                "epoch": self._epoch_idx,
                "loss": float(loss),
                "lr": float(lr) if lr is not None else float("nan"),
            }
        )
        self._batch_idx += 1

    def record_epoch(
        self,
        *,
        val_loss: float | None = None,
        train_loss: float | None = None,
        **extra: float,
    ) -> None:
        """Record per-epoch summary metrics.

        Args:
            val_loss: Validation loss at epoch end.
            train_loss: Mean training loss for the epoch. If ``None``, it is
                computed from the batches in this epoch.
            **extra: Any additional scalar metrics to persist.
        """
        if train_loss is None:
            epoch_batches = [
                b for b in self._batch_log if b["epoch"] == self._epoch_idx
            ]
            if epoch_batches:
                train_loss = float(np.mean([b["loss"] for b in epoch_batches]))
        entry = {
            "epoch": self._epoch_idx,
            "train_loss": train_loss if train_loss is not None else float("nan"),
            "val_loss": val_loss if val_loss is not None else float("nan"),
            **{k: float(v) for k, v in extra.items()},
        }
        self._epoch_log.append(entry)
        self._epoch_idx += 1

    # ── Public DataFrame accessors ────────────────────────────────────────

    def gradients_df(self) -> pl.DataFrame:
        """Polars DataFrame of per-layer gradient norms by batch."""
        if not self._grad_log:
            return pl.DataFrame(
                schema={
                    "batch": pl.Int64,
                    "layer": pl.Utf8,
                    "grad_norm": pl.Float64,
                    "grad_rms": pl.Float64,
                    "update_ratio": pl.Float64,
                }
            )
        return pl.DataFrame(self._grad_log)

    def activations_df(self) -> pl.DataFrame:
        """Polars DataFrame of per-layer activation statistics by batch."""
        if not self._act_log:
            return pl.DataFrame(
                schema={
                    "batch": pl.Int64,
                    "layer": pl.Utf8,
                    "act_kind": pl.Utf8,
                    "mean": pl.Float64,
                    "std": pl.Float64,
                    "min": pl.Float64,
                    "max": pl.Float64,
                    "dead_fraction": pl.Float64,
                    "inactivity_fraction": pl.Float64,
                }
            )
        return pl.DataFrame(self._act_log)

    def dead_neurons_df(self) -> pl.DataFrame:
        """Polars DataFrame of current per-layer dead-neuron fractions."""
        rows: list[dict[str, Any]] = []
        for name, counts in self._firing_counts.items():
            n_samples = max(self._firing_samples.get(name, 1), 1)
            dead_mask = (counts / n_samples) < 1e-6
            rows.append(
                {
                    "layer": name,
                    "n_neurons": int(counts.numel()),
                    "n_dead": int(dead_mask.sum().item()),
                    "dead_fraction": float(dead_mask.float().mean().item()),
                }
            )
        if not rows:
            return pl.DataFrame(
                schema={
                    "layer": pl.Utf8,
                    "n_neurons": pl.Int64,
                    "n_dead": pl.Int64,
                    "dead_fraction": pl.Float64,
                }
            )
        return pl.DataFrame(rows)

    def batches_df(self) -> pl.DataFrame:
        """Polars DataFrame of per-batch scalars (loss, lr)."""
        if not self._batch_log:
            return pl.DataFrame(
                schema={
                    "batch": pl.Int64,
                    "epoch": pl.Int64,
                    "loss": pl.Float64,
                    "lr": pl.Float64,
                }
            )
        return pl.DataFrame(self._batch_log)

    def epochs_df(self) -> pl.DataFrame:
        """Polars DataFrame of per-epoch summary metrics."""
        if not self._epoch_log:
            return pl.DataFrame(
                schema={
                    "epoch": pl.Int64,
                    "train_loss": pl.Float64,
                    "val_loss": pl.Float64,
                }
            )
        return pl.DataFrame(self._epoch_log)

    # ── Plots ─────────────────────────────────────────────────────────────

    def plot_loss_curves(self) -> go.Figure:
        """Loss stethoscope: train vs val curves with overfitting callout.

        Returns:
            A Plotly Figure with two traces (train / val) and annotations
            for detected overfitting epoch (if any).
        """
        epochs = self.epochs_df()
        batches = self.batches_df()
        fig = go.Figure()
        if batches.height:
            fig.add_trace(
                go.Scatter(
                    x=batches["batch"].to_list(),
                    y=batches["loss"].to_list(),
                    mode="lines",
                    name="train (per batch)",
                    line=dict(color="lightblue", width=1),
                    opacity=0.5,
                )
            )
        if epochs.height:
            fig.add_trace(
                go.Scatter(
                    x=epochs["epoch"].to_list(),
                    y=epochs["train_loss"].to_list(),
                    mode="lines+markers",
                    name="train (epoch mean)",
                    line=dict(color="steelblue", width=2),
                )
            )
            if epochs["val_loss"].is_not_null().any():
                fig.add_trace(
                    go.Scatter(
                        x=epochs["epoch"].to_list(),
                        y=epochs["val_loss"].to_list(),
                        mode="lines+markers",
                        name="val",
                        line=dict(color="firebrick", width=2),
                    )
                )
        overfit_epoch = self._detect_overfit_epoch()
        if overfit_epoch is not None:
            fig.add_vline(
                x=overfit_epoch,
                line=dict(color="orange", dash="dash"),
                annotation_text=f"overfitting suspected @ epoch {overfit_epoch}",
                annotation_position="top",
            )
        fig.update_layout(
            title="Loss Curves (Stethoscope)",
            xaxis_title="step",
            yaxis_title="loss",
            template="plotly_white",
            hovermode="x unified",
        )
        return fig

    def plot_gradient_flow(self) -> go.Figure:
        """Blood test: per-layer gradient norm over time.

        One trace per tracked parameter. Layers whose gradient norm collapses
        toward zero are the vanishing-gradient culprits.
        """
        df = self.gradients_df()
        fig = go.Figure()
        if df.height == 0:
            fig.update_layout(
                title="Gradient Flow (Blood Test) — no data",
                template="plotly_white",
            )
            return fig
        for layer in df["layer"].unique().to_list():
            sub = df.filter(pl.col("layer") == layer)
            fig.add_trace(
                go.Scatter(
                    x=sub["batch"].to_list(),
                    y=sub["grad_norm"].to_list(),
                    mode="lines",
                    name=layer,
                    hovertemplate=f"{layer}<br>batch=%{{x}}<br>‖∇‖=%{{y:.3e}}",
                )
            )
        fig.update_layout(
            title="Gradient Flow by Layer (Blood Test)",
            xaxis_title="batch",
            yaxis_title="gradient L2 norm",
            yaxis_type="log",
            template="plotly_white",
        )
        return fig

    def plot_activation_stats(self) -> go.Figure:
        """X-Ray: activation mean ± std per layer over time."""
        df = self.activations_df()
        fig = go.Figure()
        if df.height == 0:
            fig.update_layout(
                title="Activation Statistics (X-Ray) — no data",
                template="plotly_white",
            )
            return fig
        for layer in df["layer"].unique().to_list():
            sub = df.filter(pl.col("layer") == layer)
            fig.add_trace(
                go.Scatter(
                    x=sub["batch"].to_list(),
                    y=sub["mean"].to_list(),
                    mode="lines",
                    name=f"{layer} — mean",
                    hovertemplate=(
                        f"{layer}<br>batch=%{{x}}<br>mean=%{{y:.3f}}<br>"
                        "std=%{customdata:.3f}"
                    ),
                    customdata=sub["std"].to_list(),
                )
            )
        fig.update_layout(
            title="Activation Mean per Layer (X-Ray)",
            xaxis_title="batch",
            yaxis_title="activation mean",
            template="plotly_white",
        )
        return fig

    def plot_dead_neurons(self) -> go.Figure:
        """X-Ray: fraction of dead neurons per ReLU-family layer."""
        df = self.dead_neurons_df()
        fig = go.Figure()
        if df.height == 0:
            fig.update_layout(
                title="Dead Neurons (X-Ray) — no ReLU-family layers tracked",
                template="plotly_white",
            )
            return fig
        colors = [
            "firebrick" if frac > self.dead_neuron_threshold else "steelblue"
            for frac in df["dead_fraction"].to_list()
        ]
        fig.add_trace(
            go.Bar(
                x=df["layer"].to_list(),
                y=df["dead_fraction"].to_list(),
                marker_color=colors,
                text=[
                    f"{int(n)}/{int(t)}" for n, t in zip(df["n_dead"], df["n_neurons"])
                ],
                textposition="outside",
            )
        )
        fig.add_hline(
            y=self.dead_neuron_threshold,
            line=dict(color="orange", dash="dash"),
            annotation_text=f"alert threshold ({self.dead_neuron_threshold:.0%})",
        )
        fig.update_layout(
            title="Dead Neuron Fraction by Layer (X-Ray)",
            xaxis_title="layer",
            yaxis_title="fraction dead",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            showlegend=False,
        )
        return fig

    def plot_training_dashboard(self) -> go.Figure:
        """Vital signs: 2x2 dashboard (loss, grad flow, activations, LR)."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Loss (Stethoscope)",
                "Gradient Flow (Blood Test)",
                "Activation Mean (X-Ray)",
                "Learning Rate",
            ),
        )

        # (1,1) Loss
        epochs = self.epochs_df()
        batches = self.batches_df()
        if batches.height:
            fig.add_trace(
                go.Scatter(
                    x=batches["batch"].to_list(),
                    y=batches["loss"].to_list(),
                    mode="lines",
                    name="train loss",
                    line=dict(color="steelblue", width=1),
                ),
                row=1,
                col=1,
            )
        if epochs.height and epochs["val_loss"].is_not_null().any():
            # Place val loss at the last batch of each epoch for alignment.
            val_x = []
            for ep in epochs["epoch"].to_list():
                ep_batches = batches.filter(pl.col("epoch") == ep)
                val_x.append(
                    int(ep_batches["batch"].max()) if ep_batches.height else ep
                )
            fig.add_trace(
                go.Scatter(
                    x=val_x,
                    y=epochs["val_loss"].to_list(),
                    mode="lines+markers",
                    name="val loss",
                    line=dict(color="firebrick"),
                ),
                row=1,
                col=1,
            )

        # (1,2) Gradient flow
        gdf = self.gradients_df()
        if gdf.height:
            for layer in gdf["layer"].unique().to_list():
                sub = gdf.filter(pl.col("layer") == layer)
                fig.add_trace(
                    go.Scatter(
                        x=sub["batch"].to_list(),
                        y=sub["grad_norm"].to_list(),
                        mode="lines",
                        name=layer,
                        showlegend=False,
                    ),
                    row=1,
                    col=2,
                )
            fig.update_yaxes(type="log", row=1, col=2)

        # (2,1) Activation mean
        adf = self.activations_df()
        if adf.height:
            for layer in adf["layer"].unique().to_list():
                sub = adf.filter(pl.col("layer") == layer)
                fig.add_trace(
                    go.Scatter(
                        x=sub["batch"].to_list(),
                        y=sub["mean"].to_list(),
                        mode="lines",
                        name=layer,
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        # (2,2) LR
        if batches.height and batches["lr"].is_not_null().any():
            fig.add_trace(
                go.Scatter(
                    x=batches["batch"].to_list(),
                    y=batches["lr"].to_list(),
                    mode="lines",
                    name="lr",
                    line=dict(color="darkgreen"),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )

        fig.update_layout(
            title="Training Dashboard (Vital Signs)",
            template="plotly_white",
            height=720,
        )
        return fig

    def plot_lr_vs_loss(self) -> go.Figure:
        """Plot LR vs loss (useful after an :meth:`lr_range_test`)."""
        df = self.batches_df()
        fig = go.Figure()
        if df.height == 0 or df["lr"].is_null().all():
            fig.update_layout(title="LR vs Loss — no data", template="plotly_white")
            return fig
        fig.add_trace(
            go.Scatter(
                x=df["lr"].to_list(),
                y=df["loss"].to_list(),
                mode="lines",
                line=dict(color="steelblue"),
            )
        )
        fig.update_layout(
            title="Learning Rate vs Loss",
            xaxis_title="learning rate",
            yaxis_title="loss",
            xaxis_type="log",
            template="plotly_white",
        )
        return fig

    def plot_weight_distributions(self) -> go.Figure:
        """Histogram of weight values, one trace per parameter tensor."""
        fig = go.Figure()
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.ndim == 0:
                continue
            values = param.detach().cpu().flatten().numpy()
            fig.add_trace(go.Histogram(x=values, name=name, opacity=0.6))
        fig.update_layout(
            title="Weight Distributions",
            xaxis_title="value",
            yaxis_title="count",
            barmode="overlay",
            template="plotly_white",
        )
        return fig

    def plot_gradient_norms(self) -> go.Figure:
        """Mean gradient norm per layer across the run (bar chart)."""
        df = self.gradients_df()
        fig = go.Figure()
        if df.height == 0:
            fig.update_layout(title="Gradient Norms — no data", template="plotly_white")
            return fig
        summary = df.group_by("layer").agg(
            pl.col("grad_norm").mean().alias("mean_grad")
        )
        summary = summary.sort("mean_grad")
        fig.add_trace(
            go.Bar(
                x=summary["layer"].to_list(),
                y=summary["mean_grad"].to_list(),
                marker_color="steelblue",
            )
        )
        fig.update_layout(
            title="Mean Gradient Norm per Layer",
            xaxis_title="layer",
            yaxis_title="mean ‖∇‖",
            yaxis_type="log",
            template="plotly_white",
        )
        return fig

    # ── Automated report ──────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        """Print a human-readable diagnosis and return findings as a dict.

        Findings covered:
            * Gradient flow (vanishing / exploding / healthy)
            * Dead neurons (per-layer ReLU-family)
            * Loss trend (overfitting, underfitting, instability)

        Returns:
            A dict with keys ``gradient_flow``, ``dead_neurons``, ``loss_trend``
            each mapping to a dict with ``severity`` and ``message``.
        """
        findings: dict[str, Any] = {}

        # 1. Gradient flow — uses SCALE-INVARIANT per-element RMS (grad_rms)
        # and update_ratio (‖∇W‖/‖W‖), matching slide 5F thresholds.
        gdf = self.gradients_df()
        if gdf.height and "grad_rms" in gdf.columns:
            stats = gdf.group_by("layer").agg(
                pl.col("grad_rms").mean().alias("rms"),
                pl.col("update_ratio").mean().alias("ur"),
            )
            min_rms_raw = stats["rms"].min()
            max_rms_raw = stats["rms"].max()
            min_ur_raw = stats["ur"].min()
            max_ur_raw = stats["ur"].max()
            min_rms = (
                float(min_rms_raw) if isinstance(min_rms_raw, (int, float)) else None
            )
            max_rms = (
                float(max_rms_raw) if isinstance(max_rms_raw, (int, float)) else None
            )
            min_ur = float(min_ur_raw) if isinstance(min_ur_raw, (int, float)) else 0.0
            max_ur = float(max_ur_raw) if isinstance(max_ur_raw, (int, float)) else 0.0
            if min_rms is None or max_rms is None or min_rms == 0:
                severity = "UNKNOWN"
                message = "Insufficient gradient data."
            elif min_rms < 1e-5 or min_ur < 1e-4:
                # Vanishing: RMS < 1e-5 OR update ratio < 1e-4 (matches slide 5F).
                worst_layer = (
                    stats.sort("rms").row(0, named=True)["layer"]
                    if min_rms < 1e-5
                    else stats.sort("ur").row(0, named=True)["layer"]
                )
                severity = "CRITICAL"
                message = (
                    f"Vanishing gradients at '{worst_layer}' — "
                    f"min RMS = {min_rms:.2e}, min update_ratio = {min_ur:.2e}. "
                    "Fix: verify pre-norm layout (LayerNorm/RMSNorm before block), "
                    "add residual connections, switch to GELU/SiLU, or use Kaiming init."
                )
            elif max_rms > 1e-2 or max_ur > 0.1:
                # Exploding: RMS > 1e-2 OR update ratio > 0.1 (matches slide 5F).
                worst_layer = (
                    stats.sort("rms", descending=True).row(0, named=True)["layer"]
                    if max_rms > 1e-2
                    else stats.sort("ur", descending=True).row(0, named=True)["layer"]
                )
                severity = "CRITICAL"
                message = (
                    f"Exploding gradients at '{worst_layer}' — "
                    f"max RMS = {max_rms:.2e}, max update_ratio = {max_ur:.2e}. "
                    "Fix: add gradient clipping (or adaptive: ZClip/AGC), reduce LR, "
                    "verify initialization (Kaiming for ReLU, Xavier for Tanh)."
                )
            elif max_rms / max(min_rms, 1e-12) > 1e3:
                severity = "WARNING"
                message = (
                    f"Large RMS spread across layers (max/min = "
                    f"{max_rms / min_rms:.1e}). Deep layers may be learning unevenly."
                )
            else:
                severity = "HEALTHY"
                message = (
                    f"Gradient flow OK (RMS range {min_rms:.2e}–{max_rms:.2e}, "
                    f"update_ratio range {min_ur:.2e}–{max_ur:.2e})."
                )
            findings["gradient_flow"] = {"severity": severity, "message": message}
        elif gdf.height:
            # Legacy path for dataframes without the new columns.
            findings["gradient_flow"] = {
                "severity": "UNKNOWN",
                "message": (
                    "grad_rms field missing — re-run with the current library "
                    "version to get scale-invariant diagnosis."
                ),
            }
        else:
            findings["gradient_flow"] = {
                "severity": "UNKNOWN",
                "message": "No gradient tracking enabled — call track_gradients().",
            }

        # 2. Dead neurons / saturation — uses ACTIVATION-TYPE-AWARE
        # inactivity_fraction from _act_log (ReLU: |x|<eps; Tanh: |x|>0.99;
        # Sigmoid: |x|>0.99 or |x|<0.01). This catches near-dead ReLU channels
        # that the legacy exact-zero check misses post-BatchNorm.
        adf = self.activations_df()
        if adf.height and "inactivity_fraction" in adf.columns:
            # Aggregate mean inactivity per layer (averaged across batches).
            agg = (
                adf.filter(pl.col("act_kind") != "other")
                .group_by("layer")
                .agg(
                    pl.col("inactivity_fraction").mean().alias("mean_inactive"),
                    pl.col("act_kind").first().alias("kind"),
                )
                .sort("mean_inactive", descending=True)
            )
            if agg.height:
                worst = agg.row(0, named=True)
                thr = self.dead_neuron_threshold
                if worst["mean_inactive"] > thr:
                    kind = worst["kind"]
                    if kind == "relu":
                        label = "dead neurons"
                        fix = "Switch to GELU/LeakyReLU or re-initialise with Kaiming."
                    elif kind == "tanh":
                        label = "saturated (|x|>0.99)"
                        fix = "Reduce LR, add LayerNorm before, or switch to GELU."
                    elif kind == "sigmoid":
                        label = "saturated (|x|>0.99 or |x|<0.01)"
                        fix = "Reduce LR, add BN/LN, or switch to softmax+CE if classification."
                    else:
                        label = "inactive"
                        fix = "Investigate activation distribution."
                    findings["dead_neurons"] = {
                        "severity": "WARNING",
                        "message": (
                            f"'{worst['layer']}' ({kind}): "
                            f"{worst['mean_inactive']:.0%} {label}. {fix}"
                        ),
                    }
                else:
                    findings["dead_neurons"] = {
                        "severity": "HEALTHY",
                        "message": (
                            f"All {agg.height} activation layers healthy "
                            f"(worst: {worst['layer']} at "
                            f"{worst['mean_inactive']:.0%} inactive, below "
                            f"{thr:.0%} threshold)."
                        ),
                    }
            else:
                findings["dead_neurons"] = {
                    "severity": "UNKNOWN",
                    "message": "No activation layers tracked — call track_activations().",
                }
        else:
            findings["dead_neurons"] = {
                "severity": "UNKNOWN",
                "message": "No activation tracking enabled — call track_activations().",
            }

        # 3. Loss trend
        edf = self.epochs_df()
        if edf.height >= 3 and edf["val_loss"].is_not_null().any():
            overfit = self._detect_overfit_epoch()
            train_trend = self._slope(edf["train_loss"].to_list())
            if overfit is not None:
                severity = "WARNING"
                message = (
                    f"Overfitting suspected at epoch {overfit} "
                    "(val loss rising while train loss falls). "
                    "Consider dropout, weight decay, or early stopping."
                )
            elif train_trend > -1e-4 and edf.height >= 5:
                severity = "WARNING"
                message = (
                    f"Underfitting — train loss slope {train_trend:.2e}/epoch. "
                    "Consider a larger model, more epochs, or higher LR."
                )
            else:
                severity = "HEALTHY"
                message = f"Loss converging (train slope {train_trend:.2e}/epoch)."
            findings["loss_trend"] = {"severity": severity, "message": message}
        else:
            findings["loss_trend"] = {
                "severity": "UNKNOWN",
                "message": "Need at least 3 epochs with val_loss for trend analysis.",
            }

        # Human-readable summary
        print("\n" + "═" * 64)
        print("  DL Diagnostics Report — Prescription Pad")
        print("═" * 64)
        icons = {
            "HEALTHY": "✓",
            "WARNING": "!",
            "CRITICAL": "X",
            "UNKNOWN": "?",
        }
        titles = {
            "gradient_flow": "Gradient flow",
            "dead_neurons": "Dead neurons ",
            "loss_trend": "Loss trend   ",
        }
        for key, label in titles.items():
            f = findings[key]
            print(
                f"  [{icons[f['severity']]}] {label} ({f['severity']}): {f['message']}"
            )
        print("═" * 64 + "\n")
        return findings

    # ── Grad-CAM ──────────────────────────────────────────────────────────

    def grad_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        layer_name: str,
    ) -> torch.Tensor:
        """Compute a Grad-CAM heatmap for ``target_class`` from ``layer_name``.

        Args:
            input_tensor: Input batch ``(N, C, H, W)`` or ``(N, C, L)``.
            target_class: Target class index to explain.
            layer_name: ``model.named_modules()`` key of the conv layer to
                attribute. Usually the last convolutional layer.

        Returns:
            Heatmap tensor with shape matching the spatial dims of the tracked
            layer's output (``(N, H', W')`` for 2D inputs).

        Raises:
            ValueError: If ``layer_name`` is not found in the model.
        """
        target_module: nn.Module | None = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_module = module
                break
        if target_module is None:
            raise ValueError(
                f"Layer '{layer_name}' not found in model. "
                f"Available: {[n for n, _ in self.model.named_modules() if n][:10]}..."
            )

        self._gradcam_activation = None
        self._gradcam_gradient = None

        def fwd_hook(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self._gradcam_activation = out.detach()

        def bwd_hook(_mod: nn.Module, _gi: Any, go: Any) -> None:
            # go is a tuple — first element is d(output)/d(module-output)
            self._gradcam_gradient = go[0].detach()

        h1 = target_module.register_forward_hook(fwd_hook)
        h2 = target_module.register_full_backward_hook(bwd_hook)
        self._handles.grad_cam.extend([h1, h2])

        # Preserve caller's train/eval state — critical for mid-training use.
        was_training = self.model.training

        try:
            self.model.eval()
            inp = input_tensor.to(self.device)
            logits = self.model(inp)
            if logits.ndim != 2:
                raise ValueError(
                    f"grad_cam expects classification logits (N, C); got {logits.shape}"
                )
            self.model.zero_grad(set_to_none=True)
            one_hot = torch.zeros_like(logits)
            one_hot[:, target_class] = 1.0
            logits.backward(gradient=one_hot, retain_graph=False)

            if self._gradcam_activation is None or self._gradcam_gradient is None:
                raise RuntimeError(
                    "Grad-CAM hooks did not fire — layer may be unreachable "
                    "from the forward path."
                )
            activations = self._gradcam_activation  # (N, K, ...)
            gradients = self._gradcam_gradient  # (N, K, ...)
            # Global-average-pool gradients over spatial dims to get per-channel weights.
            spatial_dims = tuple(range(2, gradients.ndim))
            weights = gradients.mean(dim=spatial_dims, keepdim=True)  # (N, K, 1, ...)
            cam = (weights * activations).sum(dim=1)  # (N, ...)
            cam = torch.relu(cam)
            # Normalise per-sample to [0, 1].
            flat = cam.flatten(start_dim=1)
            mins = flat.min(dim=1, keepdim=True).values
            maxs = flat.max(dim=1, keepdim=True).values
            denom = (maxs - mins).clamp(min=1e-8)
            flat = (flat - mins) / denom
            cam = flat.view_as(cam)
            return cam.cpu()
        finally:
            # Restore caller's train/eval state BEFORE removing hooks, so
            # downstream errors in hook cleanup don't leave model in eval mode.
            if was_training:
                self.model.train()
            h1.remove()
            h2.remove()
            # Remove from bookkeeping too so detach() stays idempotent.
            self._handles.grad_cam = [
                h for h in self._handles.grad_cam if h is not h1 and h is not h2
            ]
            self._gradcam_activation = None
            self._gradcam_gradient = None

    # ── LR range test (static) ────────────────────────────────────────────

    @staticmethod
    def lr_range_test(
        model: nn.Module,
        dataloader: DataLoader,
        *,
        loss_fn: nn.Module | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.SGD,
        lr_min: float = 1e-7,
        lr_max: float = 10.0,
        steps: int = 200,
        device: torch.device | None = None,
        batch_adapter: Callable[[Any], tuple[torch.Tensor, torch.Tensor]] | None = None,
        safety_divisor: float = 10.0,
    ) -> dict[str, Any]:
        """Leslie Smith LR range test (with fastai-style safe-LR recipe).

        Trains for ``steps`` batches while exponentially increasing LR from
        ``lr_min`` to ``lr_max``. Smooths the loss curve with EMA (beta=0.98)
        before finding the steepest-descent point — this matches fastai's
        ``lr_find()`` and avoids picking a single lucky batch in the tail.

        **Critical**: returns BOTH ``min_loss_lr`` (steepest descent) AND
        ``safe_lr = min_loss_lr / safety_divisor`` (default 10). Use ``safe_lr``
        in your optimizer — ``min_loss_lr`` is the edge of instability.

        The model's weights are saved before the test and **restored** on exit
        (via state_dict deepcopy), so calling this does not corrupt your model.

        Args:
            model: The model to probe. Weights are restored after return.
            dataloader: Any DataLoader. By default yields ``(inputs, targets)``
                tuples; pass ``batch_adapter`` for HF/SSL batch formats.
            loss_fn: Loss function (REQUIRED — no silent default).
            optimizer_cls: Optimizer class (default SGD).
            lr_min, lr_max, steps: Sweep configuration.
            device: Override compute device (default: model's current device).
            batch_adapter: Callable ``batch -> (x, y)``. Default handles
                tuple/list; pass a lambda for dict-shaped batches (e.g. HF).
            safety_divisor: Divide steepest-descent LR by this to get safe LR
                (fastai default: 10).

        Returns:
            ``{"safe_lr": float,              # use this in your optimizer
                "min_loss_lr": float,          # steepest descent (edge of instability)
                "divergence_lr": float,        # where smoothed loss > 4× min
                "lrs": [...], "losses": [...], "losses_smooth": [...],
                "figure": go.Figure}``
        """
        if steps < 2:
            raise ValueError("steps must be >= 2")
        if not 0 < lr_min < lr_max:
            raise ValueError("require 0 < lr_min < lr_max")
        if loss_fn is None:
            raise ValueError(
                "loss_fn is required — no silent default. "
                "Pass nn.CrossEntropyLoss() for classification or nn.MSELoss() for regression."
            )

        import copy as _copy

        # Device: honor model's current device unless overridden.
        dev = device or next(model.parameters()).device
        if device is not None:
            model = model.to(dev)

        # Save weights for restoration.
        saved_state = _copy.deepcopy(model.state_dict())

        # Default batch adapter handles tuple/list; raises loudly on dicts.
        def _default_adapter(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                return batch[0], batch[1]
            raise ValueError(
                "lr_range_test got a non-(x,y) batch. Pass batch_adapter=... "
                "for HuggingFace-style dict batches or SSL single-tensor batches."
            )

        adapter = batch_adapter or _default_adapter

        try:
            optimizer = optimizer_cls(model.parameters(), lr=lr_min)
            gamma = (lr_max / lr_min) ** (1.0 / (steps - 1))
            lrs: list[float] = []
            losses: list[float] = []
            running_min = float("inf")  # O(1) running min, not O(n)
            model.train()
            data_iter = iter(dataloader)
            for step in range(steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                x, y = adapter(batch)
                x, y = x.to(dev), y.to(dev)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_min * (gamma**step)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                cur_lr = optimizer.param_groups[0]["lr"]
                cur_loss = float(loss.item())
                lrs.append(cur_lr)
                losses.append(cur_loss)
                if cur_loss < running_min:
                    running_min = cur_loss
                if not math.isfinite(cur_loss) or cur_loss > 10 * running_min:
                    logger.info(
                        "dldiagnostics.lr_range_test.diverged",
                        extra={"step": step, "lr": cur_lr, "loss": cur_loss},
                    )
                    break
        finally:
            # Always restore weights — no silent corruption.
            model.load_state_dict(saved_state)

        # EMA-smoothed losses (fastai convention, beta=0.98).
        beta = 0.98
        losses_smooth: list[float] = []
        ema = 0.0
        for i, ell in enumerate(losses):
            ema = beta * ema + (1 - beta) * ell
            # Bias-correct the EMA.
            losses_smooth.append(ema / (1 - beta ** (i + 1)))

        # min_loss_lr = LR at the steepest-descent point of SMOOTHED loss.
        min_loss_lr = lr_min
        divergence_lr = lr_max
        if len(losses_smooth) >= 3:
            dl = np.diff(np.array(losses_smooth))
            idx = int(np.argmin(dl))
            min_loss_lr = lrs[idx]
            # Divergence = first point where smoothed loss > 4× its minimum.
            min_smooth = min(losses_smooth)
            for i, s in enumerate(losses_smooth):
                if s > 4 * min_smooth and i > idx:
                    divergence_lr = lrs[i]
                    break

        safe_lr = min_loss_lr / safety_divisor

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=lrs,
                y=losses,
                mode="lines",
                name="raw loss",
                line=dict(color="lightgray"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=lrs,
                y=losses_smooth,
                mode="lines",
                name="smoothed",
                line=dict(color="#0D9488", width=2),
            )
        )
        fig.add_vline(
            x=safe_lr,
            line=dict(color="#22C55E", dash="dash"),
            annotation_text=f"safe_lr = {safe_lr:.2e}",
        )
        fig.add_vline(
            x=min_loss_lr,
            line=dict(color="#F59E0B", dash="dot"),
            annotation_text=f"min_loss_lr = {min_loss_lr:.2e}",
        )
        fig.update_layout(
            title="LR Range Test (smoothed) — use safe_lr, not min_loss_lr",
            xaxis_title="learning rate",
            yaxis_title="loss",
            xaxis_type="log",
            template="plotly_white",
        )
        logger.info(
            "dldiagnostics.lr_range_test.ok",
            extra={
                "safe_lr": safe_lr,
                "min_loss_lr": min_loss_lr,
                "divergence_lr": divergence_lr,
                "steps_run": len(lrs),
            },
        )
        return {
            "safe_lr": safe_lr,
            "min_loss_lr": min_loss_lr,
            "divergence_lr": divergence_lr,
            "suggested_lr": safe_lr,  # backwards-compat alias
            "lrs": lrs,
            "losses": losses,
            "losses_smooth": losses_smooth,
            "figure": fig,
        }

    # ── Hook factories (internal) ─────────────────────────────────────────

    def _make_grad_hook(self, name: str):
        """Scale-invariant gradient tracking.

        Records three metrics per layer per batch:
          - grad_norm: raw L2 norm (preserved for backwards compatibility)
          - grad_rms: per-element RMS = ‖∇‖ / sqrt(numel) — scale-invariant,
            comparable across layers of any size. This is what LLM training
            dashboards (W&B, TensorBoard) actually display.
          - update_ratio: ‖∇W‖ / ‖W‖ — the "effective LR" per layer.

        Casts to fp32 before reduction so BF16/FP16 gradients don't silently
        produce inf/NaN.
        """
        # Look up the parameter tensor for update-ratio computation.
        try:
            param = dict(self.model.named_parameters())[name]
        except KeyError:
            param = None

        def _hook(grad: torch.Tensor) -> None:
            try:
                # Handle sparse gradients (e.g. nn.Embedding(sparse=True)).
                g = grad.coalesce().values() if grad.is_sparse else grad
                # Cast to fp32 to avoid BF16/FP16 inf.
                g_f32 = g.detach().float()
                l2 = float(g_f32.norm(p=2).item())
                numel = max(g_f32.numel(), 1)
                rms = l2 / (numel**0.5)
                # Update ratio — use detached param weight norm.
                if param is not None:
                    p_norm = float(param.detach().float().norm(p=2).item())
                    upd_ratio = l2 / max(p_norm, 1e-12)
                else:
                    upd_ratio = 0.0
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "dldiagnostics.grad_hook.error",
                    extra={"layer": name, "error": str(exc)},
                )
                return
            self._grad_log.append(
                {
                    "batch": self._batch_idx,
                    "layer": name,
                    "grad_norm": l2,  # preserved for compatibility
                    "grad_rms": rms,  # scale-invariant
                    "update_ratio": upd_ratio,  # ‖∇W‖ / ‖W‖
                }
            )

        return _hook

    def _make_act_hook(self, name: str):
        """Activation statistics. Casts to fp32 to avoid BF16/FP16 overflow.

        The `inactivity_fraction` field measures activation-type-appropriate
        pathology:
          - ReLU / GELU / SiLU: fraction with |x| < 1e-6 (dead neurons)
          - Tanh: fraction with |x| > 0.99 (saturated)
          - Sigmoid: fraction with |x| > 0.99 OR |x| < 0.01 (saturated)
          - Others: 0 (no pathology definition)

        The legacy `dead_fraction` field (exact-zero count) is preserved for
        backwards compatibility but is only meaningful for ReLU.
        """
        # Determine activation type from module class name for dispatching.
        act_kind = self._classify_activation_module(name)

        def _hook(_module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.numel() == 0:
                return
            # Cast to fp32 for numerical safety (BF16/FP16 overflow on .std()).
            out = output.detach().float()
            try:
                mean = float(out.mean().item())
                std = float(out.std().item()) if out.numel() > 1 else 0.0
                mn = float(out.min().item())
                mx = float(out.max().item())
                legacy_dead = float((out == 0).float().mean().item())
                # Activation-aware inactivity/saturation metric.
                if act_kind == "relu":
                    inactivity = float((out.abs() < 1e-6).float().mean().item())
                elif act_kind == "tanh":
                    inactivity = float((out.abs() > 0.99).float().mean().item())
                elif act_kind == "sigmoid":
                    inactivity = float(
                        ((out > 0.99) | (out < 0.01)).float().mean().item()
                    )
                else:
                    inactivity = 0.0
            except RuntimeError:
                # Non-numeric tensor (e.g. mixed dtype). Skip silently.
                return
            # Guard against NaN/inf leaking into logs.
            for val_name, val in (("mean", mean), ("std", std)):
                if val != val or val in (float("inf"), float("-inf")):
                    logger.warning(
                        "dldiagnostics.act_hook.nonfinite",
                        extra={"layer": name, "field": val_name},
                    )
                    return
            self._act_log.append(
                {
                    "batch": self._batch_idx,
                    "layer": name,
                    "act_kind": act_kind,
                    "mean": mean,
                    "std": std,
                    "min": mn,
                    "max": mx,
                    "dead_fraction": legacy_dead,
                    "inactivity_fraction": inactivity,
                }
            )

        return _hook

    def _classify_activation_module(self, name: str) -> str:
        """Return one of 'relu', 'tanh', 'sigmoid', 'other' for a module name."""
        try:
            mod = dict(self.model.named_modules())[name]
        except KeyError:
            return "other"
        cls = type(mod).__name__.lower()
        if any(k in cls for k in ("relu", "gelu", "silu", "swish", "elu")):
            return "relu"
        if "tanh" in cls:
            return "tanh"
        if "sigmoid" in cls:
            return "sigmoid"
        return "other"

    def _make_dead_hook(self, name: str):
        def _hook(_module: nn.Module, _inp: Any, output: torch.Tensor) -> None:
            if not isinstance(output, torch.Tensor) or output.numel() == 0:
                return
            out = output.detach()
            # Collapse all non-channel dims. Convention: channel dim = 1 for
            # Conv outputs (N, C, ...); for Linear, output is (N, F) — here
            # we treat dim 1 as "neurons" which matches both shapes.
            if out.ndim < 2:
                return
            reduce_dims = tuple(d for d in range(out.ndim) if d != 1)
            fired = (out != 0).any(dim=reduce_dims).float().cpu()
            if name not in self._firing_counts:
                self._firing_counts[name] = torch.zeros_like(fired)
                self._firing_samples[name] = 0
            self._firing_counts[name] += fired
            self._firing_samples[name] += 1
            # Keep memory bounded by decaying old counts when window exceeded.
            if self._firing_samples[name] > self.window:
                scale = self.window / self._firing_samples[name]
                self._firing_counts[name] = self._firing_counts[name] * scale
                self._firing_samples[name] = self.window

        return _hook

    # ── Internal analysis helpers ─────────────────────────────────────────

    @staticmethod
    def _slope(series: list[float]) -> float:
        """Least-squares slope of a 1D series vs its index (ignores NaN)."""
        xs = np.arange(len(series), dtype=float)
        ys = np.asarray(series, dtype=float)
        mask = np.isfinite(ys)
        if mask.sum() < 2:
            return 0.0
        xs, ys = xs[mask], ys[mask]
        return float(np.polyfit(xs, ys, 1)[0])

    def _detect_overfit_epoch(self) -> int | None:
        """Return epoch index where val loss starts rising while train falls."""
        edf = self.epochs_df()
        if edf.height < 3:
            return None
        train = edf["train_loss"].to_list()
        val = edf["val_loss"].to_list()
        for i in range(2, len(val)):
            v_now, v_prev = val[i], val[i - 1]
            t_now, t_prev = train[i], train[i - 1]
            if any(
                x is None or not math.isfinite(x)
                for x in (v_now, v_prev, t_now, t_prev)
            ):
                continue
            if v_now > v_prev and t_now < t_prev:
                # Confirm trend persists one step back if available.
                return i
        return None


# ════════════════════════════════════════════════════════════════════════
# Standalone diagnostic checkpoint — for use AFTER a training loop
# ════════════════════════════════════════════════════════════════════════


def run_diagnostic_checkpoint(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[..., Any],
    *,
    title: str = "Model",
    n_batches: int = 8,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    show: bool = True,
    batch_adapter: Callable[[Any], tuple[torch.Tensor, ...]] | None = None,
) -> tuple[DLDiagnostics, dict[str, Any]]:
    """Run a short instrumented diagnostic pass on a TRAINED model.

    Use this between the Train and Visualise phases of an exercise. It
    attaches the four diagnostic instruments (gradients, activations,
    dead-neurons, scalar history) to the model, runs ``n_batches`` of
    forward-backward passes to populate the history, replays any
    epoch-level losses you collected during real training, and prints the
    Prescription Pad with auto-diagnosis.

    The model's weights are NOT updated — gradients are computed but no
    optimizer step is taken. The model's train/eval state is preserved.

    Args:
        model: A trained ``nn.Module``.
        dataloader: Any DataLoader whose batches the loss function accepts.
        loss_fn: Callable. The default contract is
            ``loss_fn(model, batch) -> (scalar_loss, extras_dict)`` matching
            the M5 exercise convention. Pass ``batch_adapter`` to wrap a
            different signature.
        title: Human-readable name printed in the dashboard title.
        n_batches: How many batches to instrument (default 8 — enough for
            a stable mean per layer without slowing the exercise down).
        train_losses: Optional list of per-epoch train losses captured
            during the actual training run. Replayed into the dashboard's
            stethoscope view so students see the real loss trajectory, not
            just the diagnostic-pass losses.
        val_losses: Optional list of per-epoch validation losses, same
            length as ``train_losses``.
        show: If ``True``, calls ``.show()`` on the dashboard figure.
        batch_adapter: Optional ``batch -> (loss_fn_args...)`` translator
            for non-standard batch shapes.

    Returns:
        ``(diag, findings)`` so the caller can inspect the DataFrames and
        the Prescription Pad findings dict.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("run_diagnostic_checkpoint requires an nn.Module")
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")

    diag = DLDiagnostics(model)
    diag.track_gradients().track_activations().track_dead_neurons()

    # Replay real training history into the dashboard if provided.
    if train_losses or val_losses:
        n_epochs = max(len(train_losses or []), len(val_losses or []))
        for i in range(n_epochs):
            tl = (
                float(train_losses[i])
                if train_losses and i < len(train_losses)
                else None
            )
            vl = float(val_losses[i]) if val_losses and i < len(val_losses) else None
            # Synthesise one batch entry per epoch so the per-batch stethoscope
            # has data to plot — students still see the real epoch losses.
            if tl is not None:
                diag.record_batch(loss=tl)
            diag.record_epoch(train_loss=tl, val_loss=vl)

    was_training = model.training
    try:
        model.train()  # Enable gradients + activation hooks.
        seen = 0
        for batch in dataloader:
            if seen >= n_batches:
                break
            try:
                if batch_adapter is not None:
                    args = batch_adapter(batch)
                    loss_out = loss_fn(model, *args)
                else:
                    loss_out = loss_fn(model, batch)
                # Convention: M5 loss_fn returns (loss, extras_dict). Allow
                # either a bare tensor or a tuple.
                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"loss_fn returned {type(loss).__name__}; expected Tensor"
                    )
                model.zero_grad(set_to_none=True)
                loss.backward()
                # NOTE: no optimizer.step() — diagnostic pass is read-only.
                diag.record_batch(loss=float(loss.item()))
            except Exception as exc:  # pragma: no cover - student loop variability
                logger.warning(
                    "dldiagnostics.checkpoint.batch_skipped",
                    extra={"batch_idx": seen, "error": str(exc)},
                )
            seen += 1
    finally:
        if not was_training:
            model.eval()

    # Render the dashboard and the Prescription Pad.
    fig = diag.plot_training_dashboard()
    fig.update_layout(title=f"{title} — Diagnostic Dashboard (Vital Signs)")
    if show:
        try:
            fig.show()
        except Exception as exc:  # pragma: no cover - no display in CI
            logger.info(
                "dldiagnostics.checkpoint.show_skipped",
                extra={"error": str(exc)},
            )

    findings = diag.report()
    return diag, findings


def diagnose_classifier(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    title: str = "Classifier",
    n_batches: int = 8,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    show: bool = True,
    forward_returns_tuple: bool = False,
) -> tuple[DLDiagnostics, dict[str, Any]]:
    """Convenience wrapper for ``(x, y)`` cross-entropy classifiers.

    Equivalent to :func:`run_diagnostic_checkpoint` with a built-in
    ``loss_fn`` that calls ``F.cross_entropy(model(x), y)``. Use this for
    CNN, transformer, and transfer-learning exercises.

    Args:
        model: Trained classifier ``nn.Module``.
        dataloader: Yields ``(x, y)`` tuples; ``y`` is class indices.
        title: Title for the dashboard.
        n_batches: Batches to instrument.
        train_losses, val_losses: Optional epoch histories to replay.
        show: Show the figure inline.
        forward_returns_tuple: If ``True``, ``model(x)`` returns
            ``(logits, ...)`` (e.g. attention models) — only the first
            element is used as logits.

    Returns:
        ``(diag, findings)``
    """
    import torch.nn.functional as F  # local — torch already imported

    def _cls_loss(m: nn.Module, batch: Any) -> torch.Tensor:
        x, y = batch[0], batch[1]
        out = m(x)
        logits = out[0] if forward_returns_tuple else out
        return F.cross_entropy(logits, y)

    return run_diagnostic_checkpoint(
        model,
        dataloader,
        _cls_loss,
        title=title,
        n_batches=n_batches,
        train_losses=train_losses,
        val_losses=val_losses,
        show=show,
    )


def diagnose_regressor(
    model: nn.Module,
    dataloader: DataLoader,
    *,
    title: str = "Regressor",
    n_batches: int = 8,
    train_losses: list[float] | None = None,
    val_losses: list[float] | None = None,
    show: bool = True,
    forward_returns_tuple: bool = False,
) -> tuple[DLDiagnostics, dict[str, Any]]:
    """Convenience wrapper for ``(x, y)`` MSE regressors (RNN exercises).

    Equivalent to :func:`run_diagnostic_checkpoint` with a built-in
    ``loss_fn`` that calls ``F.mse_loss(model(x), y)``.

    Args:
        forward_returns_tuple: Set ``True`` for attention models that
            return ``(predictions, attn_weights)``.
    """
    import torch.nn.functional as F

    def _reg_loss(m: nn.Module, batch: Any) -> torch.Tensor:
        x, y = batch[0], batch[1]
        out = m(x)
        pred = out[0] if forward_returns_tuple else out
        return F.mse_loss(pred, y)

    return run_diagnostic_checkpoint(
        model,
        dataloader,
        _reg_loss,
        title=title,
        n_batches=n_batches,
        train_losses=train_losses,
        val_losses=val_losses,
        show=show,
    )


__all__ = [
    "DLDiagnostics",
    "run_diagnostic_checkpoint",
    "diagnose_classifier",
    "diagnose_regressor",
]
