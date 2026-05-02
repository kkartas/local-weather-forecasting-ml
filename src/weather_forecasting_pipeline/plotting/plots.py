"""Experiment plots saved as static files."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    *,
    title: str,
    max_points: int,
) -> None:
    """Save an actual-versus-predicted line plot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(y_true), max_points)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(n), y_true[:n], label="Actual", linewidth=1.5)
    ax.plot(np.arange(n), y_pred[:n], label="Predicted", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Target value")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_residual_distribution(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path, *, title: str) -> None:
    """Save a residual distribution plot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=40, color="#2f6f73", alpha=0.85)
    ax.axvline(0.0, color="#222222", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame, path: str | Path, metric: str = "mae") -> None:
    """Save a model comparison plot for one metric."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if metrics.empty or metric not in metrics.columns:
        return
    pivot = metrics.pivot_table(index="model", columns="horizon_label", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"Model comparison by {metric.upper()}")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_error_by_horizon(metrics: pd.DataFrame, path: str | Path, metric: str = "mae") -> None:
    """Save an error-by-horizon plot."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if metrics.empty or metric not in metrics.columns:
        return
    grouped = metrics.groupby("horizon_steps", as_index=False)[metric].mean().sort_values("horizon_steps")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(grouped["horizon_steps"], grouped[metric], marker="o", color="#6b3f9e")
    ax.set_title(f"Average {metric.upper()} by forecast horizon")
    ax.set_xlabel("Horizon steps")
    ax.set_ylabel(metric.upper())
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
