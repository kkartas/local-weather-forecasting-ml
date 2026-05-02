"""Forecast evaluation metrics."""

from __future__ import annotations

import math

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true, y_pred = _aligned(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true, y_pred = _aligned(y_true, y_pred)
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float | None:
    """MAPE with near-zero true values masked.

    Returns ``None`` when all true values are near zero, because MAPE would be
    mathematically misleading.
    """
    y_true, y_pred = _aligned(y_true, y_pred)
    mask = np.abs(y_true) > epsilon
    if not np.any(mask):
        return None
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, mape_epsilon: float) -> dict[str, float | None]:
    """Compute all required metrics for one forecast output."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": safe_mape(y_true, y_pred, epsilon=mape_epsilon),
    }


def _aligned(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape}, y_pred={yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not np.any(mask):
        raise ValueError("No finite values available for metric calculation")
    return yt[mask], yp[mask]
