"""Baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PersistenceModel:
    """Forecasts the future target as the current observed target value."""

    name = "persistence"

    def __init__(self, target: str):
        self.target = target

    def fit(self, train: pd.DataFrame, target_col: str) -> "PersistenceModel":
        if self.target not in train.columns:
            raise ValueError(f"Persistence feature {self.target!r} is missing")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.target].to_numpy(dtype=np.float32)


class MovingAverageModel:
    """Forecasts with a trailing average from available target lag columns."""

    name = "moving_average"

    def __init__(self, target: str, max_lags: int = 4):
        self.target = target
        self.max_lags = max_lags
        self.lag_columns: list[str] = []

    def fit(self, train: pd.DataFrame, target_col: str) -> "MovingAverageModel":
        lag_cols = [c for c in train.columns if str(c).startswith(f"{self.target}_lag")]
        self.lag_columns = sorted(lag_cols, key=_lag_number)[: self.max_lags]
        if not self.lag_columns:
            raise ValueError(f"No lag columns available for moving-average target {self.target!r}")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.lag_columns].mean(axis=1).to_numpy(dtype=np.float32)


def make_baseline(name: str, target: str) -> PersistenceModel | MovingAverageModel:
    """Create a baseline model by configured name."""
    if name == "persistence":
        return PersistenceModel(target=target)
    if name == "moving_average":
        return MovingAverageModel(target=target)
    raise ValueError(f"Unknown baseline model: {name}")


def _lag_number(name: str) -> int:
    suffix = str(name).rsplit("_lag", maxsplit=1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 10**9
