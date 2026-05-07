"""Baseline forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLIMATOLOGY_GLOBAL_KEY = -1


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
    """Forecasts with a trailing past-only mean of the target.

    Prefers MetDataPy's pre-computed past-only rolling-mean column for the
    smallest configured rolling window (``<target>_roll<window>_mean`` from
    ``WeatherSet.rolling_features(closed='left')``) because it is a true
    consecutive-step mean of the target. Falls back to averaging available
    consecutive lag columns when no rolling mean is present.
    """

    name = "moving_average"

    def __init__(self, target: str, max_lags: int = 4):
        self.target = target
        self.max_lags = max_lags
        self.rolling_column: str | None = None
        self.lag_columns: list[str] = []

    def fit(self, train: pd.DataFrame, target_col: str) -> "MovingAverageModel":
        rolling_candidates = [
            (window, f"{self.target}_roll{window}_mean")
            for window in _detect_rolling_windows(train.columns, self.target)
        ]
        rolling_candidates.sort(key=lambda item: item[0])
        for _, name in rolling_candidates:
            if name in train.columns:
                self.rolling_column = name
                self.lag_columns = []
                return self

        lag_cols = [c for c in train.columns if str(c).startswith(f"{self.target}_lag")]
        self.lag_columns = sorted(lag_cols, key=_lag_number)[: self.max_lags]
        if not self.lag_columns:
            raise ValueError(f"No lag or rolling-mean columns available for moving-average target {self.target!r}")
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.rolling_column is not None:
            return frame[self.rolling_column].to_numpy(dtype=np.float32)
        return frame[self.lag_columns].mean(axis=1).to_numpy(dtype=np.float32)


class ClimatologyModel:
    """Forecasts the hour-of-year mean of the target observed in training.

    The lookup is keyed by ``(month, hour)``. Forecasts for unseen keys fall
    back to the overall training mean, which mirrors the standard
    meteorological climatology baseline. Fitting touches only the training
    split, satisfying the dissertation's no-future-information rule.
    """

    name = "climatology"

    def __init__(self, target: str):
        self.target = target
        self.global_mean: float = 0.0
        self.lookup: dict[tuple[int, int], float] = {}

    def fit(self, train: pd.DataFrame, target_col: str) -> "ClimatologyModel":
        if self.target not in train.columns:
            raise ValueError(f"Climatology target {self.target!r} is missing")
        if not isinstance(train.index, pd.DatetimeIndex):
            raise ValueError("Climatology baseline requires a DatetimeIndex")
        # Use the present-time observation rather than the shifted target so a
        # single fitted lookup serves every horizon: predicting at time ``t``
        # always uses the climatology of ``t``'s month/hour, regardless of
        # how far ahead the dissertation reports.
        values = train[self.target].astype(float)
        self.global_mean = float(values.mean())
        keyed = pd.DataFrame(
            {
                "value": values.to_numpy(),
                "month": train.index.month,
                "hour": train.index.hour,
            }
        )
        means = keyed.groupby(["month", "hour"], as_index=True)["value"].mean()
        self.lookup = {(int(m), int(h)): float(v) for (m, h), v in means.items()}
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Climatology baseline requires a DatetimeIndex on predict input")
        out = np.empty(len(frame), dtype=np.float32)
        for i, (month, hour) in enumerate(zip(frame.index.month, frame.index.hour)):
            out[i] = self.lookup.get((int(month), int(hour)), self.global_mean)
        return out


def make_baseline(name: str, target: str) -> PersistenceModel | MovingAverageModel | ClimatologyModel:
    """Create a baseline model by configured name."""
    if name == "persistence":
        return PersistenceModel(target=target)
    if name == "moving_average":
        return MovingAverageModel(target=target)
    if name == "climatology":
        return ClimatologyModel(target=target)
    raise ValueError(f"Unknown baseline model: {name}")


def _lag_number(name: str) -> int:
    suffix = str(name).rsplit("_lag", maxsplit=1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return 10**9


def _detect_rolling_windows(columns, target: str) -> list[int]:
    """Return rolling-window sizes available as ``<target>_roll<window>_mean`` columns."""
    prefix = f"{target}_roll"
    suffix = "_mean"
    windows: list[int] = []
    for col in columns:
        name = str(col)
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : -len(suffix)]
        try:
            windows.append(int(middle))
        except ValueError:
            continue
    return windows
