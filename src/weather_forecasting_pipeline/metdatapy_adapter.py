"""Thin MetDataPy adapter for dissertation-specific orchestration.

This module deliberately keeps meteorological preparation inside MetDataPy.
When the installed MetDataPy version does not expose a required preparation
capability, functions raise a clear error and point maintainers to
``METDATAPY.md`` instead of providing a local duplicate implementation.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd
from metdatapy import WeatherSet
from metdatapy.io import read_csv, read_parquet, to_parquet
from metdatapy.mapper import Mapper
from metdatapy.mlprep import apply_scaler, fit_scaler, make_supervised
from metdatapy.mlprep import time_split as metdatapy_time_split
from metdatapy.qc import qc_any
from metdatapy.utils import ensure_datetime_utc

LOGGER = logging.getLogger(__name__)


class MissingMetDataPyFeature(RuntimeError):
    """Raised when a required MetDataPy-owned feature is unavailable."""


def load_mapping(path: str | Path) -> dict:
    """Load a MetDataPy mapping configuration."""
    return Mapper.load(str(path))


def ingest_raw_weathercloud(raw_dir: str | Path, mapping_path: str | Path, timezone: str) -> pd.DataFrame:
    """Load raw Weathercloud data through available MetDataPy APIs.

    The installed MetDataPy 1.0.0 exposes a generic CSV reader and WeatherSet
    mapping utilities but not robust Weathercloud directory ingestion. This
    adapter supports the non-duplicative single-file path through MetDataPy and
    fails clearly for multi-file ingestion until MetDataPy provides that API.
    """
    raw_path = Path(raw_dir)
    files = sorted(raw_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {raw_path}")
    if len(files) > 1:
        raise MissingMetDataPyFeature(
            "MetDataPy directory-level Weathercloud ingestion is required for multiple CSV files. "
            "See METDATAPY.md: Weathercloud multi-file ingestion."
        )

    mapping = load_mapping(mapping_path)
    ts_col = mapping.get("ts", {}).get("col")
    if not ts_col:
        raise ValueError("Mapping config must define ts.col")

    LOGGER.info("Reading single raw CSV through MetDataPy: %s", files[0])
    raw_df = read_csv(str(files[0]))
    weather_set = WeatherSet.from_mapping(raw_df, mapping).normalize_units(mapping)

    # Timestamp localization is still performed by MetDataPy's own helper. The
    # adapter only supplies the dissertation-specific local timezone setting.
    if ts_col not in raw_df.columns:
        raise ValueError(f"Timestamp column {ts_col!r} not present in raw data")
    weather_set.df.index = ensure_datetime_utc(raw_df[ts_col], tz_hint=timezone)
    weather_set.df.index.name = "ts_utc"
    weather_set.df = weather_set.df.sort_index()
    weather_set.df = weather_set.df[~weather_set.df.index.duplicated(keep="first")]
    return weather_set.to_dataframe()


def save_interim(df: pd.DataFrame, path: str | Path) -> None:
    """Save a MetDataPy-compatible dataframe as parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    to_parquet(df, str(path))


def load_interim(path: str | Path) -> pd.DataFrame:
    """Load a MetDataPy-compatible parquet dataframe."""
    return read_parquet(str(path))


def preprocess_with_metdatapy(
    df: pd.DataFrame,
    expected_frequency: str,
    derived_metrics: Iterable[str],
    resample_rule: str | None = None,
) -> pd.DataFrame:
    """Run supported non-destructive MetDataPy preprocessing steps."""
    ws = WeatherSet(df.sort_index())
    ws.insert_missing(expected_frequency)
    ws.qc_range().qc_spike().qc_flatline()
    ws.derive(list(derived_metrics))
    ws.qc_consistency()
    ws.df = qc_any(ws.df)
    ws.calendar_features(cyclical=True)
    if resample_rule:
        ws.resample(resample_rule)
    return ws.to_dataframe()


def unavailable_feature_notes(rolling_windows: Iterable[int]) -> list[str]:
    """Return non-fatal notes for MetDataPy-owned features not yet available."""
    notes: list[str] = []
    if list(rolling_windows):
        notes.append(
            "Rolling meteorological feature generation is required by the methodology but is not "
            "exposed by MetDataPy 1.0.0; the executable pipeline uses the supported MetDataPy "
            "lag/calendar/derived/QC feature set until MetDataPy adds rolling features."
        )
    notes.append(
        "Wind direction cyclic encoding is tracked as a MetDataPy requirement; local code does not "
        "duplicate it until the official MetDataPy API is available."
    )
    return notes


def make_supervised_with_metdatapy(
    df: pd.DataFrame,
    target: str,
    horizons: Iterable[int],
    lags: Iterable[int],
) -> pd.DataFrame:
    """Create lagged supervised targets through MetDataPy."""
    if target not in df.columns:
        raise ValueError(f"Configured target {target!r} is not present in the prepared dataframe")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return make_supervised(df, targets=[target], horizons=horizons, lags=lags, drop_na=True)


def split_by_fraction_with_metdatapy(
    df: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, pd.DataFrame]:
    """Create chronological train/validation/test splits using MetDataPy boundaries."""
    if not df.index.is_monotonic_increasing:
        raise ValueError("Dataframe index must be sorted before chronological splitting")
    if len(df) < 3:
        raise ValueError("At least three rows are required for train/validation/test splitting")

    n = len(df)
    train_end_pos = max(0, min(n - 3, int(n * train_fraction) - 1))
    val_end_pos = max(train_end_pos + 1, min(n - 2, int(n * (train_fraction + validation_fraction)) - 1))
    train_end = df.index[train_end_pos]
    val_end = df.index[val_end_pos]
    splits = metdatapy_time_split(df, train_end=train_end, val_end=val_end)
    if any(part.empty for part in splits.values()):
        raise ValueError("Chronological split produced an empty train, validation, or test partition")
    return splits


def fit_apply_scaler_with_metdatapy(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    method: str,
) -> tuple[dict[str, pd.DataFrame], object]:
    """Fit scaler on train features only and apply it to all splits."""
    scale_columns = [
        c
        for c in feature_columns
        if c in splits["train"].columns
        and pd.api.types.is_numeric_dtype(splits["train"][c])
        and not pd.api.types.is_bool_dtype(splits["train"][c])
    ]
    scaler = fit_scaler(splits["train"][scale_columns], method=method, columns=scale_columns)
    scaled = {}
    for name, split in splits.items():
        scaled[name] = split.copy()
        scaled_features = apply_scaler(split[scale_columns], scaler)
        scaled[name].loc[:, scale_columns] = scaled_features[scale_columns]
    return scaled, scaler
