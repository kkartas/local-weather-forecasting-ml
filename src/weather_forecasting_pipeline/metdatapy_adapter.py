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
from metdatapy.io import read_parquet, to_parquet
from metdatapy.mapper import Mapper
from metdatapy.mlprep import apply_scaler, fit_scaler, make_supervised
from metdatapy.mlprep import time_split as metdatapy_time_split
from metdatapy.qc import qc_any
from metdatapy.weathercloud import read_weathercloud_directory

LOGGER = logging.getLogger(__name__)

ROLLING_FEATURE_COLUMNS = [
    "temp_c",
    "rh_pct",
    "pres_hpa",
    "wspd_ms",
    "gust_ms",
    "rain_mm",
    "rain_rate_mmh",
    "solar_wm2",
    "uv_index",
    "wdir_sin",
    "wdir_cos",
]


def load_mapping(path: str | Path) -> dict:
    """Load a MetDataPy mapping configuration."""
    return Mapper.load(str(path))


def ingest_raw_weathercloud(raw_dir: str | Path, mapping_path: str | Path, timezone: str) -> pd.DataFrame:
    """Load raw Weathercloud data through available MetDataPy APIs.

    MetDataPy 1.2.0 owns Weathercloud directory ingestion, delimiter/encoding
    handling, source-to-canonical mapping, timezone conversion, and unit
    normalization.
    """
    mapping = _mapping_with_timezone(load_mapping(mapping_path), timezone)
    LOGGER.info("Reading Weathercloud directory through MetDataPy: %s", raw_dir)
    df = read_weathercloud_directory(raw_dir, mapping_config=mapping, timezone=timezone)
    df = df.sort_index()
    duplicate_count = int(df.index.duplicated(keep="first").sum())
    if duplicate_count:
        LOGGER.warning("Dropping %s duplicate timestamp rows after MetDataPy ingestion", duplicate_count)
        df = df[~df.index.duplicated(keep="first")]
    return df


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
    rolling_windows: Iterable[int],
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
    ws.encode_wind_direction(drop_original=False)
    ws.rolling_features(
        columns=[col for col in ROLLING_FEATURE_COLUMNS if col in ws.df.columns],
        windows=[int(window) for window in rolling_windows],
        stats=("mean", "std", "min", "max"),
        closed="left",
    )
    if resample_rule:
        ws.resample(resample_rule)
    return ws.to_dataframe()


def _mapping_with_timezone(mapping: dict, timezone: str) -> dict:
    """Return a mapping that declares the configured source timezone for MetDataPy."""
    updated = dict(mapping)
    ts_cfg = dict(updated.get("ts") or {})
    ts_cfg.setdefault("timezone", timezone)
    updated["ts"] = ts_cfg
    return updated


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
        for col in scale_columns:
            scaled[name][col] = scaled_features[col].astype(float)
    return scaled, scaler
