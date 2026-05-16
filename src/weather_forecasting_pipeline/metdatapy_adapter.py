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
from metdatapy.mlprep import time_split_by_fraction as metdatapy_time_split_by_fraction
from metdatapy.qc import qc_any
from metdatapy.weathercloud import read_weathercloud_csv, read_weathercloud_directory
from pandas.errors import ParserError

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

    MetDataPy owns Weathercloud directory ingestion, delimiter/encoding
    handling, duplicate timestamp policy, source-to-canonical mapping,
    timezone conversion, and unit normalization. The fallback only tolerates
    Weathercloud rows with an extra trailing empty field until MetDataPy exposes
    a matching parser option.
    """
    mapping = _mapping_with_timezone(load_mapping(mapping_path), timezone)
    LOGGER.info("Reading Weathercloud directory through MetDataPy: %s", raw_dir)
    if _directory_has_utf16le_without_bom(raw_dir):
        LOGGER.warning(
            "Detected UTF-16LE Weathercloud exports without a BOM. "
            "Using trailing-field tolerant CSV parsing before MetDataPy mapping."
        )
        df, report = _read_weathercloud_directory_with_raw_fallback(raw_dir, mapping, timezone)
        if report.get("duplicate_rows"):
            LOGGER.warning(
                "Weathercloud ingestion handled %s duplicate rows across %s duplicate timestamps with policy %s",
                report["duplicate_rows"],
                report["duplicate_timestamp_count"],
                report["duplicate_policy"],
            )
        return df
    try:
        df, report = read_weathercloud_directory(
            raw_dir,
            mapping_config=mapping,
            timezone=timezone,
            duplicate_policy="keep_first",
            return_report=True,
            nonexistent="shift_forward",
            ambiguous=False,
        )
    except ParserError as exc:
        LOGGER.warning(
            "MetDataPy Weathercloud parsing failed (%s). Retrying with trailing-field tolerant CSV parsing.",
            exc,
        )
        df, report = _read_weathercloud_directory_with_raw_fallback(raw_dir, mapping, timezone)
    if report.get("duplicate_rows"):
        LOGGER.warning(
            "Weathercloud ingestion handled %s duplicate rows across %s duplicate timestamps with policy %s",
            report["duplicate_rows"],
            report["duplicate_timestamp_count"],
            report["duplicate_policy"],
        )
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
    ws.qc_range().qc_spike(causal=True).qc_flatline(causal=True)
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


def _looks_like_utf16le_without_bom(path: Path) -> bool:
    sample = path.read_bytes()[:512]
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")) or len(sample) < 4:
        return False
    even_nulls = sample[0::2].count(0)
    odd_nulls = sample[1::2].count(0)
    return odd_nulls > max(4, even_nulls * 4)


def _directory_has_utf16le_without_bom(raw_dir: str | Path) -> bool:
    directory = Path(raw_dir)
    return any(_looks_like_utf16le_without_bom(path) for path in directory.glob("*.csv"))


def _read_weathercloud_directory_with_raw_fallback(
    raw_dir: str | Path, mapping: dict, timezone: str
) -> tuple[pd.DataFrame, dict]:
    """Read Weathercloud CSV files through a trailing-field tolerant parser.

    Some Weathercloud rows contain one more trailing empty field than the
    header. The local parser keeps only the header-width columns, then delegates
    timestamp normalization, mapping, and unit conversion back to MetDataPy.
    """
    directory = Path(raw_dir)
    if not directory.is_dir():
        raise ValueError(f"Weathercloud path is not a directory: {directory}")
    csv_paths = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not csv_paths:
        raise ValueError(f"No Weathercloud CSV files found in {directory}")

    frames = []
    for csv_path in csv_paths:
        raw = _read_weathercloud_csv_raw(csv_path)
        raw = _prepare_fallback_timestamps(raw, mapping)
        frames.append(
            WeatherSet.from_mapping(
                raw,
                mapping,
                nonexistent="shift_forward",
                ambiguous=False,
            )
            .normalize_units(mapping)
            .to_dataframe()
        )
    if not frames:
        raise ValueError("No Weathercloud rows were read")
    out = pd.concat(frames).sort_index()
    rows_before = len(out)
    duplicate_mask = out.index.duplicated(keep=False)
    duplicate_rows = int(out.index.duplicated(keep="first").sum())
    duplicate_timestamp_count = int(out.index[duplicate_mask].nunique()) if duplicate_mask.any() else 0
    if duplicate_rows:
        out = out[~out.index.duplicated(keep="first")]
    out.index.name = "ts_utc"
    return out, {
        "files_read": [p.name for p in csv_paths],
        "rows_before_duplicate_handling": rows_before,
        "rows_after_duplicate_handling": len(out),
        "duplicate_rows": duplicate_rows,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "duplicate_policy": "keep_first",
    }


def _read_weathercloud_csv_raw(path: Path) -> pd.DataFrame:
    """Read a raw Weathercloud CSV, including UTF-16LE files without a BOM."""
    if not _looks_like_utf16le_without_bom(path):
        return read_weathercloud_csv(path, mapping_config=None)

    sample = path.read_bytes()[:8192].decode("utf-16le", errors="replace")
    delimiter = _detect_delimiter_from_text(sample)
    header = sample.splitlines()[0].split(delimiter)
    return _read_csv_with_header_width(path, encoding="utf-16le", delimiter=delimiter, header=header)


def _read_csv_with_header_width(path: Path, *, encoding: str, delimiter: str, header: list[str]) -> pd.DataFrame:
    """Read a CSV while ignoring surplus trailing empty fields."""
    return pd.read_csv(
        path,
        encoding=encoding,
        sep=delimiter,
        header=0,
        names=header,
        usecols=range(len(header)),
        index_col=False,
        thousands=",",
        engine="python",
    )


def _detect_delimiter_from_text(sample: str) -> str:
    candidates = [",", ";", "\t", "|"]
    lines = [line for line in sample.splitlines() if line.strip()]
    if not lines:
        return ","
    counts = {candidate: sum(line.count(candidate) for line in lines[:10]) for candidate in candidates}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def _prepare_fallback_timestamps(frame: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Parse Weathercloud day-first timestamps before MetDataPy mapping."""
    ts_col = (mapping.get("ts") or {}).get("col")
    if ts_col is None or ts_col not in frame.columns:
        return frame
    sample = frame[ts_col].dropna().astype(str).head(20)
    if not any("/" in value for value in sample):
        return frame
    out = frame.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", dayfirst=True)
    return out


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
    """Create chronological train/validation/test splits using MetDataPy fractions."""
    split_result = metdatapy_time_split_by_fraction(
        df,
        train=train_fraction,
        validation=validation_fraction,
        test=1.0 - train_fraction - validation_fraction,
        min_rows_per_split=1,
    )
    splits = {name: split_result[name] for name in ("train", "val", "test")}
    if any(split.empty for split in splits.values()):
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


def fit_target_scaler_with_metdatapy(
    train_split: pd.DataFrame,
    target_col: str,
    method: str,
) -> object:
    """Fit a single-column scaler on the training target.

    Deep-learning regressors converge much faster when the target is on the
    same scale as the standardised features. Scaling the target uses only the
    training partition (same leakage rule as the feature scaler) and is fit
    via the same MetDataPy ``fit_scaler`` API so both scalers share their
    parametrisation and can be persisted with ``joblib``.
    """
    return fit_scaler(train_split[[target_col]], method=method, columns=[target_col])


def transform_target_with_metdatapy(values: pd.Series | "np.ndarray", scaler: object, target_col: str) -> "np.ndarray":
    """Apply a fitted target scaler to a 1-D array of target values."""
    import numpy as np

    series = pd.Series(np.asarray(values, dtype=float), name=target_col)
    transformed = apply_scaler(pd.DataFrame({target_col: series}), scaler)
    return transformed[target_col].to_numpy(dtype=float)


def inverse_transform_target_with_metdatapy(
    values: "np.ndarray", scaler: object, target_col: str
) -> "np.ndarray":
    """Invert a fitted MetDataPy ScalerParams transform for a single column.

    MetDataPy does not expose an ``inverse_apply_scaler`` helper, so we
    invert the documented formulas explicitly using the per-column parameters
    stored on ``scaler.parameters[target_col]``. The MetDataPy scaler stores
    a single ``scale`` value per column whose meaning depends on ``method``:

    - ``standard``: ``scale`` = standard deviation, paired with ``mean``.
    - ``minmax``: ``scale`` = ``max - min``, paired with ``min``.
    - ``robust``: ``scale`` = interquartile range, paired with ``median``.
    """
    import numpy as np

    params = getattr(scaler, "parameters", {}).get(target_col)
    if params is None:
        raise ValueError(f"Scaler does not contain parameters for target column {target_col!r}")
    method = getattr(scaler, "method", "standard")
    arr = np.asarray(values, dtype=float)
    if method == "standard":
        return arr * float(params["scale"]) + float(params["mean"])
    if method == "minmax":
        return arr * float(params["scale"]) + float(params["min"])
    if method == "robust":
        return arr * float(params["iqr"]) + float(params["median"])
    raise ValueError(f"Unsupported scaler method for inverse transform: {method!r}")
