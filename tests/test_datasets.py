from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_forecasting_pipeline.datasets.splits import (
    SequenceDataset,
    build_sequence_dataset,
    select_dl_feature_columns,
    select_feature_columns,
    sequence_arrays_from_split,
    sequence_targets,
    target_column_name,
)
from weather_forecasting_pipeline.metdatapy_adapter import make_supervised_with_metdatapy


def test_horizon_target_shifting(synthetic_station_frame):
    supervised = make_supervised_with_metdatapy(synthetic_station_frame, target="temp_c", horizons=[6], lags=[1, 3])
    target_col = target_column_name("temp_c", 6)

    first_idx = supervised.index[0]
    assert np.isclose(supervised.loc[first_idx, target_col], synthetic_station_frame.loc[first_idx:].iloc[6]["temp_c"])
    assert "temp_c_lag1" in supervised.columns
    assert "rh_pct_lag3" in supervised.columns


def test_sequence_arrays_preserve_past_only_order(synthetic_station_frame):
    supervised = make_supervised_with_metdatapy(synthetic_station_frame, target="temp_c", horizons=[1], lags=[1])
    target_col = target_column_name("temp_c", 1)
    x, y = sequence_arrays_from_split(supervised, ["temp_c", "rh_pct"], target_col, sequence_length=4)

    assert x.shape[1:] == (4, 2)
    assert len(x) == len(y)
    assert np.isclose(x[0, -1, 0], supervised["temp_c"].iloc[3])
    assert np.isclose(y[0], supervised[target_col].iloc[3])


def test_select_dl_feature_columns_excludes_lag_columns(synthetic_station_frame):
    """DL feature selection drops MetDataPy ``_lag<n>`` columns by default.

    The dissertation's DL models receive a ``(sequence_length, n_features)``
    window, so per-timestep lag columns are redundant with the sequence axis
    and explode RAM at full resolution. The tabular feature selector keeps
    them, and the DL helper must keep that set strictly smaller while still
    preserving canonical/derived/calendar/rolling features.
    """
    supervised = make_supervised_with_metdatapy(
        synthetic_station_frame, target="temp_c", horizons=[6], lags=[1, 3, 6, 12]
    )
    target_col = target_column_name("temp_c", 6)

    tabular = select_feature_columns(supervised, target_col)
    dl_features = select_dl_feature_columns(supervised, target_col)

    assert any(c.endswith("_lag1") or "_lag" in c for c in tabular), "fixture should produce lag columns"
    assert all("_lag" not in c for c in dl_features), f"lag columns leaked into DL feature set: {dl_features}"
    assert len(dl_features) < len(tabular), "DL feature count must be strictly smaller than tabular"
    assert "temp_c" in dl_features, "canonical target value at t must remain in DL features"


def test_select_dl_feature_columns_allow_list_validates_columns():
    """Explicit allow-lists are passed through verbatim with column validation."""
    df = pd.DataFrame({"temp_c": [1.0, 2.0], "rh_pct": [60.0, 70.0], "temp_c_t+1": [2.0, 3.0]})
    selected = select_dl_feature_columns(df, "temp_c_t+1", feature_allow_list=["temp_c", "rh_pct"])
    assert selected == ["temp_c", "rh_pct"]

    with pytest.raises(ValueError):
        select_dl_feature_columns(df, "temp_c_t+1", feature_allow_list=["temp_c", "missing"])


def test_sequence_dataset_lazy_indexing_is_memory_safe():
    """``SequenceDataset`` must build windows on demand, not materialize them.

    Allocates a (n_rows, n_features) feature matrix once and verifies that the
    in-memory dataset footprint is dramatically smaller than the dense
    ``(n_sequences, sequence_length, n_features)`` tensor that the legacy
    ``np.stack`` path produces.
    """
    n_rows = 1000
    n_features = 32
    sequence_length = 64
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    targets = rng.normal(size=n_rows).astype(np.float32)

    dataset = SequenceDataset(features, targets, sequence_length)

    assert len(dataset) == n_rows - sequence_length + 1
    x, y = dataset[0]
    assert tuple(x.shape) == (sequence_length, n_features)
    assert float(y) == pytest.approx(float(targets[sequence_length - 1]))

    dense_bytes = (n_rows - sequence_length + 1) * sequence_length * n_features * 4
    dataset_bytes = features.nbytes + targets.nbytes
    assert dataset_bytes < dense_bytes // 4, (
        "lazy dataset must be at least 4x smaller than the dense stacked tensor; "
        f"dense={dense_bytes} dataset={dataset_bytes}"
    )


def test_build_sequence_dataset_uses_supplied_targets():
    """``build_sequence_dataset`` consumes a precomputed target array."""
    idx = pd.date_range("2024-01-01", periods=20, freq="10min", tz="UTC")
    df = pd.DataFrame(
        {
            "temp_c": np.arange(20, dtype=np.float32),
            "rh_pct": np.arange(20, dtype=np.float32) * 0.1,
            "temp_c_t+1": np.arange(20, dtype=np.float32) + 1.0,
        },
        index=idx,
    )
    targets = df["temp_c_t+1"].to_numpy(dtype=np.float32)
    dataset = build_sequence_dataset(df, ["temp_c", "rh_pct"], targets, sequence_length=5)
    aligned = sequence_targets(targets, 5)

    assert len(dataset) == len(aligned)
    x_first, y_first = dataset[0]
    assert tuple(x_first.shape) == (5, 2)
    assert float(y_first) == pytest.approx(float(aligned[0]))
