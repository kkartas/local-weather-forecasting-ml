"""Regression tests covering data-leakage rules of the dissertation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from weather_forecasting_pipeline.datasets.splits import (
    select_feature_columns,
    sequence_arrays_from_split,
    target_column_name,
)
from weather_forecasting_pipeline.metdatapy_adapter import (
    make_supervised_with_metdatapy,
    preprocess_with_metdatapy,
)


def test_select_feature_columns_includes_causal_qc_but_excludes_gap_flag(synthetic_station_frame):
    """Causal QC flags can be model features, but inserted gap markers cannot."""
    prepared = preprocess_with_metdatapy(
        synthetic_station_frame,
        expected_frequency="10min",
        derived_metrics=["dew_point", "vpd"],
        rolling_windows=[6],
    )
    supervised = make_supervised_with_metdatapy(prepared, target="temp_c", horizons=[6], lags=[1])
    target_col = target_column_name("temp_c", 6)

    feature_columns = select_feature_columns(supervised, target_col)

    assert "gap" not in feature_columns
    qc_features = [c for c in feature_columns if c.startswith("qc_")]
    assert qc_features, "MetDataPy causal QC flag columns should be available as features"


def test_sequence_arrays_use_only_past_observations():
    """Each sequence label must align with the last observation of the sequence.

    Build a deterministic ramp where temp_c == row index. With horizon h=1 and
    target temp_c_t+1, the label at end position e must equal row e+1, while
    every feature value in the sequence must be at most row e (no future).
    """
    n = 30
    idx = pd.date_range("2024-01-01", periods=n, freq="10min", tz="UTC")
    temp = np.arange(n, dtype=float)
    df = pd.DataFrame({"temp_c": temp, "temp_c_t+1": temp + 1.0}, index=idx)

    seq_len = 5
    x, y = sequence_arrays_from_split(df, ["temp_c"], "temp_c_t+1", sequence_length=seq_len)

    assert x.shape == (n - seq_len + 1, seq_len, 1)
    for i in range(len(x)):
        end_pos = i + seq_len - 1
        # Label is the future-shifted target at the forecast origin.
        assert y[i] == temp[end_pos] + 1.0
        # Every feature in the sequence comes from indices [end_pos - seq_len + 1 .. end_pos].
        assert np.all(x[i, :, 0] <= temp[end_pos])
        assert x[i, -1, 0] == temp[end_pos]
