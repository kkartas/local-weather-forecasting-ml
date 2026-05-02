from __future__ import annotations

import numpy as np

from weather_forecasting_pipeline.datasets.splits import sequence_arrays_from_split, target_column_name
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
