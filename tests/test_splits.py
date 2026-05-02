from __future__ import annotations

import numpy as np

from weather_forecasting_pipeline.datasets.splits import select_feature_columns, target_column_name
from weather_forecasting_pipeline.metdatapy_adapter import (
    fit_apply_scaler_with_metdatapy,
    make_supervised_with_metdatapy,
    split_by_fraction_with_metdatapy,
)


def test_chronological_split_and_scaler_fit_only_train(synthetic_station_frame):
    supervised = make_supervised_with_metdatapy(synthetic_station_frame, target="temp_c", horizons=[6], lags=[1])
    target_col = target_column_name("temp_c", 6)
    splits = split_by_fraction_with_metdatapy(supervised, train_fraction=0.7, validation_fraction=0.15)

    assert splits["train"].index.max() < splits["val"].index.min()
    assert splits["val"].index.max() < splits["test"].index.min()

    features = select_feature_columns(supervised, target_col)
    scaled, scaler = fit_apply_scaler_with_metdatapy(splits, features, method="standard")

    first_feature = features[0]
    train_mean = splits["train"][first_feature].mean()
    assert np.isclose(scaler.parameters[first_feature]["mean"], train_mean)
    assert np.isclose(float(scaled["train"][first_feature].mean()), 0.0, atol=1e-6)
