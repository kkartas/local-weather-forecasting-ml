from __future__ import annotations

import numpy as np

from weather_forecasting_pipeline.datasets.splits import arrays_from_split, select_feature_columns, target_column_name
from weather_forecasting_pipeline.metdatapy_adapter import (
    make_supervised_with_metdatapy,
    split_by_fraction_with_metdatapy,
)
from weather_forecasting_pipeline.models.baselines import make_baseline
from weather_forecasting_pipeline.models.dl_models import make_dl_model, predict_dl_model, train_dl_model
from weather_forecasting_pipeline.models.ml_models import make_ml_model


def test_baseline_and_ml_smoke(synthetic_station_frame):
    supervised = make_supervised_with_metdatapy(synthetic_station_frame, target="temp_c", horizons=[3], lags=[1, 3])
    target_col = target_column_name("temp_c", 3)
    splits = split_by_fraction_with_metdatapy(supervised, 0.7, 0.15)

    baseline = make_baseline("persistence", target="temp_c").fit(splits["train"], target_col)
    baseline_pred = baseline.predict(splits["test"])
    assert len(baseline_pred) == len(splits["test"])

    features = select_feature_columns(supervised, target_col)
    x_train, y_train = arrays_from_split(splits["train"], features, target_col)
    x_test, _ = arrays_from_split(splits["test"], features, target_col)
    model = make_ml_model("linear_regression", random_seed=42)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    assert pred.shape[0] == x_test.shape[0]


def test_dl_model_smoke():
    rng = np.random.default_rng(42)
    x_train = rng.normal(size=(20, 4, 3)).astype(np.float32)
    y_train = x_train[:, -1, 0] * 0.5
    x_val = rng.normal(size=(8, 4, 3)).astype(np.float32)
    y_val = x_val[:, -1, 0] * 0.5
    model = make_dl_model("gru", input_size=3)

    result = train_dl_model(
        model,
        x_train,
        y_train.astype(np.float32),
        x_val,
        y_val.astype(np.float32),
        max_epochs=2,
        batch_size=4,
        learning_rate=0.01,
        patience=2,
        seed=42,
    )
    pred = predict_dl_model(result.model, x_val, batch_size=4)
    assert pred.shape == y_val.shape
    assert result.epochs_trained >= 1
