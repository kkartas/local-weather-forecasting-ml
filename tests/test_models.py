from __future__ import annotations

import numpy as np

from weather_forecasting_pipeline.datasets.splits import (
    SequenceDataset,
    arrays_from_split,
    select_feature_columns,
    target_column_name,
)
from weather_forecasting_pipeline.metdatapy_adapter import (
    make_supervised_with_metdatapy,
    split_by_fraction_with_metdatapy,
)
from weather_forecasting_pipeline.models.baselines import make_baseline
from weather_forecasting_pipeline.models.dl_models import (
    make_dl_model,
    predict_dl_model,
    predict_dl_model_from_dataset,
    train_dl_model,
    train_dl_model_from_datasets,
)
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


def test_dl_model_lazy_dataset_path_smoke():
    """End-to-end lazy DL path must train on ``SequenceDataset`` inputs.

    The training pipeline now feeds the recurrent/TCN regressors through
    ``train_dl_model_from_datasets`` so the full ``(n_sequences, sequence_length,
    n_features)`` tensor is never materialized. This smoke test exercises the
    same call path with a tiny fixture so regressions in that contract are
    caught without a multi-GiB allocation.
    """
    rng = np.random.default_rng(7)
    n_rows = 60
    n_features = 4
    sequence_length = 6
    train_features = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    val_features = rng.normal(size=(n_rows // 2, n_features)).astype(np.float32)
    train_targets = train_features[:, 0] * 0.5
    val_targets = val_features[:, 0] * 0.5

    train_ds = SequenceDataset(train_features, train_targets.astype(np.float32), sequence_length)
    val_ds = SequenceDataset(val_features, val_targets.astype(np.float32), sequence_length)

    model = make_dl_model("gru", input_size=n_features, sequence_length=sequence_length)
    result = train_dl_model_from_datasets(
        model,
        train_ds,
        val_ds,
        max_epochs=2,
        batch_size=8,
        learning_rate=0.01,
        patience=2,
        seed=42,
    )
    preds = predict_dl_model_from_dataset(result.model, val_ds, batch_size=8)
    assert preds.shape == (len(val_ds),)
    assert result.epochs_trained >= 1


def test_train_dl_model_from_datasets_invokes_epoch_callback():
    rng = np.random.default_rng(123)
    n_features = 3
    sequence_length = 5
    x_train = rng.normal(size=(36, n_features)).astype(np.float32)
    y_train = (x_train[:, 0] * 0.4).astype(np.float32)
    x_val = rng.normal(size=(20, n_features)).astype(np.float32)
    y_val = (x_val[:, 0] * 0.4).astype(np.float32)

    train_ds = SequenceDataset(x_train, y_train, sequence_length=sequence_length)
    val_ds = SequenceDataset(x_val, y_val, sequence_length=sequence_length)
    model = make_dl_model("gru", input_size=n_features, sequence_length=sequence_length)
    callback_calls: list[tuple[int, int, float, float, int]] = []

    result = train_dl_model_from_datasets(
        model,
        train_ds,
        val_ds,
        max_epochs=3,
        batch_size=4,
        learning_rate=0.01,
        patience=5,
        seed=42,
        on_epoch_end=lambda epoch, max_epochs, train_loss, val_loss, patience_left: callback_calls.append(
            (epoch, max_epochs, train_loss, val_loss, patience_left)
        ),
    )

    assert result.epochs_trained == len(callback_calls)
    assert callback_calls, "Expected at least one epoch callback invocation"
    assert callback_calls[0][0] == 1
    assert all(max_epochs == 3 for _, max_epochs, _, _, _ in callback_calls)
    assert all(np.isfinite(train_loss) for _, _, train_loss, _, _ in callback_calls)
    assert all(np.isfinite(val_loss) for _, _, _, val_loss, _ in callback_calls)
    assert all(isinstance(patience_left, int) for _, _, _, _, patience_left in callback_calls)
