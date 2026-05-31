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


def test_ridge_smoke(synthetic_station_frame):
    """``ridge`` is the default linear baseline.

    Exercises the full fit/predict path so a regression in
    ``make_ml_model("ridge", ...)`` is caught by the test suite. The ridge
    wrapper uses chronological folds and is expected to expose ``alpha_``
    after fit, which is a cheap sanity check that cross-validation actually
    ran.
    """
    from weather_forecasting_pipeline.models.ml_models import ChronologicalRidgeCV

    supervised = make_supervised_with_metdatapy(
        synthetic_station_frame, target="temp_c", horizons=[3], lags=[1, 3]
    )
    target_col = target_column_name("temp_c", 3)
    splits = split_by_fraction_with_metdatapy(supervised, 0.7, 0.15)

    features = select_feature_columns(supervised, target_col)
    x_train, y_train = arrays_from_split(splits["train"], features, target_col)
    x_test, _ = arrays_from_split(splits["test"], features, target_col)
    model = make_ml_model("ridge", random_seed=42)
    assert isinstance(model, ChronologicalRidgeCV)
    assert model.n_splits == 5
    assert model.solver == "lsqr"
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    assert pred.shape[0] == x_test.shape[0]
    # The cross-validated alpha must be drawn from the configured grid.
    assert model.alpha_ in (0.1, 1.0, 10.0, 100.0)


def test_ridge_does_not_copy_center_feature_matrix():
    """Ridge must avoid sklearn's full-matrix copy during full-data fits."""
    from weather_forecasting_pipeline.models.ml_models import ChronologicalRidgeCV

    rng = np.random.default_rng(123)
    x_train = np.ascontiguousarray(rng.normal(size=(80, 12)).astype(np.float32))
    y_train = (x_train[:, 0] * 0.4 - x_train[:, 3] * 0.2 + 5.0).astype(np.float32)
    original = x_train.copy()

    model = ChronologicalRidgeCV(n_splits=3)
    model.fit(x_train, y_train)

    assert np.array_equal(x_train, original)
    assert model.estimator_.copy_X is False
    assert model.estimator_.fit_intercept is False
    assert np.isfinite(model.y_offset_)


def test_unknown_ml_model_raises():
    """Removed/unknown model names must raise rather than silently fall through."""
    import pytest

    with pytest.raises(ValueError, match="Unknown ML model"):
        make_ml_model("does_not_exist", random_seed=42)


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


def test_train_dl_model_from_datasets_applies_grad_clipping(monkeypatch):
    """``grad_clip_norm`` must call ``clip_grad_norm_`` once per optimiser step.

    Gradient clipping mitigates the exploding-gradient events that can cause
    abrupt DL training collapses on long horizons. This test pins the
    contract that the training loop invokes ``torch.nn.utils.clip_grad_norm_``
    with the configured norm and is bypassed entirely when the caller passes
    ``None``.
    """
    import torch

    from weather_forecasting_pipeline.models import dl_models

    rng = np.random.default_rng(0)
    n_features = 3
    sequence_length = 4
    x_train = rng.normal(size=(20, n_features)).astype(np.float32)
    y_train = (x_train[:, 0] * 0.3).astype(np.float32)
    x_val = rng.normal(size=(10, n_features)).astype(np.float32)
    y_val = (x_val[:, 0] * 0.3).astype(np.float32)
    train_ds = SequenceDataset(x_train, y_train, sequence_length=sequence_length)
    val_ds = SequenceDataset(x_val, y_val, sequence_length=sequence_length)

    calls: list[float] = []
    original_clip = torch.nn.utils.clip_grad_norm_

    def _spy(parameters, max_norm, **kwargs):
        calls.append(float(max_norm))
        return original_clip(parameters, max_norm, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", _spy)

    model = dl_models.make_dl_model("gru", input_size=n_features, sequence_length=sequence_length)
    dl_models.train_dl_model_from_datasets(
        model,
        train_ds,
        val_ds,
        max_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        patience=1,
        seed=42,
        grad_clip_norm=0.5,
    )
    assert calls, "clip_grad_norm_ was never invoked with grad_clip_norm=0.5"
    assert all(c == 0.5 for c in calls)

    calls.clear()
    model = dl_models.make_dl_model("gru", input_size=n_features, sequence_length=sequence_length)
    dl_models.train_dl_model_from_datasets(
        model,
        train_ds,
        val_ds,
        max_epochs=1,
        batch_size=4,
        learning_rate=0.01,
        patience=1,
        seed=42,
        grad_clip_norm=None,
    )
    assert calls == [], "clip_grad_norm_ must not be called when grad_clip_norm is None"


def test_train_dl_model_from_datasets_attaches_lr_scheduler():
    """``ReduceLROnPlateau`` must be constructed and stepped during training.

    Pins the scheduler contract so a refactor that drops the scheduler is
    caught immediately. The actual rate change
    only triggers after several non-improving epochs, so we assert the
    scheduler was constructed and is in a valid state rather than asserting
    on a learning-rate value the test fixture cannot reliably produce.
    """
    import torch

    from weather_forecasting_pipeline.models import dl_models

    constructed: list[torch.optim.lr_scheduler.ReduceLROnPlateau] = []
    original_cls = torch.optim.lr_scheduler.ReduceLROnPlateau

    class _SpyScheduler(original_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            constructed.append(self)

    torch.optim.lr_scheduler.ReduceLROnPlateau = _SpyScheduler  # type: ignore[assignment]
    try:
        rng = np.random.default_rng(1)
        n_features = 3
        sequence_length = 4
        x_train = rng.normal(size=(20, n_features)).astype(np.float32)
        y_train = (x_train[:, 0] * 0.3).astype(np.float32)
        x_val = rng.normal(size=(10, n_features)).astype(np.float32)
        y_val = (x_val[:, 0] * 0.3).astype(np.float32)
        train_ds = SequenceDataset(x_train, y_train, sequence_length=sequence_length)
        val_ds = SequenceDataset(x_val, y_val, sequence_length=sequence_length)
        model = dl_models.make_dl_model("gru", input_size=n_features, sequence_length=sequence_length)
        dl_models.train_dl_model_from_datasets(
            model,
            train_ds,
            val_ds,
            max_epochs=2,
            batch_size=4,
            learning_rate=0.01,
            patience=2,
            seed=42,
        )
    finally:
        torch.optim.lr_scheduler.ReduceLROnPlateau = original_cls  # type: ignore[assignment]

    assert constructed, "ReduceLROnPlateau was never constructed inside training"
    sched = constructed[0]
    # Documented hyperparameters of the bundle.
    assert sched.factor == dl_models.LR_SCHEDULER_FACTOR
    assert sched.patience == dl_models.LR_SCHEDULER_PATIENCE
    # min_lr is stored per-parameter-group as a list; verify the configured floor.
    assert min(sched.min_lrs) == dl_models.LR_SCHEDULER_MIN_LR


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
