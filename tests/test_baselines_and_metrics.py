"""Coverage for newer baseline and evaluation building blocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from weather_forecasting_pipeline.evaluation.metrics import persistence_skill_score
from weather_forecasting_pipeline.metdatapy_adapter import (
    fit_target_scaler_with_metdatapy,
    inverse_transform_target_with_metdatapy,
    transform_target_with_metdatapy,
)
from weather_forecasting_pipeline.models.baselines import ClimatologyModel, make_baseline
from weather_forecasting_pipeline.models.dl_models import _dilations_for_receptive_field
from weather_forecasting_pipeline.training.pipeline import _attach_persistence_skill_score


def test_climatology_uses_only_training_observations():
    idx = pd.date_range("2024-01-01", periods=72, freq="1h", tz="UTC")
    # Diurnal sine pattern: each hour-of-day has a distinct mean across days.
    temp = 10.0 + 5.0 * np.sin((idx.hour.to_numpy() / 24.0) * 2 * np.pi)
    train_idx = idx[:48]
    test_idx = idx[48:]
    train = pd.DataFrame({"temp_c": temp[:48], "temp_c_t+1": temp[:48]}, index=train_idx)
    test = pd.DataFrame({"temp_c": np.zeros(len(test_idx)), "temp_c_t+1": np.zeros(len(test_idx))}, index=test_idx)

    model = ClimatologyModel(target="temp_c").fit(train, "temp_c_t+1")
    pred = model.predict(test)

    # Predictions for each hour-of-day should equal the training mean for that
    # (month, hour). Because the train window covers the same two days that
    # the test window mirrors, predictions match the original sine curve.
    expected = 10.0 + 5.0 * np.sin((test.index.hour.to_numpy() / 24.0) * 2 * np.pi)
    np.testing.assert_allclose(pred, expected, atol=1e-5)


def test_climatology_factory_routes_through_make_baseline():
    model = make_baseline("climatology", target="temp_c")
    assert isinstance(model, ClimatologyModel)


def test_persistence_skill_score_zero_when_equal():
    assert persistence_skill_score(2.0, 2.0) == 0.0


def test_persistence_skill_score_returns_none_for_degenerate_anchor():
    assert persistence_skill_score(2.0, 0.0) is None
    assert persistence_skill_score(2.0, float("nan")) is None


def test_attach_skill_score_uses_persistence_anchor_per_horizon():
    metrics = pd.DataFrame(
        [
            {"horizon_label": "h01", "model": "persistence", "rmse": 2.0},
            {"horizon_label": "h01", "model": "linear_regression", "rmse": 1.0},
            {"horizon_label": "h12", "model": "persistence", "rmse": 4.0},
            {"horizon_label": "h12", "model": "lstm", "rmse": 4.0},
        ]
    )
    out = _attach_persistence_skill_score(metrics)

    h01_lr = out.loc[(out.horizon_label == "h01") & (out.model == "linear_regression"), "skill_score_persistence"].iloc[0]
    h12_lstm = out.loc[(out.horizon_label == "h12") & (out.model == "lstm"), "skill_score_persistence"].iloc[0]
    h01_pers = out.loc[(out.horizon_label == "h01") & (out.model == "persistence"), "skill_score_persistence"].iloc[0]

    assert h01_lr == pytest.approx(1.0 - (1.0 ** 2) / (2.0 ** 2))
    assert h12_lstm == pytest.approx(0.0)
    assert h01_pers == pytest.approx(0.0)


def test_target_scaler_round_trip_recovers_original_values():
    train = pd.DataFrame({"temp_c_t+6": np.linspace(-5.0, 35.0, 50)})
    target_col = "temp_c_t+6"
    for method in ("standard", "minmax", "robust"):
        scaler = fit_target_scaler_with_metdatapy(train, target_col, method)
        sample = np.array([-5.0, 0.0, 12.34, 35.0], dtype=float)
        scaled = transform_target_with_metdatapy(sample, scaler, target_col)
        restored = inverse_transform_target_with_metdatapy(scaled, scaler, target_col)
        np.testing.assert_allclose(restored, sample, atol=1e-6)


def test_tcn_dilation_schedule_covers_sequence_length():
    seq_len = 144
    dilations = _dilations_for_receptive_field(seq_len)
    # Receptive field for stacked 2-conv blocks of kernel 2 == 1 + sum(2*d)
    receptive_field = 1 + 2 * sum(dilations)
    assert receptive_field >= seq_len
