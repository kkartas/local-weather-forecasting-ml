from __future__ import annotations

import numpy as np

from weather_forecasting_pipeline.evaluation.metrics import evaluate_predictions, mae, rmse, safe_mape


def test_mae_rmse_correctness():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 4.0, 2.0])

    assert mae(y_true, y_pred) == 1.0
    assert np.isclose(rmse(y_true, y_pred), np.sqrt(5.0 / 3.0))


def test_safe_mape_masks_near_zero_values():
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([1.0, 8.0])

    assert safe_mape(y_true, y_pred, epsilon=1e-6) == 20.0
    assert safe_mape(np.array([0.0]), np.array([1.0]), epsilon=1e-6) is None
    assert set(evaluate_predictions(y_true, y_pred, 1e-6)) == {"mae", "rmse", "mape"}
