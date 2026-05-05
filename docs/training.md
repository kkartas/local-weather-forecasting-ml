# Model Training

Training is orchestrated by `weather_forecasting_pipeline.training.pipeline.train`.

Every configured model is trained separately for each configured forecast horizon.

## Baseline Models

Implemented in `weather_forecasting_pipeline.models.baselines`.

### Persistence

Predicts the future target as the current observed target value.

### Moving Average

Predicts the future target as a past-only mean of the target. The model
prefers the MetDataPy rolling-mean column for the smallest configured
rolling window (`<target>_roll<window>_mean`, computed with
`closed="left"` and therefore strictly past-only). When no such column
is available it falls back to averaging the target lag columns sorted by
lag number. The rolling-mean preference makes the baseline a true
consecutive-step moving average rather than an average of arbitrary
non-consecutive lags.

## Traditional Machine Learning Models

Implemented in `weather_forecasting_pipeline.models.ml_models`.

Supported names:

- `linear_regression`
- `random_forest`
- `gradient_boosting`
- `svr`

These models use tabular engineered features.

## Deep Learning Models

Implemented in `weather_forecasting_pipeline.models.dl_models`.

Supported names:

- `lstm`
- `gru`
- `tcn`

These models use sequence tensors with shape:

```text
(batch, sequence_length, n_features)
```

The sequence length is configured with:

```yaml
sequence_length: 144
```

At 10-minute resolution, `144` steps represent 24 hours of input history.

## Deep Learning Training

Deep-learning training uses:

- PyTorch
- Adam optimizer
- MSE loss on a standardised target (see "Target scaling" below)
- validation-loss early stopping
- deterministic seed setup where supported

### Target scaling

Baseline and ML models predict directly in the target's original units
because the feature scaler intentionally leaves the target column
unscaled. Recurrent and TCN regressors, however, converge much faster
when the target is also standardised, so a separate target scaler is
fit on the training partition only (same leakage rule as the feature
scaler) using MetDataPy's `fit_scaler`. During training, target values
are transformed; predictions are inverse-transformed back to original
units before metric computation, so MAE/RMSE/MAPE for DL stay directly
comparable with baseline and ML metrics. The fitted target scaler is
saved at `artifacts/scalers/target_scaler_<horizon>.joblib`.

Configured parameters:

```yaml
training:
  max_epochs: 20
  batch_size: 32
  learning_rate: 0.001
  patience: 5
  min_dl_train_rows: 300
```

If the training split has fewer rows than `min_dl_train_rows`, deep-learning models are skipped for that horizon.

## Saved Model Artifacts

Baseline and scikit-learn models:

```text
artifacts/models/<model>_<horizon>.joblib
```

PyTorch models:

```text
artifacts/models/<model>_<horizon>.pt
```

## Predictions

For each model and horizon:

```text
data/processed/predictions/predictions_<model>_<horizon>.csv
```

Columns:

- `ts_utc`
- `y_true`
- `y_pred`
