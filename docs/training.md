# Model Training

Training is orchestrated by `weather_forecasting_pipeline.training.pipeline.train`.

Every configured model is trained separately for each configured forecast horizon.

## Baseline Models

Implemented in `weather_forecasting_pipeline.models.baselines`.

### Persistence

Predicts the future target as the current observed target value.

### Moving Average

Predicts from an average of available target lag columns.

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
- MSE loss
- validation-loss early stopping
- deterministic seed setup where supported

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
