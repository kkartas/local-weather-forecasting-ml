# Configuration

Experiment settings are loaded from YAML with `weather_forecasting_pipeline.config.load_config`.

## Main Files

- `configs/default.yaml`: default methodology-oriented experiment
- `configs/smoke.yaml`: fast smoke run
- `configs/weathercloud_mapping.yaml`: Weathercloud source-to-canonical mapping

## Path Settings

```yaml
paths:
  raw_data_dir: data/raw
  interim_dir: data/interim
  processed_dir: data/processed
  artifacts_dir: artifacts
  mapping_config: configs/weathercloud_mapping.yaml
```

These paths are relative to the repository root when configs are stored in `configs/`.

## Data Settings

```yaml
data:
  source: weathercloud
  timezone: Europe/Athens
  expected_frequency: 10min
  target: temp_c
  horizons:
    h01: 6
    h03: 18
    h06: 36
    h12: 72
  optional_horizons:
    m10: 1
    h24: 144
  lags: [1, 3, 6, 12, 24, 72, 144]
  rolling_windows: [6, 18, 36, 144]
  sequence_length: 144
  derived_metrics: [dew_point, vpd, heat_index, wind_chill]
```

The source data is expected at 10-minute resolution. Horizon values are expressed in source time steps, so `6` means one hour.

Both `horizons` and `optional_horizons` are trained. The `optional_horizons` block exists so contributors can keep auxiliary horizons (such as the very short `m10` and the day-ahead `h24`) syntactically separate from the headline dissertation horizons without changing the iteration policy. Entries are merged at run time, sorted by horizon length, and each `(model, horizon)` pair is trained independently. On key collision, `horizons` wins.

## Split Settings

```yaml
split:
  train: 0.70
  validation: 0.15
  test: 0.15
```

The fractions must sum to `1.0`. Splits are chronological; no random shuffling is performed.

## Scaling Settings

```yaml
scaling:
  method: standard
```

Supported scaling methods are delegated to MetDataPy scaler utilities. The scaler is fit on the training split only and applied to validation and test splits.

Two scalers are saved per horizon:

- `artifacts/scalers/scaler_<horizon>.joblib` — feature scaler used by ML and DL models.
- `artifacts/scalers/target_scaler_<horizon>.joblib` — target-only scaler used during deep-learning training so the loss surface does not depend on the absolute magnitude of the target. Predictions are inverse-transformed back to original units before metric computation, so reported MAE/RMSE/MAPE are always in the target's natural units.

## Model Settings

```yaml
models:
  baselines: [persistence, moving_average, climatology]
  ml: [linear_regression, random_forest, gradient_boosting, svr]
  dl: [lstm, gru, tcn]
```

Each configured model is trained separately for each configured horizon.
Supported baseline names are `persistence`, `moving_average`, and
`climatology`; all three are fit on the training partition only.

## Training Settings

```yaml
training:
  max_epochs: 20
  batch_size: 32
  learning_rate: 0.001
  patience: 5
  min_dl_train_rows: 300
```

Deep-learning models use early stopping based on validation loss. If the training split is too small, deep-learning models are skipped with a warning.

## Weathercloud Mapping

`configs/weathercloud_mapping.yaml` maps raw Weathercloud columns to canonical meteorological variables.

Example:

```yaml
fields:
  temp_c:
    col: Temperature (°C)
    unit: C
  wspd_ms:
    col: Wind Speed (km/h)
    unit: km/h
```

Wind speed and gust conversion from km/h to m/s is handled by MetDataPy.
