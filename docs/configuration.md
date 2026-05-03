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
  lags: [1, 3, 6, 12, 24, 72, 144]
  rolling_windows: [6, 18, 36, 144]
  sequence_length: 144
  derived_metrics: [dew_point, vpd, heat_index, wind_chill]
```

The source data is expected at 10-minute resolution. Horizon values are expressed in source time steps, so `6` means one hour.

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

## Model Settings

```yaml
models:
  baselines: [persistence, moving_average]
  ml: [linear_regression, random_forest, gradient_boosting, svr]
  dl: [lstm, gru, tcn]
```

Each configured model is trained separately for each configured horizon.

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
