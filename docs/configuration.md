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
  dl_exclude_lag_features: true
```

The source data is expected at 10-minute resolution. Horizon values are expressed in source time steps, so `6` means one hour.

Both `horizons` and `optional_horizons` are trained. The `optional_horizons` block exists so contributors can keep auxiliary horizons (such as the very short `m10` and the day-ahead `h24`) syntactically separate from the headline dissertation horizons without changing the iteration policy. Entries are merged at run time, sorted by horizon length, and each `(model, horizon)` pair is trained independently. On key collision, `horizons` wins.

### DL feature policy keys

```yaml
data:
  dl_exclude_lag_features: true     # default; remove _lag<n> columns from DL inputs
  # dl_feature_columns: [temp_c, rh_pct, ...]  # optional explicit allow-list
```

- `dl_exclude_lag_features` (default `true`) controls whether MetDataPy
  `<col>_lag<n>` columns are excluded from per-timestep DL inputs. The
  sequence axis already encodes recent history, so the lag matrix is
  redundant for sequence models and explodes RAM at full resolution.
- `dl_feature_columns` (default unset) is an optional explicit allow-list.
  When set, the DL feature set is exactly the listed columns; when unset,
  the policy above applies. The list is validated against the supervised
  dataset at training time.
- These keys affect DL only. Tabular ML and baseline models always keep
  the wide feature set produced by `select_feature_columns()`.

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
  ml: [ridge, random_forest, gradient_boosting]
  dl: [lstm, gru, tcn]
```

Each configured model is trained separately for each configured horizon.
Supported baseline names are `persistence`, `moving_average`, and
`climatology`; all three are fit on the training partition only.

Supported ML names are `ridge` (default linear baseline; ridge regression
with alpha in `{0.1, 1, 10, 100}`, selected by five chronological
expanding-window folds on the training partition and fit with an `lsqr`
solver), `random_forest`, `gradient_boosting`, and the deprecated names
`linear_regression` and `svr` retained for backwards compatibility with
run 180526 reproduction. See CHANGES.md (2026-05-25) for the rationale
behind the substitution, the chronological ridge CV rule, and the SVR
removal.

## Training Settings

```yaml
training:
  max_epochs: 40
  batch_size: 32
  learning_rate: 0.001
  patience: 10
  grad_clip_norm: 1.0
  min_dl_train_rows: 300
  horizon_workers: 3
  torch_threads_per_worker: 2
  progress_heartbeat_seconds: 60
  progress_log_epochs: true
```

`torch_threads_per_worker` caps the intra-process BLAS/MKL/PyTorch thread
pool inside each spawned horizon worker so that
`horizon_workers * torch_threads_per_worker` does not exceed the
machine's CPU budget. Set to `null` (YAML) or omit to auto-resolve to
`max(1, cpu_count // horizon_workers)`. Has no effect when
`horizon_workers <= 1`. Required when `horizon_workers` approaches
`cpu_count` to avoid outer x inner oversubscription (CHANGES.md
2026-05-25).

Deep-learning models use early stopping based on validation loss with
patience `patience` over `max_epochs`. The training loop additionally
attaches a `ReduceLROnPlateau` learning-rate scheduler (factor 0.5,
internal patience 3, floor 1e-5) so transient validation-loss plateaus
trigger a learning-rate cut before they trigger early stopping.

`grad_clip_norm` is the L2 threshold for `torch.nn.utils.clip_grad_norm_`
applied before each optimiser step. The default `1.0` reflects the
stability bundle introduced in CHANGES.md (2026-05-25) after run 180526
showed two DL training collapses consistent with exploding gradients.
Set this key to `null` (YAML) to disable clipping entirely; omit it to
keep the documented default.

If the training split is too small, deep-learning models are skipped with a warning.

`horizon_workers` controls process-level parallelism over forecast
horizons:

- `1` (default) preserves the strictly sequential behaviour and is the
  safe choice on laptops or shared machines.
- Values greater than `1` train each horizon in its own worker process.
  The effective worker count is capped at
  `min(horizon_workers, n_horizons, cpu_count)`, and `RandomForestRegressor`
  is forced to `n_jobs=1` so the inner-loop sklearn parallelism does not
  collide with the outer process pool.
  The shipped full-run configs use `3` workers because the MetDataPy scaler
  currently applies to a wide float64 feature frame inside each worker; `6`
  workers can exceed RAM on the target host even when CPU threads are capped.
- `progress_heartbeat_seconds` (default `60`) controls ML heartbeat
  interval during blocking `.fit()` calls. Set to `0` to disable ML
  heartbeat progress lines.
- `progress_log_epochs` (default `true`) controls DL per-epoch progress
  logs. When `false`, DL logs only first epoch, last epoch, and early-stop
  epoch when applicable.

See `docs/training.md#parallel-horizon-training` for the wall-time and
memory expectations.

## Weathercloud Mapping

`configs/weathercloud_mapping.yaml` maps raw Weathercloud columns to canonical meteorological variables.

Example:

```yaml
fields:
  temp_c:
    col: temp (°C)
    unit: C
  wspd_ms:
    col: wspdavg (km/h)
    unit: km/h
```

Wind speed and gust conversion from km/h to m/s is handled by MetDataPy.
