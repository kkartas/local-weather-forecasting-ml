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

### Climatology

Predicts the future target as the training-only mean of the target keyed
by the forecast origin's `(month, hour)`. Unseen keys at inference fall
back to the global training mean. The lookup is fitted once per horizon
on `splits["train"]`, so no validation or test rows leak into the
baseline. Climatology is the standard meteorological long-horizon
anchor and complements persistence at short horizons.

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

### DL Feature Policy

DL inputs are deliberately narrower than the tabular ML feature set.

- ML and baseline models keep the wide feature matrix that
  `select_feature_columns()` produces (canonical variables, derived
  metrics, calendar cyclic features, wind-direction encoding, rolling
  statistics, causal QC flags, **and** the full `<col>_lag<n>` matrix).
- DL models read per-timestep features through `select_dl_feature_columns()`,
  which excludes every `_lag<n>` column. The sequence axis already encodes
  the same recent history, so feeding 144 timesteps × ~1.6k features
  (lag history at every step) provides no additional information while
  ballooning RAM. With the default configuration this drops the DL feature
  count from the tabular ~1.6k to roughly 50 per-timestep signals and is
  the change that lets the dissertation's full multi-year run complete
  DL training.
- Set `data.dl_exclude_lag_features: false` in YAML to retain the wide
  legacy DL feature set, or pass an explicit `data.dl_feature_columns`
  allow-list for fully transparent dissertation runs. The pipeline
  validates that every requested column exists in the prepared dataset
  before training starts.
- At the start of every DL fit the pipeline logs both feature counts and a
  conservative batch-size memory estimate so long runs are auditable from
  the terminal output alone:

```text
DL feature selection: horizon=h12 model=lstm n_dl_features=52 n_tabular_features=1632 exclude_lag_features=True
DL memory estimate: horizon=h12 model=lstm sequence_length=144 n_dl_features=52 batch_size=32 batch_bytes=958464
```

### Lazy Sequence Loading

Each DL split is wrapped in a `SequenceDataset`
(`weather_forecasting_pipeline.datasets.splits.SequenceDataset`) that holds
the per-timestep feature matrix once and builds each
`(sequence_length, n_features)` window on demand inside the PyTorch
`DataLoader`. The pipeline never materializes
`(n_sequences, sequence_length, n_features)` in memory; the legacy
`sequence_arrays_from_split` helper is kept for unit tests only.

`train_dl_model_from_datasets` and `predict_dl_model_from_dataset` are the
dataset-friendly entry points used by the pipeline; the legacy
ndarray-based `train_dl_model` / `predict_dl_model` wrappers remain for
compatibility with the smoke tests.

All three DL models apply a small dropout (`p=0.1`) between sequence
encoding and the regression head; the TCN additionally applies dropout
inside each temporal block. Dropout is fixed in code rather than
configured because the dissertation reports a single fixed
configuration per family.

The TCN's dilation schedule is selected automatically from
`sequence_length`: dilations double (1, 2, 4, ...) — capped at 64 — and
enough blocks are stacked so the receptive field is at least
`sequence_length`. With `sequence_length=144` the schedule is
`(1, 2, 4, 8, 16, 32, 64)`, giving a receptive field of 255 steps.

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

## Parallel Horizon Training

Long full-config runs are dominated by single-threaded RBF `SVR` per
horizon. Set `training.horizon_workers` greater than `1` to run each
horizon's full per-horizon pipeline (supervised build, split, scalers,
baselines, ML, DL, predictions, and per-horizon artifacts) in its own
worker process.

```yaml
training:
  horizon_workers: 4
```

- The pool uses the `spawn` start method so each worker re-imports the
  package cleanly and sklearn/torch process state stays isolated. This
  matches the Windows fork policy and avoids fork-related issues with
  PyTorch/CUDA initialization on Linux.
- Workers are capped at `min(configured, n_horizons, cpu_count)`, so
  `horizon_workers: 16` on a 6-horizon run with 8 CPUs collapses to 6.
- When `horizon_workers > 1` the pipeline forces `RandomForestRegressor`
  to `n_jobs=1` to avoid outer × inner CPU oversubscription.
- Each worker reloads `data/interim/prepared.parquet` from disk on
  start. The main process therefore prepares the data once (`ingest`,
  `preprocess`) and spawns workers only for `train`.
- All scientific outputs (artifact paths, predictions CSV layout, metric
  schema) are identical to a sequential run; only the order in which
  rows arrive in the merged `metrics.csv` may differ. The `merge` step
  (`_attach_persistence_skill_score` and `_write_metrics_and_plots`) is
  always executed once on the main process after all workers finish.
- Logs include `horizon=`, `worker=`, and `pid=` fields on every per-horizon
  stage so per-process progress is easy to follow even when workers
  interleave.

Worker logs use the same ISO-8601 UTC format as the main process. A
horizon-specific seed offset is applied inside each worker
(`set_random_seed(project.random_seed + offset)`); identical numerical
results are produced regardless of `horizon_workers` on small fixtures
(see `tests/test_parallel_horizons.py`). Float drift on RandomForest in
practice has not exceeded `1e-5` on the smoke fixture; document any
material deviation in `CHANGES.md` if it ever appears on the dissertation
configuration.

## Progress Logging

The CLI installs a single root handler with ISO-8601 UTC timestamps
(`YYYY-MM-DDTHH:MM:SSZ LEVEL logger: message`). Every pipeline stage,
horizon, and model emits matching `Stage start: ...` and
`Stage finish: ... elapsed=<seconds>s ...` log lines, so a long run is
auditable from the terminal output alone.

The training stage logs include:

- a `Train context: ...` line with the resolved horizons, target, and the
  configured baseline / ML / DL model lists,
- one start/finish pair for each horizon (`Stage start: horizon h01 steps=6 target=temp_c_t+6`),
- per-stage start/finish pairs for the supervised dataset build,
  chronological split, feature-scaler fit, and target-scaler fit,
- a `Stage start: train model family=<family> model=<name> horizon=<label> ...`
  line before every baseline, ML, and DL fit, followed by a matching
  finish line that includes `elapsed=<seconds>s`, `mae=<value>`, and
  `rmse=<value>` (DL finish lines also include `epochs_trained` and
  `best_validation_loss`),
- explicit `Skip model: family=dl model=<name> horizon=<label> reason=<...>`
  warnings when a deep-learning model cannot be trained because the train
  split has fewer rows than `min_dl_train_rows` or because sequence
  construction does not yield enough samples.

The logging additions are observational only; training data, splits,
scalers, hyperparameters, and metrics are unchanged.

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
