# Supervised Datasets And Splits

Supervised dataset creation and splitting happen during the `train` stage.

## Input

```text
data/interim/prepared.parquet
```

## Horizon Targets

For each configured horizon, MetDataPy creates a future target column:

```text
<target>_t+<steps>
```

For example, with:

```yaml
target: temp_c
horizons:
  h01: 6
```

the supervised table contains:

```text
temp_c_t+6
```

At 10-minute resolution, `6` steps is a 1-hour forecast horizon.

## Lag Features

Lag features are generated from configured lag steps:

```yaml
lags: [1, 3, 6, 12, 24, 72, 144]
```

Examples:

```text
temp_c_lag1
rh_pct_lag6
pres_hpa_lag144
```

## Supervised Output Files

For each horizon:

```text
data/processed/supervised_<horizon>.parquet
```

Examples:

```text
data/processed/supervised_h01.parquet
data/processed/supervised_h03.parquet
```

## Feature Selection

Feature columns are selected from numeric columns while excluding:

- the current horizon target column
- all other future target columns containing `_t+`
- the deterministic `gap` indicator from `WeatherSet.insert_missing`

The target-column exclusions prevent future target leakage. MetDataPy causal
QC flags (`qc_*`) are allowed as model features because spike and flatline
checks are computed with past-only windows. The `gap` marker remains excluded
so inserted missing-timestamp rows do not become an explicit model signal.

## Chronological Split

The split is chronological:

```yaml
split:
  train: 0.70
  validation: 0.15
  test: 0.15
```

The project delegates fraction-based chronological splitting to MetDataPy.

No random split and no shuffling are used.

## Split Metadata

For each horizon:

```text
data/processed/split_metadata_<horizon>.json
```

This file records:

- target column
- feature columns
- split time ranges
- row counts

## Scaling

Scaling is applied after splitting:

1. fit the **feature scaler** on train features only
2. transform train, validation, and test features using the train-fitted scaler
3. fit the **target scaler** on the training target column only and persist it for DL inverse transforms
4. save both scaler objects under `artifacts/scalers/`

Validation and test statistics are never used to fit either scaler. The
target scaler exists so DL models can train on a standardised target
without breaking the comparability of MAE/RMSE/MAPE across families
(see `docs/training.md`).
