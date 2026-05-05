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
- every column whose name starts with `qc_` (MetDataPy QC flag columns)
- the deterministic `gap` indicator from `WeatherSet.insert_missing`

The first two exclusions prevent future target leakage. The QC and `gap`
exclusions are required because MetDataPy 1.2.0 computes `qc_spike` and
`qc_flatline` with centered rolling windows; using those flags as model
features would let each row see a small number of future observations.
The exclusion is enforced by
`tests/test_leakage.py::test_select_feature_columns_excludes_qc_and_gap_flags`,
and a causal-window option in MetDataPy is tracked in `METDATAPY.md`.

## Chronological Split

The split is chronological:

```yaml
split:
  train: 0.70
  validation: 0.15
  test: 0.15
```

The project computes timestamp boundaries from these fractions and delegates boundary splitting to MetDataPy.

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
