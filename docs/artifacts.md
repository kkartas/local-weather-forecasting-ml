# Artifacts

Generated data and model outputs are intentionally separated from source code.

## Interim Data

```text
data/interim/canonical.parquet
data/interim/prepared.parquet
```

`canonical.parquet` is the post-ingestion canonical time series.

`prepared.parquet` is the post-QC and feature-engineered time series.

## Processed Data

```text
data/processed/supervised_<horizon>.parquet
data/processed/split_metadata_<horizon>.json
data/processed/predictions/
```

These files are generated during training.

## Models

```text
artifacts/models/
```

Model files use horizon-specific names so each forecast horizon has independent trained artifacts.

## Scalers

```text
artifacts/scalers/scaler_<horizon>.joblib
artifacts/scalers/target_scaler_<horizon>.joblib
```

Both scalers are fit on the training partition only and persisted as the
fitted MetDataPy `ScalerParams` objects via `joblib`, so downstream
inference can `joblib.load` and `apply_scaler` without any further
configuration.

- `scaler_<horizon>.joblib` — feature scaler used by ML and DL models.
- `target_scaler_<horizon>.joblib` — target-only scaler used during DL
  training. DL predictions are inverse-transformed back to original
  units before metric computation, so reported metrics are always in
  the target's natural units.

## Metrics

```text
artifacts/metrics/metrics.csv
artifacts/metrics/metrics.json
```

Use CSV for quick inspection and JSON for programmatic access.

## Plots

```text
artifacts/plots/
```

Plots summarize model comparison, horizon error, actual/predicted curves, and residual distributions.

## Reports

```text
artifacts/reports/summary.md
```

The summary report is intended to support dissertation Chapter 4 result writing.

## Git Tracking

Generated data and artifacts are ignored by Git, except `.gitkeep` placeholders that preserve directory structure.
