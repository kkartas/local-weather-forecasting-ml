# Pipeline Overview

The command-line interface is implemented in `weather_forecasting_pipeline.cli`.

## Stages

```text
raw Weathercloud CSV files
  -> ingest
  -> canonical parquet
  -> preprocess
  -> prepared parquet
  -> train
  -> supervised datasets, split metadata, scalers, models, predictions, metrics
  -> evaluate
  -> metrics JSON/CSV, plots, Markdown summary report
```

## Commands

### Ingest

```powershell
python -m weather_forecasting_pipeline ingest --config configs/default.yaml
```

Reads Weathercloud CSV exports from `data/raw/`, maps them to the canonical schema, normalizes units, converts timestamps to UTC, sorts data chronologically, and writes:

```text
data/interim/canonical.parquet
```

### Preprocess

```powershell
python -m weather_forecasting_pipeline preprocess --config configs/default.yaml
```

Reads canonical data, applies quality-control and feature-preparation logic, and writes:

```text
data/interim/prepared.parquet
```

### Train

```powershell
python -m weather_forecasting_pipeline train --config configs/default.yaml
```

For each horizon:

- creates a supervised learning table
- performs chronological train/validation/test splitting
- fits the scaler on train features only
- trains configured baseline, ML, and DL models
- saves models, scalers, predictions, metrics, and split metadata

### Evaluate

```powershell
python -m weather_forecasting_pipeline evaluate --config configs/default.yaml
```

Regenerates reports and plots from saved metrics and prediction files.

### Run All

```powershell
python -m weather_forecasting_pipeline run-all --config configs/default.yaml
```

Runs `ingest`, `preprocess`, `train`, and `evaluate` in order.

## Logging

Set the log level with:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/default.yaml --log-level DEBUG
```

Pipeline operations use Python logging rather than `print`.
