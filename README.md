# Local Weather Forecasting ML Pipeline

Python pipeline for short-term local weather forecasting from a single Weathercloud station export. The project builds reproducible machine-learning experiments for an MSc dissertation on observation-only, station-level forecasting.

The pipeline ingests raw 10-minute Weathercloud CSV files, prepares canonical meteorological time-series data, creates supervised forecast datasets, trains baseline/ML/deep-learning models, evaluates them by horizon, and saves experiment artifacts.

## Scope

- Uses only historical weather-station observations.
- Does not use NWP model outputs or external forecast products.
- Uses chronological train/validation/test splits.
- Fits scalers on the training split only.
- Trains each model independently for each configured forecast horizon.

## Features

- Weathercloud CSV ingestion from `data/raw/`
- canonical schema and unit normalization through MetDataPy
- timezone conversion to UTC
- quality-control flags and gap insertion
- derived meteorological features
- cyclic time and wind-direction features
- lag, rolling, and forecast-horizon features
- baseline models: persistence, moving average, climatology
- scikit-learn models: linear regression, random forest, gradient boosting, SVR
- PyTorch models: LSTM, GRU, TCN (TCN dilations auto-sized to sequence length)
- MAE, RMSE, safe MAPE, and persistence skill score evaluation
- CSV, JSON, Markdown, plot, model, and scaler artifacts

## Project Layout

```text
configs/                          Experiment and source-column mapping files
data/raw/                         Raw Weathercloud CSV exports
data/interim/                     Canonical and prepared parquet datasets
data/processed/                   Supervised datasets, split metadata, predictions
artifacts/models/                 Trained model artifacts
artifacts/scalers/                Fitted scaler artifacts
artifacts/metrics/                Metrics CSV and JSON
artifacts/plots/                  Evaluation plots
artifacts/reports/                Markdown experiment summaries
src/weather_forecasting_pipeline/ Python package
tests/                            Pytest suite
scripts/                          Convenience scripts
```

## Installation

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The project requires `metdatapy>=1.3.0` for Weathercloud ingestion and meteorological time-series preparation.

The `pip install -e .` step is required for the `python -m weather_forecasting_pipeline ...` CLI to resolve the package. If you only need to run the test suite you can skip it: `pyproject.toml` adds `src/` to `pythonpath` so `python -m pytest` works from a plain checkout.

## Data

Place one or more Weathercloud CSV exports in:

```text
data/raw/
```

The default mapping expects a timestamp column named:

```text
Date (Europe/Athens)
```

Column mapping and units are configured in:

```text
configs/weathercloud_mapping.yaml
```

Update that file if the exported Weathercloud column names differ.

## Configuration

Main experiment configuration:

```text
configs/default.yaml
```

Smoke-test configuration:

```text
configs/smoke.yaml
```

The configuration controls paths, target variable, forecast horizons, lags, rolling windows, sequence length, models, split fractions, scaling, and training parameters.

## Usage

For a step-by-step workflow from multi-year Weathercloud exports to dissertation-ready outputs, see `docs/running-the-experiment.md`.

Run the full pipeline:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/default.yaml
```

Run individual stages:

```powershell
python -m weather_forecasting_pipeline ingest --config configs/default.yaml
python -m weather_forecasting_pipeline preprocess --config configs/default.yaml
python -m weather_forecasting_pipeline train --config configs/default.yaml
python -m weather_forecasting_pipeline evaluate --config configs/default.yaml
```

Run the smaller smoke configuration:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml
```

Convenience wrapper:

```powershell
python scripts/run_experiment.py --config configs/default.yaml
```

## Outputs

Generated outputs are written to:

```text
data/interim/canonical.parquet
data/interim/prepared.parquet
data/processed/supervised_<horizon>.parquet
data/processed/split_metadata_<horizon>.json
data/processed/predictions/
artifacts/models/
artifacts/scalers/
artifacts/metrics/metrics.csv
artifacts/metrics/metrics.json
artifacts/plots/
artifacts/reports/summary.md
```

## Testing

```powershell
python -m pytest
```

The tests cover MetDataPy integration, Weathercloud ingestion, timezone conversion, supervised target shifting, chronological splits, scaler fitting, metrics, and model smoke paths.

## Documentation

Detailed documentation is available in `docs/`. The `docs/` directory is ready to be published with GitHub Pages by selecting the default branch and `/docs` as the Pages source.

## Notes

Methodology changes are recorded in `CHANGES.md`.

Remaining or future MetDataPy requirements are tracked in `METDATAPY.md`.
