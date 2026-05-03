# Local Short-Term Weather Forecasting Pipeline

This repository contains the practical software component for the MSc dissertation:

**Τοπική βραχυπρόθεσμη πρόγνωση καιρού από μετρήσεις μετεωρολογικού σταθμού με μηχανική μάθηση και νευρωνικά δίκτυα**

The project evaluates whether observation-only measurements from a single local weather station can support useful short-term local forecasting. It compares persistence and moving-average baselines, traditional machine learning models, and neural networks across configurable forecast horizons.

## Scientific Scope

- Inputs are historical local station observations only.
- Numerical Weather Prediction outputs and external forecast products are not allowed as model inputs.
- Splits are chronological and never shuffled.
- Scalers are fit only on the training split.
- Evaluation is performed on the test split after training and validation.

## MetDataPy Role

MetDataPy is the mandatory data preparation layer for this dissertation. This repository orchestrates experiments, models, artifacts, plots, and reports. It must not duplicate reusable meteorological preparation logic that belongs in MetDataPy.

MetDataPy is used for supported functionality including:

- source-to-canonical mapping
- unit normalization
- timestamp normalization
- gap insertion
- QC flagging
- derived meteorological variables
- calendar features
- supervised lag and horizon table creation
- time-safe split utilities
- scaler fit/apply utilities

Missing MetDataPy functionality required by the methodology is tracked in [METDATAPY.md](METDATAPY.md). Methodological deviations are tracked in [CHANGES.md](CHANGES.md).

## Repository Layout

```text
configs/                         Experiment and mapping configuration
data/raw/                        Raw Weathercloud CSV exports
data/interim/                    Canonical MetDataPy outputs
data/processed/                  ML-ready datasets and split metadata
artifacts/models/                Trained models
artifacts/scalers/               Fitted scalers
artifacts/metrics/               CSV and JSON metrics
artifacts/plots/                 Evaluation plots
artifacts/reports/               Markdown experiment reports
src/weather_forecasting_pipeline/ Forecasting orchestration package
tests/                           Pytest test suite
scripts/                         Convenience experiment scripts
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

MetDataPy must be installed and importable. The pipeline expects MetDataPy 1.1.0 or later.

## Configuration

The default experiment configuration is [configs/default.yaml](configs/default.yaml). Weathercloud source column mapping is configured in [configs/weathercloud_mapping.yaml](configs/weathercloud_mapping.yaml).

Place raw Weathercloud CSV exports under `data/raw/`. The dissertation methodology expects Weathercloud exports with 10-minute observations and a local timestamp column such as `Date (Europe/Athens)`.

## Commands

```powershell
python -m weather_forecasting_pipeline ingest --config configs/default.yaml
python -m weather_forecasting_pipeline preprocess --config configs/default.yaml
python -m weather_forecasting_pipeline train --config configs/default.yaml
python -m weather_forecasting_pipeline evaluate --config configs/default.yaml
python -m weather_forecasting_pipeline run-all --config configs/default.yaml
```

A convenience wrapper is also available:

```powershell
python scripts/run_experiment.py --config configs/default.yaml
```

## Artifacts

Pipeline outputs are written under `artifacts/`:

- models in `artifacts/models/`
- scalers in `artifacts/scalers/`
- metrics CSV/JSON in `artifacts/metrics/`
- plots in `artifacts/plots/`
- dissertation-oriented Markdown report in `artifacts/reports/`

Processed data and split metadata are written under `data/processed/`.

## Current Limitations

MetDataPy 1.1.0 does not yet expose every Weathercloud-specific preparation API required by the full dissertation methodology. It now covers encoding-detecting CSV reads and timezone-aware source mapping, but robust multi-file Weathercloud directory ingestion, semicolon delimiter handling, rolling features, wind-direction cyclic encoding, and `rain_rate_mmh` still need MetDataPy support before full final experiments. See [METDATAPY.md](METDATAPY.md).

The repository intentionally fails clearly when a required MetDataPy-owned capability is missing instead of silently reimplementing it locally. MetDataPy 1.1.0 is used for encoding-detecting CSV reads and timezone-aware mapping via `ts.timezone`.

## Updating MetDataPy

When a missing preparation feature is discovered:

1. Add an entry to `METDATAPY.md`.
2. Implement or update the feature in the MetDataPy library.
3. Install the updated MetDataPy version.
4. Consume the new API from `weather_forecasting_pipeline.metdatapy_adapter`.
5. Add or update integration tests.

## Reproducibility

Run the tests with:

```powershell
pytest
```

Run a smoke experiment after adding compatible raw data:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml
```

Use `configs/default.yaml` for the full configured methodology once the required MetDataPy Weathercloud and rolling-feature APIs are available.
