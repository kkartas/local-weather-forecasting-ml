# Getting Started

## Requirements

- Python 3.10 or newer
- Python 3.11 recommended
- `metdatapy>=1.3.0`
- raw Weathercloud CSV exports in `data/raw/`

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Verify The Environment

```powershell
python -m pytest
```

Expected result:

```text
29 passed
```

## Run A Smoke Experiment

The smoke configuration uses fewer models and horizons so the full command path can be checked quickly.

```powershell
python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml
```

## Run The Default Experiment

```powershell
python -m weather_forecasting_pipeline run-all --config configs/default.yaml
```

Outputs are written under `data/interim/`, `data/processed/`, and `artifacts/`.

For the full multi-year dissertation workflow, including data placement,
ingestion checks, default training, and output review, see
`running-the-experiment.md`.
