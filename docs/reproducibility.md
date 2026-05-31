# Reproducibility

This page records the information needed to reproduce the dissertation
experiments from a clean checkout.

## Environment

- Python: 3.10 or newer; Python 3.11 is recommended.
- Required data-preparation package: `metdatapy>=1.3.0`.
- Runtime install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

- Development and test install:

```powershell
pip install -e ".[dev]"
```

## Source Checkout

Record the exact commit used for a dissertation run:

```powershell
git rev-parse HEAD
git status --short
```

The working tree should be clean except for local raw data and generated
outputs ignored by `.gitignore`.

## Data Scope

Raw Weathercloud station exports are expected under `data/raw/`, but they are
not committed to the source repository. The experiments use only historical
single-station observations. Numerical Weather Prediction outputs, external
forecast products, reanalysis data, satellite data, and neighboring-station
observations are outside the dissertation methodology.

## Main Experiment

Run the primary temperature experiment with:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/default.yaml --fresh
```

The command writes generated datasets under `data/interim/` and
`data/processed/`, and model, metric, plot, and report artifacts under
`artifacts/`.

## Supplementary Targets

Relative-humidity and pressure checks are supplementary evidence, not
replacements for the primary temperature experiment:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/target_rh_ml.yaml --fresh
python -m weather_forecasting_pipeline run-all --config configs/target_pres_ml.yaml --fresh
```

## Final Snapshot

Freeze a completed run with:

```powershell
python scripts/snapshot_run.py --run-id <YYMMDD>_final
```

The snapshot should contain the configs, metrics, plots, report, models,
scalers, split metadata, `manifest.json`, and any hand-authored
`CONCLUSION.md`. Large `runs/<run_id>/` folders are ignored by git and should
be archived separately from the source repository.

If a final result combines a baseline snapshot and a delta snapshot, keep the
generated `MERGE_PROVENANCE.md` with the archived run so every metric row can
be traced to its source artifact.

## Verification

Before publishing or submitting the source repository, run:

```powershell
python -m pytest
python scripts/generate_smoke_raw_data.py
python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml --fresh
```

The test suite covers MetDataPy integration, timestamp handling,
chronological splits, leakage-prevention behavior, metrics, artifacts, and
baseline/ML/DL smoke paths.
