# Running The Experiment

This page is the step-by-step workflow for running the forecasting experiments from raw Weathercloud exports to dissertation-ready artifacts.

Use `configs/smoke.yaml` only to verify that the software path works. Use `configs/default.yaml` for the real dissertation experiment.

## 1. Prepare The Environment

Create and activate a virtual environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Verify the test suite before running experiments:

```powershell
python -m pytest
```

## 2. Add Weathercloud Data

Place one or more Weathercloud CSV exports in:

```text
data/raw/
```

Multiple files are supported and preferred for multi-year data:

```text
data/raw/weathercloud_2022.csv
data/raw/weathercloud_2023.csv
data/raw/weathercloud_2024.csv
```

Do not manually merge files unless you have a specific reason. MetDataPy reads all CSV files in `data/raw/`, concatenates them, and sorts the records chronologically.

The default mapping expects a timestamp column named:

```text
Date (Europe/Athens)
```

If your Weathercloud export uses different column names, update:

```text
configs/weathercloud_mapping.yaml
```

The default mapping is configured for Weathercloud short headers such as
`temp (°C)`, `hum (%)`, `wspdavg (km/h)`, `wspdhi (km/h)`, `wdiravg (°)`,
`bar (hPa)`, `rainrate (mm/h)`, `solarrad (W/m²)`, and `uvi`.

Keep the experiment observation-only: do not add NWP outputs, external forecast products, or data from other stations.

## 3. Run A Smoke Check

Run the small smoke configuration first. It uses fewer horizons and models, so it should finish quickly:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml
```

This confirms the CLI, dependencies, MetDataPy integration, model training, metrics, and artifact writing path. Smoke outputs are not dissertation results.

## 4. Test Full-Data Ingestion

Before starting the full default experiment, run ingestion alone on the real CSV files:

```powershell
python -m weather_forecasting_pipeline ingest --config configs/default.yaml
```

This writes:

```text
data/interim/canonical.parquet
```

Inspect the row count and date range:

```powershell
python -c "import pandas as pd; df = pd.read_parquet('data/interim/canonical.parquet'); print(len(df)); print(df.index.min()); print(df.index.max())"
```

If full-year `Europe/Athens` exports contain nonexistent or ambiguous local
daylight-saving timestamps, the adapter retries ingestion and drops only the
rows that cannot be localized deterministically. Review the log for the
dropped-row count and mention it as a data-cleaning detail if any rows are
removed in the final experiment.

If duplicate timestamps are logged, the current adapter keeps the first row for each duplicated timestamp. Review overlapping CSV exports if the duplicate count is unexpectedly high.

## 5. Run Preprocessing

Run preprocessing after ingestion succeeds:

```powershell
python -m weather_forecasting_pipeline preprocess --config configs/default.yaml
```

This writes:

```text
data/interim/prepared.parquet
```

Preprocessing uses MetDataPy for gap insertion, QC flags, derived meteorological features, calendar features, wind-direction encoding, and rolling features.

## 6. Train Models

Train all configured models and horizons:

```powershell
python -m weather_forecasting_pipeline train --config configs/default.yaml
```

The default configuration trains:

- horizons: `m10`, `h01`, `h03`, `h06`, `h12`, `h24`
- baselines: `persistence`, `moving_average`, `climatology`
- ML models: `linear_regression`, `random_forest`, `gradient_boosting`, `svr`
- DL models: `lstm`, `gru`, `tcn`

Each model is trained independently for each horizon. The run can take significantly longer than the smoke check because the default configuration uses 24-hour sequences, more horizons, more models, and up to 20 DL epochs.

Deep-learning models are skipped for a horizon if the training split has fewer rows than `training.min_dl_train_rows`.

## 7. Generate Reports And Plots

Regenerate reports and plots from saved metrics:

```powershell
python -m weather_forecasting_pipeline evaluate --config configs/default.yaml
```

This writes or refreshes:

```text
artifacts/metrics/metrics.csv
artifacts/metrics/metrics.json
artifacts/plots/
artifacts/reports/summary.md
```

## 8. Run Everything In One Command

After ingestion has already been tested on the full raw data, you can run the whole pipeline in one command:

```powershell
python -m weather_forecasting_pipeline run-all --config configs/default.yaml
```

This runs `ingest`, `preprocess`, `train`, and `evaluate` in order.

## 9. Review Outputs

Use these files for dissertation result writing:

```text
artifacts/metrics/metrics.csv
artifacts/reports/summary.md
artifacts/plots/
data/processed/split_metadata_<horizon>.json
data/processed/predictions/
```

Confirm that `artifacts/reports/summary.md` shows the default project name:

```text
nea_triglia_short_term_weather_forecasting
```

If it shows `smoke_weather_forecasting_pipeline`, the artifacts are from the smoke configuration and should not be used as dissertation results.

## 10. Common Problems

### Ingestion Fails On DST Timestamps

Full-year `Europe/Athens` Weathercloud exports can include daylight-saving
transition hours. The pipeline uses MetDataPy 1.3.0's deterministic DST
localization policy: nonexistent spring-forward labels are shifted forward and
ambiguous fall-back labels use the standard-time side. If ingestion still
fails, inspect the raw timestamp column and `configs/weathercloud_mapping.yaml`.

### Column Names Do Not Match

Update `configs/weathercloud_mapping.yaml` so each raw Weathercloud column maps to the expected canonical variable.

### Duplicate Timestamps Are Reported

Check for overlapping CSV exports. The current pipeline keeps the first duplicate timestamp after MetDataPy ingestion so the modeling index remains unique.

### DL Models Are Skipped

Check `training.min_dl_train_rows` and the number of rows after chronological splitting and sequence construction. Full multi-year data should normally be enough.

### Results Look Like Smoke Results

Check the project name and horizon set in `artifacts/reports/summary.md`. Dissertation runs should include the default horizons and model families, not only `m10`, `h01`, `linear_regression`, and `gru`.

