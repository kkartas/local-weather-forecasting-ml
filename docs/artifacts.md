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

## Run Snapshots

For dissertation archival, every training run can be frozen into a
self-contained folder under `runs/<run_id>/` via:

```powershell
python scripts/snapshot_run.py --run-id <YYMMDD>
```

The snapshot copies the configs used, the interim and processed datasets,
all trained models and scalers, the metrics CSV/JSON, the per-model
prediction CSVs, the auto-generated summary report, and the standard
pipeline plots. It then regenerates an extended analytical plot set via
`weather_forecasting_pipeline.plotting.snapshot.generate_snapshot_plots`:

```text
runs/<run_id>/plots/
    actual_vs_predicted/   scatter and time-series for winning models
    residuals/             distribution + heteroscedasticity check
    comparison/            comparison_mae / comparison_rmse /
                           error_growth_by_horizon /
                           skill_score_heatmap / best_per_family
```

Flags `--skip-svr-models`, `--skip-supervised`, `--skip-interim`, and
`--no-plots` let you trim the snapshot for size when needed. `--force`
overwrites an existing snapshot but **preserves an existing
`CONCLUSION.md`** so AI- or human-authored conclusions are never
clobbered by re-running the script.

A `manifest.json` is written alongside `README.md` for downstream
tooling. `CONCLUSION.md` is intentionally **not** auto-generated.

## Cleanup

Generated outputs can be removed safely between runs:

```powershell
python -m weather_forecasting_pipeline clean --config configs/default.yaml
```

`clean` and `--fresh` (on `train` / `run-all`) delete only the
configured `paths.interim_dir`, `paths.processed_dir`, and the
`models/`, `scalers/`, `metrics/`, `plots/`, `reports/` subtrees of
`paths.artifacts_dir`. The raw data directory is never touched. See
`docs/running-the-experiment.md#cleaning-generated-outputs-between-runs`
for guidance on when to clean.

## Git Tracking

Generated data and artifacts are ignored by Git, except `.gitkeep` placeholders that preserve directory structure.
