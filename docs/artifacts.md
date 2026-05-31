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
Snapshot analytical plots derive their axis units from `metrics.csv`, so
supplementary targets such as `rh_pct` and `pres_hpa` are labelled in
`%RH` and `hPa` rather than temperature units.

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

Snapshot time-series plots use a contiguous prediction block by default,
and callers may pass an explicit display window to
`SnapshotPaths(timeseries_start=..., timeseries_end=...)` when a known
data outage would make the default visual sample unsuitable.

Flags `--skip-supervised`, `--skip-interim`, and `--no-plots` let you trim
the snapshot for size when needed. `--skip-svr-models` is retained for
legacy snapshots that still contain SVR artifacts. `--force`
overwrites an existing snapshot but **preserves an existing
`CONCLUSION.md`** so AI- or human-authored conclusions are never
clobbered by re-running the script.

A `manifest.json` is written alongside `README.md` for downstream
tooling. `CONCLUSION.md` is intentionally **not** auto-generated.

The Google Colab notebook `notebooks/full_experiment_colab.ipynb` runs the
same snapshot script after the full experiment and zips the resulting
`runs/<run_id>/` folder under `exports/`. Copying that archive to Google
Drive is recommended for large full-run outputs.

### Delta runs + merged snapshots

When a methodology change affects only some models, retraining the
unchanged ones is wasted compute. The pipeline supports a delta-run
workflow:

1. Train only the changed models with `configs/default_delta.yaml`.
2. Snapshot the delta run as usual (`snapshot_run.py --run-id <id>_delta`).
3. Merge the delta snapshot with a baseline snapshot that contains the
   unchanged models via `scripts/merge_run_snapshots.py`, driving the
   canonical roster from `configs/default.yaml`.

```powershell
python scripts/merge_run_snapshots.py `
    --baseline runs/<baseline_id> `
    --delta runs/<id>_delta `
    --full-config configs/default.yaml `
    --output runs/<id>_final
```

The merged snapshot is a drop-in replacement for a full-run snapshot
(same layout, regenerated plot set, merged metrics CSV/JSON) and adds a
`MERGE_PROVENANCE.md` recording the source of every model and shared
artifact. Reuse is valid because the pipeline is seeded; the merger
also checks that split-metadata train/val/test boundaries match between
baseline and delta and emits warnings if they do not.

## Cleanup

Generated outputs can be removed safely between runs:

```powershell
python -m weather_forecasting_pipeline clean --config configs/default.yaml
```

`clean` and `--fresh` (on `train` / `run-all`) clear only the configured
`paths.interim_dir`, `paths.processed_dir`, and the `models/`, `scalers/`,
`metrics/`, `plots/`, `reports/` subtrees of `paths.artifacts_dir`, then
recreate those directories with `.gitkeep` placeholders. The raw data
directory is never touched. See
`docs/running-the-experiment.md#cleaning-generated-outputs-between-runs`
for guidance on when to clean.

## Git Tracking

Generated data and artifacts are ignored by Git, except `.gitkeep` placeholders that preserve directory structure.
