# Development And Testing

## Package Structure

```text
src/weather_forecasting_pipeline/
  cli.py
  config.py
  metdatapy_adapter.py
  datasets/
  evaluation/
  models/
  plotting/
  training/
  utils/
```

## Important Boundaries

MetDataPy owns reusable meteorological preparation logic. This repository owns experiment orchestration, model training, evaluation, plots, reports, and command-line execution.

Do not add local ingestion, QC, feature-engineering, lag, rolling, or horizon-generation implementations when the logic belongs in MetDataPy.

## Tests

Run:

```powershell
python -m pytest
```

Current test coverage includes:

- MetDataPy import and Weathercloud integration
- semicolon-delimited multi-file ingestion
- timezone conversion
- canonical schema expectations
- gap, QC, calendar, wind, and rolling features
- supervised horizon target shifting
- chronological splitting
- train-only scaler fitting
- MAE, RMSE, and safe MAPE
- baseline, ML, and DL smoke paths
- leakage regressions: causal QC flags may be used as model features, `gap`
  flags must not leak into the model feature set, and sequence builder must
  use only past observations
  (`tests/test_leakage.py`)
- end-to-end CLI smoke run with optional-horizon training and reloadable
  scaler artifact (`tests/test_pipeline_artifacts.py`)
- structured progress logging: ISO-timestamped formatter, per-stage,
  per-horizon, and per-model start/finish markers with `elapsed=` and
  `mae=` fields (`tests/test_logging.py`)
- climatology baseline correctness, persistence skill score behavior,
  TCN receptive-field sizing, and target-scaler round-trip
  (`tests/test_baselines_and_metrics.py`)
- DL feature selection drops `_lag<n>` columns and the lazy
  `SequenceDataset` builds windows on demand without allocating the
  dense `(n_sequences, sequence_length, n_features)` tensor
  (`tests/test_datasets.py`, `tests/test_models.py`)
- parallel horizon training: two horizons on a tiny fixture run end to
  end through `ProcessPoolExecutor` and produce a merged
  `metrics.csv` with the same rows as the sequential run
  (`tests/test_parallel_horizons.py`)
- artifact cleanup: `clean` subcommand and `--fresh` flag remove only
  generated outputs, recreate `.gitkeep` placeholders, and leave raw data
  intact
  (`tests/test_pipeline_artifacts.py`)

## Run Snapshots

`scripts/snapshot_run.py` freezes a completed training run into
`runs/<run_id>/`. It copies the configs, interim and processed datasets,
trained models, scalers, metrics, predictions, and summary report, then
calls `weather_forecasting_pipeline.plotting.snapshot.generate_snapshot_plots`
to produce an extended analytical plot set (per-model scatter, time-series,
residual diagnostics; aggregate MAE/RMSE comparisons, error-growth curves,
skill-score heatmap, best-per-family bars).

The snapshot script is the canonical way to produce dissertation-grade
per-run archives. It does **not** write `CONCLUSION.md`; that file is
authored separately and is preserved across `--force` re-runs.

## Continuous Integration

A GitHub Actions workflow at `.github/workflows/ci.yml` runs `pytest`
and a CLI smoke run on Python 3.10 and 3.11 for every push and pull
request to `master`. Keep that workflow green before merging changes.

The CLI smoke step uses `configs/smoke.yaml`, which reads from the isolated
`data/raw_smoke/` directory rather than `data/raw/`. This keeps the smoke run
fast and deterministic on a synthetic file, independent of the committed real
station exports in `data/raw/`. CI generates the synthetic file first:

```powershell
python scripts/generate_smoke_raw_data.py
python -m weather_forecasting_pipeline run-all --config configs/smoke.yaml
```

On Windows, tests that spawn `multiprocessing.Manager` or process-pool
horizon workers are skipped because some Python distributions can emit
access-violation traces during process shutdown even when assertions pass.
Linux CI still exercises those multiprocessing paths.

## Documentation Updates

When code changes affect behavior, update documentation in the same change.

Examples:

- new CLI command: update `README.md` and `docs/pipeline-overview.md`
- new config option: update `docs/configuration.md`
- new preprocessing behavior: update `docs/preprocessing-features.md`
- new model: update `docs/training.md`
- new metric or plot: update `docs/evaluation-reporting.md`
- new artifact path: update `docs/artifacts.md`
- MetDataPy integration behavior change: update the relevant data-ingestion,
  preprocessing, dataset, or training documentation
- methodology change: update the dissertation methodology and the affected
  documentation under `docs/`

## GitHub Pages

The documentation site is stored in `docs/` and includes `_config.yml` for GitHub Pages.

To publish it on GitHub:

1. Push the repository to GitHub.
2. Open repository settings.
3. Go to Pages.
4. Set the source to the default branch and `/docs`.
5. Save.

GitHub Pages will render the Markdown files as a static documentation site.
