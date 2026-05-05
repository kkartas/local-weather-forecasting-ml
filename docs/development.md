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
- leakage regressions: QC and `gap` flags must not leak into the model
  feature set; sequence builder must use only past observations
  (`tests/test_leakage.py`)
- end-to-end CLI smoke run with optional-horizon training and reloadable
  scaler artifact (`tests/test_pipeline_artifacts.py`)

## Documentation Updates

When code changes affect behavior, update documentation in the same change.

Examples:

- new CLI command: update `README.md` and `docs/pipeline-overview.md`
- new config option: update `docs/configuration.md`
- new preprocessing behavior: update `docs/preprocessing-features.md`
- new model: update `docs/training.md`
- new metric or plot: update `docs/evaluation-reporting.md`
- new artifact path: update `docs/artifacts.md`
- MetDataPy requirement change: update `METDATAPY.md`
- methodology change: update `CHANGES.md`

## GitHub Pages

The documentation site is stored in `docs/` and includes `_config.yml` for GitHub Pages.

To publish it on GitHub:

1. Push the repository to GitHub.
2. Open repository settings.
3. Go to Pages.
4. Set the source to the default branch and `/docs`.
5. Save.

GitHub Pages will render the Markdown files as a static documentation site.
