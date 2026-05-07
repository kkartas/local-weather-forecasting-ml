# Evaluation And Reporting

Evaluation is performed on the test split after training.

## Metrics

Implemented in `weather_forecasting_pipeline.evaluation.metrics`.

### MAE

Mean absolute error.

### RMSE

Root mean squared error.

### Safe MAPE

MAPE is computed only where true values are not near zero. Near-zero values are masked using:

```yaml
evaluation:
  mape_epsilon: 1.0e-6
```

If all true values are near zero, MAPE is reported as undefined.

For temperature targets near 0 °C MAPE is unstable even with masking, so
the persistence skill score (below) is the recommended cross-horizon
summary metric.

### Persistence Skill Score

For each `(target, horizon)` the metrics table also reports

```text
skill_score_persistence = 1 - rmse_model**2 / rmse_persistence**2
```

The persistence row is `0` by definition. Positive values mean the model
improves on persistence; negative values mean it underperforms it.
Horizons that do not include `persistence` in `models.baselines` get
`NaN` for this column.

## Metrics Outputs

```text
artifacts/metrics/metrics.csv
artifacts/metrics/metrics.json
```

Each row contains:

- target
- horizon label
- horizon steps
- model name
- model family
- MAE
- RMSE
- MAPE
- persistence skill score
- test row count

Deep-learning rows may also include:

- best validation loss
- epochs trained
- tabular test row count

## Markdown Report

The experiment report is written to:

```text
artifacts/reports/summary.md
```

The report contains:

- target
- no-NWP input policy
- chronological split policy
- scaling policy
- MAPE policy
- metrics table

## Plots

Generated plots are written under:

```text
artifacts/plots/
```

Plot types:

- model comparison by MAE
- average MAE by horizon
- actual vs predicted for representative prediction files
- residual distributions for representative prediction files

Plotting is skipped with a warning if the installed Matplotlib/NumPy versions are incompatible.

## Evaluate Command

```powershell
python -m weather_forecasting_pipeline evaluate --config configs/default.yaml
```

This regenerates reports and plots from saved metrics and prediction files.
