# Methodology Change Log

This file records divergences between the written dissertation methodology and the implemented forecasting pipeline.

Use this file only for changes affecting methodology, experimental design, or scientific assumptions. General implementation notes and missing MetDataPy features belong elsewhere.

## 2026-05-05 - QC flag columns excluded from model feature set

- Affected component:
  Feature selection in `weather_forecasting_pipeline.datasets.splits.select_feature_columns`.
- What changed:
  Every column whose name starts with `qc_` and the deterministic `gap`
  indicator from `WeatherSet.insert_missing` are now excluded from the model
  feature set. Previously they were passed to all ML and DL models as
  numeric features.
- Why it changed:
  MetDataPy 1.2.0 implements `qc_spike` and `qc_flatline` with
  `pandas.Series.rolling(..., center=True)` (`metdatapy/qc.py:123`,
  `:187`). Each row's QC flag therefore depends on a few observations
  after the row's own timestamp, which would let the model see roughly
  20–40 minutes of future information at 10-min cadence. Excluding these
  columns at the forecasting layer keeps the dissertation's strict
  "no future information in features" rule. A causal-window option in
  MetDataPy is tracked in `METDATAPY.md` so the local exclusion can be
  removed once it lands upstream.
- Methodology impact:
  Removes a small but real source of leakage. Reported metrics no longer
  benefit from future QC information.
- Dissertation update required:
  Yes, briefly mention in the methodology chapter that QC flags from
  MetDataPy 1.2.0 are excluded from the model feature set due to their
  centered-window implementation.

## 2026-05-05 - Optional horizons are trained alongside required horizons

- Affected component:
  Training loop in `weather_forecasting_pipeline.training.pipeline.train`.
- What changed:
  The training loop now iterates the merged set of `data.horizons` and
  `data.optional_horizons` (sorted by horizon length). Previously
  `optional_horizons` was parsed from the YAML but silently skipped, so
  `configs/default.yaml` produced no results for `m10` and `h24`.
- Why it changed:
  The dissertation requires multi-horizon coverage including the
  short-range (m10) and day-ahead (h24) horizons that the default
  configuration places under `optional_horizons`.
- Methodology impact:
  None — the configuration always intended these horizons to be trained.
- Dissertation update required:
  No.

## 2026-05-05 - DL models train on a standardised target

- Affected component:
  Deep-learning training in
  `weather_forecasting_pipeline.training.pipeline._train_dl_if_possible`.
- What changed:
  A separate target scaler is fit on the training partition only and
  applied to the DL target during training. DL predictions are
  inverse-transformed back to the target's original units before metric
  computation.
- Why it changed:
  Training on the raw target with MSE+Adam left DL models effectively at
  the mean prediction in short smoke runs, making the family
  incomparable with baselines and ML models. Reported metrics across all
  model families remain in the target's natural units.
- Methodology impact:
  Reported MAE/RMSE/MAPE for DL stay in original units, so cross-family
  comparison is unaffected. Documenting this choice in the methodology
  chapter is recommended.
- Dissertation update required:
  Yes, briefly note the DL target-scaling step.

## 2026-05-05 - Moving-average baseline uses a consecutive-step rolling mean

- Affected component:
  `weather_forecasting_pipeline.models.baselines.MovingAverageModel`.
- What changed:
  The baseline now prefers MetDataPy's past-only rolling-mean column for
  the smallest configured rolling window
  (`<target>_roll<window>_mean`, computed with `closed="left"`). It only
  falls back to averaging configured lag columns when no rolling-mean
  column is available.
- Why it changed:
  Averaging the first 4 lag columns sorted by lag number with the
  default lag set produced a non-consecutive average like
  `mean(t-1, t-3, t-6, t-12)`. That is not what a reader of the
  dissertation expects from a "moving average baseline".
- Methodology impact:
  The moving-average baseline's definition is now standard and easier to
  describe. Numbers will differ from earlier runs, so the baseline
  comparison row in dissertation Chapter 4 should be regenerated.
- Dissertation update required:
  Yes, mention the rolling-mean window used by the baseline.

## 2026-05-02 - Executable feature set follows installed MetDataPy capabilities

- Affected component:
  Feature engineering and raw Weathercloud ingestion.
- What changed:
  The executable pipeline uses only data-preparation functionality currently exposed by MetDataPy. After updating to MetDataPy 1.2.0, Weathercloud directory ingestion, delimiter/encoding handling, timezone-aware source mapping, `rain_rate_mmh`, wind-direction cyclic encoding, and rolling features are used directly.
- Why it changed:
  The dissertation requires MetDataPy to be the authoritative data preparation layer. Reimplementing missing reusable meteorological preparation logic inside this forecasting repository would violate the project architecture.
- Methodology impact:
  The final dissertation methodology remains valid. MetDataPy 1.2.0 removes the previous reduced-feature limitation for Weathercloud ingestion and the default feature-engineering set.
- Dissertation update required:
  No.
