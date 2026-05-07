# Methodology Change Log

This file records divergences between the written dissertation methodology and the implemented forecasting pipeline.

Use this file only for changes affecting methodology, experimental design, or scientific assumptions. General implementation notes and missing MetDataPy features belong elsewhere.

## 2026-05-08 - Climatology baseline added

- Affected component:
  `weather_forecasting_pipeline.models.baselines.ClimatologyModel`,
  `configs/default.yaml`.
- What changed:
  Added a hour-of-year climatology baseline that forecasts the training-only
  mean of the target keyed by ``(month, hour)`` of the forecast origin. The
  default configuration now lists `climatology` alongside `persistence` and
  `moving_average`.
- Why it changed:
  Persistence is a strong anchor at short horizons but uninformative for
  longer ones; a climatology baseline is the standard meteorological second
  anchor. Including it gives the dissertation comparison chapter a fair
  long-horizon reference and tightens the "ML beats trivial baselines" story.
- Methodology impact:
  Adds one row per horizon to the metrics table. No effect on existing model
  results because climatology fitting touches only the training partition.
- Dissertation update required:
  Yes, briefly mention the climatology baseline definition (`(month, hour)`
  lookup of training-only target means) in the methodology chapter.

## 2026-05-08 - Persistence skill score reported alongside MAE/RMSE/MAPE

- Affected component:
  `weather_forecasting_pipeline.evaluation.metrics.persistence_skill_score`,
  `weather_forecasting_pipeline.training.pipeline._attach_persistence_skill_score`.
- What changed:
  Each metrics row now carries a `skill_score_persistence` column equal to
  `1 - rmse_model**2 / rmse_persistence**2` per horizon. The persistence row
  is `0` by definition; positive values mean the model improves on
  persistence; negative values mean it underperforms.
- Why it changed:
  MAPE is unstable for temperature near 0 °C and across the sign change.
  Persistence skill score is the standard short-term forecasting summary
  metric, scale-free and robust to small target magnitudes. Reporting both
  keeps the dissertation comparison defensible without removing MAPE.
- Methodology impact:
  Purely additive; the column is derived from RMSE values already computed
  on the test partition.
- Dissertation update required:
  Yes, briefly define the skill score formula and how to interpret it.

## 2026-05-08 - TCN receptive field grows with the configured sequence length

- Affected component:
  `weather_forecasting_pipeline.models.dl_models.TCNRegressor`,
  `weather_forecasting_pipeline.models.dl_models.make_dl_model`.
- What changed:
  The TCN now picks its dilation schedule from `data.sequence_length` so the
  receptive field always covers the input window. The previous fixed
  `(1, 2, 4)` schedule had a 15-step receptive field, while the default
  configuration uses 144-step input sequences, so most of the input was
  ignored.
- Why it changed:
  Reporting TCN as a multi-hour temporal model is only defensible if the
  network can actually see the multi-hour context the dissertation describes.
- Methodology impact:
  TCN results will differ from earlier runs because the model now uses the
  full input window. Other model families are unaffected.
- Dissertation update required:
  Yes, mention the dilation schedule selection rule (doubling dilations until
  receptive field ≥ sequence length, capped at d=64).

## 2026-05-08 - Dropout regularisation added to DL models

- Affected component:
  `weather_forecasting_pipeline.models.dl_models.RecurrentRegressor`,
  `weather_forecasting_pipeline.models.dl_models.TemporalBlock`,
  `weather_forecasting_pipeline.models.dl_models.TCNRegressor`.
- What changed:
  All three DL models now include a small dropout (`p=0.1`) between sequence
  encoding and the regression head; TCN blocks additionally apply dropout
  inside each block.
- Why it changed:
  The dissertation's modest training-row budget combined with sequence
  length 144 makes the unregularised default prone to overfit, especially
  on short horizons where persistence and ML baselines are strong.
- Methodology impact:
  DL numbers will shift slightly versus earlier runs. No change to the
  scientific scope or to baseline/ML comparisons.
- Dissertation update required:
  Mention DL dropout rate where the network architectures are described.

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
