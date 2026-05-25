# Methodology Change Log

This file records divergences between the written dissertation methodology and the implemented forecasting pipeline.

Use this file only for changes affecting methodology, experimental design, or scientific assumptions. General implementation notes and missing MetDataPy features belong elsewhere.

## 2026-05-25 - Reuse unchanged-model artifacts via delta-run + merge

- Affected component:
  `configs/default_delta.yaml`,
  `scripts/merge_run_snapshots.py`,
  `runs/<run_id>/MERGE_PROVENANCE.md` documents.
- What changed:
  Added a "delta-run" workflow that retrains only the models whose
  methodology changed in the 2026-05-25 updates (ridge, lstm, gru, tcn,
  plus the deterministic baselines as a zero-cost sanity check). The
  unchanged ML models (`random_forest`, `gradient_boosting`) are reused
  bit-identically from the most recent prior run (currently `runs/180526`)
  via the new `scripts/merge_run_snapshots.py` orchestrator. The merger
  picks each model's artifacts from the delta snapshot when present and
  falls back to the baseline snapshot otherwise; models that exist in
  baseline but are not in the canonical `--full-config` roster (notably
  the dropped `linear_regression` and `svr` families) are omitted from
  the merged output. A `MERGE_PROVENANCE.md` file in the merged snapshot
  records the source of every model and shared artifact, plus warnings
  if baseline and delta split metadata disagree.
- Why it changed:
  Run 180526 burned ~17 hours of CPU time on the unchanged random forest
  and gradient boosting families. Because the data, splits, scalers,
  feature set, and model hyperparameters for those two families did not
  change between the 2026-05-18 (DL feature policy) updates and the
  2026-05-25 updates, retraining them again would produce bit-identical
  outputs and waste compute. Retraining only the changed models drops
  the next-run wall-clock from ~24-36 h to ~8-14 h.
- Methodology impact:
  The final dissertation metrics are sourced from two training runs that
  share data, splits, scalers, and seed (deterministic by construction).
  The methodology chapter should briefly acknowledge that
  `random_forest` and `gradient_boosting` rows in the final table are
  reused from run 180526, citing the determinism guarantees and the
  `MERGE_PROVENANCE.md` audit trail. No scientific result is changed by
  this decision; only the compute cost is reduced.
- Dissertation update required:
  Yes. Add one paragraph in the experimental-setup section explaining
  the delta-run + merge approach, the determinism guarantee that makes
  it valid, and that the per-row source is recorded in
  `runs/<merged_id>/MERGE_PROVENANCE.md`.

## 2026-05-25 - Thread-budget aware horizon parallelism

- Affected component:
  `weather_forecasting_pipeline.config.TrainingConfig`,
  `weather_forecasting_pipeline.training.pipeline._resolve_torch_threads_per_worker`,
  `weather_forecasting_pipeline.training.pipeline._apply_thread_cap`,
  `weather_forecasting_pipeline.training.pipeline._init_horizon_worker_progress`,
  `configs/default.yaml`, `configs/default_delta.yaml`.
- What changed:
  Each spawned horizon worker now applies a configurable cap on the
  intra-process BLAS/MKL/PyTorch thread pool. The cap is taken from
  `training.torch_threads_per_worker` (or auto-resolved to
  `max(1, cpu_count // horizon_workers)` when the YAML key is unset or
  null). The worker initializer exports `OMP_NUM_THREADS`,
  `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and calls
  `torch.set_num_threads()` before any model code runs. The default
  config now sets `horizon_workers: 6` and `torch_threads_per_worker: 2`
  for the dissertation's 12-logical-CPU target host.
- Why it changed:
  Run 180526 used `horizon_workers: 4` on a 12-CPU host because the
  codebase had no thread-cap mechanism and bumping to 6 workers would
  have let each PyTorch worker grab all 12 cores, producing severe
  outer x inner oversubscription (6 workers x 12 BLAS threads = 72
  threads competing for 12 cores). With the cap in place, 6 workers x
  2 threads = 12 threads cleanly mapped to 12 cores. This is the
  standard pattern for running multiple ML/DL processes on a single
  shared machine and is required for the dissertation's per-horizon
  parallelism to scale to the full horizon count.
- Methodology impact:
  None on the science. Models, splits, seeds, features, and metrics are
  unchanged. The change affects only wall-clock time and the number of
  CPU threads each worker is allowed to use.
- Dissertation update required:
  Optional. If the implementation chapter discusses parallelism, mention
  that the executable pipeline caps intra-worker BLAS/MKL/PyTorch threads
  so the outer worker count and the inner thread count multiply to the
  available core budget. Not a scientific point; purely an implementation
  detail.

## 2026-05-25 - SVR removed from the default ML roster

- Affected component:
  `weather_forecasting_pipeline.models.ml_models.make_ml_model`,
  `configs/default.yaml`.
- What changed:
  `svr` is no longer listed under `models.ml` in the default configuration.
  The `make_ml_model` factory still rejects unknown names but no longer
  constructs an `sklearn.svm.SVR` instance for new runs. `linear_regression`
  and the remaining ML families (`random_forest`, `gradient_boosting`) are
  unchanged. The `svr` factory branch is retained for backwards compatibility
  with explicit reproduction of run 180526 but is not exercised by any
  shipped configuration.
- Why it changed:
  Run 180526 showed that SVR never achieved the best MAE at any of the six
  configured horizons; produced **negative** persistence skill scores at the
  10-minute (-7.76), 1-hour (-0.63) and 24-hour (-0.30) horizons; and was at
  most marginally competitive at h06/h12 while remaining 6–23 % worse than
  `gradient_boosting` in MAE. Its kernelised fit cost is O(n²–n³) in the
  number of training rows, which on the dissertation's ~100k training rows
  produces ~870 MB per saved model and a fit time that dominates the run.
  The scientific contribution of keeping it is therefore zero while the
  cost is high; the model family is removed from the dissertation's final
  roster.
- Methodology impact:
  The final results table loses one row per horizon (the SVR row). All other
  models are unaffected. Run 180526 remains the cited evidence base in the
  dissertation's model-selection discussion.
- Dissertation update required:
  Yes. Document in the methodology chapter that SVR was evaluated in a
  preliminary run (180526) and excluded from the final roster on the basis
  of consistent under-performance and intractable compute cost on the
  full training set.

## 2026-05-25 - Linear regression replaced with RidgeCV

- Affected component:
  `weather_forecasting_pipeline.models.ml_models.make_ml_model`,
  `configs/default.yaml`, `configs/smoke.yaml`.
- What changed:
  Added a `ridge` ML model that constructs an `sklearn.linear_model.RidgeCV`
  with `alphas=(0.1, 1.0, 10.0, 100.0)` (efficient leave-one-out CV on the
  training split). The default configuration now lists `ridge` in place of
  `linear_regression`. The `linear_regression` factory is retained for
  backwards compatibility with run 180526 reproduction but is no longer the
  default linear baseline.
- Why it changed:
  Run 180526's `linear_regression` produced MAE values in line with the
  rest of the field (0.40–2.37 °C) but **RMSE values of 7.7, 13.4, 12.4 and
  13.9 °C** at horizons h03, h06, h12 and h24 respectively. The RMSE ≫ MAE
  gap is the signature of multicollinearity-driven extreme predictions
  on the wide 232-feature design matrix (lag+rolling stats are
  near-redundant). L2 regularisation is the standard, defensible remedy
  for this pathology; cross-validating α on the training split alone
  preserves the leakage rules in `AGENTS.md`.
- Methodology impact:
  Replaces one ML row per horizon. Tree ensembles, baselines and DL models
  are unaffected. Run 180526 remains the cited evidence base for the
  substitution.
- Dissertation update required:
  Yes. Document the substitution in the methodology chapter (RidgeCV with
  α ∈ {0.1, 1, 10, 100}, leave-one-out CV on the training partition only).
  In the results comparison, note that the OLS variant exhibited the RMSE
  explosion and that ridge regularisation is the response.

## 2026-05-25 - Deep-learning training stability: longer patience, LR scheduler, gradient clipping

- Affected component:
  `weather_forecasting_pipeline.models.dl_models.train_dl_model_from_datasets`,
  `weather_forecasting_pipeline.config.TrainingConfig`,
  `configs/default.yaml`.
- What changed:
  The DL training loop now (a) attaches a
  `torch.optim.lr_scheduler.ReduceLROnPlateau(factor=0.5, patience=3,
  min_lr=1e-5)` to the Adam optimiser and steps it after each validation
  pass; (b) clips parameter gradients with
  `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)`
  before each optimiser step; (c) reads `grad_clip_norm` from
  `TrainingConfig.grad_clip_norm` (`None` disables clipping). The default
  configuration now sets `max_epochs: 40`, `patience: 10`, and
  `grad_clip_norm: 1.0` (was `max_epochs: 20`, `patience: 5`, no clipping).
- Why it changed:
  Run 180526 produced two DL training collapses: TCN at h12 (MAE 5.62 °C,
  MAPE 117 %, early-stopped at epoch 7/20) and GRU at h24 (MAE 5.18 °C,
  MAPE 114 %, early-stopped at epoch 6/20). Both exited well before the
  loss surface had stabilised, with patience=5 firing on a transient
  validation-loss bump that a learning-rate cut would have absorbed.
  Gradient clipping is included as a low-cost insurance against the
  exploding-gradient events that produce these abrupt collapses on long
  horizons. The bundle is the standard prescription for stabilising
  sequence-model training under early stopping.
- Methodology impact:
  Affects only the DL family. The combination of longer patience, LR
  scheduling and gradient clipping changes the training trajectory but
  not the model architectures, the data, the splits or the metrics. The
  ML and baseline rows remain directly comparable to run 180526.
- Dissertation update required:
  Yes. Document the DL optimisation protocol (Adam, MSE loss, batch 32,
  max 40 epochs, validation-loss early stopping with patience 10,
  ReduceLROnPlateau with factor 0.5/patience 3/min_lr 1e-5, gradient
  clipping at L2 norm 1.0) in the methodology chapter.

## 2026-05-18 - Deep-learning feature set excludes lag columns

- Affected component:
  `weather_forecasting_pipeline.datasets.splits.select_dl_feature_columns`,
  `weather_forecasting_pipeline.training.pipeline._train_dl_if_possible`,
  `configs/default.yaml`, `configs/smoke.yaml`.
- What changed:
  Deep-learning models now read a narrower per-timestep feature set than
  tabular ML/baselines. By default `select_dl_feature_columns()` drops every
  MetDataPy `<col>_lag<n>` column from DL inputs while keeping canonical
  variables, derived metrics, calendar cyclic features, wind-direction
  encoding, rolling stats at `t`, and causal QC flags. The DL training path
  now uses a lazy `SequenceDataset` so windows are built on demand inside
  the PyTorch `DataLoader` rather than being stacked into one
  `(n_sequences, sequence_length, n_features)` tensor up front. Configurable
  via `data.dl_exclude_lag_features` (default `true`) and the optional
  `data.dl_feature_columns` allow-list. Tabular ML and baseline feature sets
  are unchanged.
- Why it changed:
  The full multi-year run failed during DL training with
  `numpy.core._exceptions._ArrayMemoryError` while attempting to allocate
  ~83 GiB for the train tensor on the m10 horizon. The sequence axis already
  encodes recent history, so feeding the wide tabular `_lag<n>` matrix at
  every timestep adds no information while making DL training infeasible on
  the dissertation hardware.
- Methodology impact:
  DL numbers will differ from any prior wide-feature DL attempts because
  the per-timestep feature vector is narrower. Tabular ML and baseline
  metrics are unaffected because their feature set has not changed. No prior
  DL run had successfully completed on the full configuration, so previous
  partial DL numbers were not used as dissertation references; the first
  full DL results are the ones produced under this new feature policy.
- Dissertation update required:
  Yes. Document the DL feature selection rule (lags excluded; per-timestep
  features only), the lazy `SequenceDataset` loading path, and the
  resulting RAM expectations in the methodology chapter.

## 2026-05-16 - MetDataPy 1.3.0 ingestion and causal QC migration

- Affected component:
  `weather_forecasting_pipeline.metdatapy_adapter`,
  `weather_forecasting_pipeline.datasets.splits`,
  `requirements.txt`, `pyproject.toml`.
- What changed:
  The project now requires `metdatapy>=1.3.0`. Ingestion uses MetDataPy's
  official DST localization options (`nonexistent="shift_forward"`,
  `ambiguous=False`), duplicate timestamp policy/reporting, and UTF-16LE
  no-BOM detection. Preprocessing calls causal QC windows for spike and
  flatline checks, and QC flag columns are now eligible model features. The
  split helper now delegates fraction-based chronological splitting to
  MetDataPy. A narrow local CSV parser fallback remains only for Weathercloud
  rows with surplus trailing empty fields.
- Why it changed:
  MetDataPy 1.3.0 implements several features previously tracked as upstream
  blockers in `METDATAPY.md`, allowing the forecasting repository to remove
  local DST and split orchestration logic and to use leakage-safe QC flags.
- Methodology impact:
  The observation-only scope, target, horizons, splits, and metrics are
  unchanged. QC flags may now be used as model inputs because they are computed
  with causal windows. Metrics should be regenerated because the feature set
  changes versus earlier QC-excluding runs.
- Dissertation update required:
  Yes, update the methodology to note MetDataPy 1.3.0, deterministic DST
  localization, duplicate handling, causal QC flags, and QC feature inclusion.

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

Superseded by the 2026-05-16 MetDataPy 1.3.0 migration above. QC flags were
excluded while MetDataPy only exposed centered spike/flatline windows; they are
eligible features again now that causal QC windows are used.

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
