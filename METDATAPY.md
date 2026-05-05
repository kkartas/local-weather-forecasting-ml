# MetDataPy Feature Requirements

This file tracks MetDataPy features required by the dissertation forecasting project that are missing, incomplete, or recently resolved in the currently installed MetDataPy library.

Current installed version inspected: `metdatapy==1.2.0`.

## Active Missing Or Incomplete Features

### 2026-05-05 - Causal-window option for QC spike and flatline checks

- Required feature:
  Allow `qc_spike` and `qc_flatline` to compute their rolling local
  median/MAD/variance with `center=False, closed="left"` (or an equivalent
  `causal=True` toggle) so the resulting QC flags only depend on past
  observations.
- Reason:
  MetDataPy 1.2.0 implements both checks with `pandas.Series.rolling(window,
  center=True, ...)` (`metdatapy/qc.py:123` and `:187`). When the resulting
  QC flag columns are used as model features, each row sees roughly half a
  window of future observations (≈20–40 minutes at 10-min cadence). That
  violates the dissertation's "no future information in features" rule.
- Workaround in this repository:
  `weather_forecasting_pipeline.datasets.splits.select_feature_columns`
  excludes every column whose name starts with `qc_`, plus the deterministic
  `gap` indicator from `insert_missing`. Tests in
  `tests/test_leakage.py::test_select_feature_columns_excludes_qc_and_gap_flags`
  guard the rule.
- Expected input:
  Existing dataframes; behaviour controlled per-call via a `causal=True`
  argument or via the `WeatherSet` configuration.
- Expected output:
  QC flag columns whose value at row `t` only depends on observations at or
  before `t`.
- Suggested API:
  `qc_spike(df, columns=None, window=9, thresh=6.0, causal=False)`
  `qc_flatline(df, columns=None, window=5, tol=0.0, causal=False)`
  with equivalent toggles surfaced via `WeatherSet.qc_spike()` and
  `WeatherSet.qc_flatline()`.
- Priority:
  Medium
- Blocking status:
  Does not block experiments because QC features are excluded from the model
  feature set in this repository. Resolving this in MetDataPy would let the
  forecasting layer drop the local exclusion and benefit from QC information
  in the feature set without leakage.
- Forecasting pipeline usage:
  Required by `weather_forecasting_pipeline.datasets.splits.select_feature_columns`
  before QC flags can be re-included as features.
- Dissertation update required:
  No.

### 2026-05-03 - Duplicate timestamp handling policy in Weathercloud ingestion

- Required feature:
  Provide an explicit duplicate timestamp handling policy for Weathercloud directory ingestion.
- Reason:
  Duplicate timestamp handling is reusable meteorological time-series preparation logic and should be owned by MetDataPy. The forecasting repository currently warns and keeps the first duplicate timestamp after MetDataPy ingestion because model training requires a unique chronological index.
- Expected input:
  One or more Weathercloud CSV exports that may contain overlapping time periods.
- Expected output:
  Canonical dataframe with deterministic duplicate handling plus metadata or warnings describing the number of duplicate rows and the selected policy.
- Suggested API:
  `read_weathercloud_directory(path, mapping_config, timezone="Europe/Athens", delimiter=None, duplicate_policy="keep_first", return_report=False)`
- Priority:
  Medium
- Blocking status:
  Does not block smoke experiments, but should be resolved before final multi-file experiments with overlapping exports.
- Forecasting pipeline usage:
  Required by `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud`.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

### 2026-05-03 - Fraction-based chronological split utility

- Required feature:
  Provide a reusable chronological train/validation/test split helper based on fractions or proportions.
- Reason:
  Time-safe splitting is reusable ML preparation logic. MetDataPy currently provides boundary-based `time_split`; this repository still computes fraction boundaries locally and then delegates boundary splitting to MetDataPy.
- Expected input:
  Sorted datetime-indexed dataframe and split fractions such as `0.70, 0.15, 0.15`.
- Expected output:
  Chronologically ordered non-overlapping train, validation, and test dataframes plus split metadata.
- Suggested API:
  `metdatapy.time_split_by_fraction(df, train=0.70, validation=0.15, test=0.15, min_rows_per_split=1)`
- Priority:
  Low
- Blocking status:
  Does not block experiments because the current repository orchestration is small and explicit.
- Forecasting pipeline usage:
  Used before scaler fitting and model training.
- Dissertation update required:
  No.

### 2026-05-03 - DST-safe local timestamp normalization

- Required feature:
  Support deterministic localization of naive local station timestamps across daylight-saving transitions.
- Reason:
  Full-year Weathercloud exports for `Europe/Athens` can include timestamps around DST transition days. Current MetDataPy 1.2.0 timestamp normalization raises on nonexistent local timestamps such as `2024-03-31 03:00` in `Europe/Athens`. DST-safe timezone normalization is reusable meteorological ingestion logic and belongs in MetDataPy.
- Expected input:
  Weathercloud CSV exports with a naive local timestamp column and `timezone="Europe/Athens"`.
- Expected output:
  UTC `ts_utc` index with a documented policy for nonexistent and ambiguous local times.
- Suggested API:
  `read_weathercloud_directory(..., nonexistent="shift_forward", ambiguous="infer")` and equivalent options in `ensure_datetime_utc`.
- Priority:
  High
- Blocking status:
  May block full-year Weathercloud ingestion if rows exist during DST transition hours.
- Forecasting pipeline usage:
  Required by `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud` for yearly CSV exports.
- Dissertation update required:
  No if implemented in MetDataPy before final experiments; yes if final experiments exclude or manually adjust DST-transition rows.

## Resolved In MetDataPy 1.2.0

### 2026-05-03 - Weathercloud multi-file ingestion

- Required feature:
  Support loading and concatenating multiple Weathercloud CSV exports from a directory.
- Resolution:
  MetDataPy 1.2.0 provides `metdatapy.read_weathercloud_directory` and `metdatapy.weathercloud.read_weathercloud_directory`.
- Forecasting pipeline usage:
  `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud` now calls MetDataPy directory ingestion directly.
- Remaining gap:
  Duplicate timestamp policy should be moved fully into MetDataPy, tracked above.
- Dissertation update required:
  No.

### 2026-05-03 - Weathercloud delimiter support

- Required feature:
  Robust Weathercloud CSV reading with support for semicolon-delimited exports.
- Resolution:
  MetDataPy 1.2.0 `read_csv`, `read_weathercloud_csv`, and `read_weathercloud_directory` accept a `delimiter` argument and auto-detect common delimiters.
- Forecasting pipeline usage:
  Weathercloud ingestion now delegates delimiter handling to MetDataPy.
- Remaining gap:
  None identified for delimiter handling.
- Dissertation update required:
  No.

### 2026-05-03 - Complete canonical Weathercloud schema support

- Required feature:
  Include `rain_rate_mmh` in canonical schema mapping, unit normalization, QC bounds, and artifact metadata.
- Resolution:
  MetDataPy 1.2.0 includes `rain_rate_mmh` in canonical variables, plausible bounds, unit converters, mapper support, resampling, and NetCDF metadata.
- Forecasting pipeline usage:
  `configs/weathercloud_mapping.yaml` now includes an optional `rain_rate_mmh` mapping.
- Remaining gap:
  None identified.
- Dissertation update required:
  No.

### 2026-05-03 - Wind direction cyclic encoding

- Required feature:
  Convert `wdir_deg` into `wdir_sin` and `wdir_cos` before modeling and rolling statistics.
- Resolution:
  MetDataPy 1.2.0 provides `WeatherSet.encode_wind_direction(drop_original=False)`.
- Forecasting pipeline usage:
  `preprocess_with_metdatapy` now calls `encode_wind_direction` before rolling feature generation.
- Remaining gap:
  None identified.
- Dissertation update required:
  No.

### 2026-05-03 - Rolling meteorological feature generation

- Required feature:
  Generate past-only rolling mean, standard deviation, minimum, and maximum for configured variables and windows.
- Resolution:
  MetDataPy 1.2.0 provides `WeatherSet.rolling_features(columns, windows, stats=("mean", "std", "min", "max"), closed="left")`.
- Forecasting pipeline usage:
  `preprocess_with_metdatapy` now calls MetDataPy rolling feature generation for the configured windows.
- Remaining gap:
  None identified.
- Dissertation update required:
  No.

## Resolved In MetDataPy 1.1.0

### 2026-05-03 - Encoding-detecting generic CSV ingestion

- Required feature:
  Read common real-world CSV encodings used by Weathercloud exports.
- Resolution:
  MetDataPy 1.1.0 `metdatapy.io.read_csv(path, ts_col=None, nrows=None)` detects common encodings before calling pandas.
- Forecasting pipeline usage:
  Weathercloud ingestion still relies on MetDataPy CSV reading through the 1.2.0 Weathercloud helpers.
- Remaining gap:
  None identified after MetDataPy 1.2.0 added delimiter support.
- Dissertation update required:
  No.

### 2026-05-03 - Local timezone-aware source mapping

- Required feature:
  Source ingestion should parse a local timestamp column such as `Date (Europe/Athens)`, localize it to `Europe/Athens`, and convert it to UTC.
- Resolution:
  MetDataPy 1.1.0 `WeatherSet.from_mapping` reads `ts.timezone` from the mapping and delegates timestamp normalization to MetDataPy.
- Forecasting pipeline usage:
  `configs/weathercloud_mapping.yaml` declares `ts.timezone: Europe/Athens`.
- Remaining gap:
  None for timestamp normalization.
- Dissertation update required:
  No.
