# MetDataPy Feature Requirements

This file tracks MetDataPy features required by the dissertation forecasting project that are missing, incomplete, or recently resolved in the currently installed MetDataPy library.

Current installed version inspected: `metdatapy==1.1.0`.

## Active Missing Or Incomplete Features

### 2026-05-03 - Weathercloud multi-file ingestion

- Required feature:
  Support loading and concatenating multiple Weathercloud CSV exports from a directory.
- Reason:
  Multi-file Weathercloud ingestion is reusable meteorological data preparation logic and therefore belongs in MetDataPy, not in the forecasting experiment repository.
- Expected input:
  Directory containing one or more Weathercloud CSV files.
- Expected output:
  Canonical dataframe with `ts_utc` index and normalized meteorological columns.
- Suggested API:
  `metdatapy.read_weathercloud_directory(path, mapping_config, timezone="Europe/Athens")`
- Priority:
  High
- Blocking status:
  Blocks full Weathercloud raw-data ingestion when more than one export is present.
- Forecasting pipeline usage:
  Required by the `ingest` and `run-all` commands before preprocessing and feature generation.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

### 2026-05-03 - Weathercloud delimiter support

- Required feature:
  Robust Weathercloud CSV reading with support for semicolon-delimited exports, preferably by explicit delimiter configuration or delimiter detection.
- Reason:
  Source-specific ingestion robustness belongs in MetDataPy as part of the official data preparation layer.
- Expected input:
  Weathercloud CSV path with configurable or auto-detected delimiter.
- Expected output:
  Raw dataframe preserving rows and source columns for canonical mapping.
- Suggested API:
  `metdatapy.read_weathercloud_csv(path, mapping_config=None, timezone="Europe/Athens", delimiter=None)`
- Priority:
  High
- Blocking status:
  Blocks reliable ingestion of real semicolon-separated Weathercloud exports.
- Forecasting pipeline usage:
  Required by `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud`.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

### 2026-05-03 - Complete canonical Weathercloud schema support

- Required feature:
  Include `rain_rate_mmh` in canonical schema mapping, unit normalization, QC bounds, and artifact metadata.
- Reason:
  Canonical meteorological schema management belongs in MetDataPy.
- Expected input:
  Weathercloud rain-rate column and mapping configuration.
- Expected output:
  Canonical `rain_rate_mmh` column in the prepared dataframe.
- Suggested API:
  Extend MetDataPy canonical field definitions and unit converters with `rain_rate_mmh`.
- Priority:
  Medium
- Blocking status:
  Blocks full canonical schema coverage required by the dissertation methodology.
- Forecasting pipeline usage:
  Used as a feature and QC variable when available.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

### 2026-05-03 - Wind direction cyclic encoding

- Required feature:
  Convert `wdir_deg` into `wdir_sin` and `wdir_cos` before modeling and rolling statistics.
- Reason:
  Meteorological cyclic direction encoding is reusable feature preparation logic.
- Expected input:
  Canonical dataframe with optional `wdir_deg`.
- Expected output:
  Dataframe with `wdir_sin` and `wdir_cos`, preserving `wdir_deg` for reference.
- Suggested API:
  `WeatherSet.encode_wind_direction(drop_original=False)`
- Priority:
  High
- Blocking status:
  Blocks full feature engineering methodology for wind direction.
- Forecasting pipeline usage:
  Required before feature dataset creation.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

### 2026-05-03 - Rolling meteorological feature generation

- Required feature:
  Generate past-only rolling mean, standard deviation, minimum, and maximum for configured variables and windows.
- Reason:
  Rolling time-series feature generation is reusable ML preparation logic and must avoid future leakage consistently across projects.
- Expected input:
  Canonical dataframe, list of columns, list of integer windows, and aggregations.
- Expected output:
  Dataframe with rolling features computed without future observations.
- Suggested API:
  `WeatherSet.rolling_features(columns, windows, stats=("mean", "std", "min", "max"), closed="left")`
- Priority:
  High
- Blocking status:
  Blocks full tabular feature set from the dissertation methodology.
- Forecasting pipeline usage:
  Required before supervised dataset creation.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

## Resolved In MetDataPy 1.1.0

### 2026-05-03 - Encoding-detecting generic CSV ingestion

- Required feature:
  Read common real-world CSV encodings used by Weathercloud exports.
- Resolution:
  MetDataPy 1.1.0 `metdatapy.io.read_csv(path, ts_col=None, nrows=None)` detects common encodings before calling pandas.
- Forecasting pipeline usage:
  `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud` continues to call MetDataPy `read_csv`.
- Remaining gap:
  Delimiter support is still incomplete and is tracked separately above.
- Dissertation update required:
  No.

### 2026-05-03 - Local timezone-aware source mapping

- Required feature:
  Source ingestion should parse a local timestamp column such as `Date (Europe/Athens)`, localize it to `Europe/Athens`, and convert it to UTC.
- Resolution:
  MetDataPy 1.1.0 `WeatherSet.from_mapping` reads `ts.timezone` from the mapping and delegates timestamp normalization to MetDataPy.
- Forecasting pipeline usage:
  `configs/weathercloud_mapping.yaml` now declares `ts.timezone: Europe/Athens`, and the adapter no longer manually overrides the canonical index.
- Remaining gap:
  None for single-file timestamp normalization.
- Dissertation update required:
  No.
