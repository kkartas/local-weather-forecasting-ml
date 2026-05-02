# MetDataPy Feature Requirements

This file tracks MetDataPy features required by the dissertation forecasting project that are missing or incomplete in the currently installed MetDataPy library.

## 2026-05-02 - Weathercloud multi-file ingestion

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

## 2026-05-02 - Weathercloud delimiter and encoding support

- Required feature:
  Robust Weathercloud CSV reading with support for semicolon delimiters and encodings including UTF-8, UTF-8 with BOM, UTF-16, UTF-16 LE, and related real export formats.
- Reason:
  Source-specific ingestion robustness belongs in MetDataPy as part of the official data preparation layer.
- Expected input:
  Weathercloud CSV path with configurable or auto-detected delimiter and encoding.
- Expected output:
  Raw dataframe preserving rows and source columns for canonical mapping.
- Suggested API:
  `metdatapy.read_weathercloud_csv(path, mapping_config=None, timezone="Europe/Athens", encoding=None, delimiter=None)`
- Priority:
  High
- Blocking status:
  Blocks reliable ingestion of real Weathercloud exports.
- Forecasting pipeline usage:
  Required by `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud`.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

## 2026-05-02 - Local timezone-aware Weathercloud timestamp normalization

- Required feature:
  Source ingestion should parse a local timestamp column such as `Date (Europe/Athens)`, localize it to `Europe/Athens`, and convert it to UTC.
- Reason:
  Timestamp normalization is reusable meteorological time-series preparation and is already conceptually part of MetDataPy.
- Expected input:
  Raw dataframe or CSV with local Weathercloud timestamp column and timezone configuration.
- Expected output:
  `ts_utc` datetime index in UTC without silently treating local timestamps as UTC.
- Suggested API:
  `WeatherSet.from_csv(path, mapping, timezone="Europe/Athens")` or `read_weathercloud_csv(..., timezone="Europe/Athens")`
- Priority:
  High
- Blocking status:
  Blocks correct local-to-UTC conversion for Weathercloud exports with naive local timestamps.
- Forecasting pipeline usage:
  Required during ingestion and canonicalization.
- Dissertation update required:
  No, if implemented in MetDataPy as planned.

## 2026-05-02 - Complete canonical Weathercloud schema support

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

## 2026-05-02 - Wind direction cyclic encoding

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

## 2026-05-02 - Rolling meteorological feature generation

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
