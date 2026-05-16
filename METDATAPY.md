# MetDataPy Feature Requirements

This file tracks MetDataPy features required by the dissertation forecasting
project that are missing, incomplete, or recently resolved in the currently
installed MetDataPy library.

Current installed version inspected: `metdatapy==1.3.0`.

## Active Missing Or Incomplete Features

### 2026-05-16 - Tolerant parsing for surplus trailing Weathercloud fields

- Required feature:
  Read Weathercloud CSV rows that contain one more trailing empty field than
  the header declares.
- Reason:
  Some monthly Weathercloud exports include rows such as
  `timestamp;;;;;;;;;;;;;;;;;;;`, which pandas rejects with
  `ParserError: Expected 19 fields ... saw 20` when called through
  MetDataPy 1.3.0's standard CSV reader.
- Workaround in this repository:
  `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud`
  first calls MetDataPy's Weathercloud directory reader. If pandas raises this
  parser error, the adapter retries with a narrow header-width parser that
  ignores surplus trailing fields, then delegates timestamp normalization,
  canonical mapping, and unit normalization back to MetDataPy.
- Expected input:
  Weathercloud semicolon-delimited CSV exports with occasional surplus trailing
  empty fields.
- Expected output:
  A correctly decoded raw dataframe whose declared header columns are visible
  to MetDataPy mapping.
- Suggested API:
  Add a Weathercloud reader option such as
  `read_weathercloud_csv(..., tolerate_trailing_empty_fields=True)`, or make
  this tolerance the default for Weathercloud exports.
- Priority:
  Medium
- Blocking status:
  Does not block this repository's experiments because the local adapter
  fallback handles the observed export format.
- Forecasting pipeline usage:
  Used by `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud`
  only after normal MetDataPy parsing raises `pandas.errors.ParserError`.
- Dissertation update required:
  No.

## Resolved In MetDataPy 1.3.0

### 2026-05-16 - UTF-16LE Weathercloud exports without BOM

- Required feature:
  Detect and read UTF-16LE Weathercloud CSV exports that do not include a
  byte-order mark.
- Resolution:
  MetDataPy 1.3.0 detects UTF-16LE/UTF-16BE by NUL-byte distribution when no
  BOM is present.
- Forecasting pipeline usage:
  Normal Weathercloud ingestion now relies on MetDataPy for UTF-16LE no-BOM
  detection. The local fallback may still explicitly read UTF-16LE when handling
  surplus trailing fields.
- Remaining gap:
  Surplus trailing empty fields are tracked above.
- Dissertation update required:
  No.

### 2026-05-16 - DST-safe local timestamp normalization

- Required feature:
  Support deterministic localization of naive local station timestamps across
  daylight-saving transitions.
- Resolution:
  MetDataPy 1.3.0 exposes `nonexistent` and `ambiguous` options on
  `ensure_datetime_utc`, `WeatherSet.from_mapping`,
  `read_weathercloud_csv`, and `read_weathercloud_directory`.
- Forecasting pipeline usage:
  `ingest_raw_weathercloud` passes `nonexistent="shift_forward"` and
  `ambiguous=False` for deterministic `Europe/Athens` handling.
- Remaining gap:
  None identified for DST handling.
- Dissertation update required:
  Mention the deterministic DST policy if final methodology discusses
  timestamp normalization details.

### 2026-05-16 - Duplicate timestamp handling policy in Weathercloud ingestion

- Required feature:
  Provide an explicit duplicate timestamp handling policy for Weathercloud
  directory ingestion.
- Resolution:
  MetDataPy 1.3.0 provides `duplicate_policy` and `return_report` in
  `read_weathercloud_directory`.
- Forecasting pipeline usage:
  `ingest_raw_weathercloud` uses `duplicate_policy="keep_first"` and logs the
  duplicate count from the MetDataPy report.
- Remaining gap:
  The local trailing-field parser mirrors the same keep-first policy when the
  standard MetDataPy parser cannot read the raw file.
- Dissertation update required:
  Report duplicate counts as a data-cleaning detail if present in final runs.

### 2026-05-16 - Fraction-based chronological split utility

- Required feature:
  Provide a reusable chronological train/validation/test split helper based on
  fractions or proportions.
- Resolution:
  MetDataPy 1.3.0 provides `metdatapy.mlprep.time_split_by_fraction`.
- Forecasting pipeline usage:
  `split_by_fraction_with_metdatapy` delegates directly to MetDataPy's
  fraction-based splitter.
- Remaining gap:
  None identified.
- Dissertation update required:
  No.

### 2026-05-16 - Causal-window option for QC spike and flatline checks

- Required feature:
  Allow `qc_spike` and `qc_flatline` to compute rolling local statistics with
  past-only windows.
- Resolution:
  MetDataPy 1.3.0 adds `causal=True` to `qc_spike`, `qc_flatline`, and the
  corresponding `WeatherSet` methods.
- Forecasting pipeline usage:
  `preprocess_with_metdatapy` calls `qc_spike(causal=True)` and
  `qc_flatline(causal=True)`. Because these flags are now past-only, `qc_*`
  columns may be selected as model features.
- Remaining gap:
  None identified for causal QC features.
- Dissertation update required:
  Yes, note that QC flags are generated causally and may be used as inputs.

## Resolved In MetDataPy 1.2.0

### 2026-05-03 - Weathercloud multi-file ingestion

- Resolution:
  MetDataPy 1.2.0 provides `metdatapy.read_weathercloud_directory` and
  `metdatapy.weathercloud.read_weathercloud_directory`.

### 2026-05-03 - Weathercloud delimiter support

- Resolution:
  MetDataPy 1.2.0 accepts a `delimiter` argument and auto-detects common
  delimiters.

### 2026-05-03 - Complete canonical Weathercloud schema support

- Resolution:
  MetDataPy 1.2.0 includes `rain_rate_mmh` in canonical variables, plausible
  bounds, unit converters, mapper support, resampling, and metadata.

### 2026-05-03 - Wind direction cyclic encoding

- Resolution:
  MetDataPy 1.2.0 provides `WeatherSet.encode_wind_direction(drop_original=False)`.

### 2026-05-03 - Rolling meteorological feature generation

- Resolution:
  MetDataPy 1.2.0 provides past-only rolling feature generation via
  `WeatherSet.rolling_features(..., closed="left")`.

## Resolved In MetDataPy 1.1.0

### 2026-05-03 - Encoding-detecting generic CSV ingestion

- Resolution:
  MetDataPy 1.1.0 `metdatapy.io.read_csv` detects common encodings before
  calling pandas. MetDataPy 1.3.0 extends this with UTF-16LE/UTF-16BE no-BOM
  detection.

### 2026-05-03 - Local timezone-aware source mapping

- Resolution:
  MetDataPy 1.1.0 `WeatherSet.from_mapping` reads `ts.timezone` from mapping
  files and delegates timestamp normalization to MetDataPy.
