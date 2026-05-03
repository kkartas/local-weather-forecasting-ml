# Data Ingestion

Ingestion is orchestrated by `weather_forecasting_pipeline.metdatapy_adapter.ingest_raw_weathercloud`.

## Input

Place one or more Weathercloud CSV exports in:

```text
data/raw/
```

The default mapping expects:

```text
Date (Europe/Athens)
```

as the local timestamp column.

## MetDataPy Integration

The project uses MetDataPy 1.2.0 for:

- Weathercloud directory ingestion
- CSV encoding detection
- delimiter handling
- source-to-canonical schema mapping
- timezone conversion
- unit normalization

The adapter calls:

```python
metdatapy.weathercloud.read_weathercloud_directory(...)
```

## Canonical Output

The ingest stage writes:

```text
data/interim/canonical.parquet
```

The canonical dataframe is indexed by `ts_utc`, a UTC timestamp index.

Typical canonical variables include:

- `temp_c`
- `rh_pct`
- `pres_hpa`
- `wspd_ms`
- `gust_ms`
- `wdir_deg`
- `rain_mm`
- `rain_rate_mmh`
- `solar_wm2`
- `uv_index`

Only variables present in the raw files and mapping are included.

## Duplicate Timestamps

After MetDataPy ingestion, the adapter sorts by timestamp and checks for duplicate timestamps. If duplicates exist, the current project behavior is:

- log a warning with the duplicate count
- keep the first row for each duplicated timestamp

This keeps the modeling index unique. A future MetDataPy duplicate-policy API is tracked in `METDATAPY.md`.

## No Raw Data Mutation

Raw CSV exports under `data/raw/` are read but not modified.
