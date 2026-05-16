# Preprocessing And Feature Engineering

Preprocessing is orchestrated by `weather_forecasting_pipeline.metdatapy_adapter.preprocess_with_metdatapy`.

## Input

```text
data/interim/canonical.parquet
```

## Output

```text
data/interim/prepared.parquet
```

## Processing Steps

### Chronological Sorting

The dataframe is sorted by `ts_utc` before preparation.

### Missing Timestamp Insertion

MetDataPy inserts missing timestamps according to the configured frequency:

```yaml
expected_frequency: 10min
```

Inserted rows receive the `gap` flag.

### Quality Control

MetDataPy applies non-destructive QC flags:

- range checks
- spike checks
- flatline checks
- physical consistency checks
- aggregate `qc_any`

Suspicious values are flagged. Rows are not dropped at this stage.

Spike and flatline checks use MetDataPy's causal QC mode, so each QC flag at
forecast origin `t` depends only on observations available at or before `t`.
The resulting `qc_*` columns may be used as model features without future-data
leakage. The deterministic `gap` marker from missing timestamp insertion is
still excluded from model features.

### Derived Meteorological Features

Configured derived metrics are added through MetDataPy:

```yaml
derived_metrics: [dew_point, vpd, heat_index, wind_chill]
```

Possible output columns include:

- `dew_point_c`
- `vpd_kpa`
- `heat_index_c`
- `wind_chill_c`

### Calendar Features

MetDataPy calendar features include:

- `hour`
- `weekday`
- `month`
- `doy`
- `hour_sin`
- `hour_cos`
- `doy_sin`
- `doy_cos`

### Wind Direction Encoding

Wind direction is cyclic, so `wdir_deg` is encoded as:

- `wdir_sin`
- `wdir_cos`

The original `wdir_deg` is preserved for reference.

### Rolling Features

MetDataPy creates past-only rolling features using:

```python
WeatherSet.rolling_features(..., closed="left")
```

The default rolling windows are:

```yaml
rolling_windows: [6, 18, 36, 144]
```

At 10-minute resolution these correspond to:

- 1 hour
- 3 hours
- 6 hours
- 24 hours

Rolling statistics:

- mean
- standard deviation
- minimum
- maximum

Rolling features are computed for available major meteorological variables and cyclic wind-direction components.

## Leakage Control

Rolling features use prior observations only. The current observation and future observations are excluded from rolling windows.
