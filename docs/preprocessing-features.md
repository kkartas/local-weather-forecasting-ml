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

> **Leakage caveat:** MetDataPy 1.2.0 implements `qc_spike` and `qc_flatline`
> with `pandas.Series.rolling(..., center=True)`, so the resulting flag at row
> `t` depends on a few observations after `t`. To keep the strict
> "no future information in features" rule, the forecasting layer excludes
> every column whose name starts with `qc_` (and the deterministic `gap`
> indicator) from the model feature set in
> `weather_forecasting_pipeline.datasets.splits.select_feature_columns`. The
> exclusion is enforced by
> `tests/test_leakage.py::test_select_feature_columns_excludes_qc_and_gap_flags`.
> A causal-window option in MetDataPy is tracked in `METDATAPY.md` so this
> local exclusion can be removed once it lands upstream.

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
