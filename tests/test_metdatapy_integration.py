from __future__ import annotations

import pandas as pd

from weather_forecasting_pipeline.metdatapy_adapter import ingest_raw_weathercloud, preprocess_with_metdatapy


def test_metdatapy_imports():
    import metdatapy

    assert hasattr(metdatapy, "WeatherSet")
    assert hasattr(metdatapy, "make_supervised")


def test_weathercloud_single_csv_ingestion_through_metdatapy(weathercloud_csv):
    raw_dir, mapping_path = weathercloud_csv
    df = ingest_raw_weathercloud(raw_dir, mapping_path, timezone="Europe/Athens")

    assert df.index.name == "ts_utc"
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert {"temp_c", "rh_pct", "pres_hpa", "wspd_ms", "gust_ms", "wdir_deg"}.issubset(df.columns)
    assert df["wspd_ms"].iloc[0] == 1.0
    assert df.index[0] == pd.Timestamp("2023-12-31 22:00:00", tz="UTC")


def test_preprocess_adds_gap_qc_and_calendar_features(synthetic_station_frame):
    prepared = preprocess_with_metdatapy(
        synthetic_station_frame.drop(synthetic_station_frame.index[5]),
        expected_frequency="10min",
        derived_metrics=["dew_point", "vpd", "heat_index", "wind_chill"],
    )

    assert "gap" in prepared.columns
    assert bool(prepared["gap"].iloc[5])
    assert "qc_any" in prepared.columns
    assert {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}.issubset(prepared.columns)
    assert {"dew_point_c", "vpd_kpa", "heat_index_c", "wind_chill_c"}.issubset(prepared.columns)
