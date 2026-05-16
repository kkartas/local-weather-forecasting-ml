from __future__ import annotations

import pandas as pd
import pytest
import yaml
from metdatapy import WeatherSet

from weather_forecasting_pipeline.metdatapy_adapter import ingest_raw_weathercloud, preprocess_with_metdatapy


def test_metdatapy_imports():
    import metdatapy

    assert hasattr(metdatapy, "WeatherSet")
    assert hasattr(metdatapy, "make_supervised")


def test_weathercloud_directory_semicolon_ingestion_through_metdatapy(weathercloud_csv):
    raw_dir, mapping_path = weathercloud_csv
    df = ingest_raw_weathercloud(raw_dir, mapping_path, timezone="Europe/Athens")

    assert df.index.name == "ts_utc"
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates
    assert {"temp_c", "rh_pct", "pres_hpa", "wspd_ms", "gust_ms", "wdir_deg", "rain_rate_mmh"}.issubset(df.columns)
    assert len(df) == 4
    assert df["wspd_ms"].iloc[0] == 1.0
    assert df.index[0] == pd.Timestamp("2023-12-31 22:00:00", tz="UTC")


@pytest.mark.parametrize(
    ("rows", "expected_index", "expected_values"),
    [
        (
            ["2024-03-31 02:50", "2024-03-31 03:00", "2024-03-31 03:10", "2024-03-31 04:00"],
            ["2024-03-31 00:50", "2024-03-31 01:00"],
            [10.0, 10.1],
        ),
        (
            ["2024-10-27 02:50", "2024-10-27 03:00", "2024-10-27 03:10", "2024-10-27 04:00"],
            ["2024-10-26 23:50", "2024-10-27 01:00", "2024-10-27 01:10", "2024-10-27 02:00"],
            [10.0, 10.1, 10.2, 10.3],
        ),
    ],
)
def test_weathercloud_ingestion_handles_athens_dst_transition_rows(
    tmp_path, rows, expected_index, expected_values
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mapping_path = tmp_path / "mapping.yaml"
    mapping = {
        "version": 1,
        "ts": {"col": "Date (Europe/Athens)"},
        "fields": {"temp_c": {"col": "Temperature (°C)", "unit": "C"}},
    }
    mapping_path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    pd.DataFrame(
        {
            "Date (Europe/Athens)": rows,
            "Temperature (°C)": [10.0, 10.1, 10.2, 10.3],
        }
    ).to_csv(raw_dir / "weathercloud_dst.csv", index=False, sep=";")

    df = ingest_raw_weathercloud(raw_dir, mapping_path, timezone="Europe/Athens")

    assert df.index.tolist() == [pd.Timestamp(value, tz="UTC") for value in expected_index]
    assert df["temp_c"].tolist() == expected_values


def test_weathercloud_ingestion_handles_utf16le_short_weathercloud_headers(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mapping_path = tmp_path / "mapping.yaml"
    mapping = {
        "version": 1,
        "ts": {"col": "Date (Europe/Athens)"},
        "fields": {
            "temp_c": {"col": "temp (°C)", "unit": "C"},
            "rh_pct": {"col": "hum (%)", "unit": "%"},
            "pres_hpa": {"col": "bar (hPa)", "unit": "hpa"},
            "wspd_ms": {"col": "wspdavg (km/h)", "unit": "km/h"},
            "gust_ms": {"col": "wspdhi (km/h)", "unit": "km/h"},
            "wdir_deg": {"col": "wdiravg (°)", "unit": "deg"},
            "rain_mm": {"col": "rain (mm)", "unit": "mm"},
            "rain_rate_mmh": {"col": "rainrate (mm/h)", "unit": "mm/h"},
            "solar_wm2": {"col": "solarrad (W/m²)", "unit": "W/m2"},
            "uv_index": {"col": "uvi", "unit": "index"},
        },
    }
    mapping_path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    csv_text = (
        "Date (Europe/Athens);temp (°C);hum (%);bar (hPa);wspdavg (km/h);"
        "wspdhi (km/h);wdiravg (°);rain (mm);rainrate (mm/h);solarrad (W/m²);uvi;\n"
        "01/03/2023 00:00:00;12.0;80;1010.0;3.6;7.2;90;0.0;0.0;0.0;0;\n"
        "01/03/2023 00:05:00;;;;;;;;;;;\n"
        "01/03/2023 00:10:00;12.5;81;1010.1;7.2;10.8;100;0.0;0.0;0.0;0;\n"
    )
    (raw_dir / "weathercloud_utf16le.csv").write_bytes(csv_text.encode("utf-16le"))

    df = ingest_raw_weathercloud(raw_dir, mapping_path, timezone="Europe/Athens")

    assert df.index[0] == pd.Timestamp("2023-02-28 22:00:00", tz="UTC")
    assert df["temp_c"].dropna().tolist() == [12.0, 12.5]
    assert df["wspd_ms"].dropna().tolist() == pytest.approx([1.0, 2.0])
    assert df["gust_ms"].dropna().tolist() == pytest.approx([2.0, 3.0])


def test_metdatapy_mapping_timezone_normalizes_local_time():
    raw = pd.DataFrame({"Date (Europe/Athens)": ["2024-01-01 00:00"], "Temperature (°C)": [10.0]})
    mapping = {
        "version": 1,
        "ts": {"col": "Date (Europe/Athens)", "timezone": "Europe/Athens"},
        "fields": {"temp_c": {"col": "Temperature (°C)", "unit": "C"}},
    }

    df = WeatherSet.from_mapping(raw, mapping).normalize_units(mapping).to_dataframe()

    assert df.index[0] == pd.Timestamp("2023-12-31 22:00:00", tz="UTC")
    assert df["temp_c"].iloc[0] == 10.0


def test_preprocess_adds_gap_qc_and_calendar_features(synthetic_station_frame):
    prepared = preprocess_with_metdatapy(
        synthetic_station_frame.drop(synthetic_station_frame.index[5]),
        expected_frequency="10min",
        derived_metrics=["dew_point", "vpd", "heat_index", "wind_chill"],
        rolling_windows=[6],
    )

    assert "gap" in prepared.columns
    assert bool(prepared["gap"].iloc[5])
    assert "qc_any" in prepared.columns
    assert {"hour_sin", "hour_cos", "doy_sin", "doy_cos"}.issubset(prepared.columns)
    assert {"wdir_sin", "wdir_cos", "temp_c_roll6_mean", "wdir_sin_roll6_mean"}.issubset(prepared.columns)
    assert {"dew_point_c", "vpd_kpa", "heat_index_c", "wind_chill_c"}.issubset(prepared.columns)
