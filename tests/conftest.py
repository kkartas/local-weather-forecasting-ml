from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def synthetic_station_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01 00:00", periods=220, freq="10min", tz="UTC")
    x = np.arange(len(idx), dtype=float)
    return pd.DataFrame(
        {
            "temp_c": 12.0 + np.sin(x / 12.0),
            "rh_pct": 70.0 + np.cos(x / 18.0),
            "pres_hpa": 1015.0 + np.sin(x / 40.0),
            "wspd_ms": 2.0 + np.abs(np.sin(x / 7.0)),
            "gust_ms": 3.0 + np.abs(np.sin(x / 5.0)),
            "wdir_deg": (x * 15.0) % 360.0,
            "rain_mm": np.zeros(len(idx)),
            "solar_wm2": np.maximum(0.0, 500.0 * np.sin((idx.hour.to_numpy() - 6) / 12.0 * np.pi)),
            "uv_index": np.maximum(0.0, 5.0 * np.sin((idx.hour.to_numpy() - 6) / 12.0 * np.pi)),
        },
        index=pd.DatetimeIndex(idx, name="ts_utc"),
    )


@pytest.fixture
def weathercloud_csv(tmp_path: Path) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    mapping_path = tmp_path / "mapping.yaml"
    mapping = {
        "version": 1,
        "ts": {"col": "Date (Europe/Athens)"},
        "fields": {
            "temp_c": {"col": "Temperature (°C)", "unit": "C"},
            "rh_pct": {"col": "Humidity (%)", "unit": "%"},
            "pres_hpa": {"col": "Pressure (hPa)", "unit": "hpa"},
            "wspd_ms": {"col": "Wind Speed (km/h)", "unit": "km/h"},
            "gust_ms": {"col": "Wind Gust (km/h)", "unit": "km/h"},
            "wdir_deg": {"col": "Wind Direction (°)", "unit": "deg"},
            "rain_mm": {"col": "Rain (mm)", "unit": "mm"},
            "solar_wm2": {"col": "Solar Radiation (W/m²)", "unit": "W/m2"},
            "uv_index": {"col": "UV Index", "unit": "index"},
        },
    }
    mapping_path.write_text(yaml.safe_dump(mapping, allow_unicode=True), encoding="utf-8")
    frame = pd.DataFrame(
        {
            "Date (Europe/Athens)": ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20"],
            "Temperature (°C)": [10.0, 10.5, 11.0],
            "Humidity (%)": [80.0, 81.0, 82.0],
            "Pressure (hPa)": [1010.0, 1010.1, 1010.2],
            "Wind Speed (km/h)": [3.6, 7.2, 10.8],
            "Wind Gust (km/h)": [7.2, 10.8, 14.4],
            "Wind Direction (°)": [90.0, 100.0, 110.0],
            "Rain (mm)": [0.0, 0.0, 0.1],
            "Solar Radiation (W/m²)": [0.0, 0.0, 0.0],
            "UV Index": [0.0, 0.0, 0.0],
        }
    )
    frame.to_csv(raw_dir / "weathercloud.csv", index=False)
    return raw_dir, mapping_path
