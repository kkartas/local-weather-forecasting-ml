"""Tests for ISO-timestamped progress logging across the pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from weather_forecasting_pipeline.cli import (
    _HANDLER_FLAG,
    LOG_DATEFMT,
    LOG_FORMAT,
    _configure_logging,
    main as cli_main,
)


ISO_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s")


def _write_synthetic_weathercloud_csv(path: Path, n: int = 220) -> None:
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="10min")
    x = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {
            "Date (Europe/Athens)": idx.strftime("%Y-%m-%d %H:%M"),
            "Temperature (°C)": 12.0 + np.sin(x / 12.0),
            "Humidity (%)": 70.0 + np.cos(x / 18.0),
            "Pressure (hPa)": 1015.0 + np.sin(x / 40.0),
            "Wind Speed (km/h)": 3.6 + np.abs(np.sin(x / 7.0)),
            "Wind Gust (km/h)": 7.2 + np.abs(np.sin(x / 5.0)),
            "Wind Direction (°)": (x * 15.0) % 360.0,
            "Rain (mm)": np.zeros(n),
            "Rain Rate (mm/h)": np.zeros(n),
            "Solar Radiation (W/m²)": np.zeros(n),
            "UV Index": np.zeros(n),
        }
    )
    frame.to_csv(path, sep=";", index=False)


@pytest.fixture
def smoke_config(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_synthetic_weathercloud_csv(raw_dir / "weathercloud.csv")

    mapping_path = tmp_path / "mapping.yaml"
    mapping_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "ts": {"col": "Date (Europe/Athens)", "timezone": "Europe/Athens"},
                "fields": {
                    "temp_c": {"col": "Temperature (°C)", "unit": "C"},
                    "rh_pct": {"col": "Humidity (%)", "unit": "%"},
                    "pres_hpa": {"col": "Pressure (hPa)", "unit": "hpa"},
                    "wspd_ms": {"col": "Wind Speed (km/h)", "unit": "km/h"},
                    "gust_ms": {"col": "Wind Gust (km/h)", "unit": "km/h"},
                    "wdir_deg": {"col": "Wind Direction (°)", "unit": "deg"},
                    "rain_mm": {"col": "Rain (mm)", "unit": "mm"},
                    "rain_rate_mmh": {"col": "Rain Rate (mm/h)", "unit": "mm/h"},
                    "solar_wm2": {"col": "Solar Radiation (W/m²)", "unit": "W/m2"},
                    "uv_index": {"col": "UV Index", "unit": "index"},
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    config = {
        "project": {"name": "logging_smoke", "random_seed": 42},
        "paths": {
            "raw_data_dir": str(raw_dir),
            "interim_dir": str(tmp_path / "interim"),
            "processed_dir": str(tmp_path / "processed"),
            "artifacts_dir": str(tmp_path / "artifacts"),
            "mapping_config": str(mapping_path),
        },
        "data": {
            "source": "weathercloud",
            "timezone": "Europe/Athens",
            "expected_frequency": "10min",
            "resample_rule": None,
            "target": "temp_c",
            "horizons": {"h01": 6},
            "optional_horizons": {"m10": 1},
            "lags": [1, 3, 6],
            "rolling_windows": [6],
            "sequence_length": 8,
            "derived_metrics": ["dew_point", "vpd"],
        },
        "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
        "scaling": {"method": "standard"},
        "models": {
            "baselines": ["persistence", "moving_average"],
            "ml": ["linear_regression"],
            "dl": [],
        },
        "training": {
            "max_epochs": 1,
            "batch_size": 16,
            "learning_rate": 0.01,
            "patience": 1,
            "min_dl_train_rows": 80,
        },
        "evaluation": {"mape_epsilon": 1.0e-6, "plot_max_points": 100},
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_configure_logging_installs_iso_timestamped_handler():
    """CLI logging setup must install an ISO-timestamped handler exactly once."""
    _configure_logging("INFO")
    _configure_logging("INFO")  # repeated calls must not duplicate our handler.
    root = logging.getLogger()

    managed = [h for h in root.handlers if getattr(h, _HANDLER_FLAG, False)]
    assert len(managed) == 1, "expected exactly one managed CLI handler"

    formatter = managed[0].formatter
    assert formatter is not None
    assert formatter._fmt == LOG_FORMAT
    assert formatter.datefmt == LOG_DATEFMT
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "hello", None, None)
    formatted = formatter.format(record)
    assert ISO_TIMESTAMP_RE.match(formatted), formatted


def test_cli_run_all_smoke_emits_progress_logs(smoke_config, caplog):
    """The CLI must emit start/finish logs for stages, horizons, and per model."""
    with caplog.at_level(logging.INFO):
        cli_main(["run-all", "--config", str(smoke_config), "--log-level", "INFO"])

    messages = [record.getMessage() for record in caplog.records]

    assert any(m.startswith("Run start: command=run-all") for m in messages)
    assert any(m.startswith("Run finish: command=run-all") for m in messages)

    assert any(m.startswith("Stage start: ingest") for m in messages)
    assert any(m.startswith("Stage finish: ingest") for m in messages)
    assert any(m.startswith("Stage start: preprocess") for m in messages)
    assert any(m.startswith("Stage finish: preprocess") for m in messages)
    assert any(m.startswith("Train context: project=logging_smoke") for m in messages)
    assert any(m.startswith("Stage start: train") for m in messages)
    assert any(m.startswith("Stage finish: train") for m in messages)
    assert any(m.startswith("Stage start: evaluate") for m in messages)
    assert any(m.startswith("Stage finish: evaluate") for m in messages)

    for horizon in ("m10", "h01"):
        assert any(
            m.startswith(f"Stage start: horizon {horizon}") for m in messages
        ), f"missing horizon-start log for {horizon}"
        assert any(
            m.startswith(f"Stage finish: horizon {horizon}") for m in messages
        ), f"missing horizon-finish log for {horizon}"
        assert any(
            m.startswith(f"Stage start: build supervised {horizon}") for m in messages
        ), f"missing build-supervised log for {horizon}"
        assert any(
            m.startswith(f"Stage start: split {horizon}") for m in messages
        ), f"missing split log for {horizon}"
        assert any(
            m.startswith(f"Stage start: fit feature scaler {horizon}") for m in messages
        ), f"missing feature-scaler log for {horizon}"
        assert any(
            m.startswith(f"Stage start: fit target scaler {horizon}") for m in messages
        ), f"missing target-scaler log for {horizon}"
        for family, model in (("baseline", "persistence"), ("baseline", "moving_average"), ("ml", "linear_regression")):
            start_prefix = f"Stage start: train model family={family} model={model} horizon={horizon}"
            finish_marker = f"family={family} model={model} horizon={horizon}"
            assert any(m.startswith(start_prefix) for m in messages), f"missing start log: {start_prefix}"
            assert any(
                m.startswith("Stage finish: train model") and finish_marker in m for m in messages
            ), f"missing finish log with {finish_marker}"


def test_train_logs_elapsed_seconds_and_mae(smoke_config, caplog):
    """Each per-model finish line must report `elapsed=` and the model's MAE."""
    with caplog.at_level(logging.INFO):
        cli_main(["run-all", "--config", str(smoke_config), "--log-level", "INFO"])

    finish_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Stage finish: train model")
    ]
    assert finish_lines, "no per-model finish logs were emitted"

    elapsed_re = re.compile(r"\belapsed=\d+\.\d{2}s\b")
    mae_re = re.compile(r"\bmae=\d")
    for line in finish_lines:
        assert elapsed_re.search(line), f"missing elapsed= field: {line}"
        assert mae_re.search(line), f"missing mae= field: {line}"
