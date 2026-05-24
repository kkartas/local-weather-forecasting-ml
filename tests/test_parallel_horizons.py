"""Smoke test for process-level parallel horizon training.

The dissertation pipeline can train each forecast horizon in a separate
worker process. This test verifies that running with ``horizon_workers: 2``
on a tiny synthetic fixture produces the same artifacts and metric rows as
the sequential default for both horizons.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_forecasting_pipeline.cli import main as cli_main


def _write_synthetic_weathercloud_csv(path: Path, n: int = 320) -> None:
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


def _write_mapping(path: Path) -> None:
    path.write_text(
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


def _build_config(
    tmp: Path, raw_dir: Path, mapping: Path, *, horizon_workers: int
) -> dict:
    return {
        "project": {"name": "parallel_smoke", "random_seed": 42},
        "paths": {
            "raw_data_dir": str(raw_dir),
            "interim_dir": str(tmp / "interim"),
            "processed_dir": str(tmp / "processed"),
            "artifacts_dir": str(tmp / "artifacts"),
            "mapping_config": str(mapping),
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
            "dl_exclude_lag_features": True,
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
            "horizon_workers": horizon_workers,
        },
        "evaluation": {"mape_epsilon": 1.0e-6, "plot_max_points": 100},
    }


def _write_environment(tmp: Path, *, horizon_workers: int) -> Path:
    raw_dir = tmp / "raw"
    raw_dir.mkdir()
    _write_synthetic_weathercloud_csv(raw_dir / "weathercloud.csv")
    mapping = tmp / "mapping.yaml"
    _write_mapping(mapping)
    config_path = tmp / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(_build_config(tmp, raw_dir, mapping, horizon_workers=horizon_workers)),
        encoding="utf-8",
    )
    return config_path


def test_parallel_horizon_training_produces_both_horizon_metrics(tmp_path: Path):
    """Two horizons trained in parallel must each appear in the merged metrics.

    Validates the end-to-end parallel flow: process-pool spawning, per-worker
    config rehydration, per-horizon artifact paths, and the single merged
    metrics.csv that the main process writes after all workers finish.
    """
    config_path = _write_environment(tmp_path, horizon_workers=2)
    cli_main(["run-all", "--config", str(config_path), "--log-level", "INFO"])

    artifacts = tmp_path / "artifacts"
    metrics_csv = artifacts / "metrics" / "metrics.csv"
    assert metrics_csv.exists(), "merged metrics.csv must be written after parallel runs"
    metrics = pd.read_csv(metrics_csv)

    assert set(metrics["horizon_label"]) == {"m10", "h01"}
    for horizon in ("m10", "h01"):
        rows = metrics[metrics["horizon_label"] == horizon]
        # Each horizon must run all configured models (2 baselines + 1 ML).
        assert set(rows["model"]) == {"persistence", "moving_average", "linear_regression"}
        assert (rows["mae"] >= 0).all()

        assert (tmp_path / "processed" / f"split_metadata_{horizon}.json").exists()
        assert (tmp_path / "processed" / f"supervised_{horizon}.parquet").exists()
        assert (artifacts / "scalers" / f"scaler_{horizon}.joblib").exists()
        for model in ("persistence", "moving_average", "linear_regression"):
            assert (artifacts / "models" / f"{model}_{horizon}.joblib").exists()


def test_parallel_and_sequential_metrics_match_on_small_data(tmp_path: Path):
    """Parallel and sequential runs must produce identical metric rows.

    Process-level parallelism reorders worker completions but must not change
    the scientific outputs. Each horizon row is keyed by ``(horizon_label,
    model)``; the values are compared after sorting on those keys so order
    differences alone cannot cause a false positive.
    """
    seq_dir = tmp_path / "sequential"
    par_dir = tmp_path / "parallel"
    seq_dir.mkdir()
    par_dir.mkdir()

    seq_config = _write_environment(seq_dir, horizon_workers=1)
    par_config = _write_environment(par_dir, horizon_workers=2)

    cli_main(["run-all", "--config", str(seq_config), "--log-level", "WARNING"])
    cli_main(["run-all", "--config", str(par_config), "--log-level", "WARNING"])

    sort_keys = ["horizon_label", "model"]
    seq_metrics = pd.read_csv(seq_dir / "artifacts" / "metrics" / "metrics.csv").sort_values(sort_keys).reset_index(drop=True)
    par_metrics = pd.read_csv(par_dir / "artifacts" / "metrics" / "metrics.csv").sort_values(sort_keys).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        seq_metrics[["horizon_label", "model", "model_family"]],
        par_metrics[["horizon_label", "model", "model_family"]],
    )
    for col in ("mae", "rmse"):
        assert np.allclose(
            seq_metrics[col].to_numpy(dtype=float),
            par_metrics[col].to_numpy(dtype=float),
            atol=1e-5,
        ), f"{col} mismatch between sequential and parallel runs"


def test_parallel_main_logs_horizon_completion_progress(tmp_path: Path, caplog):
    config_path = _write_environment(tmp_path, horizon_workers=2)
    with caplog.at_level(logging.INFO):
        cli_main(["run-all", "--config", str(config_path), "--log-level", "INFO"])
    messages = [r.getMessage() for r in caplog.records]

    # In parallel mode, workers emit per-model train start lines; the parent
    # process should only report per-horizon completion rollups.
    assert not any("Stage start: train model" in m for m in messages)
    assert any(m.startswith("Horizon worker complete: horizon=") for m in messages)
    assert any(m == "Horizons complete: 1/2" for m in messages)
    assert any(m == "Horizons complete: 2/2" for m in messages)
