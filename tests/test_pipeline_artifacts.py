"""End-to-end smoke test verifying CLI behaviour and artifact integrity."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from metdatapy.mlprep import apply_scaler

from weather_forecasting_pipeline.cli import main as cli_main
from weather_forecasting_pipeline.config import ExperimentConfig, load_config
from weather_forecasting_pipeline.training.pipeline import _resolved_horizons


def _write_synthetic_weathercloud_csv(path: Path, n: int = 220) -> None:
    """Write a small Weathercloud-style CSV with semicolon delimiter."""
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


def _write_smoke_config(tmp: Path, raw_dir: Path, mapping: Path) -> Path:
    config = {
        "project": {"name": "cli_smoke", "random_seed": 42},
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
    config_path = tmp / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture
def smoke_environment(tmp_path: Path) -> tuple[Path, ExperimentConfig]:
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

    config_path = _write_smoke_config(tmp_path, raw_dir, mapping_path)
    return config_path, load_config(config_path)


def test_cli_run_all_smoke_writes_expected_artifacts(smoke_environment):
    """Invoking the CLI end-to-end produces the documented artifact tree."""
    config_path, config = smoke_environment

    cli_main(["run-all", "--config", str(config_path)])

    metrics_csv = config.paths.artifacts_dir / "metrics" / "metrics.csv"
    assert metrics_csv.exists()
    metrics = pd.read_csv(metrics_csv)

    # Both required and optional horizons must train.
    assert set(metrics["horizon_label"]) == {"m10", "h01"}
    # All baseline + ML models must report finite MAE.
    assert metrics["mae"].notna().all()
    assert (metrics["mae"] >= 0).all()

    # The summary report and per-horizon split metadata must exist.
    assert (config.paths.artifacts_dir / "reports" / "summary.md").exists()
    for horizon in ("m10", "h01"):
        assert (config.paths.processed_dir / f"split_metadata_{horizon}.json").exists()


def test_persisted_scaler_reloads_and_transforms(smoke_environment):
    """The saved scaler must load and reproduce the in-memory transform."""
    config_path, config = smoke_environment
    cli_main(["run-all", "--config", str(config_path)])

    scaler_path = config.paths.artifacts_dir / "scalers" / "scaler_h01.joblib"
    assert scaler_path.exists()

    scaler = joblib.load(scaler_path)
    # Must expose MetDataPy ScalerParams attributes (method, columns, parameters).
    assert hasattr(scaler, "parameters")
    assert hasattr(scaler, "method")

    sample_col = next(iter(scaler.parameters))
    sample_df = pd.DataFrame({sample_col: [0.0, 1.0, 2.0]})
    transformed = apply_scaler(sample_df, scaler)
    assert transformed.shape == sample_df.shape
    assert pd.api.types.is_numeric_dtype(transformed[sample_col])


def test_resolved_horizons_includes_optional(smoke_environment):
    """`optional_horizons` entries must end up in the trained horizon mapping."""
    _, config = smoke_environment
    resolved = _resolved_horizons(config)
    assert resolved == {"m10": 1, "h01": 6}


def test_clean_command_removes_generated_outputs_only(smoke_environment):
    """``clean`` must wipe outputs and restore tracked placeholders.

    The raw data directory contains the Weathercloud exports that drive every
    experiment, so it is intentionally outside the scope of cleanup. The clean
    command also stays scoped to the configured paths so a misconfigured
    artifacts directory cannot remove unrelated user files. The empty
    directory skeleton is recreated because Git tracks these paths through
    ``.gitkeep`` placeholders.
    """
    config_path, config = smoke_environment

    cli_main(["run-all", "--config", str(config_path)])
    raw_files = list(config.paths.raw_data_dir.iterdir())
    assert raw_files, "smoke fixture must populate the raw data directory"

    cli_main(["clean", "--config", str(config_path)])

    placeholder_dirs = [
        config.paths.interim_dir,
        config.paths.processed_dir,
        config.paths.artifacts_dir / "models",
        config.paths.artifacts_dir / "scalers",
        config.paths.artifacts_dir / "metrics",
        config.paths.artifacts_dir / "plots",
        config.paths.artifacts_dir / "reports",
    ]
    for path in placeholder_dirs:
        assert path.is_dir()
        assert sorted(child.name for child in path.iterdir()) == [".gitkeep"]
        assert (path / ".gitkeep").read_bytes() == b"\n"
    # Raw inputs must remain untouched.
    assert config.paths.raw_data_dir.exists()
    assert sorted(p.name for p in config.paths.raw_data_dir.iterdir()) == sorted(p.name for p in raw_files)


def test_fresh_flag_resets_outputs_before_run(smoke_environment):
    """``--fresh`` on ``run-all`` must clear stale outputs before regenerating them."""
    config_path, config = smoke_environment

    cli_main(["run-all", "--config", str(config_path)])
    metrics_csv = config.paths.artifacts_dir / "metrics" / "metrics.csv"
    sentinel = config.paths.processed_dir / "leftover_from_old_run.txt"
    sentinel.write_text("stale", encoding="utf-8")

    cli_main(["run-all", "--config", str(config_path), "--fresh"])

    assert metrics_csv.exists(), "metrics must be regenerated after fresh run"
    assert not sentinel.exists(), "fresh run must drop stale processed outputs"
