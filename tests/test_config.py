from pathlib import Path

import yaml

from weather_forecasting_pipeline.config import load_config


def test_load_config_progress_logging_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"name": "cfg_test", "random_seed": 42},
                "paths": {
                    "raw_data_dir": str(tmp_path / "raw"),
                    "interim_dir": str(tmp_path / "interim"),
                    "processed_dir": str(tmp_path / "processed"),
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "mapping_config": str(tmp_path / "mapping.yaml"),
                },
                "data": {
                    "source": "weathercloud",
                    "timezone": "Europe/Athens",
                    "expected_frequency": "10min",
                    "resample_rule": None,
                    "target": "temp_c",
                    "horizons": {"h01": 6},
                    "optional_horizons": {},
                    "lags": [1, 3],
                    "rolling_windows": [6],
                    "sequence_length": 8,
                    "derived_metrics": [],
                },
                "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
                "scaling": {"method": "standard"},
                "models": {"baselines": ["persistence"], "ml": [], "dl": []},
                "training": {
                    "max_epochs": 2,
                    "batch_size": 8,
                    "learning_rate": 0.01,
                    "patience": 1,
                    "min_dl_train_rows": 50,
                },
                "evaluation": {"mape_epsilon": 1.0e-6, "plot_max_points": 100},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(config_path)
    assert cfg.training.progress_heartbeat_seconds == 60
    assert cfg.training.progress_log_epochs is True
