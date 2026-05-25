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
    # Missing ``grad_clip_norm`` key falls back to the documented default of
    # 1.0 (CHANGES.md 2026-05-25). Explicit ``null`` would disable clipping.
    assert cfg.training.grad_clip_norm == 1.0


def test_load_config_grad_clip_norm_explicit_values(tmp_path: Path):
    """``grad_clip_norm`` accepts a numeric override or explicit ``null``."""
    base_payload = {
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
            "grad_clip_norm": 2.5,
        },
        "evaluation": {"mape_epsilon": 1.0e-6, "plot_max_points": 100},
    }

    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(yaml.safe_dump(base_payload), encoding="utf-8")
    explicit_cfg = load_config(explicit_path)
    assert explicit_cfg.training.grad_clip_norm == 2.5

    disabled_payload = {**base_payload, "training": {**base_payload["training"], "grad_clip_norm": None}}
    disabled_path = tmp_path / "disabled.yaml"
    disabled_path.write_text(yaml.safe_dump(disabled_payload), encoding="utf-8")
    disabled_cfg = load_config(disabled_path)
    assert disabled_cfg.training.grad_clip_norm is None


def test_load_config_torch_threads_per_worker(tmp_path: Path):
    """`torch_threads_per_worker` accepts explicit int, explicit null, or omission."""
    base_payload = {
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

    # Missing key -> auto (None at load time; resolved by pipeline later).
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text(yaml.safe_dump(base_payload), encoding="utf-8")
    missing_cfg = load_config(missing_path)
    assert missing_cfg.training.torch_threads_per_worker is None

    # Explicit numeric override is honoured verbatim.
    explicit_payload = {
        **base_payload,
        "training": {**base_payload["training"], "torch_threads_per_worker": 3},
    }
    explicit_path = tmp_path / "explicit.yaml"
    explicit_path.write_text(yaml.safe_dump(explicit_payload), encoding="utf-8")
    explicit_cfg = load_config(explicit_path)
    assert explicit_cfg.training.torch_threads_per_worker == 3

    # Explicit null also resolves to None (auto).
    null_payload = {
        **base_payload,
        "training": {**base_payload["training"], "torch_threads_per_worker": None},
    }
    null_path = tmp_path / "null.yaml"
    null_path.write_text(yaml.safe_dump(null_payload), encoding="utf-8")
    null_cfg = load_config(null_path)
    assert null_cfg.training.torch_threads_per_worker is None
