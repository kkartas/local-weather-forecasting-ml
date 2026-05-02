"""Configuration loading for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_seed: int = 42


@dataclass(frozen=True)
class PathConfig:
    raw_data_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    mapping_config: Path


@dataclass(frozen=True)
class DataConfig:
    source: str
    timezone: str
    expected_frequency: str
    resample_rule: str | None
    target: str
    horizons: dict[str, int]
    optional_horizons: dict[str, int]
    lags: list[int]
    rolling_windows: list[int]
    sequence_length: int
    derived_metrics: list[str]


@dataclass(frozen=True)
class SplitConfig:
    train: float
    validation: float
    test: float


@dataclass(frozen=True)
class ScalingConfig:
    method: str


@dataclass(frozen=True)
class ModelConfig:
    baselines: list[str]
    ml: list[str]
    dl: list[str]


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    min_dl_train_rows: int


@dataclass(frozen=True)
class EvaluationConfig:
    mape_epsilon: float
    plot_max_points: int


@dataclass(frozen=True)
class ExperimentConfig:
    project: ProjectConfig
    paths: PathConfig
    data: DataConfig
    split: SplitConfig
    scaling: ScalingConfig
    models: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key: {key}")
    return mapping[key]


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate the YAML experiment configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    paths = _require(raw, "paths")
    base = config_path.parent.parent if config_path.parent.name == "configs" else Path.cwd()

    split_raw = _require(raw, "split")
    split_total = float(split_raw["train"]) + float(split_raw["validation"]) + float(split_raw["test"])
    if abs(split_total - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    data_raw = _require(raw, "data")
    if data_raw.get("source") != "weathercloud":
        raise ValueError("Only Weathercloud station data is supported by this dissertation pipeline")

    models_raw = _require(raw, "models")
    training_raw = _require(raw, "training")
    evaluation_raw = _require(raw, "evaluation")
    project_raw = _require(raw, "project")
    scaling_raw = _require(raw, "scaling")

    return ExperimentConfig(
        project=ProjectConfig(
            name=str(project_raw["name"]),
            random_seed=int(project_raw.get("random_seed", 42)),
        ),
        paths=PathConfig(
            raw_data_dir=base / paths["raw_data_dir"],
            interim_dir=base / paths["interim_dir"],
            processed_dir=base / paths["processed_dir"],
            artifacts_dir=base / paths["artifacts_dir"],
            mapping_config=base / paths["mapping_config"],
        ),
        data=DataConfig(
            source=str(data_raw["source"]),
            timezone=str(data_raw["timezone"]),
            expected_frequency=str(data_raw["expected_frequency"]),
            resample_rule=data_raw.get("resample_rule"),
            target=str(data_raw["target"]),
            horizons={str(k): int(v) for k, v in data_raw["horizons"].items()},
            optional_horizons={str(k): int(v) for k, v in data_raw.get("optional_horizons", {}).items()},
            lags=[int(v) for v in data_raw["lags"]],
            rolling_windows=[int(v) for v in data_raw.get("rolling_windows", [])],
            sequence_length=int(data_raw["sequence_length"]),
            derived_metrics=list(data_raw.get("derived_metrics", [])),
        ),
        split=SplitConfig(
            train=float(split_raw["train"]),
            validation=float(split_raw["validation"]),
            test=float(split_raw["test"]),
        ),
        scaling=ScalingConfig(method=str(scaling_raw["method"])),
        models=ModelConfig(
            baselines=list(models_raw.get("baselines", [])),
            ml=list(models_raw.get("ml", [])),
            dl=list(models_raw.get("dl", [])),
        ),
        training=TrainingConfig(
            max_epochs=int(training_raw["max_epochs"]),
            batch_size=int(training_raw["batch_size"]),
            learning_rate=float(training_raw["learning_rate"]),
            patience=int(training_raw["patience"]),
            min_dl_train_rows=int(training_raw.get("min_dl_train_rows", 300)),
        ),
        evaluation=EvaluationConfig(
            mape_epsilon=float(evaluation_raw.get("mape_epsilon", 1e-6)),
            plot_max_points=int(evaluation_raw.get("plot_max_points", 500)),
        ),
    )


def ensure_directories(config: ExperimentConfig) -> None:
    """Create configured output directories if needed."""
    for path in [
        config.paths.raw_data_dir,
        config.paths.interim_dir,
        config.paths.processed_dir,
        config.paths.artifacts_dir / "models",
        config.paths.artifacts_dir / "scalers",
        config.paths.artifacts_dir / "metrics",
        config.paths.artifacts_dir / "plots",
        config.paths.artifacts_dir / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)
