"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
import logging
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from weather_forecasting_pipeline.config import ExperimentConfig, ensure_directories
from weather_forecasting_pipeline.datasets.splits import (
    arrays_from_split,
    make_split_metadata,
    save_split_metadata,
    select_feature_columns,
    sequence_arrays_from_split,
    target_column_name,
)
from weather_forecasting_pipeline.evaluation.metrics import evaluate_predictions
from weather_forecasting_pipeline.metdatapy_adapter import (
    fit_apply_scaler_with_metdatapy,
    ingest_raw_weathercloud,
    load_interim,
    make_supervised_with_metdatapy,
    preprocess_with_metdatapy,
    save_interim,
    split_by_fraction_with_metdatapy,
)
from weather_forecasting_pipeline.models.baselines import make_baseline
from weather_forecasting_pipeline.models.dl_models import (
    make_dl_model,
    predict_dl_model,
    save_torch_model,
    train_dl_model,
)
from weather_forecasting_pipeline.models.ml_models import make_ml_model
from weather_forecasting_pipeline.utils.reproducibility import set_random_seed

LOGGER = logging.getLogger(__name__)


def canonical_path(config: ExperimentConfig) -> Path:
    return config.paths.interim_dir / "canonical.parquet"


def prepared_path(config: ExperimentConfig) -> Path:
    return config.paths.interim_dir / "prepared.parquet"


def ingest(config: ExperimentConfig) -> Path:
    """Run raw-to-canonical ingestion through MetDataPy."""
    ensure_directories(config)
    df = ingest_raw_weathercloud(config.paths.raw_data_dir, config.paths.mapping_config, config.data.timezone)
    output = canonical_path(config)
    save_interim(df, output)
    LOGGER.info("Saved canonical data: %s rows -> %s", len(df), output)
    return output


def preprocess(config: ExperimentConfig) -> Path:
    """Run supported MetDataPy preprocessing and feature preparation."""
    ensure_directories(config)
    source = canonical_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Canonical input not found: {source}. Run ingest first.")
    df = load_interim(source)
    prepared = preprocess_with_metdatapy(
        df,
        expected_frequency=config.data.expected_frequency,
        derived_metrics=config.data.derived_metrics,
        rolling_windows=config.data.rolling_windows,
        resample_rule=config.data.resample_rule,
    )
    output = prepared_path(config)
    save_interim(prepared, output)
    LOGGER.info("Saved prepared data: %s rows -> %s", len(prepared), output)
    return output


def train(config: ExperimentConfig) -> pd.DataFrame:
    """Train configured models and write predictions, model artifacts, and metrics."""
    ensure_directories(config)
    set_random_seed(config.project.random_seed)
    source = prepared_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Prepared input not found: {source}. Run preprocess first.")
    prepared = load_interim(source)

    all_metrics: list[dict[str, Any]] = []
    predictions_dir = config.paths.processed_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    horizons_to_train = _resolved_horizons(config)

    for horizon_label, horizon_steps in horizons_to_train.items():
        target_col = target_column_name(config.data.target, horizon_steps)
        LOGGER.info("Preparing horizon %s (%s steps)", horizon_label, horizon_steps)
        supervised = make_supervised_with_metdatapy(
            prepared,
            target=config.data.target,
            horizons=[horizon_steps],
            lags=config.data.lags,
        )
        if target_col not in supervised.columns:
            raise ValueError(f"Expected supervised target column missing: {target_col}")

        horizon_data_path = config.paths.processed_dir / f"supervised_{horizon_label}.parquet"
        supervised.to_parquet(horizon_data_path, index=True)

        splits = split_by_fraction_with_metdatapy(
            supervised,
            train_fraction=config.split.train,
            validation_fraction=config.split.validation,
        )
        feature_columns = select_feature_columns(supervised, target_col)
        metadata = make_split_metadata(splits, target_col, feature_columns)
        save_split_metadata(metadata, config.paths.processed_dir / f"split_metadata_{horizon_label}.json")

        scaled_splits, scaler = fit_apply_scaler_with_metdatapy(splits, feature_columns, config.scaling.method)
        # Persist the fitted scaler object directly so downstream tooling can
        # reload it for inference; the previous indirection only stored repr().
        joblib.dump(scaler, config.paths.artifacts_dir / "scalers" / f"scaler_{horizon_label}.joblib")

        x_train, y_train = arrays_from_split(scaled_splits["train"], feature_columns, target_col)
        x_val, y_val = arrays_from_split(scaled_splits["val"], feature_columns, target_col)
        x_test, y_test = arrays_from_split(scaled_splits["test"], feature_columns, target_col)

        for model_name in config.models.baselines:
            model = make_baseline(model_name, target=config.data.target).fit(splits["train"], target_col)
            y_pred = model.predict(splits["test"])
            all_metrics.append(
                _record_result(config, horizon_label, horizon_steps, model_name, "baseline", y_test, y_pred)
            )
            _save_predictions(predictions_dir, horizon_label, model_name, y_test, y_pred, splits["test"].index)
            _save_baseline_artifact(config, horizon_label, model_name, model)

        for model_name in config.models.ml:
            model = make_ml_model(model_name, config.project.random_seed)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test).astype(np.float32)
            all_metrics.append(_record_result(config, horizon_label, horizon_steps, model_name, "ml", y_test, y_pred))
            _save_predictions(predictions_dir, horizon_label, model_name, y_test, y_pred, splits["test"].index)
            joblib.dump(model, config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.joblib")

        for model_name in config.models.dl:
            dl_metrics = _train_dl_if_possible(
                config,
                model_name,
                horizon_label,
                horizon_steps,
                scaled_splits,
                feature_columns,
                target_col,
                y_test,
                predictions_dir,
            )
            all_metrics.extend(dl_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    _write_metrics_and_plots(config, metrics_df)
    return metrics_df


def evaluate(config: ExperimentConfig) -> pd.DataFrame:
    """Regenerate summary report and aggregate plots from saved metrics."""
    metrics_csv = config.paths.artifacts_dir / "metrics" / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv}. Run train first.")
    metrics_df = pd.read_csv(metrics_csv)
    _write_metrics_and_plots(config, metrics_df)
    return metrics_df


def run_all(config: ExperimentConfig) -> pd.DataFrame:
    """Run ingest, preprocess, train, and evaluate."""
    ingest(config)
    preprocess(config)
    metrics = train(config)
    evaluate(config)
    return metrics


def _resolved_horizons(config: ExperimentConfig) -> dict[str, int]:
    """Merge required and optional horizons in deterministic order.

    The configuration exposes ``horizons`` (always trained) and an optional
    ``optional_horizons`` block. Earlier versions parsed the optional block but
    never iterated it, so configurations such as ``configs/default.yaml`` that
    placed ``m10`` and ``h24`` under ``optional_horizons`` silently produced
    no results for those horizons. The dissertation requires multi-horizon
    coverage, so both blocks are now trained. ``horizons`` wins on key
    collision and the merged mapping is sorted by horizon length so the
    summary report and plots have a consistent horizon axis.
    """
    merged: dict[str, int] = {}
    for label, steps in (config.data.optional_horizons or {}).items():
        merged[label] = int(steps)
    for label, steps in config.data.horizons.items():
        merged[label] = int(steps)
    return dict(sorted(merged.items(), key=lambda kv: kv[1]))


def _train_dl_if_possible(
    config: ExperimentConfig,
    model_name: str,
    horizon_label: str,
    horizon_steps: int,
    scaled_splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    target_col: str,
    y_test_tabular: np.ndarray,
    predictions_dir: Path,
) -> list[dict[str, Any]]:
    if len(scaled_splits["train"]) < config.training.min_dl_train_rows:
        LOGGER.warning(
            "Skipping %s for %s: train rows (%s) below min_dl_train_rows (%s)",
            model_name,
            horizon_label,
            len(scaled_splits["train"]),
            config.training.min_dl_train_rows,
        )
        return []

    x_train, y_train = sequence_arrays_from_split(
        scaled_splits["train"], feature_columns, target_col, config.data.sequence_length
    )
    x_val, y_val = sequence_arrays_from_split(scaled_splits["val"], feature_columns, target_col, config.data.sequence_length)
    x_test, y_test = sequence_arrays_from_split(
        scaled_splits["test"], feature_columns, target_col, config.data.sequence_length
    )
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        LOGGER.warning("Skipping %s for %s: insufficient rows after sequence construction", model_name, horizon_label)
        return []

    model = make_dl_model(model_name, input_size=len(feature_columns))
    result = train_dl_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        max_epochs=config.training.max_epochs,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        patience=config.training.patience,
        seed=config.project.random_seed,
    )
    y_pred = predict_dl_model(result.model, x_test, batch_size=config.training.batch_size)
    save_torch_model(result.model, config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.pt")
    _save_predictions(
        predictions_dir,
        horizon_label,
        model_name,
        y_test,
        y_pred,
        scaled_splits["test"].index[-len(y_test) :],
    )
    row = _record_result(config, horizon_label, horizon_steps, model_name, "dl", y_test, y_pred)
    row["best_validation_loss"] = result.best_validation_loss
    row["epochs_trained"] = result.epochs_trained
    row["tabular_test_rows"] = len(y_test_tabular)
    return [row]


def _record_result(
    config: ExperimentConfig,
    horizon_label: str,
    horizon_steps: int,
    model_name: str,
    model_family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = evaluate_predictions(y_true, y_pred, mape_epsilon=config.evaluation.mape_epsilon)
    return {
        "target": config.data.target,
        "horizon_label": horizon_label,
        "horizon_steps": horizon_steps,
        "model": model_name,
        "model_family": model_family,
        **metrics,
        "n_test": len(y_true),
    }


def _save_predictions(
    predictions_dir: Path,
    horizon_label: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    index: pd.Index,
) -> None:
    frame = pd.DataFrame({"ts_utc": index.astype(str), "y_true": y_true, "y_pred": y_pred})
    frame.to_csv(predictions_dir / f"predictions_{model_name}_{horizon_label}.csv", index=False)


def _save_baseline_artifact(config: ExperimentConfig, horizon_label: str, model_name: str, model: object) -> None:
    joblib.dump(model, config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.joblib")


def _write_metrics_and_plots(config: ExperimentConfig, metrics_df: pd.DataFrame) -> None:
    metrics_dir = config.paths.artifacts_dir / "metrics"
    plots_dir = config.paths.artifacts_dir / "plots"
    reports_dir = config.paths.artifacts_dir / "reports"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(metrics_dir / "metrics.csv", index=False)
    with (metrics_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics_df.where(pd.notna(metrics_df), None).to_dict(orient="records"), fh, indent=2)
    _write_markdown_report(config, metrics_df, reports_dir / "summary.md")

    if not _plotting_environment_compatible():
        LOGGER.warning(
            "Plot generation skipped because the installed Matplotlib/NumPy versions are incompatible. "
            "Install project requirements in a clean environment to enable plots."
        )
        return

    try:
        from weather_forecasting_pipeline.plotting.plots import (
            plot_actual_vs_predicted,
            plot_error_by_horizon,
            plot_metric_comparison,
            plot_residual_distribution,
        )
    except Exception as exc:
        LOGGER.warning("Plotting is unavailable in this environment; metrics and reports were still written: %s", exc)
        return

    try:
        plot_metric_comparison(metrics_df, plots_dir / "model_comparison_mae.png", metric="mae")
        plot_error_by_horizon(metrics_df, plots_dir / "error_by_horizon_mae.png", metric="mae")

        predictions_dir = config.paths.processed_dir / "predictions"
        for pred_file in sorted(predictions_dir.glob("predictions_*_*.csv"))[:4]:
            pred = pd.read_csv(pred_file)
            stem = pred_file.stem.replace("predictions_", "")
            plot_actual_vs_predicted(
                pred["y_true"].to_numpy(),
                pred["y_pred"].to_numpy(),
                plots_dir / f"actual_vs_predicted_{stem}.png",
                title=f"Actual vs predicted: {stem}",
                max_points=config.evaluation.plot_max_points,
            )
            plot_residual_distribution(
                pred["y_true"].to_numpy(),
                pred["y_pred"].to_numpy(),
                plots_dir / f"residuals_{stem}.png",
                title=f"Residuals: {stem}",
            )
    except Exception as exc:
        LOGGER.warning("Plot generation failed; metrics and reports were still written: %s", exc)


def _write_markdown_report(config: ExperimentConfig, metrics_df: pd.DataFrame, path: Path) -> None:
    lines = [
        f"# Experiment Summary: {config.project.name}",
        "",
        f"- Target: `{config.data.target}`",
        "- Input policy: observation-only local station data; no NWP inputs.",
        "- Split policy: chronological 70/15/15 by default; no shuffling.",
        "- Scaling policy: fitted on training features only and applied to validation/test.",
        "- MAPE policy: masks near-zero true values; reports blank when undefined.",
        "",
        "## Metrics",
        "",
    ]
    if metrics_df.empty:
        lines.append("No metrics were produced.")
    else:
        display_cols = ["model_family", "model", "horizon_label", "horizon_steps", "mae", "rmse", "mape", "n_test"]
        lines.extend(_markdown_table(metrics_df[display_cols].sort_values(["horizon_steps", "model_family", "model"])))
    lines.extend(
        [
            "",
            "## MetDataPy Notes",
            "",
            "The executable pipeline uses the preparation functionality exposed by the installed MetDataPy version. "
            "Remaining MetDataPy requirements are tracked in `METDATAPY.md`.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> list[str]:
    """Render a small GitHub-flavored Markdown table without extra dependencies."""
    columns = [str(c) for c in df.columns]
    rows = [[_format_cell(value) for value in row] for row in df.to_numpy()]
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(row) + " |")
    return output


def _format_cell(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _plotting_environment_compatible() -> bool:
    """Avoid importing a known incompatible Matplotlib/NumPy binary pair."""
    try:
        numpy_version = metadata.version("numpy")
        matplotlib_version = metadata.version("matplotlib")
    except metadata.PackageNotFoundError:
        return False
    numpy_major = int(numpy_version.split(".", maxsplit=1)[0])
    matplotlib_major_minor = tuple(int(part) for part in matplotlib_version.split(".")[:2])
    return not (numpy_major >= 2 and matplotlib_major_minor < (3, 8))
