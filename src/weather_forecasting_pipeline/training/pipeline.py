"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

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
from weather_forecasting_pipeline.evaluation.metrics import evaluate_predictions, persistence_skill_score
from weather_forecasting_pipeline.metdatapy_adapter import (
    fit_apply_scaler_with_metdatapy,
    fit_target_scaler_with_metdatapy,
    ingest_raw_weathercloud,
    inverse_transform_target_with_metdatapy,
    load_interim,
    make_supervised_with_metdatapy,
    preprocess_with_metdatapy,
    save_interim,
    split_by_fraction_with_metdatapy,
    transform_target_with_metdatapy,
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


@contextmanager
def _log_stage(name: str, **context: Any) -> Iterator[dict[str, Any]]:
    """Log start, finish, and elapsed time for a pipeline stage.

    The context dict can be mutated by the caller (e.g. to record output
    sizes) and its key=value pairs are appended to the finish line. The
    stage name is the same on the start and finish lines so log filters
    (e.g. ``Stage finish: train``) line up unambiguously.
    """
    started = time.perf_counter()
    LOGGER.info("Stage start: %s%s", name, _fmt_context(context))
    payload: dict[str, Any] = dict(context)
    try:
        yield payload
    except Exception:
        elapsed = time.perf_counter() - started
        LOGGER.exception("Stage error: %s elapsed=%.2fs%s", name, elapsed, _fmt_context(payload))
        raise
    elapsed = time.perf_counter() - started
    LOGGER.info("Stage finish: %s elapsed=%.2fs%s", name, elapsed, _fmt_context(payload))


def _fmt_context(payload: dict[str, Any]) -> str:
    """Render a context dict as `key=value` pairs for log messages."""
    if not payload:
        return ""
    return " " + " ".join(f"{key}={_fmt_value(value)}" for key, value in payload.items())


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if np.isfinite(value) else "nan"
    return str(value)


def canonical_path(config: ExperimentConfig) -> Path:
    return config.paths.interim_dir / "canonical.parquet"


def prepared_path(config: ExperimentConfig) -> Path:
    return config.paths.interim_dir / "prepared.parquet"


def ingest(config: ExperimentConfig) -> Path:
    """Run raw-to-canonical ingestion through MetDataPy."""
    ensure_directories(config)
    with _log_stage("ingest", project=config.project.name) as ctx:
        df = ingest_raw_weathercloud(config.paths.raw_data_dir, config.paths.mapping_config, config.data.timezone)
        output = canonical_path(config)
        save_interim(df, output)
        LOGGER.info("Saved canonical data: %s rows -> %s", len(df), output)
        ctx["rows"] = len(df)
        ctx["output"] = output
    return output


def preprocess(config: ExperimentConfig) -> Path:
    """Run supported MetDataPy preprocessing and feature preparation."""
    ensure_directories(config)
    source = canonical_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Canonical input not found: {source}. Run ingest first.")
    with _log_stage("preprocess", project=config.project.name) as ctx:
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
        ctx["rows_in"] = len(df)
        ctx["rows_out"] = len(prepared)
        ctx["output"] = output
    return output


def train(config: ExperimentConfig) -> pd.DataFrame:
    """Train configured models and write predictions, model artifacts, and metrics."""
    ensure_directories(config)
    set_random_seed(config.project.random_seed)
    source = prepared_path(config)
    if not source.exists():
        raise FileNotFoundError(f"Prepared input not found: {source}. Run preprocess first.")

    horizons_to_train = _resolved_horizons(config)
    _log_train_run_context(config, horizons_to_train)

    with _log_stage("train", project=config.project.name, horizons=len(horizons_to_train)) as train_ctx:
        prepared = load_interim(source)

        all_metrics: list[dict[str, Any]] = []
        predictions_dir = config.paths.processed_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for horizon_label, horizon_steps in horizons_to_train.items():
            target_col = target_column_name(config.data.target, horizon_steps)
            with _log_stage(
                f"horizon {horizon_label}",
                steps=horizon_steps,
                target=target_col,
            ) as horizon_ctx:
                with _log_stage(f"build supervised {horizon_label}", target=target_col) as build_ctx:
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
                    build_ctx["rows"] = len(supervised)
                    build_ctx["columns"] = len(supervised.columns)
                    build_ctx["output"] = horizon_data_path

                with _log_stage(f"split {horizon_label}") as split_ctx:
                    splits = split_by_fraction_with_metdatapy(
                        supervised,
                        train_fraction=config.split.train,
                        validation_fraction=config.split.validation,
                    )
                    feature_columns = select_feature_columns(supervised, target_col)
                    split_metadata = make_split_metadata(splits, target_col, feature_columns)
                    save_split_metadata(
                        split_metadata,
                        config.paths.processed_dir / f"split_metadata_{horizon_label}.json",
                    )
                    split_ctx["train"] = len(splits["train"])
                    split_ctx["val"] = len(splits["val"])
                    split_ctx["test"] = len(splits["test"])
                    split_ctx["features"] = len(feature_columns)

                with _log_stage(f"fit feature scaler {horizon_label}", method=config.scaling.method):
                    scaled_splits, scaler = fit_apply_scaler_with_metdatapy(
                        splits, feature_columns, config.scaling.method
                    )
                    # Persist the fitted scaler object directly so downstream tooling can
                    # reload it for inference; the previous indirection only stored repr().
                    joblib.dump(
                        scaler,
                        config.paths.artifacts_dir / "scalers" / f"scaler_{horizon_label}.joblib",
                    )

                with _log_stage(f"fit target scaler {horizon_label}", method=config.scaling.method):
                    # The target is intentionally kept unscaled in scaled_splits so baseline
                    # and ML models predict directly in original units. DL models train far
                    # better on a standardised target, so a separate target scaler is fit
                    # on the training partition only and persisted for inference.
                    target_scaler = fit_target_scaler_with_metdatapy(
                        splits["train"], target_col, config.scaling.method
                    )
                    joblib.dump(
                        target_scaler,
                        config.paths.artifacts_dir / "scalers" / f"target_scaler_{horizon_label}.joblib",
                    )

                x_train, y_train = arrays_from_split(scaled_splits["train"], feature_columns, target_col)
                x_test, y_test = arrays_from_split(scaled_splits["test"], feature_columns, target_col)

                horizon_models = 0
                for model_name in config.models.baselines:
                    with _log_stage(
                        "train model",
                        family="baseline",
                        model=model_name,
                        horizon=horizon_label,
                    ) as model_ctx:
                        model = make_baseline(model_name, target=config.data.target).fit(
                            splits["train"], target_col
                        )
                        y_pred = model.predict(splits["test"])
                        result = _record_result(
                            config, horizon_label, horizon_steps, model_name, "baseline", y_test, y_pred
                        )
                        all_metrics.append(result)
                        _save_predictions(
                            predictions_dir, horizon_label, model_name, y_test, y_pred, splits["test"].index
                        )
                        _save_baseline_artifact(config, horizon_label, model_name, model)
                        model_ctx["mae"] = result.get("mae")
                        model_ctx["rmse"] = result.get("rmse")
                    horizon_models += 1

                for model_name in config.models.ml:
                    with _log_stage(
                        "train model",
                        family="ml",
                        model=model_name,
                        horizon=horizon_label,
                        n_train=len(x_train),
                        n_features=x_train.shape[1] if x_train.ndim == 2 else len(feature_columns),
                    ) as model_ctx:
                        model = make_ml_model(model_name, config.project.random_seed)
                        model.fit(x_train, y_train)
                        y_pred = model.predict(x_test).astype(np.float32)
                        result = _record_result(
                            config, horizon_label, horizon_steps, model_name, "ml", y_test, y_pred
                        )
                        all_metrics.append(result)
                        _save_predictions(
                            predictions_dir, horizon_label, model_name, y_test, y_pred, splits["test"].index
                        )
                        joblib.dump(
                            model,
                            config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.joblib",
                        )
                        model_ctx["mae"] = result.get("mae")
                        model_ctx["rmse"] = result.get("rmse")
                    horizon_models += 1

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
                        target_scaler,
                    )
                    all_metrics.extend(dl_metrics)
                    horizon_models += 1 if dl_metrics else 0

                horizon_ctx["models_trained"] = horizon_models

        with _log_stage("write metrics and plots"):
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df = _attach_persistence_skill_score(metrics_df)
            _write_metrics_and_plots(config, metrics_df)

        train_ctx["models_trained"] = len(metrics_df)
    return metrics_df


def evaluate(config: ExperimentConfig) -> pd.DataFrame:
    """Regenerate summary report and aggregate plots from saved metrics."""
    metrics_csv = config.paths.artifacts_dir / "metrics" / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_csv}. Run train first.")
    with _log_stage("evaluate", project=config.project.name, source=metrics_csv) as ctx:
        metrics_df = pd.read_csv(metrics_csv)
        if "skill_score_persistence" not in metrics_df.columns:
            metrics_df = _attach_persistence_skill_score(metrics_df)
        _write_metrics_and_plots(config, metrics_df)
        ctx["rows"] = len(metrics_df)
    return metrics_df


def run_all(config: ExperimentConfig) -> pd.DataFrame:
    """Run ingest, preprocess, train, and evaluate."""
    with _log_stage("run-all", project=config.project.name):
        ingest(config)
        preprocess(config)
        metrics = train(config)
        evaluate(config)
    return metrics


def _log_train_run_context(config: ExperimentConfig, horizons: dict[str, int]) -> None:
    """Log resolved horizons and configured model families before training starts."""
    horizon_summary = ", ".join(f"{label}={steps}" for label, steps in horizons.items())
    LOGGER.info(
        "Train context: project=%s target=%s horizons=[%s] baselines=%s ml=%s dl=%s",
        config.project.name,
        config.data.target,
        horizon_summary,
        list(config.models.baselines),
        list(config.models.ml),
        list(config.models.dl),
    )


def _attach_persistence_skill_score(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Add a per-horizon skill-score column referenced to persistence.

    For each horizon, the persistence row's RMSE is the skill anchor; every
    other model in that horizon receives ``1 - rmse_model**2 / rmse_pers**2``.
    Persistence's own row gets ``0`` by definition. Horizons without a
    persistence row (e.g. when the baseline list does not include it) get
    ``NaN``. The column complements MAPE, which is unstable near 0 °C.
    """
    if metrics_df.empty or "horizon_label" not in metrics_df.columns:
        return metrics_df
    out = metrics_df.copy()
    out["skill_score_persistence"] = np.nan
    for horizon, group in out.groupby("horizon_label"):
        persistence_rows = group[group["model"] == "persistence"]
        if persistence_rows.empty:
            continue
        anchor_rmse = float(persistence_rows.iloc[0]["rmse"])
        for idx in group.index:
            score = persistence_skill_score(float(out.at[idx, "rmse"]), anchor_rmse)
            if score is not None:
                out.at[idx, "skill_score_persistence"] = score
    return out


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
    target_scaler: object,
) -> list[dict[str, Any]]:
    if len(scaled_splits["train"]) < config.training.min_dl_train_rows:
        LOGGER.warning(
            "Skip model: family=dl model=%s horizon=%s reason=min_dl_train_rows "
            "train_rows=%s min_dl_train_rows=%s",
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
    x_test, y_test_orig = sequence_arrays_from_split(
        scaled_splits["test"], feature_columns, target_col, config.data.sequence_length
    )
    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        LOGGER.warning(
            "Skip model: family=dl model=%s horizon=%s reason=insufficient_sequences "
            "train_seq=%s val_seq=%s test_seq=%s sequence_length=%s",
            model_name,
            horizon_label,
            len(x_train),
            len(x_val),
            len(x_test),
            config.data.sequence_length,
        )
        return []

    with _log_stage(
        "train model",
        family="dl",
        model=model_name,
        horizon=horizon_label,
        train_seq=len(x_train),
        val_seq=len(x_val),
        sequence_length=config.data.sequence_length,
        max_epochs=config.training.max_epochs,
    ) as model_ctx:
        # Train the recurrent/TCN regressor on a standardised target so the loss
        # surface does not depend on the absolute magnitude of temp_c. Predictions
        # are inverse-transformed back to original units before metric computation
        # so DL results stay directly comparable with baseline and ML models.
        y_train_scaled = transform_target_with_metdatapy(y_train, target_scaler, target_col).astype(np.float32)
        y_val_scaled = transform_target_with_metdatapy(y_val, target_scaler, target_col).astype(np.float32)

        model = make_dl_model(
            model_name, input_size=len(feature_columns), sequence_length=config.data.sequence_length
        )
        result = train_dl_model(
            model,
            x_train,
            y_train_scaled,
            x_val,
            y_val_scaled,
            max_epochs=config.training.max_epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            patience=config.training.patience,
            seed=config.project.random_seed,
        )
        y_pred_scaled = predict_dl_model(result.model, x_test, batch_size=config.training.batch_size)
        y_pred = inverse_transform_target_with_metdatapy(y_pred_scaled, target_scaler, target_col).astype(np.float32)
        save_torch_model(result.model, config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.pt")
        _save_predictions(
            predictions_dir,
            horizon_label,
            model_name,
            y_test_orig,
            y_pred,
            scaled_splits["test"].index[-len(y_test_orig) :],
        )
        row = _record_result(config, horizon_label, horizon_steps, model_name, "dl", y_test_orig, y_pred)
        row["best_validation_loss"] = result.best_validation_loss
        row["epochs_trained"] = result.epochs_trained
        row["tabular_test_rows"] = len(y_test_tabular)
        model_ctx["mae"] = row.get("mae")
        model_ctx["rmse"] = row.get("rmse")
        model_ctx["epochs_trained"] = result.epochs_trained
        model_ctx["best_validation_loss"] = result.best_validation_loss
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
        display_cols = [
            "model_family",
            "model",
            "horizon_label",
            "horizon_steps",
            "mae",
            "rmse",
            "mape",
            "skill_score_persistence",
            "n_test",
        ]
        available_cols = [c for c in display_cols if c in metrics_df.columns]
        lines.extend(
            _markdown_table(
                metrics_df[available_cols].sort_values(["horizon_steps", "model_family", "model"])
            )
        )
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
