"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd

from weather_forecasting_pipeline.config import ExperimentConfig, ensure_directories, load_config
from weather_forecasting_pipeline.datasets.splits import (
    arrays_from_split,
    build_sequence_dataset,
    estimate_sequence_batch_bytes,
    make_split_metadata,
    save_split_metadata,
    select_dl_feature_columns,
    select_feature_columns,
    sequence_targets,
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
    predict_dl_model_from_dataset,
    save_torch_model,
    train_dl_model_from_datasets,
)
from weather_forecasting_pipeline.models.ml_models import make_ml_model
from weather_forecasting_pipeline.training.progress import (
    SharedTrainingProgressTracker,
    TrainingProgressTracker,
    heartbeat_during,
)
from weather_forecasting_pipeline.utils.reproducibility import set_random_seed

LOGGER = logging.getLogger(__name__)

# Worker logging format mirrors the CLI handler so multi-process runs keep
# the ISO-8601 UTC layout enforced by ``cli._configure_logging``. Defining
# the format here avoids a circular import between training and cli at
# package import time.
_WORKER_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_WORKER_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"
_worker_progress_tracker: TrainingProgressTracker | SharedTrainingProgressTracker | None = None


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
    horizon_workers = _resolve_horizon_workers(config, len(horizons_to_train))
    per_horizon_models = len(config.models.baselines) + len(config.models.ml) + len(config.models.dl)
    total_models = len(horizons_to_train) * per_horizon_models
    LOGGER.info(
        "Train plan: models=%s horizons=%s per_horizon=%s horizon_workers=%s target=%s",
        total_models,
        len(horizons_to_train),
        per_horizon_models,
        horizon_workers,
        config.data.target,
    )

    with _log_stage(
        "train",
        project=config.project.name,
        horizons=len(horizons_to_train),
        horizon_workers=horizon_workers,
    ) as train_ctx:
        prepared = load_interim(source)

        all_metrics: list[dict[str, Any]] = []
        predictions_dir = config.paths.processed_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        if horizon_workers <= 1:
            progress_tracker: TrainingProgressTracker | SharedTrainingProgressTracker = (
                TrainingProgressTracker(total_models=total_models)
            )
            for horizon_label, horizon_steps in horizons_to_train.items():
                rows = _train_one_horizon(
                    config=config,
                    prepared=prepared,
                    horizon_label=horizon_label,
                    horizon_steps=horizon_steps,
                    predictions_dir=predictions_dir,
                    parallel_horizon_workers=1,
                    progress_tracker=progress_tracker,
                )
                all_metrics.extend(rows)
        else:
            with mp.Manager() as manager:
                progress_tracker = SharedTrainingProgressTracker(total_models=total_models, manager=manager)
                all_metrics.extend(
                    _train_horizons_in_parallel(
                        config=config,
                        horizons=horizons_to_train,
                        horizon_workers=horizon_workers,
                        progress_tracker=progress_tracker,
                    )
                )

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
        "Train context: project=%s target=%s horizons=[%s] baselines=%s ml=%s dl=%s "
        "horizon_workers=%s dl_exclude_lag_features=%s",
        config.project.name,
        config.data.target,
        horizon_summary,
        list(config.models.baselines),
        list(config.models.ml),
        list(config.models.dl),
        config.training.horizon_workers,
        config.data.dl_exclude_lag_features,
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


def _resolve_horizon_workers(config: ExperimentConfig, n_horizons: int) -> int:
    """Cap configured ``horizon_workers`` at the number of horizons and CPUs.

    Process-level parallelism is bounded by the smaller of the configured
    value, the number of horizons to train, and the number of available CPUs.
    Values below ``1`` collapse to sequential mode so a misconfigured
    ``horizon_workers: 0`` does not silently disable training.
    """
    requested = max(1, int(config.training.horizon_workers))
    cpu = max(1, os.cpu_count() or 1)
    return max(1, min(requested, max(1, n_horizons), cpu))


def _train_one_horizon(
    *,
    config: ExperimentConfig,
    prepared: pd.DataFrame,
    horizon_label: str,
    horizon_steps: int,
    predictions_dir: Path,
    parallel_horizon_workers: int,
    progress_tracker: TrainingProgressTracker | SharedTrainingProgressTracker | None,
) -> list[dict[str, Any]]:
    """Run the full per-horizon pipeline and return its metric rows.

    Extracted so the same function powers both the sequential ``train`` loop
    and the per-process worker used when ``training.horizon_workers > 1``.
    All side effects (artifacts, predictions, split metadata) are scoped to
    this horizon, so workers do not contend on the same files.
    """
    target_col = target_column_name(config.data.target, horizon_steps)
    horizon_metrics: list[dict[str, Any]] = []
    rf_n_jobs = 1 if parallel_horizon_workers > 1 else -1
    worker_pid = os.getpid()

    with _log_stage(
        f"horizon {horizon_label}",
        steps=horizon_steps,
        target=target_col,
        worker=parallel_horizon_workers,
        pid=worker_pid,
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
            slot: dict[str, int] | None = None
            if progress_tracker is not None:
                slot = progress_tracker.start_model()
            with _log_stage(
                "train model",
                family="baseline",
                model=model_name,
                horizon=horizon_label,
                run=f"{slot['run']}/{slot['total']}" if slot is not None else "n/a",
                remaining=slot["remaining"] if slot is not None else "n/a",
            ) as model_ctx:
                model = make_baseline(model_name, target=config.data.target).fit(
                    splits["train"], target_col
                )
                y_pred = model.predict(splits["test"])
                result = _record_result(
                    config, horizon_label, horizon_steps, model_name, "baseline", y_test, y_pred
                )
                horizon_metrics.append(result)
                _save_predictions(
                    predictions_dir, horizon_label, model_name, y_test, y_pred, splits["test"].index
                )
                _save_baseline_artifact(config, horizon_label, model_name, model)
                model_ctx["mae"] = result.get("mae")
                model_ctx["rmse"] = result.get("rmse")
                if progress_tracker is not None:
                    done = progress_tracker.finish_model()
                    model_ctx["run_completed"] = f"{done['run_completed']}/{done['total']}"
                    model_ctx["remaining"] = done["remaining"]
            horizon_models += 1

        for model_name in config.models.ml:
            slot = None
            if progress_tracker is not None:
                slot = progress_tracker.start_model()
            with _log_stage(
                "train model",
                family="ml",
                model=model_name,
                horizon=horizon_label,
                n_train=len(x_train),
                n_features=x_train.shape[1] if x_train.ndim == 2 else len(feature_columns),
                run=f"{slot['run']}/{slot['total']}" if slot is not None else "n/a",
                remaining=slot["remaining"] if slot is not None else "n/a",
            ) as model_ctx:
                model = make_ml_model(
                    model_name, config.project.random_seed, rf_n_jobs=rf_n_jobs
                )
                with heartbeat_during(
                    config.training.progress_heartbeat_seconds,
                    lambda elapsed: LOGGER.info(
                        "Stage progress: train model family=ml model=%s horizon=%s "
                        "run=%s remaining=%s heartbeat elapsed=%ss status=fitting",
                        model_name,
                        horizon_label,
                        f"{slot['run']}/{slot['total']}" if slot is not None else "n/a",
                        slot["remaining"] if slot is not None else "n/a",
                        elapsed,
                    ),
                ):
                    model.fit(x_train, y_train)
                y_pred = model.predict(x_test).astype(np.float32)
                result = _record_result(
                    config, horizon_label, horizon_steps, model_name, "ml", y_test, y_pred
                )
                horizon_metrics.append(result)
                _save_predictions(
                    predictions_dir, horizon_label, model_name, y_test, y_pred, splits["test"].index
                )
                joblib.dump(
                    model,
                    config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.joblib",
                )
                model_ctx["mae"] = result.get("mae")
                model_ctx["rmse"] = result.get("rmse")
                if progress_tracker is not None:
                    done = progress_tracker.finish_model()
                    model_ctx["run_completed"] = f"{done['run_completed']}/{done['total']}"
                    model_ctx["remaining"] = done["remaining"]
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
                progress_tracker,
            )
            horizon_metrics.extend(dl_metrics)
            horizon_models += 1 if dl_metrics else 0

        horizon_ctx["models_trained"] = horizon_models

    return horizon_metrics


def _train_horizons_in_parallel(
    *,
    config: ExperimentConfig,
    horizons: dict[str, int],
    horizon_workers: int,
    progress_tracker: SharedTrainingProgressTracker,
) -> list[dict[str, Any]]:
    """Train each horizon in its own worker process.

    The pool uses the ``spawn`` start method so each worker re-imports the
    package cleanly (matching Windows behaviour and keeping sklearn/torch
    process state isolated). Workers reload the prepared parquet from disk
    independently because pickling the full multi-year dataframe to every
    worker would dwarf the actual training input.
    """
    config_path = _resolve_config_path_for_workers(config)
    if config_path is None:
        raise RuntimeError(
            "Parallel horizon training requires the original YAML config path; "
            "pass the loaded config alongside its source path or call train() "
            "from the CLI."
        )
    LOGGER.info(
        "Parallel horizon training: workers=%s horizons=%s config=%s",
        horizon_workers,
        list(horizons.keys()),
        config_path,
    )
    metrics: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=horizon_workers,
        mp_context=ctx,
        initializer=_init_horizon_worker_progress,
        initargs=(progress_tracker,),
    ) as executor:
        futures = {
            executor.submit(
                _train_horizon_worker_entry,
                str(config_path),
                horizon_label,
                int(horizon_steps),
                horizon_workers,
            ): horizon_label
            for horizon_label, horizon_steps in horizons.items()
        }
        completed_horizons = 0
        total_horizons = len(horizons)
        for future in as_completed(futures):
            horizon_label = futures[future]
            try:
                rows = future.result()
            except Exception:
                LOGGER.exception("Horizon worker failed: horizon=%s", horizon_label)
                raise
            completed_horizons += 1
            snapshot = _tracker_progress_snapshot(progress_tracker)
            LOGGER.info(
                "Horizon worker complete: horizon=%s rows=%s%s",
                horizon_label,
                len(rows),
                _fmt_context(snapshot) if snapshot is not None else "",
            )
            LOGGER.info("Horizons complete: %s/%s", completed_horizons, total_horizons)
            metrics.extend(rows)
    return metrics


def _resolve_config_path_for_workers(config: ExperimentConfig) -> Path | None:
    """Return a YAML path the workers can ``load_config`` from.

    Stored on the config dataclass when it is loaded via :func:`load_config`
    in :mod:`weather_forecasting_pipeline.cli`; falls back to environment
    variable when callers want to drive ``train()`` directly.
    """
    env_path = os.environ.get("WFP_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
    cached = getattr(config, "_source_path", None)
    if cached is not None:
        return Path(cached)
    return None


def _train_horizon_worker_entry(
    config_path: str,
    horizon_label: str,
    horizon_steps: int,
    horizon_workers: int,
) -> list[dict[str, Any]]:
    """Process-pool entry point that trains a single horizon end-to-end.

    Re-configures logging in the worker (each spawned process has a fresh
    Python state), reloads the YAML config and the prepared parquet from
    disk, applies a horizon-specific seed offset for reproducibility, and
    delegates to :func:`_train_one_horizon`.
    """
    _configure_worker_logging()
    config = load_config(config_path)
    seed_offset = sum(ord(c) for c in horizon_label) % 1000
    set_random_seed(config.project.random_seed + seed_offset)
    prepared = load_interim(prepared_path(config))
    predictions_dir = config.paths.processed_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Horizon worker start: horizon=%s steps=%s pid=%s seed=%s",
        horizon_label,
        horizon_steps,
        os.getpid(),
        config.project.random_seed + seed_offset,
    )
    return _train_one_horizon(
        config=config,
        prepared=prepared,
        horizon_label=horizon_label,
        horizon_steps=horizon_steps,
        predictions_dir=predictions_dir,
        parallel_horizon_workers=horizon_workers,
        progress_tracker=_worker_progress_tracker,
    )


def _init_horizon_worker_progress(tracker: SharedTrainingProgressTracker) -> None:
    """Attach shared progress tracker to worker globals and logging."""
    global _worker_progress_tracker
    _worker_progress_tracker = tracker
    _configure_worker_logging()


def _configure_worker_logging() -> None:
    """Install the project's ISO log format inside a worker process."""
    logging.Formatter.converter = time.gmtime
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt=_WORKER_LOG_FORMAT, datefmt=_WORKER_LOG_DATEFMT))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


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
    progress_tracker: TrainingProgressTracker | SharedTrainingProgressTracker | None,
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

    dl_feature_columns = select_dl_feature_columns(
        scaled_splits["train"],
        target_col,
        exclude_lag_features=config.data.dl_exclude_lag_features,
        feature_allow_list=config.data.dl_feature_columns,
    )
    if not dl_feature_columns:
        LOGGER.warning(
            "Skip model: family=dl model=%s horizon=%s reason=no_dl_features "
            "n_tabular_features=%s",
            model_name,
            horizon_label,
            len(feature_columns),
        )
        return []

    seq_len = config.data.sequence_length
    n_train_rows = len(scaled_splits["train"])
    n_val_rows = len(scaled_splits["val"])
    n_test_rows = len(scaled_splits["test"])
    if min(n_train_rows, n_val_rows, n_test_rows) < seq_len:
        LOGGER.warning(
            "Skip model: family=dl model=%s horizon=%s reason=insufficient_sequences "
            "train_rows=%s val_rows=%s test_rows=%s sequence_length=%s",
            model_name,
            horizon_label,
            n_train_rows,
            n_val_rows,
            n_test_rows,
            seq_len,
        )
        return []

    LOGGER.info(
        "DL feature selection: horizon=%s model=%s n_dl_features=%s "
        "n_tabular_features=%s exclude_lag_features=%s",
        horizon_label,
        model_name,
        len(dl_feature_columns),
        len(feature_columns),
        config.data.dl_exclude_lag_features,
    )

    estimated_batch_bytes = estimate_sequence_batch_bytes(
        seq_len, len(dl_feature_columns), config.training.batch_size
    )
    LOGGER.info(
        "DL memory estimate: horizon=%s model=%s sequence_length=%s n_dl_features=%s "
        "batch_size=%s batch_bytes=%s",
        horizon_label,
        model_name,
        seq_len,
        len(dl_feature_columns),
        config.training.batch_size,
        estimated_batch_bytes,
    )

    # The training and validation targets are scaled (DL trains on a
    # standardised target). The test-side target stays in original units so
    # metric computation matches baselines/ML directly.
    y_train_full = scaled_splits["train"][target_col].to_numpy(dtype=np.float32)
    y_val_full = scaled_splits["val"][target_col].to_numpy(dtype=np.float32)
    y_test_full = scaled_splits["test"][target_col].to_numpy(dtype=np.float32)
    y_train_scaled = transform_target_with_metdatapy(
        y_train_full, target_scaler, target_col
    ).astype(np.float32)
    y_val_scaled = transform_target_with_metdatapy(
        y_val_full, target_scaler, target_col
    ).astype(np.float32)

    train_dataset = build_sequence_dataset(
        scaled_splits["train"], dl_feature_columns, y_train_scaled, seq_len
    )
    val_dataset = build_sequence_dataset(
        scaled_splits["val"], dl_feature_columns, y_val_scaled, seq_len
    )
    test_dataset = build_sequence_dataset(
        scaled_splits["test"], dl_feature_columns, y_test_full, seq_len
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        LOGGER.warning(
            "Skip model: family=dl model=%s horizon=%s reason=insufficient_sequences "
            "train_seq=%s val_seq=%s test_seq=%s sequence_length=%s",
            model_name,
            horizon_label,
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
            seq_len,
        )
        return []

    slot: dict[str, int] | None = None
    if progress_tracker is not None:
        slot = progress_tracker.start_model()

    with _log_stage(
        "train model",
        family="dl",
        model=model_name,
        horizon=horizon_label,
        train_seq=len(train_dataset),
        val_seq=len(val_dataset),
        sequence_length=seq_len,
        max_epochs=config.training.max_epochs,
        n_dl_features=len(dl_feature_columns),
        run=f"{slot['run']}/{slot['total']}" if slot is not None else "n/a",
        remaining=slot["remaining"] if slot is not None else "n/a",
    ) as model_ctx:
        model = make_dl_model(
            model_name, input_size=len(dl_feature_columns), sequence_length=seq_len
        )

        def _on_epoch_end(
            epoch: int, max_epochs: int, train_loss: float, val_loss: float, patience_left: int
        ) -> None:
            should_log = config.training.progress_log_epochs or epoch == 1 or epoch == max_epochs
            should_log = should_log or patience_left == 0
            if not should_log:
                return
            LOGGER.info(
                "Stage progress: train model family=dl model=%s horizon=%s "
                "run=%s remaining=%s epoch=%s/%s train_loss=%.6f val_loss=%.6f patience_left=%s",
                model_name,
                horizon_label,
                f"{slot['run']}/{slot['total']}" if slot is not None else "n/a",
                slot["remaining"] if slot is not None else "n/a",
                epoch,
                max_epochs,
                train_loss,
                val_loss,
                patience_left,
            )

        result = train_dl_model_from_datasets(
            model,
            train_dataset,
            val_dataset,
            max_epochs=config.training.max_epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            patience=config.training.patience,
            seed=config.project.random_seed,
            grad_clip_norm=config.training.grad_clip_norm,
            on_epoch_end=_on_epoch_end,
        )
        y_pred_scaled = predict_dl_model_from_dataset(
            result.model, test_dataset, batch_size=config.training.batch_size
        )
        y_pred = inverse_transform_target_with_metdatapy(
            y_pred_scaled, target_scaler, target_col
        ).astype(np.float32)
        # Test-side ground truth in original units, aligned with sequence end positions.
        y_test_orig = sequence_targets(y_test_full, seq_len)
        save_torch_model(
            result.model, config.paths.artifacts_dir / "models" / f"{model_name}_{horizon_label}.pt"
        )
        _save_predictions(
            predictions_dir,
            horizon_label,
            model_name,
            y_test_orig,
            y_pred,
            scaled_splits["test"].index[-len(y_test_orig) :],
        )
        row = _record_result(
            config, horizon_label, horizon_steps, model_name, "dl", y_test_orig, y_pred
        )
        row["best_validation_loss"] = result.best_validation_loss
        row["epochs_trained"] = result.epochs_trained
        row["tabular_test_rows"] = len(y_test_tabular)
        row["n_dl_features"] = len(dl_feature_columns)
        model_ctx["mae"] = row.get("mae")
        model_ctx["rmse"] = row.get("rmse")
        model_ctx["epochs_trained"] = result.epochs_trained
        model_ctx["best_validation_loss"] = result.best_validation_loss
        if progress_tracker is not None:
            done = progress_tracker.finish_model()
            model_ctx["run_completed"] = f"{done['run_completed']}/{done['total']}"
            model_ctx["remaining"] = done["remaining"]
    return [row]


def _tracker_progress_snapshot(
    tracker: TrainingProgressTracker | SharedTrainingProgressTracker | None,
) -> dict[str, int] | None:
    """Best-effort read of completed/remaining counters for summary logs."""
    if tracker is None:
        return None
    completed_value = getattr(tracker, "_completed", None)
    total_models = getattr(tracker, "total_models", None)
    if completed_value is None or total_models is None:
        return None
    try:
        completed = int(completed_value.value) if hasattr(completed_value, "value") else int(completed_value)
        total = int(total_models)
    except (TypeError, ValueError):
        return None
    return {"run_completed": completed, "remaining": max(total - completed, 0)}


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
