"""Command-line interface for the dissertation forecasting pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

from weather_forecasting_pipeline.config import ExperimentConfig, load_config
from weather_forecasting_pipeline.training.pipeline import evaluate, ingest, preprocess, run_all, train

LOGGER = logging.getLogger("weather_forecasting_pipeline.cli")

# Use ISO 8601 timestamps (UTC, with offset) on every log line so long runs
# remain auditable from terminal output alone. `time.gmtime` keeps timestamps
# in UTC regardless of the operator's locale, which matches the canonical
# `ts_utc` index used throughout the pipeline.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"


_HANDLER_FLAG = "_weather_forecasting_pipeline_handler"


def _configure_logging(level: str) -> None:
    """Install a single timestamped root handler.

    `logging.basicConfig` is a no-op when handlers already exist (for example,
    when pytest captures logs), so the handler is reconfigured explicitly to
    ensure the dissertation format applies in every entry point. Only handlers
    previously installed by this function are replaced, so external handlers
    (such as pytest's `caplog` capture handler) keep receiving records.
    """
    logging.Formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper()))
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_FLAG, False):
            root.removeHandler(existing)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT))
    setattr(handler, _HANDLER_FLAG, True)
    root.addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weather_forecasting_pipeline")
    parser.add_argument(
        "command",
        choices=["ingest", "preprocess", "train", "evaluate", "run-all", "clean"],
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to experiment YAML configuration")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Remove generated outputs (data/interim, data/processed, artifacts) before "
            "running the selected command. Only valid with train / run-all / clean."
        ),
    )
    return parser


def _resolve_generated_paths(config: ExperimentConfig) -> list[Path]:
    """Return the configured generated paths that exist on disk.

    Raw data (``config.paths.raw_data_dir``) and YAML configurations are
    intentionally excluded so cleanup never removes the source-of-truth
    Weathercloud exports a contributor placed in ``data/raw/``.
    """
    candidates = [
        config.paths.interim_dir,
        config.paths.processed_dir,
        config.paths.artifacts_dir / "models",
        config.paths.artifacts_dir / "scalers",
        config.paths.artifacts_dir / "metrics",
        config.paths.artifacts_dir / "plots",
        config.paths.artifacts_dir / "reports",
    ]
    return [p for p in candidates if p.exists()]


def _clean_generated_outputs(config: ExperimentConfig) -> None:
    """Delete generated outputs while leaving raw data and configs intact.

    Used by ``clean`` and by ``--fresh`` on ``train`` / ``run-all``. Raw data
    is never touched: re-ingesting raw exports is cheap, but recovering raw
    exports is not. On Windows, indexing services or sync clients can hold a
    directory handle even after every file underneath has been removed, so a
    second pass deletes contents only when ``rmtree`` cannot drop the
    enclosing directory itself.
    """
    raw_dir = config.paths.raw_data_dir.resolve()
    for path in _resolve_generated_paths(config):
        resolved = path.resolve()
        if resolved == raw_dir or raw_dir in resolved.parents:
            LOGGER.warning("Refusing to delete path inside raw data directory: %s", resolved)
            continue
        LOGGER.info("Removing generated path: %s", resolved)
        try:
            if resolved.is_dir():
                shutil.rmtree(resolved)
            else:
                resolved.unlink()
        except PermissionError as exc:
            if not resolved.is_dir():
                LOGGER.warning("Skipping locked path %s: %s", resolved, exc)
                continue
            # Fall back to deleting children only and leave the directory
            # itself in place so file handles held by Windows services do
            # not abort the cleanup.
            LOGGER.warning(
                "Directory removal blocked (likely Windows file lock); "
                "clearing contents of %s instead: %s",
                resolved,
                exc,
            )
            for child in resolved.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
                except PermissionError as child_exc:
                    LOGGER.warning("Skipping locked child %s: %s", child, child_exc)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    config = load_config(args.config)
    # Expose the resolved config path so parallel-horizon workers can rehydrate
    # the same configuration without round-tripping the dataclass through
    # pickle. Frozen dataclasses cannot carry an extra path attribute, and
    # dumping/reloading YAML keeps the source of truth on disk.
    os.environ["WFP_CONFIG_PATH"] = str(Path(args.config).resolve())
    LOGGER.info(
        "Run start: command=%s config=%s project=%s fresh=%s",
        args.command,
        args.config,
        config.project.name,
        bool(args.fresh),
    )
    if args.fresh and args.command not in {"train", "run-all", "clean"}:
        raise ValueError("--fresh is only valid with train, run-all, or clean")
    if args.fresh and args.command in {"train", "run-all"}:
        _clean_generated_outputs(config)
    if args.command == "ingest":
        ingest(config)
    elif args.command == "preprocess":
        preprocess(config)
    elif args.command == "train":
        train(config)
    elif args.command == "evaluate":
        evaluate(config)
    elif args.command == "run-all":
        run_all(config)
    elif args.command == "clean":
        _clean_generated_outputs(config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    LOGGER.info("Run finish: command=%s project=%s", args.command, config.project.name)


if __name__ == "__main__":
    main()
