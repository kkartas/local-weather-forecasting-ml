"""Command-line interface for the dissertation forecasting pipeline."""

from __future__ import annotations

import argparse
import logging
import time

from weather_forecasting_pipeline.config import load_config
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
    parser.add_argument("command", choices=["ingest", "preprocess", "train", "evaluate", "run-all"])
    parser.add_argument("--config", default="configs/default.yaml", help="Path to experiment YAML configuration")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)
    config = load_config(args.config)
    LOGGER.info(
        "Run start: command=%s config=%s project=%s",
        args.command,
        args.config,
        config.project.name,
    )
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
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    LOGGER.info("Run finish: command=%s project=%s", args.command, config.project.name)


if __name__ == "__main__":
    main()
