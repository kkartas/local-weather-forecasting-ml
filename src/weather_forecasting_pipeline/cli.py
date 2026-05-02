"""Command-line interface for the dissertation forecasting pipeline."""

from __future__ import annotations

import argparse
import logging

from weather_forecasting_pipeline.config import load_config
from weather_forecasting_pipeline.training.pipeline import evaluate, ingest, preprocess, run_all, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weather_forecasting_pipeline")
    parser.add_argument("command", choices=["ingest", "preprocess", "train", "evaluate", "run-all"])
    parser.add_argument("--config", default="configs/default.yaml", help="Path to experiment YAML configuration")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(levelname)s:%(name)s:%(message)s")
    config = load_config(args.config)
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


if __name__ == "__main__":
    main()
