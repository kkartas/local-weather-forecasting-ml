"""Convenience wrapper for running the full dissertation experiment."""

from __future__ import annotations

import argparse

from weather_forecasting_pipeline.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(["run-all", "--config", args.config])
