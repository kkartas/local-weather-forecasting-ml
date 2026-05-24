"""Generate synthetic Weathercloud CSV data for CI and local smoke runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _load_mapping_columns(mapping_path: Path) -> tuple[str, list[str]]:
    mapping = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    ts_col = mapping["ts"]["col"]
    field_cols = [field["col"] for field in mapping["fields"].values()]
    return ts_col, field_cols


def write_synthetic_weathercloud_csv(
    output_path: Path,
    *,
    mapping_path: Path,
    n: int = 220,
) -> None:
    """Write a semicolon-delimited Weathercloud export matching the mapping config."""
    ts_col, field_cols = _load_mapping_columns(mapping_path)
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="10min")
    x = np.arange(n, dtype=float)

    values_by_col = {
        ts_col: idx.strftime("%Y-%m-%d %H:%M"),
        field_cols[0]: 12.0 + np.sin(x / 12.0),
        field_cols[1]: 70.0 + np.cos(x / 18.0),
        field_cols[2]: 1015.0 + np.sin(x / 40.0),
        field_cols[3]: 3.6 + np.abs(np.sin(x / 7.0)),
        field_cols[4]: 7.2 + np.abs(np.sin(x / 5.0)),
        field_cols[5]: (x * 15.0) % 360.0,
        field_cols[6]: np.zeros(n),
        field_cols[7]: np.zeros(n),
        field_cols[8]: np.zeros(n),
        field_cols[9]: np.zeros(n),
    }
    frame = pd.DataFrame(values_by_col)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, sep=";", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("configs/weathercloud_mapping.yaml"),
        help="Weathercloud column mapping used by configs/smoke.yaml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/smoke_weathercloud.csv"),
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=220,
        help="Number of 10-minute observations to generate.",
    )
    args = parser.parse_args()
    write_synthetic_weathercloud_csv(
        args.output,
        mapping_path=args.mapping,
        n=args.rows,
    )
    print(f"Wrote synthetic Weathercloud CSV: {args.output}")


if __name__ == "__main__":
    main()
