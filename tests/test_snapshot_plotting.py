from __future__ import annotations

import pandas as pd
import pytest

from weather_forecasting_pipeline.plotting.snapshot import _select_timeseries_sample
from weather_forecasting_pipeline.plotting.snapshot import (
    SnapshotPaths,
    _set_current_value_labels,
)
import weather_forecasting_pipeline.plotting.snapshot as snapshot_plotting


def _predictions_frame(times: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(times, utc=True),
            "y_true": range(len(times)),
            "y_pred": range(len(times)),
        }
    )


def test_timeseries_sample_defaults_to_largest_contiguous_block():
    early = pd.date_range("2025-12-01 00:00:00+00:00", periods=4, freq="10min")
    later = pd.date_range("2026-02-10 00:00:00+00:00", periods=12, freq="10min")
    df = _predictions_frame([*early.astype(str), *later.astype(str)])

    sample, label = _select_timeseries_sample(df, max_points=6, max_gap="1h")

    assert len(sample) == 6
    assert sample["ts_utc"].min() >= pd.Timestamp("2026-02-10 00:00:00+00:00")
    assert label == "2026-02-10"


def test_timeseries_sample_honours_explicit_window_with_naive_dates():
    times = pd.date_range("2026-02-15 00:00:00+00:00", periods=72, freq="1h")
    df = _predictions_frame(times.astype(str).tolist())

    sample, label = _select_timeseries_sample(
        df,
        start="2026-02-16",
        end="2026-02-17 23:59:59",
    )

    assert sample["ts_utc"].min() == pd.Timestamp("2026-02-16 00:00:00+00:00")
    assert sample["ts_utc"].max() == pd.Timestamp("2026-02-17 23:00:00+00:00")
    assert label == "2026-02-16 to 2026-02-17"


def test_timeseries_sample_rejects_empty_window():
    df = _predictions_frame(["2026-02-15 00:00:00+00:00"])

    with pytest.raises(ValueError, match="No prediction rows"):
        _select_timeseries_sample(df, start="2026-03-01", end="2026-03-02")


def test_snapshot_value_labels_follow_metrics_target(tmp_path):
    metrics = pd.DataFrame(
        [
            {
                "target": "rh_pct",
                "horizon_label": "h01",
                "model": "persistence",
                "mae": 1.0,
            }
        ]
    )
    metrics_csv = tmp_path / "metrics.csv"
    metrics.to_csv(metrics_csv, index=False)

    _set_current_value_labels(
        SnapshotPaths(
            predictions_dir=tmp_path,
            metrics_csv=metrics_csv,
            plots_dir=tmp_path / "plots",
        )
    )

    assert snapshot_plotting._CURRENT_VALUE_LABEL == "Relative humidity"
    assert snapshot_plotting._CURRENT_VALUE_UNIT == "%RH"
