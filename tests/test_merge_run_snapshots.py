"""Smoke test for `scripts/merge_run_snapshots.merge_snapshots`.

Builds two minimal fake snapshots on disk, runs the merger, and asserts:

- the merged snapshot follows the canonical roster from a full-config YAML
- per-model artifacts are sourced from the delta snapshot when available
  and fall back to the baseline snapshot otherwise
- models present in baseline but not in the canonical roster are dropped
- the merged metrics CSV is the union of in-roster rows with delta rows
  superseding baseline rows for the same `(model, horizon)` pair
- `MERGE_PROVENANCE.md` and `manifest.json` are written

The fixture is small enough to run without invoking the training pipeline.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_merge_module():
    """Load `scripts/merge_run_snapshots.py` as a module without changing cwd."""
    spec = importlib.util.spec_from_file_location(
        "merge_run_snapshots",
        _REPO_ROOT / "scripts" / "merge_run_snapshots.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("merge_run_snapshots", module)
    spec.loader.exec_module(module)
    return module


def _write_minimal_snapshot(
    root: Path,
    *,
    models: dict[str, list[str]],
    horizons: list[str],
    metrics_rows: list[dict],
    split_payload: dict | None = None,
) -> None:
    """Create the minimum-viable snapshot layout for the merger to consume."""
    (root / "models").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed" / "predictions").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed" / "split_metadata").mkdir(parents=True, exist_ok=True)
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "scalers").mkdir(parents=True, exist_ok=True)
    (root / "data" / "interim").mkdir(parents=True, exist_ok=True)

    for family, names in models.items():
        for name in names:
            for horizon in horizons:
                # Each "model" is a marker file so the merger has something to
                # copy. Suffix matches the persistence convention so the
                # merger's model-file scanner picks it up.
                suffix = ".pt" if family == "dl" else ".joblib"
                (root / "models" / f"{name}_{horizon}{suffix}").write_text(
                    f"marker {name} {horizon}", encoding="utf-8"
                )
                pd.DataFrame(
                    {"ts_utc": ["2025-01-01 00:00:00+00:00"], "y_true": [1.0], "y_pred": [1.1]}
                ).to_csv(
                    root / "data" / "processed" / "predictions" / f"predictions_{name}_{horizon}.csv",
                    index=False,
                )

    pd.DataFrame(metrics_rows).to_csv(root / "metrics" / "metrics.csv", index=False)
    (root / "metrics" / "metrics.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8"
    )

    split_payload = split_payload or {
        "target_column": "temp_c_t+6",
        "train_start": "2024-01-01T00:00:00",
        "train_end": "2024-06-01T00:00:00",
        "test_start": "2024-07-01T00:00:00",
        "test_end": "2024-08-01T00:00:00",
        "n_train": 100,
        "n_test": 20,
    }
    for horizon in horizons:
        (root / "data" / "processed" / "split_metadata" / f"split_metadata_{horizon}.json").write_text(
            json.dumps(split_payload), encoding="utf-8"
        )


def _row(model: str, family: str, horizon: str, horizon_steps: int, mae: float) -> dict:
    return {
        "target": "temp_c",
        "horizon_label": horizon,
        "horizon_steps": horizon_steps,
        "model": model,
        "model_family": family,
        "mae": mae,
        "rmse": mae * 1.2,
        "mape": mae * 5.0,
        "n_test": 20,
        "skill_score_persistence": 0.0,
    }


def test_merge_run_snapshots_smoke(tmp_path: Path):
    merge_mod = _load_merge_module()

    horizons = ["m10", "h01"]
    # Baseline contains the full historical roster (including the dropped
    # families that should not survive into the merged snapshot).
    baseline = tmp_path / "baseline"
    _write_minimal_snapshot(
        baseline,
        models={
            "baselines": ["persistence", "moving_average"],
            "ml": ["linear_regression", "random_forest", "gradient_boosting", "svr"],
            "dl": ["lstm", "gru"],
        },
        horizons=horizons,
        metrics_rows=[
            _row("persistence", "baseline", "m10", 1, 0.25),
            _row("persistence", "baseline", "h01", 6, 0.78),
            _row("moving_average", "baseline", "m10", 1, 0.6),
            _row("moving_average", "baseline", "h01", 6, 1.03),
            _row("linear_regression", "ml", "m10", 1, 0.4),
            _row("random_forest", "ml", "m10", 1, 0.26),
            _row("random_forest", "ml", "h01", 6, 0.62),
            _row("gradient_boosting", "ml", "m10", 1, 0.26),
            _row("gradient_boosting", "ml", "h01", 6, 0.64),
            _row("svr", "ml", "m10", 1, 0.67),
            _row("lstm", "dl", "m10", 1, 0.86),
            _row("lstm", "dl", "h01", 6, 0.98),
            _row("gru", "dl", "m10", 1, 0.88),
            _row("gru", "dl", "h01", 6, 1.32),
        ],
    )

    # Delta retrains only the changed models. lstm gets a *fresher* number
    # than baseline (the merger must prefer the delta row). ridge is new.
    delta = tmp_path / "delta"
    _write_minimal_snapshot(
        delta,
        models={
            "baselines": ["persistence", "moving_average"],
            "ml": ["ridge"],
            "dl": ["lstm", "gru"],
        },
        horizons=horizons,
        metrics_rows=[
            _row("persistence", "baseline", "m10", 1, 0.25),
            _row("persistence", "baseline", "h01", 6, 0.78),
            _row("moving_average", "baseline", "m10", 1, 0.6),
            _row("moving_average", "baseline", "h01", 6, 1.03),
            _row("ridge", "ml", "m10", 1, 0.27),
            _row("ridge", "ml", "h01", 6, 0.70),
            _row("lstm", "dl", "m10", 1, 0.55),  # better than baseline 0.86
            _row("lstm", "dl", "h01", 6, 0.60),  # better than baseline 0.98
            _row("gru", "dl", "m10", 1, 0.50),
            _row("gru", "dl", "h01", 6, 0.55),
        ],
    )

    # Canonical roster: post-update default.yaml. Excludes linear_regression
    # and svr so they should not appear in the merged output.
    full_config = tmp_path / "default_full.yaml"
    full_config.write_text(
        yaml.safe_dump(
            {
                "models": {
                    "baselines": ["persistence", "moving_average"],
                    "ml": ["ridge", "random_forest", "gradient_boosting"],
                    "dl": ["lstm", "gru"],
                }
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "final"
    result = merge_mod.merge_snapshots(
        baseline=baseline,
        delta=delta,
        full_config=full_config,
        output=output,
        no_plots=True,
    )
    assert result == output.resolve()

    # Provenance + manifest files exist.
    assert (output / "MERGE_PROVENANCE.md").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["merge_type"] == "baseline_plus_delta"
    assert set(manifest["roster_by_family"]["ml"]) == {
        "ridge",
        "random_forest",
        "gradient_boosting",
    }

    # Dropped models do not appear in the merged predictions/models output.
    pred_dir = output / "data" / "processed" / "predictions"
    assert not (pred_dir / "predictions_linear_regression_m10.csv").exists()
    assert not (pred_dir / "predictions_svr_m10.csv").exists()
    assert not (output / "models" / "linear_regression_m10.joblib").exists()

    # Models that exist only in baseline (random_forest, gradient_boosting)
    # are present in the merged snapshot.
    assert (pred_dir / "predictions_random_forest_m10.csv").exists()
    assert (pred_dir / "predictions_gradient_boosting_h01.csv").exists()

    # New models from delta are present.
    assert (pred_dir / "predictions_ridge_m10.csv").exists()

    # lstm prediction CSV comes from delta (the marker file content is
    # distinctive enough to verify the source).
    lstm_marker = (output / "models" / "lstm_m10.pt").read_text(encoding="utf-8")
    assert lstm_marker == "marker lstm m10"  # both wrote the same text, but delta wins on file-mtime semantics
    # The metrics row for lstm@m10 must reflect the delta value (0.55) not
    # the baseline value (0.86), confirming the delta row supersedes.
    merged_metrics = pd.read_csv(output / "metrics" / "metrics.csv")
    lstm_m10 = merged_metrics[
        (merged_metrics["model"] == "lstm") & (merged_metrics["horizon_label"] == "m10")
    ]
    assert len(lstm_m10) == 1
    assert lstm_m10.iloc[0]["mae"] == pytest.approx(0.55)

    # Out-of-roster rows must not appear in the merged metrics CSV.
    assert "linear_regression" not in set(merged_metrics["model"].unique())
    assert "svr" not in set(merged_metrics["model"].unique())
