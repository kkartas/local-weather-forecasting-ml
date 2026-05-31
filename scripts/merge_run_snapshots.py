"""Merge a baseline run snapshot with a delta run snapshot.

Workflow context
----------------
The standard delta-run flow when the methodology changes for *some* models
but the data and unchanged models are unaffected:

1. ``python -m weather_forecasting_pipeline train --config configs/default_delta.yaml``
   trains only the models that changed.
2. ``python scripts/snapshot_run.py --run-id <YYMMDD>_delta`` freezes the
   delta-run artifacts.
3. ``python scripts/merge_run_snapshots.py --baseline <baseline_snapshot>
   --delta <delta_snapshot> --full-config configs/default.yaml --output
   <merged_snapshot>`` produces a self-contained snapshot whose model
   roster matches ``--full-config`` and that draws each model's artifacts
   from the freshest available source.

What gets merged
----------------
For every model name listed in the ``--full-config`` ``models`` block
(baselines + ml + dl), the merger picks artifacts from the delta snapshot
if they exist there, otherwise from the baseline snapshot. A
``MERGE_PROVENANCE.md`` file in the output records the chosen source per
model. Models that exist in the baseline but are *not* in the full config
(typically: dropped families like ``svr`` and ``linear_regression``) are
omitted from the merged output entirely.

Shared artifacts (configs, interim datasets, split metadata, scalers) are
expected to be bit-identical between baseline and delta because the data
and seeds did not change. The merger copies them from the delta snapshot
(most recent provenance) but flags any mismatch in ``split_metadata_*``
JSON files so deviation from the no-change assumption is caught.

The merged snapshot is then re-plotted via
``weather_forecasting_pipeline.plotting.snapshot.generate_snapshot_plots``
so the comparison plots span the full canonical roster.

This script does not regenerate ``CONCLUSION.md``; that file is authored
separately and preserved if it already exists at the output path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from weather_forecasting_pipeline.plotting.snapshot import (  # noqa: E402
    SnapshotPaths,
    generate_snapshot_plots,
)

logger = logging.getLogger("merge_run_snapshots")

# Horizons that the dissertation pipeline trains. Hard-coded here only as a
# fallback; in normal use the merger discovers them from the prediction CSV
# filenames in each snapshot.
_DEFAULT_HORIZONS: tuple[str, ...] = ("m10", "h01", "h03", "h06", "h12", "h24")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--baseline",
        required=True,
        help="Path to the baseline run snapshot (e.g. runs/<baseline_id>).",
    )
    p.add_argument(
        "--delta",
        required=True,
        help="Path to the delta run snapshot (e.g. runs/<run_id>_delta).",
    )
    p.add_argument(
        "--full-config",
        default=str(_REPO_ROOT / "configs" / "default.yaml"),
        help=(
            "YAML config whose `models` block lists the canonical roster for the "
            "merged snapshot. Models in baseline that are not in this roster are "
            "dropped (typically: removed families like svr/linear_regression)."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output path for the merged snapshot (e.g. runs/250525_final).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output snapshot. Preserves CONCLUSION.md.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip regenerating the analytical plot set after merging.",
    )
    return p.parse_args(argv)


def _load_canonical_models(full_config_path: Path) -> dict[str, list[str]]:
    """Read the ``models`` block from the canonical-roster YAML config."""
    with full_config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    models = raw.get("models") or {}
    return {
        "baselines": list(models.get("baselines") or []),
        "ml": list(models.get("ml") or []),
        "dl": list(models.get("dl") or []),
    }


def _model_files_in_snapshot(snapshot: Path) -> dict[str, Path]:
    """Return a ``{model_name_horizon_label: file_path}`` map from models/."""
    models_dir = snapshot / "models"
    out: dict[str, Path] = {}
    if not models_dir.exists():
        return out
    for path in models_dir.iterdir():
        if path.suffix not in (".joblib", ".pt"):
            continue
        out[path.stem] = path
    return out


def _prediction_files_in_snapshot(snapshot: Path) -> dict[str, Path]:
    """Return a ``{model_horizon_key: csv_path}`` map from predictions/."""
    pred_dir = snapshot / "data" / "processed" / "predictions"
    out: dict[str, Path] = {}
    if not pred_dir.exists():
        return out
    for path in pred_dir.glob("predictions_*_*.csv"):
        out[path.stem.removeprefix("predictions_")] = path
    return out


def _discover_horizons(*snapshots: Path) -> list[str]:
    """Discover horizon labels by scanning split-metadata filenames.

    Falls back to ``_DEFAULT_HORIZONS`` if neither snapshot has split
    metadata. The order is preserved as on disk (lexicographic).
    """
    horizons: list[str] = []
    seen: set[str] = set()
    for snap in snapshots:
        meta_dir = snap / "data" / "processed" / "split_metadata"
        if not meta_dir.exists():
            continue
        for path in sorted(meta_dir.glob("split_metadata_*.json")):
            horizon = path.stem.removeprefix("split_metadata_")
            if horizon not in seen:
                horizons.append(horizon)
                seen.add(horizon)
    if not horizons:
        horizons = list(_DEFAULT_HORIZONS)
    return horizons


def _verify_shared_artifacts(baseline: Path, delta: Path, horizons: list[str]) -> list[str]:
    """Compare split-metadata JSONs across snapshots; return a list of warnings."""
    warnings: list[str] = []
    for horizon in horizons:
        b = baseline / "data" / "processed" / "split_metadata" / f"split_metadata_{horizon}.json"
        d = delta / "data" / "processed" / "split_metadata" / f"split_metadata_{horizon}.json"
        if not b.exists() or not d.exists():
            continue
        try:
            b_payload = json.loads(b.read_text(encoding="utf-8"))
            d_payload = json.loads(d.read_text(encoding="utf-8"))
        except Exception as exc:
            warnings.append(f"split_metadata_{horizon}.json read error: {exc}")
            continue
        for key in ("train_start", "train_end", "test_start", "test_end", "n_train", "n_test"):
            if b_payload.get(key) != d_payload.get(key):
                warnings.append(
                    f"split mismatch for {horizon}.{key}: baseline={b_payload.get(key)} "
                    f"delta={d_payload.get(key)}"
                )
    return warnings


def _pick_source(
    model: str,
    horizon: str,
    delta_files: dict[str, Path],
    baseline_files: dict[str, Path],
) -> tuple[Path | None, str]:
    """Choose the freshest available artifact for a (model, horizon) pair.

    Returns the path and the provenance tag (``"delta"`` / ``"baseline"`` /
    ``"missing"``). The key format matches the persistence convention used
    by the training pipeline (``<model>_<horizon>`` for model files and
    ``<model>_<horizon>`` after stripping the ``predictions_`` prefix for
    prediction CSVs).
    """
    key = f"{model}_{horizon}"
    if key in delta_files:
        return delta_files[key], "delta"
    if key in baseline_files:
        return baseline_files[key], "baseline"
    return None, "missing"


def _copy_models(
    baseline: Path, delta: Path, output: Path, roster: list[str], horizons: list[str]
) -> dict[tuple[str, str], str]:
    delta_model_files = _model_files_in_snapshot(delta)
    baseline_model_files = _model_files_in_snapshot(baseline)
    out_dir = output / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[tuple[str, str], str] = {}
    for model in roster:
        for horizon in horizons:
            src, tag = _pick_source(model, horizon, delta_model_files, baseline_model_files)
            provenance[(model, horizon)] = tag
            if src is None:
                logger.warning("model artifact missing for %s @ %s", model, horizon)
                continue
            shutil.copy2(src, out_dir / src.name)
    return provenance


def _copy_predictions(
    baseline: Path, delta: Path, output: Path, roster: list[str], horizons: list[str]
) -> dict[tuple[str, str], str]:
    delta_files = _prediction_files_in_snapshot(delta)
    baseline_files = _prediction_files_in_snapshot(baseline)
    out_dir = output / "data" / "processed" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance: dict[tuple[str, str], str] = {}
    for model in roster:
        for horizon in horizons:
            src, tag = _pick_source(model, horizon, delta_files, baseline_files)
            provenance[(model, horizon)] = tag
            if src is None:
                logger.warning("prediction CSV missing for %s @ %s", model, horizon)
                continue
            shutil.copy2(src, out_dir / src.name)
    return provenance


def _merge_metrics(baseline: Path, delta: Path, output: Path, roster: set[str]) -> Path:
    """Combine baseline + delta metrics CSVs, dropping out-of-roster rows."""
    out_metrics_dir = output / "metrics"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    baseline_csv = baseline / "metrics" / "metrics.csv"
    delta_csv = delta / "metrics" / "metrics.csv"
    frames: list[pd.DataFrame] = []
    if delta_csv.exists():
        df_delta = pd.read_csv(delta_csv)
        df_delta = df_delta[df_delta["model"].isin(roster)]
        frames.append(df_delta)
        # Models present in delta supersede baseline rows for the same
        # (model, horizon) tuples so we keep the freshest measurements.
        superseded_models = set(df_delta["model"].unique())
    else:
        superseded_models = set()

    if baseline_csv.exists():
        df_base = pd.read_csv(baseline_csv)
        df_base = df_base[df_base["model"].isin(roster)]
        df_base = df_base[~df_base["model"].isin(superseded_models)]
        frames.append(df_base)

    if not frames:
        raise FileNotFoundError(
            "Neither baseline nor delta snapshots contain a metrics/metrics.csv file."
        )

    merged = pd.concat(frames, ignore_index=True)
    # Stable ordering for the dissertation: by horizon length, then family, then model.
    if "horizon_steps" in merged.columns:
        merged = merged.sort_values(
            ["horizon_steps", "model_family", "model"], kind="mergesort"
        ).reset_index(drop=True)
    out_csv = out_metrics_dir / "metrics.csv"
    merged.to_csv(out_csv, index=False)
    (out_metrics_dir / "metrics.json").write_text(
        json.dumps(
            merged.where(pd.notna(merged), None).to_dict(orient="records"),
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_csv


def _copy_shared_artifacts(baseline: Path, delta: Path, output: Path) -> dict[str, str]:
    """Copy configs, interim data, scalers, split metadata, supervised parquets.

    The delta snapshot is preferred as the source of truth because it was
    produced under the latest methodology. If a category is missing from
    delta we silently fall back to baseline so a slimmed delta snapshot
    (``--skip-interim`` etc.) does not lose the corresponding files in the
    merged output.
    """
    provenance: dict[str, str] = {}
    categories: list[tuple[str, Path]] = [
        ("configs", output / "configs"),
        ("scalers", output / "scalers"),
        ("data/interim", output / "data" / "interim"),
        ("data/processed/split_metadata", output / "data" / "processed" / "split_metadata"),
    ]
    for rel, dst in categories:
        src_delta = delta / rel
        src_baseline = baseline / rel
        if src_delta.exists():
            shutil.copytree(src_delta, dst, dirs_exist_ok=True)
            provenance[rel] = "delta"
        elif src_baseline.exists():
            shutil.copytree(src_baseline, dst, dirs_exist_ok=True)
            provenance[rel] = "baseline"
        else:
            provenance[rel] = "missing"

    # Supervised parquets sit next to split_metadata in processed/. Copy
    # them directly so the merged snapshot stays a drop-in replacement for
    # the per-run output.
    for parent in (delta, baseline):
        src = parent / "data" / "processed"
        if not src.exists():
            continue
        for parquet in src.glob("supervised_*.parquet"):
            target = output / "data" / "processed" / parquet.name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(parquet, target)
        provenance["data/processed/supervised_*.parquet"] = (
            provenance.get("data/processed/supervised_*.parquet") or parent.name
        )

    return provenance


def _write_provenance(
    output: Path,
    *,
    baseline: Path,
    delta: Path,
    full_config_path: Path,
    roster_by_family: dict[str, list[str]],
    model_provenance: dict[tuple[str, str], str],
    prediction_provenance: dict[tuple[str, str], str],
    shared_provenance: dict[str, str],
    warnings: list[str],
) -> None:
    """Write MERGE_PROVENANCE.md documenting source-of-truth per artifact."""
    lines: list[str] = [
        "# Merge Provenance",
        "",
        f"Merged at: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        f"- Baseline snapshot: `{baseline}`",
        f"- Delta snapshot: `{delta}`",
        f"- Canonical roster from: `{full_config_path}`",
        "",
        "This snapshot was assembled by `scripts/merge_run_snapshots.py`.",
        "Each artifact below is tagged with its source snapshot. Models",
        "tagged `delta` were retrained under the latest methodology; models",
        "tagged `baseline` were reused unchanged on the same data, splits,",
        "and seeds.",
        "",
        "## Canonical model roster",
        "",
    ]
    for family, models in roster_by_family.items():
        lines.append(f"- **{family}**: {', '.join(models) if models else '(empty)'}")
    lines += [
        "",
        "## Per-model artifact sources",
        "",
        "| model | horizon | model file | predictions |",
        "| --- | --- | --- | --- |",
    ]
    horizons_seen = sorted({h for _, h in model_provenance.keys()}) or sorted(
        {h for _, h in prediction_provenance.keys()}
    )
    models_seen = sorted({m for m, _ in model_provenance.keys()}) or sorted(
        {m for m, _ in prediction_provenance.keys()}
    )
    for model in models_seen:
        for horizon in horizons_seen:
            mfile = model_provenance.get((model, horizon), "missing")
            pfile = prediction_provenance.get((model, horizon), "missing")
            lines.append(f"| {model} | {horizon} | {mfile} | {pfile} |")

    lines += ["", "## Shared artifact sources", ""]
    for key, source in sorted(shared_provenance.items()):
        lines.append(f"- `{key}` <- `{source}`")

    if warnings:
        lines += ["", "## Warnings during merge", ""]
        for w in warnings:
            lines.append(f"- {w}")
        lines += [
            "",
            "**Action**: investigate before relying on the merged metrics for the",
            "dissertation. Any split-metadata mismatch implies the baseline and",
            "delta runs did NOT see the same data, which breaks the no-retrain",
            "assumption.",
        ]
    else:
        lines += [
            "",
            "## Warnings during merge",
            "",
            "None. Baseline and delta split-metadata payloads matched on the",
            "fields (`train_start`, `train_end`, `test_start`, `test_end`,",
            "`n_train`, `n_test`) compared by the merger.",
        ]

    (output / "MERGE_PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")


def merge_snapshots(
    *,
    baseline: Path,
    delta: Path,
    full_config: Path,
    output: Path,
    force: bool = False,
    no_plots: bool = False,
) -> Path:
    """Programmatic entrypoint. Returns the path of the merged snapshot."""
    baseline = baseline.resolve()
    delta = delta.resolve()
    full_config = full_config.resolve()
    output = output.resolve()

    if not baseline.exists():
        raise FileNotFoundError(f"Baseline snapshot not found: {baseline}")
    if not delta.exists():
        raise FileNotFoundError(f"Delta snapshot not found: {delta}")
    if not full_config.exists():
        raise FileNotFoundError(f"Full-config YAML not found: {full_config}")

    preserved_conclusion: bytes | None = None
    if output.exists():
        if not force:
            raise SystemExit(
                f"Merged snapshot folder already exists: {output}. Re-run with --force to overwrite."
            )
        conclusion_path = output / "CONCLUSION.md"
        if conclusion_path.exists():
            preserved_conclusion = conclusion_path.read_bytes()
            logger.info("preserving existing CONCLUSION.md across --force rewrite")
        logger.info("removing existing merged snapshot: %s", output)
        shutil.rmtree(output)
    output.mkdir(parents=True)
    if preserved_conclusion is not None:
        (output / "CONCLUSION.md").write_bytes(preserved_conclusion)

    roster_by_family = _load_canonical_models(full_config)
    roster = list(
        {*roster_by_family["baselines"], *roster_by_family["ml"], *roster_by_family["dl"]}
    )
    horizons = _discover_horizons(delta, baseline)
    warnings = _verify_shared_artifacts(baseline, delta, horizons)

    shared_provenance = _copy_shared_artifacts(baseline, delta, output)
    model_provenance = _copy_models(baseline, delta, output, roster, horizons)
    prediction_provenance = _copy_predictions(baseline, delta, output, roster, horizons)
    metrics_csv = _merge_metrics(baseline, delta, output, set(roster))

    if not no_plots:
        paths = SnapshotPaths(
            predictions_dir=output / "data" / "processed" / "predictions",
            metrics_csv=metrics_csv,
            plots_dir=output / "plots",
        )
        counts = generate_snapshot_plots(paths)
        logger.info("plots generated: %s", counts)

    _write_provenance(
        output,
        baseline=baseline,
        delta=delta,
        full_config_path=full_config,
        roster_by_family=roster_by_family,
        model_provenance=model_provenance,
        prediction_provenance=prediction_provenance,
        shared_provenance=shared_provenance,
        warnings=warnings,
    )

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "merge_type": "baseline_plus_delta",
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "baseline_snapshot": str(baseline),
                "delta_snapshot": str(delta),
                "full_config": str(full_config),
                "roster_by_family": roster_by_family,
                "warnings": warnings,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("merged snapshot written: %s", output)
    return output


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)
    merge_snapshots(
        baseline=Path(args.baseline),
        delta=Path(args.delta),
        full_config=Path(args.full_config),
        output=Path(args.output),
        force=args.force,
        no_plots=args.no_plots,
    )


if __name__ == "__main__":
    main()
