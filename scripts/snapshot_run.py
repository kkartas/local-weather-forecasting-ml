"""Snapshot the latest training run into ``runs/<run_id>/``.

Usage::

    python scripts/snapshot_run.py                       # uses today's YYMMDD
    python scripts/snapshot_run.py --run-id <run_id>     # explicit id
    python scripts/snapshot_run.py --skip-svr-models     # save disk space
    python scripts/snapshot_run.py --skip-supervised     # exclude wide parquets
    python scripts/snapshot_run.py --force               # overwrite existing snapshot

What it does
------------
1. Validates that the expected artifact folders exist under ``--root``.
2. Creates ``runs/<run_id>/`` with the following layout:

   - ``configs/``    snapshot of YAMLs used by the run
   - ``data/interim/``     canonical + prepared parquets
   - ``data/processed/predictions/``    per-model y_true/y_pred CSVs
   - ``data/processed/split_metadata/`` train/val/test boundaries per horizon
   - ``data/processed/supervised_*.parquet`` (unless --skip-supervised)
   - ``models/``     all trained models (unless --skip-svr-models)
   - ``scalers/``    feature and target scalers
   - ``metrics/``    metrics.csv + metrics.json
   - ``reports/``    auto-generated summary.md
   - ``plots/``      regenerated analytical plot set
   - ``README.md``   inventory + reproduction instructions (NOT a conclusion)

3. Regenerates the analytical plot set via
   ``weather_forecasting_pipeline.plotting.snapshot.generate_snapshot_plots``.

CONCLUSION.md is **not** produced automatically. It is meant to be authored
afterwards by an AI agent (or human) using the snapshotted artifacts as
evidence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import sys
from pathlib import Path

# Ensure the in-tree package import works when the script is invoked directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from weather_forecasting_pipeline.plotting.snapshot import (  # noqa: E402
    SnapshotPaths,
    generate_snapshot_plots,
)

logger = logging.getLogger("snapshot_run")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run-id",
        default=dt.date.today().strftime("%y%m%d"),
        help="Snapshot folder name under runs/ (default: today's YYMMDD).",
    )
    p.add_argument(
        "--root",
        default=str(_REPO_ROOT),
        help="Repository root containing artifacts/, data/, configs/ (default: auto-detected).",
    )
    p.add_argument(
        "--runs-dir",
        default=None,
        help="Override the parent runs/ directory (default: <root>/runs).",
    )
    p.add_argument(
        "--skip-svr-models",
        action="store_true",
        help="Exclude SVR .joblib model files (saves ~5 GB on the default config).",
    )
    p.add_argument(
        "--skip-supervised",
        action="store_true",
        help="Exclude supervised_*.parquet files (saves ~2 GB on the default config).",
    )
    p.add_argument(
        "--skip-interim",
        action="store_true",
        help="Exclude data/interim/*.parquet files.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Copy artifacts only; do not regenerate plots.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing runs/<run_id>/ folder.",
    )
    return p.parse_args(argv)


def _copytree(src: Path, dst: Path, *, ignore_globs: tuple[str, ...] = ()) -> int:
    """Copy ``src`` into ``dst``, skipping files matching any of ``ignore_globs``.

    Returns the number of files copied.
    """
    if not src.exists():
        logger.warning("source missing, skipping: %s", src)
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in src.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        if any(path.match(pat) for pat in ignore_globs):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _copy_globs(src_dir: Path, patterns: tuple[str, ...], dst_dir: Path) -> int:
    if not src_dir.exists():
        logger.warning("source missing, skipping: %s", src_dir)
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for pat in patterns:
        for path in src_dir.glob(pat):
            shutil.copy2(path, dst_dir / path.name)
            count += 1
    return count


def _write_readme(dst: Path, run_id: str, manifest: dict[str, int]) -> None:
    """Write an inventory README. This is *not* a conclusion document."""
    lines = [
        f"# Run Snapshot `{run_id}`",
        "",
        f"Created: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "This folder is a frozen snapshot of the training run. It contains the",
        "configuration, raw inputs, model artifacts, metrics, predictions, and",
        "the regenerated plot set used for analysis.",
        "",
        "`CONCLUSION.md` is intentionally absent — it should be authored",
        "separately (by an AI agent or human reviewer) using the files in this",
        "folder as evidence.",
        "",
        "## Inventory",
        "",
        "| Item | Count |",
        "| --- | --- |",
    ]
    for key, count in sorted(manifest.items()):
        lines.append(f"| {key} | {count} |")
    lines += [
        "",
        "## Layout",
        "",
        "```",
        "configs/                snapshot of YAMLs used by the run",
        "data/interim/           canonical + prepared parquets",
        "data/processed/",
        "    predictions/        per-model y_true/y_pred CSVs",
        "    split_metadata/     train/val/test boundaries per horizon",
        "    supervised_*.parquet  (omitted if --skip-supervised)",
        "models/                 trained models (.joblib / .pt)",
        "scalers/                StandardScaler for features + targets",
        "metrics/                metrics.csv + metrics.json",
        "reports/                summary.md (auto-generated)",
        "plots/",
        "    actual_vs_predicted/   scatter + time-series",
        "    residuals/             distribution + heteroscedasticity",
        "    comparison/            MAE, RMSE, error growth, skill heatmap, best-per-family",
        "```",
        "",
        "## Reproduce the plots",
        "",
        "```bash",
        f"python scripts/snapshot_run.py --run-id {run_id} --no-plots --force  # re-copy artifacts only",
        f"python -c \"from weather_forecasting_pipeline.plotting.snapshot import SnapshotPaths, generate_snapshot_plots; "
        f"from pathlib import Path; p = Path('runs/{run_id}'); "
        f"generate_snapshot_plots(SnapshotPaths(predictions_dir=p/'data/processed/predictions', "
        f"metrics_csv=p/'metrics/metrics.csv', plots_dir=p/'plots'))\"",
        "```",
    ]
    (dst / "README.md").write_text("\n".join(lines), encoding="utf-8")


def snapshot_run(
    *,
    run_id: str,
    root: Path,
    runs_dir: Path | None = None,
    skip_svr_models: bool = False,
    skip_supervised: bool = False,
    skip_interim: bool = False,
    no_plots: bool = False,
    force: bool = False,
) -> Path:
    """Programmatic entrypoint. Returns the path of the created snapshot."""
    root = root.resolve()
    runs_dir = (runs_dir or (root / "runs")).resolve()
    target = runs_dir / run_id

    # Preserve any hand- or AI-authored CONCLUSION.md across --force re-runs so
    # the snapshot can be regenerated from updated artifacts without losing the
    # written analysis.
    preserved_conclusion: bytes | None = None
    if target.exists():
        if not force:
            raise SystemExit(
                f"Snapshot folder already exists: {target}. Re-run with --force to overwrite."
            )
        conclusion_path = target / "CONCLUSION.md"
        if conclusion_path.exists():
            preserved_conclusion = conclusion_path.read_bytes()
            logger.info("preserving existing CONCLUSION.md across --force rewrite")
        logger.info("removing existing snapshot: %s", target)
        shutil.rmtree(target)
    target.mkdir(parents=True)
    if preserved_conclusion is not None:
        (target / "CONCLUSION.md").write_bytes(preserved_conclusion)

    artifacts = root / "artifacts"
    data = root / "data"
    configs = root / "configs"

    manifest: dict[str, int] = {}

    # 1. Configs.
    manifest["configs"] = _copy_globs(configs, ("*.yaml", "*.yml"), target / "configs")

    # 2. Models (optionally excluding the very large SVR files).
    model_ignore = ("svr_*.joblib",) if skip_svr_models else ()
    manifest["models"] = _copytree(artifacts / "models", target / "models", ignore_globs=model_ignore)

    # 3. Scalers, metrics, reports.
    manifest["scalers"] = _copytree(artifacts / "scalers", target / "scalers")
    manifest["metrics"] = _copytree(artifacts / "metrics", target / "metrics")
    manifest["reports"] = _copytree(artifacts / "reports", target / "reports")

    # 4. Existing pipeline plots (the snapshot plot generator augments these,
    # it does not replace them; we still keep the model_comparison_mae.png and
    # error_by_horizon_mae.png that the standard pipeline emits).
    manifest["plots_inherited"] = _copytree(artifacts / "plots", target / "plots")

    # 5. Data — interim parquets.
    if not skip_interim:
        manifest["data_interim"] = _copy_globs(
            data / "interim", ("*.parquet",), target / "data" / "interim"
        )

    # 6. Data — predictions and split metadata always go in.
    manifest["predictions"] = _copytree(
        data / "processed" / "predictions", target / "data" / "processed" / "predictions"
    )
    manifest["split_metadata"] = _copy_globs(
        data / "processed", ("split_metadata_*.json",),
        target / "data" / "processed" / "split_metadata",
    )

    # 7. Data — supervised parquets (large).
    if not skip_supervised:
        manifest["supervised_parquets"] = _copy_globs(
            data / "processed", ("supervised_*.parquet",), target / "data" / "processed"
        )

    # 8. Plots — regenerate the analytical set unless suppressed.
    if not no_plots:
        paths = SnapshotPaths(
            predictions_dir=target / "data" / "processed" / "predictions",
            metrics_csv=target / "metrics" / "metrics.csv",
            plots_dir=target / "plots",
        )
        counts = generate_snapshot_plots(paths)
        manifest["plots_scatter"] = counts.get("scatter", 0)
        manifest["plots_timeseries"] = counts.get("timeseries", 0)
        manifest["plots_residuals"] = counts.get("residuals", 0)
        manifest["plots_comparison"] = counts.get("comparison", 0)

    # 9. README.
    _write_readme(target, run_id, manifest)

    # 10. Manifest JSON for downstream tooling.
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "root": str(root),
                "skip_svr_models": skip_svr_models,
                "skip_supervised": skip_supervised,
                "skip_interim": skip_interim,
                "no_plots": no_plots,
                "counts": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("snapshot created: %s", target)
    return target


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    snapshot_run(
        run_id=args.run_id,
        root=Path(args.root),
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
        skip_svr_models=args.skip_svr_models,
        skip_supervised=args.skip_supervised,
        skip_interim=args.skip_interim,
        no_plots=args.no_plots,
        force=args.force,
    )


if __name__ == "__main__":
    main()
