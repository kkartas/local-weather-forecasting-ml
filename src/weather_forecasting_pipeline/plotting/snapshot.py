"""Run-snapshot analytical plots.

Produces the full plot set used in dissertation run reports:

- scatter and time-series of actual vs predicted (per winning model x horizon)
- residual analysis (distribution + heteroscedasticity check)
- model-comparison bars (MAE, RMSE)
- error-growth curves by horizon
- skill-score heatmap
- best-per-family bar chart

All inputs are read from per-model prediction CSVs persisted by the training
pipeline (``data/processed/predictions/predictions_<model>_<horizon>.csv``)
plus the aggregate ``metrics.csv`` written by the evaluation step.

The output directory layout is::

    <plots_dir>/
        actual_vs_predicted/
            scatter_<model>_<horizon>.png
            timeseries_<model>_<horizon>.png
        residuals/
            residuals_<model>_<horizon>.png
        comparison/
            comparison_mae.png
            comparison_rmse.png
            error_growth_by_horizon.png
            skill_score_heatmap.png
            best_per_family.png

The module never re-runs inference: it consumes the already-saved prediction
CSVs so it is cheap to re-execute against any historical run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Order matters: drives the x-axis of every horizon plot.
DEFAULT_HORIZONS: tuple[str, ...] = ("m10", "h01", "h03", "h06", "h12", "h24")

HORIZON_LABELS: dict[str, str] = {
    "m10": "10 min",
    "h01": "1 h",
    "h03": "3 h",
    "h06": "6 h",
    "h12": "12 h",
    "h24": "24 h",
}

# Default focus models for the per-model plots. ``persistence`` is always
# included so each per-horizon comparison has the baseline alongside the
# winners. Other models are still covered in the aggregate comparison plots.
DEFAULT_FOCUS_MODELS: tuple[str, ...] = (
    "gradient_boosting",
    "random_forest",
    "lstm",
    "persistence",
)

# Order in which models appear in the comparison plots (when present).
DEFAULT_MODEL_ORDER: tuple[str, ...] = (
    "persistence",
    "moving_average",
    "climatology",
    "linear_regression",
    "random_forest",
    "gradient_boosting",
    "svr",
    "lstm",
    "gru",
    "tcn",
)


def _pyplot():
    """Lazy import of matplotlib so the module loads without a display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


@dataclass
class SnapshotPaths:
    """Resolved input and output paths for the snapshot plotting routine."""

    predictions_dir: Path
    metrics_csv: Path
    plots_dir: Path
    horizons: Sequence[str] = field(default_factory=lambda: DEFAULT_HORIZONS)
    focus_models: Sequence[str] = field(default_factory=lambda: DEFAULT_FOCUS_MODELS)
    model_order: Sequence[str] = field(default_factory=lambda: DEFAULT_MODEL_ORDER)

    def actual_vs_predicted_dir(self) -> Path:
        return self.plots_dir / "actual_vs_predicted"

    def residuals_dir(self) -> Path:
        return self.plots_dir / "residuals"

    def comparison_dir(self) -> Path:
        return self.plots_dir / "comparison"


def _load_predictions(predictions_dir: Path, model: str, horizon: str) -> pd.DataFrame:
    path = predictions_dir / f"predictions_{model}_{horizon}.csv"
    df = pd.read_csv(path, parse_dates=["ts_utc"])
    return df.dropna(subset=["y_true", "y_pred"])


def _plot_scatter(out: Path, model: str, horizon: str, df: pd.DataFrame) -> None:
    plt = _pyplot()
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(y_true, y_pred, s=3, alpha=0.15, color="#1f77b4", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="y = x")
    ax.set_xlabel("Observed (°C)")
    ax.set_ylabel("Predicted (°C)")
    ax.set_title(f"{model} @ {HORIZON_LABELS.get(horizon, horizon)} — Actual vs Predicted")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ax.text(
        0.98, 0.02,
        f"MAE={mae:.3f} °C\nRMSE={rmse:.3f} °C\nn={len(df):,}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_timeseries(out: Path, model: str, horizon: str, df: pd.DataFrame, max_points: int = 1500) -> None:
    plt = _pyplot()
    df = df.sort_values("ts_utc")
    if len(df) > max_points:
        df = df.iloc[:max_points]

    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    ax.plot(df["ts_utc"], df["y_true"], color="black", linewidth=0.9, label="Observed")
    ax.plot(df["ts_utc"], df["y_pred"], color="#d62728", linewidth=0.9, alpha=0.85, label="Predicted")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"{model} @ {HORIZON_LABELS.get(horizon, horizon)} — Test sample (first {len(df):,} points)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_residuals(out: Path, model: str, horizon: str, df: pd.DataFrame) -> None:
    plt = _pyplot()
    df = df.sort_values("ts_utc")
    resid = (df["y_pred"] - df["y_true"]).to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))

    ax = axes[0]
    ax.hist(resid, bins=80, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.axvline(resid.mean(), color="red", linewidth=1.0, linestyle="--",
               label=f"bias={resid.mean():+.3f}")
    ax.set_xlabel("Error = pred - obs (°C)")
    ax.set_ylabel("Frequency")
    ax.set_title("Error distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    # Heteroscedasticity check: predicted vs residual.
    sample = df.sample(min(8000, len(df)), random_state=42)
    ax.scatter(sample["y_pred"], (sample["y_pred"] - sample["y_true"]),
               s=3, alpha=0.2, color="#9467bd", rasterized=True)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xlabel("Predicted (°C)")
    ax.set_ylabel("Error (°C)")
    ax.set_title("Error vs Predicted")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model} @ {HORIZON_LABELS.get(horizon, horizon)} — Residual analysis", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_comparison(
    out: Path, metrics: pd.DataFrame, metric: str, ylabel: str,
    horizons: Sequence[str], model_order: Sequence[str],
) -> None:
    plt = _pyplot()
    pivot = metrics.pivot_table(index="horizon_label", columns="model", values=metric)
    pivot = pivot.reindex(list(horizons))
    cols = [m for m in model_order if m in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(13.0, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.85, colormap="tab10")
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Model comparison by horizon — {metric.upper()}")
    ax.set_xticklabels([HORIZON_LABELS.get(h, h) for h in pivot.index], rotation=0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_error_growth(
    out: Path, metrics: pd.DataFrame,
    horizons: Sequence[str], model_order: Sequence[str],
) -> None:
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for model in model_order:
        sub = metrics[metrics["model"] == model].set_index("horizon_label").reindex(list(horizons))
        if "mae" not in sub.columns or sub["mae"].isna().all():
            continue
        ax.plot(
            [HORIZON_LABELS.get(h, h) for h in horizons],
            sub["mae"].to_numpy(),
            marker="o", label=model,
        )
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Error growth by horizon — all models")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_skill_heatmap(
    out: Path, metrics: pd.DataFrame,
    horizons: Sequence[str], model_order: Sequence[str],
) -> None:
    if "skill_score_persistence" not in metrics.columns:
        logger.warning("skill_score_persistence missing from metrics; skipping heatmap.")
        return
    plt = _pyplot()
    pivot = metrics.pivot_table(index="model", columns="horizon_label", values="skill_score_persistence")
    pivot = pivot.reindex(
        index=[m for m in model_order if m in pivot.index],
        columns=list(horizons),
    )
    # Clip extreme negatives (e.g. linear_regression at -25) so the color
    # scale still resolves the interesting band [-2, +1].
    clipped = pivot.clip(lower=-2.0, upper=1.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    im = ax.imshow(clipped.values, cmap="RdYlGn", vmin=-2.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels([HORIZON_LABELS.get(h, h) for h in horizons])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Skill score vs persistence (clipped to [-2, 1] for display)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if pd.isna(val):
                continue
            color = "white" if abs(clipped.values[i, j]) > 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, label="skill")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _plot_best_per_family(
    out: Path, metrics: pd.DataFrame, horizons: Sequence[str],
) -> None:
    plt = _pyplot()
    rows = []
    for horizon in horizons:
        sub = metrics[metrics["horizon_label"] == horizon]
        for family in ("baseline", "ml", "dl"):
            fam = sub[sub["model_family"] == family]
            if fam.empty:
                continue
            row = fam.loc[fam["mae"].idxmin()]
            rows.append(
                {"horizon": horizon, "family": family, "model": row["model"], "mae": row["mae"]}
            )
    if not rows:
        return
    summary = pd.DataFrame(rows)
    pivot = summary.pivot(index="horizon", columns="family", values="mae").reindex(list(horizons))

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    pivot.plot(
        kind="bar", ax=ax, width=0.78,
        color={"baseline": "#888888", "ml": "#1f77b4", "dl": "#d62728"},
    )
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("MAE (°C) — best model of family")
    ax.set_title("Best model per family and horizon")
    ax.set_xticklabels([HORIZON_LABELS.get(h, h) for h in pivot.index], rotation=0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Family")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def generate_snapshot_plots(paths: SnapshotPaths) -> dict[str, int]:
    """Generate the full snapshot plot set.

    Returns a counter dictionary so callers can log/report what was produced.
    """
    avp_dir = paths.actual_vs_predicted_dir()
    res_dir = paths.residuals_dir()
    cmp_dir = paths.comparison_dir()
    for d in (avp_dir, res_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    counter = {"scatter": 0, "timeseries": 0, "residuals": 0, "comparison": 0}

    # Per-model x horizon plots.
    for model in paths.focus_models:
        for horizon in paths.horizons:
            try:
                df = _load_predictions(paths.predictions_dir, model, horizon)
            except FileNotFoundError:
                logger.info("predictions missing for %s %s — skipping", model, horizon)
                continue
            _plot_scatter(avp_dir / f"scatter_{model}_{horizon}.png", model, horizon, df)
            counter["scatter"] += 1
            _plot_timeseries(avp_dir / f"timeseries_{model}_{horizon}.png", model, horizon, df)
            counter["timeseries"] += 1
            _plot_residuals(res_dir / f"residuals_{model}_{horizon}.png", model, horizon, df)
            counter["residuals"] += 1

    # Aggregate comparison plots.
    if paths.metrics_csv.exists():
        metrics = pd.read_csv(paths.metrics_csv)
        _plot_metric_comparison(
            cmp_dir / "comparison_mae.png", metrics, "mae", "MAE (°C)",
            paths.horizons, paths.model_order,
        )
        _plot_metric_comparison(
            cmp_dir / "comparison_rmse.png", metrics, "rmse", "RMSE (°C)",
            paths.horizons, paths.model_order,
        )
        _plot_error_growth(
            cmp_dir / "error_growth_by_horizon.png", metrics,
            paths.horizons, paths.model_order,
        )
        _plot_skill_heatmap(
            cmp_dir / "skill_score_heatmap.png", metrics,
            paths.horizons, paths.model_order,
        )
        _plot_best_per_family(cmp_dir / "best_per_family.png", metrics, paths.horizons)
        counter["comparison"] = 5
    else:
        logger.warning("metrics csv missing at %s — comparison plots skipped", paths.metrics_csv)

    return counter
