#!/usr/bin/env python3
"""
Plot static -> coagent pipeline training histories.

Layout: 4 metrics × 2 stages (static | coagent), one colored line per run.
X-axis resets to generation 0 for each stage so both panels are directly comparable.

Usage:
  python plot_timeline.py                        # up to 3 runs from models/pipeline_runs
  python plot_timeline.py --all                  # all run_* directories
  python plot_timeline.py --runs 10 --smooth 15  # custom run count and smoothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


STAGES = ("static_stage", "coagent_stage")
STAGE_TITLES = {
    "static_stage": "Static (evader vs fixed searchers)",
    "coagent_stage": "Co-agent (co-evolution)",
}
METRICS = [
    ("evader_score",   "Evader Score"),
    ("searcher_score", "Searcher Score"),
    ("found_rate",     "Found Rate"),
    ("avg_attempts",   "Avg Attempts"),
]


def natural_run_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.name.split("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (10**9, path.name)


def load_history(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} did not contain a list")
    return data


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    window = min(window, len(values))
    kernel = np.ones(window) / window
    left = window // 2
    right = window - left - 1
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def discover_runs(base: Path, limit: int | None) -> list[Path]:
    runs = sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=natural_run_key,
    )
    return runs if limit is None else runs[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/pipeline_runs"))
    parser.add_argument("--runs", type=int, default=None, help="max run_N directories to plot (default: all)")
    parser.add_argument("--all", action="store_true", help="plot all discovered run_N directories (same as omitting --runs)")
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("models/pipeline_timeline.png"))
    parser.add_argument("--no-raw", action="store_true", help="hide faint unsmoothed lines")
    args = parser.parse_args()

    if not args.base.exists():
        raise SystemExit(f"Missing pipeline directory: {args.base}")

    run_dirs = discover_runs(args.base, args.runs)
    if not run_dirs:
        raise SystemExit(f"No run_* directories found under {args.base}")

    # Load all data: data[run_name][stage][metric] = np.ndarray
    # Also load optional static sub-metrics for coagent records.
    runs: list[str] = []
    data: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for run_dir in run_dirs:
        run_name = run_dir.name
        run_data: dict[str, dict[str, np.ndarray]] = {}
        for stage in STAGES:
            history_path = run_dir / stage / "training_history.json"
            if not history_path.exists():
                print(f"warning: missing {history_path}")
                continue
            history = load_history(history_path)
            if not history:
                print(f"warning: empty {history_path}")
                continue
            metrics_dict = {
                metric: np.array([float(row.get(metric, np.nan)) for row in history])
                for metric, _ in METRICS
            }
            # Carry optional hybrid sub-metrics written only during coagent training.
            for key in ("static_evader_score", "static_found_rate"):
                vals = [row.get(key) for row in history]
                if any(v is not None for v in vals):
                    metrics_dict[key] = np.array([float(v) if v is not None else np.nan for v in vals])
            run_data[stage] = metrics_dict
        if run_data:
            runs.append(run_name)
            data[run_name] = run_data

    if not runs:
        raise SystemExit("No run data found to plot.")

    print(f"Loaded {len(runs)} run(s): {', '.join(runs)}")

    n_metrics = len(METRICS)
    n_stages = len(STAGES)
    fig, axes = plt.subplots(
        n_metrics, n_stages,
        figsize=(14, 10),
        sharex="col",
        squeeze=False,
    )
    fig.suptitle("Pipeline Training History — Static → Co-agent", fontsize=14, fontweight="bold")

    palette = plt.cm.tab10.colors

    for col, stage in enumerate(STAGES):
        axes[0, col].set_title(STAGE_TITLES[stage], fontsize=11)

        # Map coagent metrics to their static sub-metric counterpart (if any).
        static_overlay = {
            "evader_score": "static_evader_score",
            "found_rate":   "static_found_rate",
        }

        for row, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, col]
            overlay_key = static_overlay.get(metric) if stage == "coagent_stage" else None

            for i, run_name in enumerate(runs):
                stage_data = data[run_name].get(stage)
                if stage_data is None or metric not in stage_data:
                    continue
                y = stage_data[metric]
                x = np.arange(len(y))
                color = palette[i % len(palette)]
                label = run_name if col == 0 else None

                if not args.no_raw:
                    ax.plot(x, y, color=color, alpha=0.18, linewidth=0.8)
                ax.plot(x, smooth(y, args.smooth), color=color, linewidth=1.6, label=label)

                # Overlay static sub-metric as a dashed line in the coagent column.
                if overlay_key and overlay_key in stage_data:
                    ys = stage_data[overlay_key]
                    valid = ~np.isnan(ys)
                    if valid.any():
                        xs = np.arange(len(ys))[valid]
                        ys_valid = ys[valid]
                        if not args.no_raw:
                            ax.plot(xs, ys_valid, color=color, alpha=0.12, linewidth=0.7, linestyle="--")
                        ax.plot(xs, smooth(ys_valid, args.smooth), color=color,
                                linewidth=1.2, linestyle="--", alpha=0.7,
                                label=(f"{run_name} (vs static)" if col == 1 and i == 0 else None))

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            if metric == "found_rate":
                ax.set_ylim(-0.03, 1.03)
            if overlay_key:
                ax.set_ylabel(f"{ylabel}\n(solid=ML, dashed=static)")

        axes[-1, col].set_xlabel("Generation")

    # Single legend on the left column, top panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="best", fontsize=7, framealpha=0.75)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"Saved -> {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
