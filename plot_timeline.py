#!/usr/bin/env python3
"""
Visualize ShellGame training histories across scanned directory structures.

The script auto-discovers histories in layouts such as:
  - models/*/run_N/{static_stage,coagent_stage}/training_history.json
  - models/checkpoints/vN/training_history.json
  - models/core/*/training_history.json

It produces:
  1) A multi-panel timeline plot with staged and single-stage columns.
  2) A compact insights scatter plot (final and delta metric relationships).

Examples:
  python plot_timeline.py
  python plot_timeline.py --roots models --runs 6 --smooth 12
  python plot_timeline.py --mode staged --out models/staged_timeline.png
  python plot_timeline.py --scan-only
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from training_layout import (
    HistoryTarget,
    STAGED_STAGES,
    natural_label_key,
    scan_histories,
    summarize_targets,
    to_display_path,
)


PLOT_METRICS = [
    ("evader_score", "Evader Score"),
    ("searcher_score", "Searcher Score"),
    ("found_rate", "Found Rate"),
    ("avg_attempts", "Avg Attempts"),
    ("advantage", "Evader Advantage"),
]
OPTIONAL_METRICS = ("static_evader_score", "static_found_rate", "evader_fitness_std", "searcher_fitness_std")
STAGE_TITLES = {
    "static_stage": "Static Stage",
    "coagent_stage": "Coagent Stage",
    "single_stage": "Single-Stage Histories",
}


def load_history(path: Path) -> list[dict]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{path} did not contain a JSON list")
    return payload


def _to_float_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _extract_metric(history: list[dict], metric: str) -> np.ndarray:
    if metric == "advantage":
        values = []
        for record in history:
            evader = _to_float_or_nan(record.get("evader_score"))
            searcher = _to_float_or_nan(record.get("searcher_score"))
            values.append(evader - searcher if math.isfinite(evader) and math.isfinite(searcher) else float("nan"))
        return np.array(values, dtype=float)

    return np.array([_to_float_or_nan(record.get(metric)) for record in history], dtype=float)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()

    arr = np.asarray(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 2:
        return arr.copy()

    idx = np.arange(len(arr))
    filled = arr.copy()
    filled[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])

    window = min(window, len(arr))
    kernel = np.ones(window, dtype=float) / window
    left = window // 2
    right = window - left - 1
    padded = np.pad(filled, (left, right), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[~valid] = np.nan
    return smoothed


def first_last_delta(values: np.ndarray) -> tuple[float, float, float]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return math.nan, math.nan, math.nan
    first = float(valid[0])
    last = float(valid[-1])
    return first, last, last - first


def print_scan_report(targets: list[HistoryTarget]) -> None:
    summary = summarize_targets(targets)
    print("Structure Scan")
    print(f"  discovered histories: {summary['totals']['histories']}")
    print("  structures: " + ", ".join(f"{name}={count}" for name, count in sorted(summary["by_structure"].items())))
    print("  stages: " + ", ".join(f"{name}={count}" for name, count in sorted(summary["by_stage"].items())))
    print("  families: " + ", ".join(f"{name}={count}" for name, count in sorted(summary["by_family"].items())))
    print("  examples:")
    for target in targets[:8]:
        print(f"    - {target.experiment}/{target.stage} -> {to_display_path(target.history_path)}")


def filter_targets(targets: list[HistoryTarget], mode: str, runs_limit: int | None) -> list[HistoryTarget]:
    if mode == "staged":
        filtered = [target for target in targets if target.structure == "run_stage"]
    elif mode == "single":
        filtered = [target for target in targets if target.structure != "run_stage"]
    else:
        filtered = targets[:]

    if runs_limit is None:
        return filtered

    run_stage_targets = [target for target in filtered if target.structure == "run_stage"]
    non_run_targets = [target for target in filtered if target.structure != "run_stage"]

    groups = sorted({target.group for target in run_stage_targets}, key=natural_label_key)
    allowed = set(groups[:runs_limit])

    return [
        target
        for target in run_stage_targets
        if target.group in allowed
    ] + non_run_targets


def build_data(targets: list[HistoryTarget]) -> tuple[dict[tuple[str, str], dict[str, np.ndarray]], list[str]]:
    data: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    warnings: list[str] = []

    for target in targets:
        key = (target.experiment, target.stage)
        try:
            history = load_history(target.history_path)
        except Exception as exc:  # noqa: BLE001 - present readable warning and continue
            warnings.append(f"failed to parse {to_display_path(target.history_path)}: {exc}")
            continue

        if not history:
            warnings.append(f"empty history: {to_display_path(target.history_path)}")
            continue

        metrics = {metric: _extract_metric(history, metric) for metric, _ in PLOT_METRICS}
        for metric in OPTIONAL_METRICS:
            metrics[metric] = _extract_metric(history, metric)

        data[key] = metrics

    return data, warnings


def short_name(experiment: str) -> str:
    parts = experiment.split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return experiment


def plot_timeline(
    targets: list[HistoryTarget],
    data: dict[tuple[str, str], dict[str, np.ndarray]],
    smooth_window: int,
    out_path: Path,
    no_raw: bool,
    show: bool,
    max_series: int | None,
) -> None:
    if not data:
        raise SystemExit("No valid history data found to plot.")

    stage_order = [stage for stage in (*STAGED_STAGES, "single_stage") if any(target.stage == stage for target in targets)]
    if not stage_order:
        raise SystemExit("No stages discovered to plot.")

    experiments = sorted({target.experiment for target in targets}, key=natural_label_key)
    palette = plt.cm.tab20(np.linspace(0, 1, max(len(experiments), 2)))
    colors = {experiment: palette[index % len(palette)] for index, experiment in enumerate(experiments)}

    n_rows = len(PLOT_METRICS)
    n_cols = len(stage_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 2.2 * n_rows),
        sharex="col",
        squeeze=False,
    )

    fig.suptitle("ShellGame Training Timeline (Auto-Scanned)", fontsize=14, fontweight="bold")

    legend_handles = []
    legend_labels: list[str] = []

    for col, stage in enumerate(stage_order):
        stage_targets = [
            target
            for target in sorted(targets, key=lambda current: natural_label_key(current.experiment))
            if target.stage == stage and (target.experiment, target.stage) in data
        ]
        if max_series is not None:
            stage_targets = stage_targets[:max_series]

        axes[0, col].set_title(STAGE_TITLES.get(stage, stage), fontsize=11)

        overlay_map = {
            "evader_score": "static_evader_score",
            "found_rate": "static_found_rate",
        }

        for row, (metric, ylabel) in enumerate(PLOT_METRICS):
            ax = axes[row, col]
            finals: list[float] = []

            for target in stage_targets:
                key = (target.experiment, target.stage)
                series = data[key][metric]
                if not np.isfinite(series).any():
                    continue

                x = np.arange(len(series))
                color = colors[target.experiment]
                label = short_name(target.experiment) if row == 0 else None

                if not no_raw:
                    ax.plot(x, series, color=color, alpha=0.17, linewidth=0.8)

                smoothed = smooth(series, smooth_window)
                line, = ax.plot(x, smoothed, color=color, linewidth=1.6, label=label)
                if row == 0 and label and label not in legend_labels:
                    legend_labels.append(label)
                    legend_handles.append(line)

                finite_values = series[np.isfinite(series)]
                if len(finite_values):
                    finals.append(float(finite_values[-1]))

                overlay_key = overlay_map.get(metric) if stage == "coagent_stage" else None
                if overlay_key is not None:
                    overlay = data[key][overlay_key]
                    valid = np.isfinite(overlay)
                    if valid.any():
                        xs = np.arange(len(overlay))[valid]
                        ys = overlay[valid]
                        if not no_raw:
                            ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.7, linestyle="--")
                        ax.plot(
                            xs,
                            smooth(ys, smooth_window),
                            color=color,
                            linewidth=1.15,
                            linestyle="--",
                            alpha=0.72,
                        )

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

            if metric == "found_rate":
                ax.set_ylim(-0.03, 1.03)
            if metric == "advantage":
                ax.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.45)

            if finals:
                median_final = float(np.median(np.array(finals, dtype=float)))
                spread = float(max(finals) - min(finals))
                digits = 3 if metric == "found_rate" else 2
                ax.text(
                    0.01,
                    0.95,
                    f"n={len(finals)} med={median_final:.{digits}f} spread={spread:.{digits}f}",
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    alpha=0.82,
                )

        axes[-1, col].set_xlabel("Generation")

    if legend_handles:
        columns = min(4, max(1, len(legend_handles) // 4 + 1))
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=columns,
            fontsize=8,
            framealpha=0.8,
            bbox_to_anchor=(0.5, 0.995),
        )

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved timeline -> {to_display_path(out_path)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_insights(
    targets: list[HistoryTarget],
    data: dict[tuple[str, str], dict[str, np.ndarray]],
    out_path: Path,
    show: bool,
) -> None:
    stage_points: dict[str, list[tuple[str, float, float, float, float]]] = {stage: [] for stage in (*STAGED_STAGES, "single_stage")}

    for target in targets:
        key = (target.experiment, target.stage)
        if key not in data:
            continue

        ev_first, ev_last, ev_delta = first_last_delta(data[key]["evader_score"])
        fd_first, fd_last, fd_delta = first_last_delta(data[key]["found_rate"])

        if not (math.isfinite(ev_last) and math.isfinite(fd_last)):
            continue

        stage_points.setdefault(target.stage, []).append((target.experiment, fd_last, ev_last, fd_delta, ev_delta))

    has_points = any(stage_points[stage] for stage in stage_points)
    if not has_points:
        print("Skipping insights plot: no finite final evader_score/found_rate pairs.")
        return

    stage_colors = {
        "static_stage": "#1f77b4",
        "coagent_stage": "#d62728",
        "single_stage": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    ax_final = axes[0, 0]
    ax_delta = axes[0, 1]

    for stage in (*STAGED_STAGES, "single_stage"):
        points = stage_points.get(stage, [])
        if not points:
            continue

        found_final = np.array([item[1] for item in points], dtype=float)
        evader_final = np.array([item[2] for item in points], dtype=float)
        found_delta = np.array([item[3] for item in points], dtype=float)
        evader_delta = np.array([item[4] for item in points], dtype=float)

        color = stage_colors.get(stage, "#444444")
        label = STAGE_TITLES.get(stage, stage)

        ax_final.scatter(found_final, evader_final, s=48, alpha=0.78, label=label, color=color)
        ax_delta.scatter(found_delta, evader_delta, s=48, alpha=0.78, label=label, color=color)

        if len(points) <= 12:
            for experiment, fd_last, ev_last, fd_delta, ev_delta in points:
                label_text = short_name(experiment)
                ax_final.annotate(label_text, (fd_last, ev_last), fontsize=7, alpha=0.8)
                ax_delta.annotate(label_text, (fd_delta, ev_delta), fontsize=7, alpha=0.8)

    ax_final.set_title("Final Metric Relationship")
    ax_final.set_xlabel("Final Found Rate")
    ax_final.set_ylabel("Final Evader Score")
    ax_final.grid(True, alpha=0.25)

    ax_delta.set_title("Training Delta Relationship")
    ax_delta.set_xlabel("Found Rate Delta")
    ax_delta.set_ylabel("Evader Score Delta")
    ax_delta.axhline(0.0, color="#666666", linewidth=0.8, alpha=0.45)
    ax_delta.axvline(0.0, color="#666666", linewidth=0.8, alpha=0.45)
    ax_delta.grid(True, alpha=0.25)

    handles, labels = ax_final.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(handles)), fontsize=8, framealpha=0.8)

    fig.suptitle("ShellGame Training Insights")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print(f"Saved insights -> {to_display_path(out_path)}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("models")])
    parser.add_argument("--base", type=Path, default=None, help="legacy alias for a single root")
    parser.add_argument("--mode", choices=("auto", "all", "staged", "single"), default="auto")
    parser.add_argument("--runs", type=int, default=None, help="max staged run groups to plot")
    parser.add_argument("--max-series", type=int, default=None, help="max plotted series per stage")
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("models/training_timeline.png"))
    parser.add_argument("--insights-out", type=Path, default=None)
    parser.add_argument("--no-raw", action="store_true", help="hide faint unsmoothed lines")
    parser.add_argument("--scan-only", action="store_true", help="only print discovered structures")
    parser.add_argument("--show", action="store_true", help="show interactive matplotlib windows")
    args = parser.parse_args()

    roots = [args.base] if args.base is not None else args.roots
    targets = scan_histories(roots, recursive=True)
    if not targets:
        roots_display = ", ".join(to_display_path(root.resolve()) for root in roots)
        raise SystemExit(f"No training_history.json files found under: {roots_display}")

    print_scan_report(targets)
    if args.scan_only:
        raise SystemExit(0)

    mode = "all" if args.mode == "auto" else args.mode
    targets = filter_targets(targets, mode=mode, runs_limit=args.runs)
    if not targets:
        raise SystemExit("No histories matched the selected mode/filter.")

    data, warnings = build_data(targets)
    if warnings:
        print("Warnings")
        for warning in warnings:
            print(f"  - {warning}")

    by_stage = Counter(target.stage for target in targets)
    print("Plot Selection")
    print(f"  histories: {len(targets)}")
    print("  stages: " + ", ".join(f"{stage}={count}" for stage, count in sorted(by_stage.items())))

    plot_timeline(
        targets=targets,
        data=data,
        smooth_window=max(1, args.smooth),
        out_path=args.out,
        no_raw=args.no_raw,
        show=args.show,
        max_series=args.max_series,
    )

    insights_out = args.insights_out
    if insights_out is None:
        insights_out = args.out.with_name(f"{args.out.stem}_insights{args.out.suffix}")

    plot_insights(
        targets=targets,
        data=data,
        out_path=insights_out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
