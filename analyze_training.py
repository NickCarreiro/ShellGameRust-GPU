#!/usr/bin/env python3
"""
Analyze ShellGame training histories and surface actionable anomalies.

This script now scans model directory structures automatically. It supports:
  - staged runs:     models/*/run_N/{static_stage,coagent_stage}/training_history.json
  - checkpoints:     models/checkpoints/vN/training_history.json
  - core archives:   models/core/*/training_history.json
  - generic layouts: any directory containing training_history.json

Examples:
  python analyze_training.py
  python analyze_training.py --show
  python analyze_training.py --no-visuals
  python analyze_training.py --roots models --json-out models/analysis_report.json
  python analyze_training.py --runs 4 --strict
  python analyze_training.py --scan-only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from training_layout import (
    HistoryTarget,
    natural_label_key,
    scan_histories,
    summarize_targets,
    to_display_path,
)


PRIMARY_METRICS = (
    "searcher_score",
    "evader_score",
    "found_rate",
    "avg_attempts",
)
OPTIONAL_METRICS = (
    "evader_fitness_std",
    "searcher_fitness_std",
    "static_evader_score",
    "static_found_rate",
)
SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "failure": 3}


@dataclass
class Issue:
    severity: str
    experiment: str
    stage: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class HistorySummary:
    experiment: str
    group: str
    family: str
    stage: str
    structure: str
    path: str
    records: int
    first_generation: int | None
    last_generation: int | None
    metrics: dict[str, dict[str, float]]


def severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.get(severity, 99)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def metric_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": math.nan,
            "max": math.nan,
            "first": math.nan,
            "last": math.nan,
            "mean": math.nan,
            "delta": math.nan,
            "std": math.nan,
        }
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "first": values[0],
        "last": values[-1],
        "mean": avg,
        "delta": values[-1] - values[0],
        "std": math.sqrt(variance),
    }


def add_issue(
    issues: list[Issue],
    severity: str,
    experiment: str,
    stage: str,
    message: str,
    **detail: Any,
) -> None:
    issues.append(Issue(severity=severity, experiment=experiment, stage=stage, message=message, detail=detail))


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("history root is not a list")
    return payload


def _tail(values: list[float], width: int = 5) -> list[float]:
    if not values:
        return []
    return values[-min(width, len(values)) :]


def _check_generation_sequence(
    target: HistoryTarget,
    generations: list[int],
    expected_generations: int | None,
    records: int,
    issues: list[Issue],
) -> tuple[int | None, int | None]:
    if not generations:
        return None, None

    if generations != sorted(generations):
        add_issue(
            issues,
            "failure",
            target.experiment,
            target.stage,
            "generation sequence is not monotonically non-decreasing",
            first=generations[0],
            last=generations[-1],
        )

    unique = sorted(set(generations))
    if unique:
        expected_seq = list(range(unique[0], unique[-1] + 1))
        missing = sorted(set(expected_seq) - set(generations))
        duplicates = sorted(gen for gen in unique if generations.count(gen) > 1)
        if missing or duplicates:
            add_issue(
                issues,
                "failure",
                target.experiment,
                target.stage,
                "generation sequence has gaps or duplicates",
                first=unique[0],
                last=unique[-1],
                missing=missing[:20],
                duplicates=duplicates[:20],
            )

    if generations[0] != 0:
        add_issue(
            issues,
            "warning",
            target.experiment,
            target.stage,
            "history does not start at generation 0",
            first=generations[0],
        )

    if expected_generations is not None:
        expected_records = expected_generations + 1
        if records < expected_records:
            add_issue(
                issues,
                "warning",
                target.experiment,
                target.stage,
                "history appears incomplete",
                records=records,
                expected_records=expected_records,
            )
        elif records > expected_records:
            add_issue(
                issues,
                "warning",
                target.experiment,
                target.stage,
                "history has more records than expected",
                records=records,
                expected_records=expected_records,
            )

    return generations[0], generations[-1]


def diagnose_metrics(
    target: HistoryTarget,
    values: dict[str, list[float]],
    stats: dict[str, dict[str, float]],
    issues: list[Issue],
) -> None:
    found = values.get("found_rate", [])
    static_found = values.get("static_found_rate", [])
    attempts = values.get("avg_attempts", [])
    searcher = values.get("searcher_score", [])
    evader = values.get("evader_score", [])

    def _check_probability(metric_name: str, series: list[float]) -> None:
        if series and any(value < -1e-9 or value > 1.0 + 1e-9 for value in series):
            add_issue(
                issues,
                "failure",
                target.experiment,
                target.stage,
                f"{metric_name} outside [0, 1]",
                min=min(series),
                max=max(series),
            )

    _check_probability("found_rate", found)
    _check_probability("static_found_rate", static_found)

    for metric in ("evader_fitness_std", "searcher_fitness_std"):
        series = values.get(metric, [])
        if series and min(series) < -1e-9:
            add_issue(
                issues,
                "failure",
                target.experiment,
                target.stage,
                f"{metric} contains negative values",
                min=min(series),
            )

    found_tail = _tail(found)
    if len(found_tail) >= 3 and max(found_tail) <= 0.02:
        add_issue(
            issues,
            "warning",
            target.experiment,
            target.stage,
            "found_rate collapsed near zero in the tail",
            tail_mean=mean(found_tail),
            interpretation="evader may be overpowering searchers or searcher reward is weak",
        )
    if len(found_tail) >= 3 and min(found_tail) >= 0.98:
        add_issue(
            issues,
            "warning",
            target.experiment,
            target.stage,
            "found_rate saturated near one in the tail",
            tail_mean=mean(found_tail),
            interpretation="searcher may be overpowering evader or evader policy collapsed",
        )

    for metric in PRIMARY_METRICS:
        series = values.get(metric, [])
        if len(series) >= 5 and max(series) - min(series) < 1e-9:
            add_issue(
                issues,
                "warning",
                target.experiment,
                target.stage,
                f"{metric} is perfectly flat",
                value=series[-1],
            )

    if attempts and min(attempts) <= 0.0:
        add_issue(
            issues,
            "failure",
            target.experiment,
            target.stage,
            "avg_attempts is non-positive",
            min=min(attempts),
        )

    if stats.get("searcher_score", {}).get("delta", 0.0) < -50:
        add_issue(
            issues,
            "warning",
            target.experiment,
            target.stage,
            "searcher score regressed sharply",
            delta=stats["searcher_score"]["delta"],
        )
    if stats.get("evader_score", {}).get("delta", 0.0) < -50:
        add_issue(
            issues,
            "warning",
            target.experiment,
            target.stage,
            "evader score regressed sharply",
            delta=stats["evader_score"]["delta"],
        )

    if searcher and evader and len(searcher) == len(evader):
        advantage = [ev - sr for ev, sr in zip(evader, searcher)]
        if advantage[-1] - advantage[0] < -80:
            add_issue(
                issues,
                "info",
                target.experiment,
                target.stage,
                "evader advantage narrowed strongly over training",
                first=advantage[0],
                last=advantage[-1],
                delta=advantage[-1] - advantage[0],
            )
        if advantage[-1] - advantage[0] > 80:
            add_issue(
                issues,
                "info",
                target.experiment,
                target.stage,
                "evader advantage widened strongly over training",
                first=advantage[0],
                last=advantage[-1],
                delta=advantage[-1] - advantage[0],
            )

    if found and static_found:
        gap = found[-1] - static_found[-1]
        if abs(gap) >= 0.2:
            add_issue(
                issues,
                "info",
                target.experiment,
                target.stage,
                "final found_rate differs strongly from static_found_rate",
                found_rate=found[-1],
                static_found_rate=static_found[-1],
                gap=gap,
            )


def analyze_history(
    target: HistoryTarget,
    expected_generations: int | None,
    issues: list[Issue],
) -> HistorySummary | None:
    path = target.history_path
    if not path.exists():
        add_issue(
            issues,
            "failure",
            target.experiment,
            target.stage,
            "missing training_history.json",
            path=to_display_path(path),
        )
        return None

    try:
        history = load_history(path)
    except Exception as exc:  # noqa: BLE001 - preserve parse error detail in report
        add_issue(
            issues,
            "failure",
            target.experiment,
            target.stage,
            "could not parse training history",
            path=to_display_path(path),
            error=str(exc),
        )
        return None

    if not history:
        add_issue(
            issues,
            "failure",
            target.experiment,
            target.stage,
            "training history is empty",
            path=to_display_path(path),
        )
        return HistorySummary(
            experiment=target.experiment,
            group=target.group,
            family=target.family,
            stage=target.stage,
            structure=target.structure,
            path=to_display_path(path),
            records=0,
            first_generation=None,
            last_generation=None,
            metrics={},
        )

    generations: list[int] = []
    values: dict[str, list[float]] = defaultdict(list)

    for idx, record in enumerate(history):
        if not isinstance(record, dict):
            add_issue(
                issues,
                "failure",
                target.experiment,
                target.stage,
                "history record is not an object",
                index=idx,
            )
            continue

        generation = record.get("generation")
        if not isinstance(generation, int):
            add_issue(
                issues,
                "failure",
                target.experiment,
                target.stage,
                "record has invalid generation",
                index=idx,
                value=generation,
            )
        else:
            generations.append(generation)

        for metric in PRIMARY_METRICS:
            value = finite_float(record.get(metric))
            if value is None:
                add_issue(
                    issues,
                    "failure",
                    target.experiment,
                    target.stage,
                    f"record has invalid {metric}",
                    index=idx,
                    value=record.get(metric),
                )
            else:
                values[metric].append(value)

        for metric in OPTIONAL_METRICS:
            raw = record.get(metric)
            if raw is None:
                continue
            value = finite_float(raw)
            if value is None:
                add_issue(
                    issues,
                    "warning",
                    target.experiment,
                    target.stage,
                    f"record has non-numeric {metric}",
                    index=idx,
                    value=raw,
                )
            else:
                values[metric].append(value)

    first_gen, last_gen = _check_generation_sequence(
        target=target,
        generations=generations,
        expected_generations=expected_generations,
        records=len(history),
        issues=issues,
    )

    stats = {metric: metric_stats(series) for metric, series in values.items()}
    diagnose_metrics(target, values, stats, issues)

    return HistorySummary(
        experiment=target.experiment,
        group=target.group,
        family=target.family,
        stage=target.stage,
        structure=target.structure,
        path=to_display_path(path),
        records=len(history),
        first_generation=first_gen,
        last_generation=last_gen,
        metrics=stats,
    )


def median(values: list[float]) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def robust_outliers(values: list[tuple[str, float]]) -> list[tuple[str, float, float]]:
    finite = [(name, value) for name, value in values if math.isfinite(value)]
    if len(finite) < 3:
        return []

    nums = [value for _, value in finite]
    med = median(nums)
    deviations = [abs(value - med) for value in nums]
    mad = median(deviations)

    if mad <= 1e-12:
        spread = sample_std(nums)
        if spread <= 1e-12:
            return []
        avg = mean(nums)
        return [
            (name, value, (value - avg) / spread)
            for name, value in finite
            if abs((value - avg) / spread) >= 2.5
        ]

    return [
        (name, value, 0.6745 * (value - med) / mad)
        for name, value in finite
        if abs(0.6745 * (value - med) / mad) >= 3.5
    ]


def analyze_handoffs(summaries: list[HistorySummary], issues: list[Issue]) -> None:
    by_group: dict[str, dict[str, HistorySummary]] = defaultdict(dict)
    for summary in summaries:
        by_group[summary.group][summary.stage] = summary

    for group, stage_map in by_group.items():
        static = stage_map.get("static_stage")
        coagent = stage_map.get("coagent_stage")
        if not static or not coagent:
            continue

        static_end_evader = static.metrics.get("evader_score", {}).get("last", math.nan)
        coagent_start_evader = coagent.metrics.get("evader_score", {}).get("first", math.nan)
        if math.isfinite(static_end_evader) and math.isfinite(coagent_start_evader):
            delta = coagent_start_evader - static_end_evader
            if abs(delta) > 75:
                add_issue(
                    issues,
                    "warning",
                    group,
                    "handoff",
                    "large evader score jump between static end and coagent start",
                    static_end=static_end_evader,
                    coagent_start=coagent_start_evader,
                    delta=delta,
                )

        static_end_found = static.metrics.get("found_rate", {}).get("last", math.nan)
        coagent_start_found = coagent.metrics.get("found_rate", {}).get("first", math.nan)
        if math.isfinite(static_end_found) and math.isfinite(coagent_start_found):
            delta = coagent_start_found - static_end_found
            if abs(delta) > 0.35:
                add_issue(
                    issues,
                    "info",
                    group,
                    "handoff",
                    "large found_rate jump between static end and coagent start",
                    static_end=static_end_found,
                    coagent_start=coagent_start_found,
                    delta=delta,
                )


def analyze_cross_run(summaries: list[HistorySummary], issues: list[Issue]) -> None:
    by_stage: dict[str, list[HistorySummary]] = defaultdict(list)
    for summary in summaries:
        by_stage[summary.stage].append(summary)

    for stage, stage_summaries in by_stage.items():
        if len(stage_summaries) < 2:
            continue

        for metric in PRIMARY_METRICS:
            finals = [
                (summary.experiment, summary.metrics.get(metric, {}).get("last", math.nan))
                for summary in stage_summaries
            ]
            deltas = [
                (summary.experiment, summary.metrics.get(metric, {}).get("delta", math.nan))
                for summary in stage_summaries
            ]
            final_vals = [value for _, value in finals if math.isfinite(value)]
            delta_vals = [value for _, value in deltas if math.isfinite(value)]

            if len(final_vals) >= 2:
                spread = max(final_vals) - min(final_vals)
                avg = mean(final_vals)
                relative_spread = spread / max(abs(avg), 1.0)

                if metric == "found_rate" and spread >= 0.25:
                    add_issue(
                        issues,
                        "warning",
                        "cross_run",
                        stage,
                        "found_rate final values vary widely across histories",
                        values=dict(finals),
                        spread=spread,
                        interpretation="training may be seed-sensitive or one run diverged",
                    )
                elif metric != "found_rate" and relative_spread >= 0.75 and spread >= 25:
                    add_issue(
                        issues,
                        "warning",
                        "cross_run",
                        stage,
                        f"{metric} final values vary widely across histories",
                        values=dict(finals),
                        spread=spread,
                        relative_spread=relative_spread,
                    )

            for run_name, value, robust_z in robust_outliers(finals):
                add_issue(
                    issues,
                    "warning",
                    run_name,
                    stage,
                    f"{metric} final value is a cross-run outlier",
                    value=value,
                    robust_z=robust_z,
                    all_values=dict(finals),
                )

            for run_name, value, robust_z in robust_outliers(deltas):
                add_issue(
                    issues,
                    "warning",
                    run_name,
                    stage,
                    f"{metric} training delta is a cross-run outlier",
                    value=value,
                    robust_z=robust_z,
                    all_deltas=dict(deltas),
                )

            if len(delta_vals) >= 3:
                positive = sum(1 for value in delta_vals if value > 0)
                negative = sum(1 for value in delta_vals if value < 0)
                if positive and negative and min(positive, negative) == 1:
                    add_issue(
                        issues,
                        "info",
                        "cross_run",
                        stage,
                        f"{metric} improves in some histories and regresses in others",
                        deltas=dict(deltas),
                        interpretation="compare seeds, opponent pools, and tree generation settings",
                    )


def _final_metric(summary: HistorySummary, metric: str) -> float:
    return summary.metrics.get(metric, {}).get("last", math.nan)


def _delta_metric(summary: HistorySummary, metric: str) -> float:
    return summary.metrics.get(metric, {}).get("delta", math.nan)


def _format_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def print_scan_report(targets: list[HistoryTarget]) -> None:
    scan_summary = summarize_targets(targets)
    print("Structure Scan")
    print(f"  discovered histories: {scan_summary['totals']['histories']}")

    by_structure = scan_summary["by_structure"]
    if by_structure:
        rendered = ", ".join(f"{key}={by_structure[key]}" for key in sorted(by_structure))
        print(f"  structures: {rendered}")

    by_stage = scan_summary["by_stage"]
    if by_stage:
        rendered = ", ".join(f"{key}={by_stage[key]}" for key in sorted(by_stage))
        print(f"  stages: {rendered}")

    by_family = scan_summary["by_family"]
    if by_family:
        rendered = ", ".join(f"{key}={by_family[key]}" for key in sorted(by_family))
        print(f"  families: {rendered}")

    print("  examples:")
    for target in targets[:8]:
        print(f"    - {target.experiment}/{target.stage} -> {to_display_path(target.history_path)}")


def print_leaderboards(summaries: list[HistorySummary]) -> None:
    by_stage: dict[str, list[HistorySummary]] = defaultdict(list)
    for summary in summaries:
        by_stage[summary.stage].append(summary)

    print("\nInsights")
    for stage in sorted(by_stage):
        stage_summaries = by_stage[stage]
        if not stage_summaries:
            continue
        print(f"  {stage}:")

        evader_rank = sorted(
            stage_summaries,
            key=lambda summary: _final_metric(summary, "evader_score"),
            reverse=True,
        )[:3]
        print("    top final evader_score:")
        for summary in evader_rank:
            value = _final_metric(summary, "evader_score")
            print(f"      {summary.experiment}: {_format_float(value, 2)}")

        found_rank = sorted(
            stage_summaries,
            key=lambda summary: _final_metric(summary, "found_rate"),
        )[:3]
        print("    lowest final found_rate:")
        for summary in found_rank:
            value = _final_metric(summary, "found_rate")
            print(f"      {summary.experiment}: {_format_float(value, 3)}")

        delta_rank = sorted(
            stage_summaries,
            key=lambda summary: _delta_metric(summary, "evader_score"),
            reverse=True,
        )[:3]
        print("    largest evader_score delta:")
        for summary in delta_rank:
            value = _delta_metric(summary, "evader_score")
            print(f"      {summary.experiment}: {_format_float(value, 2)}")


def print_report(summaries: list[HistorySummary], issues: list[Issue]) -> int:
    by_severity = {severity: 0 for severity in SEVERITY_ORDER}
    for issue in issues:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1

    print("\nSummary")
    print(f"  histories analyzed: {len(summaries)}")
    print(
        "  issues: "
        + ", ".join(
            f"{severity}={by_severity.get(severity, 0)}"
            for severity in ("failure", "warning", "info")
        )
    )

    print("\nHistory Metrics")
    for summary in summaries:
        evader = summary.metrics.get("evader_score", {})
        searcher = summary.metrics.get("searcher_score", {})
        found = summary.metrics.get("found_rate", {})
        attempts = summary.metrics.get("avg_attempts", {})
        print(
            f"  {summary.experiment}/{summary.stage}: records={summary.records} "
            f"gen={summary.first_generation}->{summary.last_generation} "
            f"evader={_format_float(evader.get('first', math.nan), 2)}->{_format_float(evader.get('last', math.nan), 2)} "
            f"searcher={_format_float(searcher.get('first', math.nan), 2)}->{_format_float(searcher.get('last', math.nan), 2)} "
            f"found={_format_float(found.get('first', math.nan), 3)}->{_format_float(found.get('last', math.nan), 3)} "
            f"attempts={_format_float(attempts.get('first', math.nan), 2)}->{_format_float(attempts.get('last', math.nan), 2)}"
        )

    print_leaderboards(summaries)

    print("\nDiagnoses")
    if not issues:
        print("  OK: no failures, warnings, or notable anomalies detected")
    else:
        for issue in sorted(
            issues,
            key=lambda current: (
                -severity_rank(current.severity),
                natural_label_key(current.experiment),
                current.stage,
                current.message,
            ),
        ):
            detail = f" {issue.detail}" if issue.detail else ""
            print(f"  {issue.severity.upper()} {issue.experiment}/{issue.stage}: {issue.message}{detail}")

    return 2 if by_severity.get("failure", 0) else 1 if by_severity.get("warning", 0) else 0


def write_json_report(
    path: Path,
    roots: list[Path],
    targets: list[HistoryTarget],
    summaries: list[HistorySummary],
    issues: list[Issue],
) -> None:
    payload = {
        "roots": [to_display_path(root) for root in roots],
        "scan": summarize_targets(targets),
        "targets": [
            {
                "history_path": to_display_path(target.history_path),
                "root": to_display_path(target.root),
                "experiment": target.experiment,
                "group": target.group,
                "family": target.family,
                "stage": target.stage,
                "structure": target.structure,
            }
            for target in targets
        ],
        "summaries": [summary.__dict__ for summary in summaries],
        "issues": [issue.__dict__ for issue in issues],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote JSON report -> {to_display_path(path)}")


def _filter_targets(targets: list[HistoryTarget], runs_limit: int | None, include_all_runs: bool) -> list[HistoryTarget]:
    if runs_limit is None or include_all_runs:
        return targets

    staged = [target for target in targets if target.structure == "run_stage"]
    non_staged = [target for target in targets if target.structure != "run_stage"]

    run_groups = sorted({target.group for target in staged}, key=natural_label_key)
    allowed_groups = set(run_groups[:runs_limit])

    return [
        target
        for target in targets
        if target.structure != "run_stage" or target.group in allowed_groups
    ]


def _add_expected_run_issues(targets: list[HistoryTarget], expected_runs: int, issues: list[Issue]) -> None:
    run_targets = [target for target in targets if target.structure == "run_stage"]
    by_family: dict[str, set[str]] = defaultdict(set)
    for target in run_targets:
        by_family[target.family].add(target.group.split("/", 1)[-1])

    for family, run_names in sorted(by_family.items()):
        expected_names = {f"run_{i}" for i in range(1, expected_runs + 1)}
        missing_names = sorted(expected_names - run_names, key=natural_label_key)
        for missing in missing_names:
            add_issue(
                issues,
                "failure",
                f"{family}/{missing}",
                "pipeline",
                "missing expected run directory",
                expected_runs=expected_runs,
            )


def write_visualizations(
    targets: list[HistoryTarget],
    *,
    out_path: Path,
    insights_out: Path | None,
    mode: str,
    runs_limit: int | None,
    smooth_window: int,
    max_series: int | None,
    no_raw: bool,
    show: bool,
) -> None:
    try:
        from plot_timeline import (
            build_data,
            filter_targets,
            plot_insights,
            plot_timeline,
        )
    except Exception as exc:  # noqa: BLE001 - keep analysis usable if plotting deps are missing
        print(f"\nVisualization skipped: could not import plotting helpers: {exc}")
        return

    plot_mode = "all" if mode == "auto" else mode
    plot_targets = filter_targets(targets, mode=plot_mode, runs_limit=runs_limit)
    if not plot_targets:
        print("\nVisualization skipped: no histories matched the selected plot filter.")
        return

    data, warnings = build_data(plot_targets)
    if warnings:
        print("\nVisualization warnings")
        for warning in warnings:
            print(f"  - {warning}")

    if insights_out is None:
        insights_out = out_path.with_name(f"{out_path.stem}_insights{out_path.suffix}")

    print("\nVisualizations")
    try:
        plot_timeline(
            targets=plot_targets,
            data=data,
            smooth_window=max(1, smooth_window),
            out_path=out_path,
            no_raw=no_raw,
            show=False,
            max_series=max_series,
        )
        plot_insights(
            targets=plot_targets,
            data=data,
            out_path=insights_out,
            show=False,
        )
        if show:
            open_visualization_windows([out_path, insights_out])
    except Exception as exc:  # noqa: BLE001 - report cleanly instead of hiding analysis output
        print(f"Visualization failed: {exc}")


def should_auto_show_windows() -> bool:
    if os.environ.get("ANALYZE_TRAINING_NO_WINDOW") == "1":
        return False
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return sys.stdout.isatty() and has_display


def open_visualization_windows(paths: list[Path]) -> None:
    opener = shutil.which("xdg-open")
    for path in paths:
        resolved = path.resolve()
        try:
            if opener:
                subprocess.Popen(
                    [opener, str(resolved)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            else:
                webbrowser.open(resolved.as_uri())
            print(f"Opened window -> {to_display_path(resolved)}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not open window for {to_display_path(resolved)}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("models")])
    parser.add_argument("--base", type=Path, default=None, help="legacy alias for a single root")
    parser.add_argument("--runs", type=int, default=None, help="limit discovered run_N groups")
    parser.add_argument("--all", action="store_true", help="include all run_N groups")
    parser.add_argument("--expect-runs", type=int, default=None, help="flag missing run_1..run_N directories")
    parser.add_argument("--scan-only", action="store_true", help="only print discovered structures")
    parser.add_argument("--static-generations", type=int, default=None)
    parser.add_argument("--coagent-generations", type=int, default=None)
    parser.add_argument("--single-generations", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit non-zero on warnings as well as failures")
    parser.add_argument("--no-visuals", action="store_true", help="skip timeline and insights PNG generation")
    parser.add_argument("--plot-mode", choices=("auto", "all", "staged", "single"), default="auto")
    parser.add_argument("--plot-out", type=Path, default=Path("models/training_timeline.png"))
    parser.add_argument("--insights-out", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=10, help="smoothing window for generated plots")
    parser.add_argument("--max-series", type=int, default=None, help="max plotted series per stage")
    parser.add_argument("--no-raw", action="store_true", help="hide faint unsmoothed plot lines")
    parser.add_argument(
        "--show",
        "--window",
        dest="show",
        action="store_true",
        default=None,
        help="open generated visualizations in image windows after saving",
    )
    parser.add_argument("--no-window", dest="show", action="store_false", help="save plots without opening windows")
    args = parser.parse_args()
    show_windows = should_auto_show_windows() if args.show is None else args.show

    roots = [args.base] if args.base is not None else args.roots
    targets = scan_histories(roots, recursive=True)

    if not targets:
        roots_display = ", ".join(to_display_path(root.resolve()) for root in roots)
        raise SystemExit(f"No training_history.json files found under: {roots_display}")

    targets = _filter_targets(targets, runs_limit=args.runs, include_all_runs=args.all)

    print_scan_report(targets)
    if args.scan_only:
        raise SystemExit(0)

    issues: list[Issue] = []
    if args.expect_runs is not None:
        _add_expected_run_issues(targets, expected_runs=args.expect_runs, issues=issues)

    summaries: list[HistorySummary] = []

    for target in targets:
        if target.stage == "static_stage":
            expected_generations = args.static_generations
        elif target.stage == "coagent_stage":
            expected_generations = args.coagent_generations
        else:
            expected_generations = args.single_generations

        summary = analyze_history(
            target=target,
            expected_generations=expected_generations,
            issues=issues,
        )
        if summary is not None:
            summaries.append(summary)

    analyze_handoffs(summaries, issues)
    analyze_cross_run(summaries, issues)

    exit_code = print_report(summaries, issues)
    if args.json_out:
        write_json_report(path=args.json_out, roots=roots, targets=targets, summaries=summaries, issues=issues)

    if not args.no_visuals:
        write_visualizations(
            targets=targets,
            out_path=args.plot_out,
            insights_out=args.insights_out,
            mode=args.plot_mode,
            runs_limit=args.runs,
            smooth_window=args.smooth,
            max_series=args.max_series,
            no_raw=args.no_raw,
            show=show_windows,
        )

    if args.strict:
        raise SystemExit(exit_code)
    raise SystemExit(2 if any(issue.severity == "failure" for issue in issues) else 0)


if __name__ == "__main__":
    main()
