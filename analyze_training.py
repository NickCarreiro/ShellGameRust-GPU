#!/usr/bin/env python3
"""
Analyze ShellGame pipeline training histories for failures and suspicious runs.

Designed for output from:
  ./train_static_then_coagent.sh 3

Checks include:
  - missing run/stage/history files
  - missing or duplicate generation records
  - non-finite metric values
  - suspiciously flat metrics
  - found-rate collapse/saturation
  - score regressions and stage-to-stage handoff anomalies
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


STAGES = ("static_stage", "coagent_stage")
METRICS = ("searcher_score", "evader_score", "found_rate", "avg_attempts")
SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "failure": 3}


@dataclass
class Issue:
    severity: str
    run: str
    stage: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageSummary:
    run: str
    stage: str
    path: str
    records: int
    first_generation: int | None
    last_generation: int | None
    metrics: dict[str, dict[str, float]]


def natural_run_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.name.split("_", 1)[1]), path.name)
    except (IndexError, ValueError):
        return (10**9, path.name)


def severity_rank(severity: str) -> int:
    return SEVERITY_ORDER.get(severity, 99)


def add_issue(
    issues: list[Issue],
    severity: str,
    run: str,
    stage: str,
    message: str,
    **detail: Any,
) -> None:
    issues.append(Issue(severity, run, stage, message, detail))


def load_history(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("history root is not a list")
    return data


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
        return {"min": math.nan, "max": math.nan, "first": math.nan, "last": math.nan, "mean": math.nan, "delta": math.nan}
    return {
        "min": min(values),
        "max": max(values),
        "first": values[0],
        "last": values[-1],
        "mean": mean(values),
        "delta": values[-1] - values[0],
    }


def analyze_stage(
    run_name: str,
    stage: str,
    path: Path,
    expected_generations: int | None,
    issues: list[Issue],
) -> StageSummary | None:
    if not path.exists():
        add_issue(issues, "failure", run_name, stage, "missing training_history.json", path=str(path))
        return None

    try:
        history = load_history(path)
    except Exception as exc:
        add_issue(issues, "failure", run_name, stage, "could not parse training history", path=str(path), error=str(exc))
        return None

    if not history:
        add_issue(issues, "failure", run_name, stage, "training history is empty", path=str(path))
        return StageSummary(run_name, stage, str(path), 0, None, None, {})

    generations: list[int] = []
    metric_values: dict[str, list[float]] = {metric: [] for metric in METRICS}

    for idx, record in enumerate(history):
        if not isinstance(record, dict):
            add_issue(issues, "failure", run_name, stage, "history record is not an object", index=idx)
            continue

        gen = record.get("generation")
        if not isinstance(gen, int):
            add_issue(issues, "failure", run_name, stage, "record has invalid generation", index=idx, value=gen)
        else:
            generations.append(gen)

        for metric in METRICS:
            value = finite_float(record.get(metric))
            if value is None:
                add_issue(issues, "failure", run_name, stage, f"record has invalid {metric}", index=idx, value=record.get(metric))
            else:
                metric_values[metric].append(value)

    if generations:
        expected = list(range(generations[0], generations[0] + len(generations)))
        if generations != expected:
            missing = sorted(set(range(min(generations), max(generations) + 1)) - set(generations))
            duplicates = sorted(g for g in set(generations) if generations.count(g) > 1)
            add_issue(
                issues,
                "failure",
                run_name,
                stage,
                "generation sequence has gaps, duplicates, or ordering issues",
                first=generations[0],
                last=generations[-1],
                missing=missing[:20],
                duplicates=duplicates[:20],
            )

        if generations[0] != 0:
            add_issue(issues, "warning", run_name, stage, "history does not start at generation 0", first=generations[0])

        if expected_generations is not None:
            expected_records = expected_generations + 1
            if len(history) < expected_records:
                add_issue(
                    issues,
                    "warning",
                    run_name,
                    stage,
                    "stage appears incomplete",
                    records=len(history),
                    expected_records=expected_records,
                )
            elif len(history) > expected_records:
                add_issue(
                    issues,
                    "warning",
                    run_name,
                    stage,
                    "stage has more records than expected",
                    records=len(history),
                    expected_records=expected_records,
                )

    stats = {metric: metric_stats(values) for metric, values in metric_values.items()}
    diagnose_metrics(run_name, stage, metric_values, stats, issues)

    return StageSummary(
        run=run_name,
        stage=stage,
        path=str(path),
        records=len(history),
        first_generation=generations[0] if generations else None,
        last_generation=generations[-1] if generations else None,
        metrics=stats,
    )


def diagnose_metrics(
    run_name: str,
    stage: str,
    values: dict[str, list[float]],
    stats: dict[str, dict[str, float]],
    issues: list[Issue],
) -> None:
    found = values.get("found_rate", [])
    attempts = values.get("avg_attempts", [])
    searcher = values.get("searcher_score", [])
    evader = values.get("evader_score", [])

    if found:
        if any(v < -1e-9 or v > 1.0 + 1e-9 for v in found):
            add_issue(issues, "failure", run_name, stage, "found_rate outside [0, 1]", min=min(found), max=max(found))

        tail = found[-min(5, len(found)) :]
        if len(tail) >= 3 and max(tail) <= 0.02:
            add_issue(
                issues,
                "warning",
                run_name,
                stage,
                "found_rate collapsed near zero in the tail",
                tail_mean=mean(tail),
                interpretation="evader may be overpowering searchers or searcher reward may lack gradient",
            )
        if len(tail) >= 3 and min(tail) >= 0.98:
            add_issue(
                issues,
                "warning",
                run_name,
                stage,
                "found_rate saturated near one in the tail",
                tail_mean=mean(tail),
                interpretation="searcher may be overpowering evader or evader policy collapsed",
            )

    for metric, series in values.items():
        if len(series) >= 5:
            span = max(series) - min(series)
            if span < 1e-9:
                add_issue(issues, "warning", run_name, stage, f"{metric} is perfectly flat", value=series[0])

    if attempts and len(attempts) >= 3:
        if min(attempts) <= 0.0:
            add_issue(issues, "failure", run_name, stage, "avg_attempts is non-positive", min=min(attempts))
        tail_attempts = attempts[-min(5, len(attempts)) :]
        if max(tail_attempts) - min(tail_attempts) < 1e-9:
            add_issue(issues, "info", run_name, stage, "avg_attempts flat in tail", value=tail_attempts[-1])

    if searcher and evader and len(searcher) == len(evader):
        if stats["searcher_score"]["delta"] < -50:
            add_issue(
                issues,
                "warning",
                run_name,
                stage,
                "searcher score regressed sharply",
                delta=stats["searcher_score"]["delta"],
            )
        if stats["evader_score"]["delta"] < -50:
            add_issue(
                issues,
                "warning",
                run_name,
                stage,
                "evader score regressed sharply",
                delta=stats["evader_score"]["delta"],
            )


def analyze_handoff(run_name: str, summaries: dict[str, StageSummary], issues: list[Issue]) -> None:
    static = summaries.get("static_stage")
    coagent = summaries.get("coagent_stage")
    if not static or not coagent:
        return

    static_end_evader = static.metrics.get("evader_score", {}).get("last")
    coagent_start_evader = coagent.metrics.get("evader_score", {}).get("first")
    if math.isfinite(static_end_evader) and math.isfinite(coagent_start_evader):
        delta = coagent_start_evader - static_end_evader
        if abs(delta) > 75:
            add_issue(
                issues,
                "warning",
                run_name,
                "handoff",
                "large evader score jump between static end and coagent start",
                static_end=static_end_evader,
                coagent_start=coagent_start_evader,
                delta=delta,
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
        return [(name, value, (value - mean(nums)) / spread) for name, value in finite if abs((value - mean(nums)) / spread) >= 2.5]
    return [
        (name, value, 0.6745 * (value - med) / mad)
        for name, value in finite
        if abs(0.6745 * (value - med) / mad) >= 3.5
    ]


def analyze_cross_run(summaries: list[StageSummary], issues: list[Issue]) -> None:
    by_stage: dict[str, list[StageSummary]] = {stage: [] for stage in STAGES}
    by_run: dict[str, dict[str, StageSummary]] = {}
    for summary in summaries:
        by_stage.setdefault(summary.stage, []).append(summary)
        by_run.setdefault(summary.run, {})[summary.stage] = summary

    for stage, stage_summaries in by_stage.items():
        if len(stage_summaries) < 2:
            continue

        for metric in METRICS:
            finals = [
                (summary.run, summary.metrics.get(metric, {}).get("last", math.nan))
                for summary in stage_summaries
            ]
            deltas = [
                (summary.run, summary.metrics.get(metric, {}).get("delta", math.nan))
                for summary in stage_summaries
            ]
            final_nums = [value for _, value in finals if math.isfinite(value)]
            delta_nums = [value for _, value in deltas if math.isfinite(value)]

            if len(final_nums) >= 2:
                spread = max(final_nums) - min(final_nums)
                avg = mean(final_nums)
                relative_spread = spread / max(abs(avg), 1.0)
                if metric == "found_rate" and spread >= 0.25:
                    add_issue(
                        issues,
                        "warning",
                        "cross_run",
                        stage,
                        "found_rate final values vary widely across runs",
                        values=dict(finals),
                        spread=spread,
                        interpretation="training is seed-sensitive or one run diverged",
                    )
                elif metric != "found_rate" and relative_spread >= 0.75 and spread >= 25:
                    add_issue(
                        issues,
                        "warning",
                        "cross_run",
                        stage,
                        f"{metric} final values vary widely across runs",
                        values=dict(finals),
                        spread=spread,
                        relative_spread=relative_spread,
                    )

            for run_name, value, score in robust_outliers(finals):
                add_issue(
                    issues,
                    "warning",
                    run_name,
                    stage,
                    f"{metric} final value is a cross-run outlier",
                    value=value,
                    robust_z=score,
                    all_values=dict(finals),
                )

            for run_name, value, score in robust_outliers(deltas):
                add_issue(
                    issues,
                    "warning",
                    run_name,
                    stage,
                    f"{metric} training delta is a cross-run outlier",
                    value=value,
                    robust_z=score,
                    all_deltas=dict(deltas),
                )

            if len(delta_nums) >= 2:
                positive = sum(1 for value in delta_nums if value > 0)
                negative = sum(1 for value in delta_nums if value < 0)
                if positive and negative and min(positive, negative) == 1 and len(delta_nums) >= 3:
                    add_issue(
                        issues,
                        "info",
                        "cross_run",
                        stage,
                        f"{metric} improves in some runs and regresses in others",
                        deltas=dict(deltas),
                        interpretation="compare seeds and generated trees for sensitivity",
                    )

        tail_found = []
        for summary in stage_summaries:
            found_last = summary.metrics.get("found_rate", {}).get("last", math.nan)
            tail_found.append((summary.run, found_last))
        finite_tail_found = [value for _, value in tail_found if math.isfinite(value)]
        if len(finite_tail_found) >= 3:
            if max(finite_tail_found) <= 0.05:
                add_issue(
                    issues,
                    "warning",
                    "cross_run",
                    stage,
                    "all runs ended with near-zero found_rate",
                    values=dict(tail_found),
                    interpretation="evader may be too strong, searcher signal too weak, or attempt budget too tight",
                )
            if min(finite_tail_found) >= 0.95:
                add_issue(
                    issues,
                    "warning",
                    "cross_run",
                    stage,
                    "all runs ended with near-one found_rate",
                    values=dict(tail_found),
                    interpretation="searcher may be too strong or evader training collapsed",
                )

    if len(by_run) >= 2:
        handoff_deltas: list[tuple[str, float]] = []
        for run_name, stages in by_run.items():
            static = stages.get("static_stage")
            coagent = stages.get("coagent_stage")
            if not static or not coagent:
                continue
            static_evader = static.metrics.get("evader_score", {}).get("last", math.nan)
            coagent_evader = coagent.metrics.get("evader_score", {}).get("first", math.nan)
            if math.isfinite(static_evader) and math.isfinite(coagent_evader):
                handoff_deltas.append((run_name, coagent_evader - static_evader))

        if len(handoff_deltas) >= 2:
            nums = [value for _, value in handoff_deltas]
            if max(nums) - min(nums) >= 50:
                add_issue(
                    issues,
                    "warning",
                    "cross_run",
                    "handoff",
                    "static-to-coagent evader score jump varies widely across runs",
                    deltas=dict(handoff_deltas),
                    interpretation="resume handoff may be inconsistent or coagent initial opponent differs strongly by seed",
                )


def discover_runs(base: Path, expected_runs: int | None) -> list[Path]:
    if not base.exists():
        raise SystemExit(f"Missing pipeline directory: {base}")
    runs = sorted((p for p in base.iterdir() if p.is_dir() and p.name.startswith("run_")), key=natural_run_key)
    return runs[:expected_runs] if expected_runs is not None else runs


def print_report(summaries: list[StageSummary], issues: list[Issue]) -> int:
    by_severity = {severity: 0 for severity in SEVERITY_ORDER}
    for issue in issues:
        by_severity[issue.severity] = by_severity.get(issue.severity, 0) + 1

    print("\nSummary")
    print(f"  stages analyzed: {len(summaries)}")
    print(
        "  issues: "
        + ", ".join(f"{severity}={by_severity.get(severity, 0)}" for severity in ("failure", "warning", "info"))
    )

    print("\nStage Metrics")
    for summary in summaries:
        evader = summary.metrics.get("evader_score", {})
        searcher = summary.metrics.get("searcher_score", {})
        found = summary.metrics.get("found_rate", {})
        attempts = summary.metrics.get("avg_attempts", {})
        print(
            f"  {summary.run}/{summary.stage}: records={summary.records} "
            f"gen={summary.first_generation}->{summary.last_generation} "
            f"evader={evader.get('first', math.nan):.2f}->{evader.get('last', math.nan):.2f} "
            f"searcher={searcher.get('first', math.nan):.2f}->{searcher.get('last', math.nan):.2f} "
            f"found={found.get('first', math.nan):.3f}->{found.get('last', math.nan):.3f} "
            f"attempts={attempts.get('first', math.nan):.2f}->{attempts.get('last', math.nan):.2f}"
        )

    if issues:
        print("\nDiagnoses")
        for issue in sorted(issues, key=lambda i: (-severity_rank(i.severity), i.run, i.stage, i.message)):
            detail = f" {issue.detail}" if issue.detail else ""
            print(f"  {issue.severity.upper()} {issue.run}/{issue.stage}: {issue.message}{detail}")
    else:
        print("\nDiagnoses")
        print("  OK: no failures, warnings, or notable anomalies detected")

    return 2 if by_severity.get("failure", 0) else 1 if by_severity.get("warning", 0) else 0


def write_json_report(path: Path, summaries: list[StageSummary], issues: list[Issue]) -> None:
    payload = {
        "summaries": [summary.__dict__ for summary in summaries],
        "issues": [issue.__dict__ for issue in issues],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote JSON report -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("models/pipeline_runs"))
    parser.add_argument("--runs", type=int, default=3, help="expected number of run_N directories")
    parser.add_argument("--all", action="store_true", help="analyze all discovered run_N directories")
    parser.add_argument("--static-generations", type=int, default=None)
    parser.add_argument("--coagent-generations", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit non-zero on warnings as well as failures")
    args = parser.parse_args()

    expected_runs = None if args.all else args.runs
    run_dirs = discover_runs(args.base, expected_runs)
    if not run_dirs:
        raise SystemExit(f"No run_* directories found under {args.base}")

    issues: list[Issue] = []
    summaries: list[StageSummary] = []

    if expected_runs is not None:
        expected_names = {f"run_{i}" for i in range(1, expected_runs + 1)}
        present_names = {p.name for p in run_dirs}
        missing = sorted(expected_names - present_names, key=lambda name: int(name.split("_")[1]))
        for run_name in missing:
            add_issue(
                issues,
                "failure",
                run_name,
                "pipeline",
                "missing expected run directory",
                base=str(args.base),
            )

    for run_dir in run_dirs:
        per_run: dict[str, StageSummary] = {}
        for stage in STAGES:
            expected_generations = (
                args.static_generations if stage == "static_stage" else args.coagent_generations
            )
            summary = analyze_stage(
                run_dir.name,
                stage,
                run_dir / stage / "training_history.json",
                expected_generations,
                issues,
            )
            if summary is not None:
                summaries.append(summary)
                per_run[stage] = summary
        analyze_handoff(run_dir.name, per_run, issues)

    analyze_cross_run(summaries, issues)

    exit_code = print_report(summaries, issues)
    if args.json_out:
        write_json_report(args.json_out, summaries, issues)

    if args.strict:
        raise SystemExit(exit_code)
    raise SystemExit(2 if any(issue.severity == "failure" for issue in issues) else 0)


if __name__ == "__main__":
    main()
