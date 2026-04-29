#!/usr/bin/env python3
"""
Real-time ShellGame training monitor.

The monitor follows the files produced by train_iterate.sh:
  - training_history.json
  - train_attempt*.log / evaluate_attempt*.log
  - resource_logs/<checkpoint>_train_attempt*.csv
  - resource_logs/<checkpoint>_evaluate_attempt*.csv

Typical use:
  python analyze_training.py
  python analyze_training.py --watch
  python analyze_training.py --window
  python analyze_training.py --target models/core/my_checkpoint_iter1 --watch
  python analyze_training.py --list
  python analyze_training.py --once --json-out models/monitor_snapshot.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# This script is a read-only monitor — it never runs the model or needs a GPU.
# Hiding CUDA devices prevents accidental GPU access and keeps this process out
# of nvidia-smi's compute client list, which would otherwise block GPU recovery.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from training_layout import scan_histories, to_display_path


METRICS = (
    "generation",
    "found_rate",
    "static_found_rate",
    "avg_attempts",
    "avg_max_attempts",
    "survival_budget_ratio",
    "static_survival_budget_ratio",
    "evader_score",
    "searcher_score",
    "escape_score",
    "static_evader_score",
    "static_escape_score",
    "evader_fitness_std",
    "searcher_fitness_std",
    "population_size",
    "min_nodes",
    "max_nodes",
)

RESOURCE_FIELDS = (
    "gpu_util_pct",
    "gpu_mem_used_mib",
    "gpu_power_w",
    "gpu_graphics_clock_mhz",
    "trainer_cpu_pct",
    "trainer_mem_pct",
    "trainer_rss_kib",
)

LOG_PATTERNS = {
    "generation": re.compile(r"Generation\s+(\d+)\s*/\s*(\d+)"),
    "backend": re.compile(r"backend:\s*([A-Za-z0-9_-]+)"),
    "accelerator": re.compile(r"accelerator:\s*(.+)"),
    "gpu_batching": re.compile(r"gpu batching:\s*(.+)"),
    "resource_log": re.compile(r"resource log:\s*(.+)"),
}


@dataclass
class RunArtifacts:
    run_dir: Path
    label: str
    history_path: Path | None = None
    train_logs: list[Path] = field(default_factory=list)
    eval_logs: list[Path] = field(default_factory=list)
    health_logs: list[Path] = field(default_factory=list)
    resource_logs: list[Path] = field(default_factory=list)
    mtime: float = 0.0

    def latest_train_log(self) -> Path | None:
        return newest_file(self.train_logs)

    def latest_eval_log(self) -> Path | None:
        return newest_file(self.eval_logs)

    def latest_resource_log(self) -> Path | None:
        sampled = [path for path in self.resource_logs if csv_has_data_rows(path)]
        return newest_file(sampled) or newest_file(self.resource_logs)


@dataclass
class ResourceSummary:
    path: Path | None
    samples: int = 0
    last: dict[str, float | str] = field(default_factory=dict)
    maxes: dict[str, float] = field(default_factory=dict)
    recent_maxes: dict[str, float] = field(default_factory=dict)
    series: dict[str, list[float]] = field(default_factory=dict)
    timestamps: list[str] = field(default_factory=list)
    newest_timestamp: str | None = None


def newest_file(paths: Iterable[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def csv_has_data_rows(path: Path) -> bool:
    try:
        with path.open(newline="") as handle:
            next(handle, None)
            return next(handle, None) is not None
    except OSError:
        return False


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def fmt_float(value: Any, digits: int = 3, missing: str = "-") -> str:
    number = finite_float(value)
    if number is None:
        return missing
    return f"{number:.{digits}f}"


def fmt_pct(value: Any, digits: int = 1) -> str:
    number = finite_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}%"


def fmt_age(timestamp: float | None) -> str:
    if not timestamp:
        return "-"
    age = max(0.0, time.time() - timestamp)
    if age < 60:
        return f"{age:.0f}s ago"
    if age < 3600:
        return f"{age / 60:.1f}m ago"
    if age < 86400:
        return f"{age / 3600:.1f}h ago"
    return f"{age / 86400:.1f}d ago"


def fmt_path(path: Path | None) -> str:
    return to_display_path(path) if path else "-"


def ascii_sparkline(values: list[float], width: int = 34) -> str:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return "-"
    if len(clean) > width:
        stride = len(clean) / width
        sampled = [clean[int(i * stride)] for i in range(width)]
    else:
        sampled = clean

    lo = min(sampled)
    hi = max(sampled)
    if abs(hi - lo) < 1e-12:
        return "-" * len(sampled)

    ramp = " .:-=+*#%@"
    span = hi - lo
    out = []
    for value in sampled:
        idx = int(round((value - lo) / span * (len(ramp) - 1)))
        out.append(ramp[max(0, min(idx, len(ramp) - 1))])
    return "".join(out)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return clamp(numerator / denominator, 0.0, 1.0)


def bounded_escape_component(value: float | None) -> float | None:
    if value is None:
        return None
    # The trainer's escape score is intentionally near a 0..1000 scale:
    # escape_rate * 1000 + budget/reward tie-breakers.  Cap instead of
    # stretching so a single great static check cannot hide learned collapse.
    return clamp(value / 1000.0, 0.0, 1.0)


def record_training_effectiveness_score(record: dict[str, Any]) -> float | None:
    """Return a bounded 0..100 score for evader training quality.

    The score favors evaders that survive both learned and static searchers,
    spend most of the search budget, earn high escape scores, and do not show a
    huge learned/static matchup gap.  It is derived at analysis time so old
    histories can be graphed without rewriting their JSON files.
    """
    found_rate = finite_float(record.get("found_rate"))
    if found_rate is None:
        return None
    found_rate = clamp(found_rate, 0.0, 1.0)

    static_found_rate = finite_float(record.get("static_found_rate"))
    static_found_rate = found_rate if static_found_rate is None else clamp(static_found_rate, 0.0, 1.0)

    survival = finite_float(record.get("survival_budget_ratio"))
    if survival is None:
        survival = ratio_or_none(
            finite_float(record.get("avg_attempts")),
            finite_float(record.get("avg_max_attempts")),
        )
    survival = clamp(survival if survival is not None else 1.0 - found_rate, 0.0, 1.0)

    static_survival = finite_float(record.get("static_survival_budget_ratio"))
    static_survival = clamp(static_survival if static_survival is not None else survival, 0.0, 1.0)

    escape_component = bounded_escape_component(finite_float(record.get("escape_score")))
    if escape_component is None:
        evader_score = finite_float(record.get("evader_score")) or 0.0
        estimated_escape = (1.0 - found_rate) * 1000.0 + survival * 120.0 + evader_score * 0.05
        escape_component = bounded_escape_component(estimated_escape) or 0.0

    static_escape_component = bounded_escape_component(finite_float(record.get("static_escape_score")))
    if static_escape_component is None:
        static_escape_component = escape_component

    learned_resilience = 0.65 * (1.0 - found_rate) + 0.35 * survival
    static_resilience = 0.65 * (1.0 - static_found_rate) + 0.35 * static_survival
    matchup_stability = clamp(1.0 - abs(found_rate - static_found_rate) / 0.50, 0.0, 1.0)

    score = 100.0 * (
        0.45 * learned_resilience
        + 0.25 * static_resilience
        + 0.15 * escape_component
        + 0.08 * static_escape_component
        + 0.07 * matchup_stability
    )
    return clamp(score, 0.0, 100.0)


def training_effectiveness_series(history: list[dict[str, Any]]) -> list[float]:
    out: list[float] = []
    for record in history:
        score = record_training_effectiveness_score(record)
        if score is not None:
            out.append(score)
    return out


def snapshot_series(history: list[dict[str, Any]]) -> dict[str, list[float]]:
    series = {metric: metric_series(history, metric) for metric in METRICS}
    series["training_effectiveness_score"] = training_effectiveness_series(history)
    return series


def read_text_tail(path: Path | None, lines: int) -> list[str]:
    if path is None or not path.exists():
        return []
    tail: deque[str] = deque(maxlen=max(0, lines))
    try:
        with path.open("r", errors="replace") as handle:
            for line in handle:
                tail.append(line.rstrip("\n"))
    except OSError:
        return []
    return list(tail)


def read_text(path: Path | None, max_bytes: int = 256_000) -> str:
    if path is None or not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(max(0, size - max_bytes))
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


def load_history(path: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if path is None or not path.exists():
        return [], "missing training_history.json"
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - live files may be mid-write
        return [], f"could not read training_history.json: {exc}"
    if not isinstance(payload, list):
        return [], "training_history.json root is not a list"
    records = [record for record in payload if isinstance(record, dict)]
    if len(records) != len(payload):
        return records, "some training records are not objects"
    return records, None


def metric_series(history: list[dict[str, Any]], metric: str) -> list[float]:
    out: list[float] = []
    for record in history:
        value = finite_float(record.get(metric))
        if value is not None:
            out.append(value)
    return out


def latest_record(history: list[dict[str, Any]]) -> dict[str, Any]:
    return history[-1] if history else {}


def run_label(run_dir: Path) -> str:
    try:
        return str(run_dir.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(run_dir)


def collect_artifacts_for_dir(run_dir: Path, resource_roots: list[Path]) -> RunArtifacts:
    run_dir = run_dir.expanduser().resolve()
    history_path = run_dir / "training_history.json"
    if not history_path.exists():
        history_path = None

    train_logs = sorted(run_dir.glob("train_attempt*.log"))
    eval_logs = sorted(run_dir.glob("evaluate_attempt*.log"))
    health_logs = sorted(run_dir.glob("cuda_health_*attempt*.log"))

    resource_logs: list[Path] = []
    basename = run_dir.name
    resource_patterns = [
        f"{basename}_train_attempt*.csv",
        f"{basename}_evaluate_attempt*.csv",
    ]
    for resource_root in resource_roots:
        root = resource_root.expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        if root.exists():
            for pattern in resource_patterns:
                resource_logs.extend(sorted(root.glob(pattern)))

    resource_logs.extend(sorted(run_dir.glob("*train_attempt*.csv")))
    resource_logs.extend(sorted(run_dir.glob("*evaluate_attempt*.csv")))
    resource_logs = sorted(set(path.resolve() for path in resource_logs))

    all_paths = [path for path in [history_path] if path is not None]
    all_paths += train_logs + eval_logs + health_logs + resource_logs
    mtime = max((path.stat().st_mtime for path in all_paths if path.exists()), default=run_dir.stat().st_mtime if run_dir.exists() else 0.0)

    return RunArtifacts(
        run_dir=run_dir,
        label=run_label(run_dir),
        history_path=history_path,
        train_logs=train_logs,
        eval_logs=eval_logs,
        health_logs=health_logs,
        resource_logs=resource_logs,
        mtime=mtime,
    )


def roots_from_args(args: argparse.Namespace) -> list[Path]:
    if args.base is not None:
        return [args.base]
    return args.roots


def discover_runs(roots: list[Path], resource_roots: list[Path]) -> list[RunArtifacts]:
    dirs: set[Path] = set()

    targets = scan_histories(roots, recursive=True)
    dirs.update(target.history_path.parent.resolve() for target in targets)

    for root in roots:
        root = root.expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        if not root.exists():
            continue
        if root.is_file():
            dirs.add(root.parent.resolve())
            continue
        for pattern in ("train_attempt*.log", "evaluate_attempt*.log"):
            dirs.update(path.parent.resolve() for path in root.rglob(pattern))

    runs = [collect_artifacts_for_dir(run_dir, resource_roots) for run_dir in dirs]
    return sorted(runs, key=lambda run: run.mtime, reverse=True)


def resolve_target(args: argparse.Namespace) -> RunArtifacts | None:
    resource_roots = args.resource_roots
    target = args.target_opt or args.target
    if target is not None:
        path = target.expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.is_file():
            if path.name == "training_history.json" or path.suffix in {".log", ".csv"}:
                path = path.parent
            else:
                raise SystemExit(f"Target file is not a known training artifact: {to_display_path(path)}")
        if not path.exists():
            raise SystemExit(f"Target does not exist: {to_display_path(path)}")
        return collect_artifacts_for_dir(path, resource_roots)

    runs = discover_runs(roots_from_args(args), resource_roots)
    return runs[0] if runs else None


def parse_log_facts(artifact: RunArtifacts) -> dict[str, Any]:
    text = "\n".join(
        part
        for part in (
            read_text(artifact.latest_train_log()),
            read_text(artifact.latest_eval_log()),
        )
        if part
    )

    facts: dict[str, Any] = {
        "cuda_ready": "CUDA device 0 ready" in text or "GPU batch inference active" in text,
        "cuda_failed": bool(re.search(r"CUDA_ERROR|CUDA .*failed|cudarc|unspecified launch failure", text, re.I)),
        "cpu_forced": "CPU forced" in text or "SHELLGAME_FORCE_CPU" in text,
        "backend": None,
        "accelerator": None,
        "gpu_batching": None,
        "latest_generation_from_log": None,
        "expected_generations": None,
    }

    for match in LOG_PATTERNS["backend"].finditer(text):
        facts["backend"] = match.group(1)
    for match in LOG_PATTERNS["accelerator"].finditer(text):
        facts["accelerator"] = match.group(1).strip()
    for match in LOG_PATTERNS["gpu_batching"].finditer(text):
        facts["gpu_batching"] = match.group(1).strip()
    for match in LOG_PATTERNS["generation"].finditer(text):
        facts["latest_generation_from_log"] = int(match.group(1))
        facts["expected_generations"] = int(match.group(2))

    return facts


def read_resource_summary(path: Path | None, recent_samples: int) -> ResourceSummary:
    summary = ResourceSummary(path=path)
    if path is None or not path.exists():
        return summary

    recent: deque[dict[str, float]] = deque(maxlen=max(1, recent_samples))
    maxes = {field: 0.0 for field in RESOURCE_FIELDS}
    series = {field: [] for field in RESOURCE_FIELDS}
    timestamps: list[str] = []
    last_raw: dict[str, str] = {}

    try:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                summary.samples += 1
                last_raw = {key: value for key, value in row.items() if key is not None}
                numeric_row: dict[str, float] = {}
                for field in RESOURCE_FIELDS:
                    value = finite_float(row.get(field))
                    if value is None:
                        continue
                    numeric_row[field] = value
                    series[field].append(value)
                    maxes[field] = max(maxes.get(field, 0.0), value)
                timestamp = str(row.get("timestamp") or "")
                if timestamp:
                    timestamps.append(timestamp)
                if numeric_row:
                    recent.append(numeric_row)
    except OSError:
        return summary

    summary.last = dict(last_raw)
    summary.maxes = maxes
    summary.series = series
    summary.timestamps = timestamps
    recent_maxes = {field: 0.0 for field in RESOURCE_FIELDS}
    for row in recent:
        for field, value in row.items():
            recent_maxes[field] = max(recent_maxes.get(field, 0.0), value)
    summary.recent_maxes = recent_maxes
    summary.newest_timestamp = str(last_raw.get("timestamp") or "") or None
    return summary


def active_training_processes() -> list[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", r"ml_self_play|train_iterate\.sh"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    lines = []
    for line in result.stdout.splitlines():
        if "analyze_training.py" in line:
            continue
        if "pgrep -af" in line:
            continue
        lines.append(line.strip())
    return lines


def active_process_env_flags(process_lines: list[str]) -> dict[str, dict[str, str]]:
    interesting = {
        "RESOURCE_LOGGING",
        "RESOURCE_LOG_INTERVAL",
        "RESOURCE_LOG_DIR",
        "USE_CUDA",
        "SHELLGAME_FORCE_CPU",
        "CUDA_LAUNCH_BLOCKING",
        "REQUIRE_CUDA",
    }
    flags: dict[str, dict[str, str]] = {}
    for line in process_lines:
        raw_pid = line.split(maxsplit=1)[0] if line else ""
        if not raw_pid.isdigit():
            continue
        env_path = Path("/proc") / raw_pid / "environ"
        try:
            payload = env_path.read_bytes().decode(errors="replace")
        except OSError:
            continue
        values: dict[str, str] = {}
        for item in payload.split("\0"):
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            if key in interesting:
                values[key] = value
        if values:
            flags[raw_pid] = values
    return flags


def build_alerts(
    artifact: RunArtifacts,
    history: list[dict[str, Any]],
    history_error: str | None,
    resource: ResourceSummary,
    log_facts: dict[str, Any],
    active_processes: list[str],
    target_active_processes: list[str],
    env_flags: dict[str, dict[str, str]],
    stale_seconds: float,
) -> list[tuple[str, str]]:
    alerts: list[tuple[str, str]] = []

    if history_error:
        alerts.append(("failure", history_error))

    active = bool(target_active_processes)
    if active and artifact.history_path and artifact.history_path.exists():
        age = time.time() - artifact.history_path.stat().st_mtime
        if age > stale_seconds:
            alerts.append(("warning", f"history has not changed for {age:.0f}s while training process is active"))

    if log_facts.get("cuda_failed"):
        alerts.append(("failure", "latest logs contain CUDA failure markers"))

    if log_facts.get("cpu_forced"):
        alerts.append(("warning", "logs indicate CPU mode or forced CPU fallback"))

    cuda_expected = bool(log_facts.get("cuda_ready") or log_facts.get("backend") == "cuda")
    target_pids = {line.split(maxsplit=1)[0] for line in target_active_processes if line.split(maxsplit=1)[0].isdigit()}
    resource_logging_disabled = any(
        flags.get("RESOURCE_LOGGING") == "0"
        for pid, flags in env_flags.items()
        if pid in target_pids
    )
    if cuda_expected and resource.path is None:
        if resource_logging_disabled:
            alerts.append(("warning", "CUDA appears enabled, but RESOURCE_LOGGING=0 disabled resource CSV capture"))
        elif active:
            alerts.append(("warning", "CUDA appears enabled but no resource CSV was found for this run"))
        else:
            alerts.append(("info", "no resource CSV found for this historical run"))

    if active and cuda_expected and resource.samples > 0:
        recent_gpu = resource.recent_maxes.get("gpu_util_pct", 0.0)
        recent_power = resource.recent_maxes.get("gpu_power_w", 0.0)
        if recent_gpu <= 0.0 and recent_power < 8.0:
            alerts.append(("warning", "recent resource samples show no GPU burst; check whether this phase is CPU-bound"))

    if history:
        last = latest_record(history)
        found = finite_float(last.get("found_rate"))
        static_found = finite_float(last.get("static_found_rate"))
        avg_attempts = finite_float(last.get("avg_attempts"))
        if found is not None and not 0.0 <= found <= 1.0:
            alerts.append(("failure", f"found_rate outside [0, 1]: {found:.3f}"))
        if static_found is not None and not 0.0 <= static_found <= 1.0:
            alerts.append(("failure", f"static_found_rate outside [0, 1]: {static_found:.3f}"))
        if avg_attempts is not None and avg_attempts <= 0:
            alerts.append(("failure", f"avg_attempts is non-positive: {avg_attempts:.3f}"))

        tail_found = metric_series(history, "found_rate")[-5:]
        if len(tail_found) >= 3 and max(tail_found) <= 0.02:
            alerts.append(("warning", "found_rate tail is near zero; searcher may be overmatched"))
        if len(tail_found) >= 3 and min(tail_found) >= 0.98:
            alerts.append(("warning", "found_rate tail is saturated; evader may be collapsing"))

        if found is not None and static_found is not None and abs(found - static_found) >= 0.25:
            alerts.append(("info", f"learned/static found_rate gap is {found - static_found:+.3f}"))

    if not alerts:
        alerts.append(("ok", "no active alerts"))
    return alerts


def make_snapshot(args: argparse.Namespace, artifact: RunArtifacts) -> dict[str, Any]:
    history, history_error = load_history(artifact.history_path)
    record = dict(latest_record(history))
    effectiveness = record_training_effectiveness_score(record)
    if effectiveness is not None:
        record["training_effectiveness_score"] = effectiveness
    resource = read_resource_summary(artifact.latest_resource_log(), args.recent_samples)
    active = active_training_processes()
    target_active = [line for line in active if str(artifact.run_dir) in line]
    env_flags = active_process_env_flags(active)
    log_facts = parse_log_facts(artifact)
    expected_generations = args.expected_generations or log_facts.get("expected_generations")

    generation = record.get("generation")
    if generation is None:
        generation = log_facts.get("latest_generation_from_log")

    alerts = build_alerts(
        artifact=artifact,
        history=history,
        history_error=history_error,
        resource=resource,
        log_facts=log_facts,
        active_processes=active,
        target_active_processes=target_active,
        env_flags=env_flags,
        stale_seconds=args.stale_seconds,
    )

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target": {
            "label": artifact.label,
            "run_dir": fmt_path(artifact.run_dir),
            "history": fmt_path(artifact.history_path),
            "latest_train_log": fmt_path(artifact.latest_train_log()),
            "latest_eval_log": fmt_path(artifact.latest_eval_log()),
            "latest_resource_log": fmt_path(resource.path),
            "mtime": artifact.mtime,
        },
        "state": {
            "active_processes": active,
            "target_active_processes": target_active,
            "active_env_flags": env_flags,
            "history_error": history_error,
            "records": len(history),
            "generation": generation,
            "expected_generations": expected_generations,
            "backend": log_facts.get("backend"),
            "accelerator": log_facts.get("accelerator"),
            "gpu_batching": log_facts.get("gpu_batching"),
            "cuda_ready": log_facts.get("cuda_ready"),
            "cpu_forced": log_facts.get("cpu_forced"),
            "cuda_failed": log_facts.get("cuda_failed"),
        },
        "latest_record": record,
        "series": snapshot_series(history),
        "resource": {
            "path": fmt_path(resource.path),
            "samples": resource.samples,
            "last": resource.last,
            "maxes": resource.maxes,
            "recent_maxes": resource.recent_maxes,
            "series": resource.series,
            "timestamps": resource.timestamps,
            "newest_timestamp": resource.newest_timestamp,
        },
        "logs": {
            "train_tail": read_text_tail(artifact.latest_train_log(), args.tail),
            "eval_tail": read_text_tail(artifact.latest_eval_log(), args.tail),
        },
        "alerts": [{"severity": severity, "message": message} for severity, message in alerts],
    }


def metric_line(snapshot: dict[str, Any], metric: str, digits: int = 3) -> str:
    record = snapshot["latest_record"]
    series = snapshot["series"].get(metric, [])
    current = finite_float(record.get(metric))
    first = series[0] if series else None
    delta = current - first if current is not None and first is not None else None
    trend = ascii_sparkline(series)
    return f"{metric:28} {fmt_float(current, digits):>10} {fmt_float(delta, digits):>10}  {trend}"


def resource_value(resource: dict[str, Any], section: str, key: str) -> Any:
    return resource.get(section, {}).get(key)


def aligned_binary_series(
    left: list[float],
    right: list[float],
    op: Any,
) -> list[float]:
    count = min(len(left), len(right))
    out: list[float] = []
    for idx in range(count):
        try:
            value = op(left[idx], right[idx])
        except ZeroDivisionError:
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def graph_series_from_snapshot(snapshot: dict[str, Any]) -> list[tuple[str, list[float], str]]:
    series = snapshot.get("series", {})
    resource = snapshot.get("resource", {}).get("series", {})
    charts: list[tuple[str, list[float], str]] = []

    def add(label: str, values: list[float] | None, group: str) -> None:
        clean = [value for value in values or [] if math.isfinite(value)]
        if clean:
            charts.append((label, clean, group))

    add("training_effectiveness_score", series.get("training_effectiveness_score"), "Derived")

    for metric in METRICS:
        add(metric, series.get(metric), "Training")

    add(
        "learned_minus_static_found_rate",
        aligned_binary_series(series.get("found_rate", []), series.get("static_found_rate", []), lambda a, b: a - b),
        "Derived",
    )
    add(
        "evader_minus_searcher_score",
        aligned_binary_series(series.get("evader_score", []), series.get("searcher_score", []), lambda a, b: a - b),
        "Derived",
    )
    add(
        "escape_minus_static_escape",
        aligned_binary_series(series.get("escape_score", []), series.get("static_escape_score", []), lambda a, b: a - b),
        "Derived",
    )
    add(
        "attempt_budget_used",
        aligned_binary_series(series.get("avg_attempts", []), series.get("avg_max_attempts", []), lambda a, b: a / b if b else math.nan),
        "Derived",
    )
    add(
        "survival_budget_gap",
        aligned_binary_series(
            series.get("survival_budget_ratio", []),
            series.get("static_survival_budget_ratio", []),
            lambda a, b: a - b,
        ),
        "Derived",
    )
    add(
        "fitness_std_gap",
        aligned_binary_series(
            series.get("evader_fitness_std", []),
            series.get("searcher_fitness_std", []),
            lambda a, b: a - b,
        ),
        "Derived",
    )

    for field in RESOURCE_FIELDS:
        add(f"resource/{field}", resource.get(field), "Resource")

    return charts


def render_snapshot(snapshot: dict[str, Any], args: argparse.Namespace, *, clear: bool) -> str:
    target = snapshot["target"]
    state = snapshot["state"]
    resource = snapshot["resource"]
    record = snapshot["latest_record"]

    lines: list[str] = []
    if clear:
        lines.append("\033[2J\033[H")

    lines.append(f"ShellGame Training Monitor    {snapshot['generated_at']}")
    lines.append("=" * 78)
    lines.append(f"target:      {target['label']}")
    lines.append(f"history:     {target['history']} ({fmt_age(Path(target['history']).stat().st_mtime if target['history'] != '-' and Path(target['history']).exists() else None)})")
    lines.append(f"train log:   {target['latest_train_log']}")
    lines.append(f"eval log:    {target['latest_eval_log']}")
    lines.append(f"resource:    {target['latest_resource_log']}")
    lines.append("")

    gen = state.get("generation")
    expected = state.get("expected_generations")
    if gen is not None and expected:
        progress = f"{gen}/{expected} ({(float(gen) / max(float(expected), 1.0)) * 100:.1f}%)"
    else:
        progress = str(gen) if gen is not None else "-"

    active = state.get("active_processes") or []
    target_active = state.get("target_active_processes") or []
    lines.append("State")
    lines.append(f"  active processes: {len(active)}")
    lines.append(f"  target processes: {len(target_active)}")
    for process in active[:3]:
        lines.append(f"    {process}")
    env_flags = state.get("active_env_flags") or {}
    for pid, flags in list(env_flags.items())[:3]:
        rendered_flags = " ".join(f"{key}={value}" for key, value in sorted(flags.items()))
        lines.append(f"    env[{pid}]: {rendered_flags}")
    lines.append(f"  records:          {state.get('records', 0)}")
    lines.append(f"  generation:       {progress}")
    lines.append(f"  population:       {record.get('population_size', '-')}")
    lines.append(f"  nodes:            {record.get('min_nodes', '-') }..{record.get('max_nodes', '-')}")
    lines.append(f"  backend:          {state.get('backend') or '-'}")
    lines.append(f"  accelerator:      {state.get('accelerator') or '-'}")
    lines.append(f"  gpu batching:     {state.get('gpu_batching') or '-'}")
    lines.append("")

    lines.append("Metrics                    current      delta   trend")
    lines.append("-" * 78)
    for metric, digits in (
        ("training_effectiveness_score", 2),
        ("found_rate", 3),
        ("static_found_rate", 3),
        ("avg_attempts", 2),
        ("avg_max_attempts", 2),
        ("survival_budget_ratio", 3),
        ("evader_score", 2),
        ("searcher_score", 2),
        ("escape_score", 2),
        ("static_escape_score", 2),
        ("evader_fitness_std", 2),
        ("searcher_fitness_std", 2),
    ):
        if snapshot["series"].get(metric) or metric in record:
            lines.append(metric_line(snapshot, metric, digits))
    lines.append("")

    lines.append("GPU / Resource Samples")
    lines.append("-" * 78)
    lines.append(f"  csv samples:      {resource.get('samples', 0)}")
    lines.append(f"  last sample:      {resource.get('newest_timestamp') or '-'}")
    lines.append(
        "  current:          "
        f"gpu={fmt_pct(resource_value(resource, 'last', 'gpu_util_pct'))} "
        f"power={fmt_float(resource_value(resource, 'last', 'gpu_power_w'), 2)}W "
        f"clock={fmt_float(resource_value(resource, 'last', 'gpu_graphics_clock_mhz'), 0)}MHz "
        f"mem={fmt_float(resource_value(resource, 'last', 'gpu_mem_used_mib'), 0)}MiB "
        f"trainer_cpu={fmt_pct(resource_value(resource, 'last', 'trainer_cpu_pct'))}"
    )
    lines.append(
        "  recent max:       "
        f"gpu={fmt_pct(resource_value(resource, 'recent_maxes', 'gpu_util_pct'))} "
        f"power={fmt_float(resource_value(resource, 'recent_maxes', 'gpu_power_w'), 2)}W "
        f"clock={fmt_float(resource_value(resource, 'recent_maxes', 'gpu_graphics_clock_mhz'), 0)}MHz "
        f"mem={fmt_float(resource_value(resource, 'recent_maxes', 'gpu_mem_used_mib'), 0)}MiB "
        f"trainer_cpu={fmt_pct(resource_value(resource, 'recent_maxes', 'trainer_cpu_pct'))}"
    )
    lines.append(
        "  run max:          "
        f"gpu={fmt_pct(resource_value(resource, 'maxes', 'gpu_util_pct'))} "
        f"power={fmt_float(resource_value(resource, 'maxes', 'gpu_power_w'), 2)}W "
        f"clock={fmt_float(resource_value(resource, 'maxes', 'gpu_graphics_clock_mhz'), 0)}MHz "
        f"mem={fmt_float(resource_value(resource, 'maxes', 'gpu_mem_used_mib'), 0)}MiB"
    )
    lines.append("")

    lines.append("Alerts")
    lines.append("-" * 78)
    for alert in snapshot["alerts"]:
        lines.append(f"  {alert['severity'].upper():7} {alert['message']}")
    lines.append("")

    if args.tail > 0:
        train_tail = snapshot["logs"].get("train_tail", [])
        eval_tail = snapshot["logs"].get("eval_tail", [])
        if train_tail:
            lines.append("Train Log Tail")
            lines.append("-" * 78)
            lines.extend(f"  {line}" for line in train_tail[-args.tail :])
            lines.append("")
        if eval_tail:
            lines.append("Eval Log Tail")
            lines.append("-" * 78)
            lines.extend(f"  {line}" for line in eval_tail[-args.tail :])
            lines.append("")

    if args.watch:
        lines.append(f"refresh: {args.interval}s    quit: Ctrl-C    force one-shot: --once")
    return "\n".join(lines)


def write_json_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")


def print_run_list(runs: list[RunArtifacts], limit: int) -> None:
    print("Discovered Training Runs")
    print(f"  count: {len(runs)}")
    for idx, run in enumerate(runs[:limit], start=1):
        hist = run.history_path
        history, error = load_history(hist)
        record = latest_record(history)
        generation = record.get("generation", "-")
        found = fmt_float(record.get("found_rate"), 3)
        evader = fmt_float(record.get("evader_score"), 2)
        resource = fmt_path(run.latest_resource_log())
        marker = "!" if error else " "
        print(
            f"{idx:2}. {marker} {fmt_age(run.mtime):>8} "
            f"gen={generation!s:>4} found={found:>6} evader={evader:>8} "
            f"{run.label}"
        )
        print(f"      history:  {fmt_path(hist)}")
        print(f"      resource: {resource}")


def alert_exit_code(snapshot: dict[str, Any], strict: bool) -> int:
    severities = {alert["severity"] for alert in snapshot.get("alerts", [])}
    if "failure" in severities:
        return 2
    if strict and "warning" in severities:
        return 1
    return 0


def path_list_from_text(raw: str, default: list[Path]) -> list[Path]:
    raw = raw.strip()
    if not raw:
        return default
    try:
        parts = shlex.split(raw)
    except ValueError:
        parts = raw.split()
    return [Path(part) for part in parts]


def optional_int(raw: str) -> int | None:
    raw = raw.strip()
    if not raw:
        return None
    return int(raw)


def run_monitor_window(base_args: argparse.Namespace) -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Could not start monitor window: {exc}") from exc

    class MonitorWindow:
        COLORS = {
            "bg": "#0f172a",
            "panel": "#111827",
            "panel_2": "#172033",
            "card": "#162033",
            "card_2": "#0b1220",
            "border": "#2d3748",
            "grid": "#273449",
            "text": "#e5e7eb",
            "muted": "#94a3b8",
            "subtle": "#64748b",
            "accent": "#38bdf8",
            "blue": "#60a5fa",
            "teal": "#2dd4bf",
            "purple": "#c084fc",
            "green": "#22c55e",
            "yellow": "#f59e0b",
            "red": "#ef4444",
        }
        GROUP_COLORS = {
            "Training": "#60a5fa",
            "Derived": "#c084fc",
            "Resource": "#2dd4bf",
        }
        OVERVIEW_CHARTS = {
            "training_effectiveness_score",
            "found_rate",
            "static_found_rate",
            "survival_budget_ratio",
            "avg_attempts",
            "evader_score",
            "searcher_score",
            "escape_score",
            "learned_minus_static_found_rate",
            "attempt_budget_used",
            "resource/gpu_util_pct",
            "resource/gpu_power_w",
            "resource/trainer_cpu_pct",
        }

        def __init__(self) -> None:
            self.root = tk.Tk()
            self.root.title("ShellGame Training Monitor")
            self.root.geometry("1480x940")
            self.root.minsize(1040, 680)
            self.root.configure(bg=self.COLORS["bg"])
            self.root.option_add("*Font", "{DejaVu Sans} 10")
            self._configure_style(ttk)

            self.target_var = tk.StringVar(value=str(base_args.target_opt or base_args.target or ""))
            self.roots_var = tk.StringVar(value=" ".join(str(path) for path in roots_from_args(base_args)))
            self.resource_roots_var = tk.StringVar(value=" ".join(str(path) for path in base_args.resource_roots))
            self.interval_var = tk.StringVar(value=str(base_args.interval))
            self.tail_var = tk.StringVar(value=str(base_args.tail))
            self.recent_samples_var = tk.StringVar(value=str(base_args.recent_samples))
            self.stale_seconds_var = tk.StringVar(value=str(base_args.stale_seconds))
            self.expected_generations_var = tk.StringVar(
                value="" if base_args.expected_generations is None else str(base_args.expected_generations)
            )
            self.auto_latest_var = tk.BooleanVar(value=not bool(self.target_var.get().strip()))
            self.watch_var = tk.BooleanVar(value=True)
            self.auto_scroll_var = tk.BooleanVar(value=True)
            self.freeze_graphs_var = tk.BooleanVar(value=False)
            self.graph_group_var = tk.StringVar(value="Overview")
            self.graph_filter_var = tk.StringVar(value="")
            self.chart_limit_var = tk.StringVar(value="12")
            self.chart_columns_var = tk.StringVar(value="Auto")
            self.chart_height_var = tk.StringVar(value="150")
            self.graph_points_var = tk.StringVar(value="300")
            self.run_scan_interval_var = tk.StringVar(value="15")
            self.status_var = tk.StringVar(value="Ready")

            self.runs: list[RunArtifacts] = []
            self.last_snapshot: dict[str, Any] | None = None
            self.current_artifact: RunArtifacts | None = None
            self.after_id: str | None = None
            self.resize_after_id: str | None = None
            self.last_run_scan_time = 0.0
            self.last_graph_signature: tuple[Any, ...] | None = None
            self.updating_runs = False
            self.kpi_labels: dict[str, Any] = {}
            self.kpi_values: dict[str, Any] = {}

            self._build_ui(tk, ttk, filedialog, messagebox)
            self.refresh_runs(select_latest=self.auto_latest_var.get())
            self.refresh_snapshot()
            self._schedule_if_watching()

        def _configure_style(self, ttk: Any) -> None:
            style = ttk.Style(self.root)
            try:
                style.theme_use("clam")
            except Exception:
                pass

            colors = self.COLORS
            style.configure(".", background=colors["bg"], foreground=colors["text"], borderwidth=0)
            style.configure("App.TFrame", background=colors["bg"])
            style.configure("Panel.TFrame", background=colors["panel"])
            style.configure("Card.TFrame", background=colors["card"])
            style.configure("Toolbar.TFrame", background=colors["panel_2"])
            style.configure("TLabel", background=colors["bg"], foreground=colors["text"])
            style.configure("Muted.TLabel", background=colors["bg"], foreground=colors["muted"])
            style.configure("Title.TLabel", background=colors["bg"], foreground=colors["text"], font="{DejaVu Sans} 16 bold")
            style.configure("Subtitle.TLabel", background=colors["bg"], foreground=colors["muted"])
            style.configure("Panel.TLabel", background=colors["panel"], foreground=colors["text"])
            style.configure("PanelMuted.TLabel", background=colors["panel"], foreground=colors["muted"])
            style.configure("Toolbar.TLabel", background=colors["panel_2"], foreground=colors["text"])
            style.configure(
                "TEntry",
                fieldbackground=colors["card_2"],
                foreground=colors["text"],
                insertcolor=colors["text"],
                bordercolor=colors["border"],
                lightcolor=colors["border"],
                darkcolor=colors["border"],
                padding=4,
            )
            style.configure(
                "TCombobox",
                fieldbackground=colors["card_2"],
                foreground=colors["text"],
                arrowcolor=colors["text"],
                bordercolor=colors["border"],
                padding=3,
            )
            style.configure(
                "TButton",
                background=colors["panel_2"],
                foreground=colors["text"],
                bordercolor=colors["border"],
                focusthickness=0,
                padding=(10, 5),
            )
            style.map("TButton", background=[("active", "#23314b")], foreground=[("disabled", colors["subtle"])])
            style.configure("Accent.TButton", background="#075985", foreground="#e0f2fe", padding=(12, 6))
            style.map("Accent.TButton", background=[("active", "#0369a1")])
            style.configure("TCheckbutton", background=colors["bg"], foreground=colors["text"])
            style.configure("Panel.TCheckbutton", background=colors["panel_2"], foreground=colors["text"])
            style.map("TCheckbutton", background=[("active", colors["bg"])])
            style.configure("TNotebook", background=colors["bg"], borderwidth=0)
            style.configure("TNotebook.Tab", background=colors["panel"], foreground=colors["muted"], padding=(12, 7))
            style.map(
                "TNotebook.Tab",
                background=[("selected", colors["panel_2"]), ("active", "#1e293b")],
                foreground=[("selected", colors["text"]), ("active", colors["text"])],
            )
            style.configure(
                "Treeview",
                background=colors["card_2"],
                fieldbackground=colors["card_2"],
                foreground=colors["text"],
                rowheight=25,
                bordercolor=colors["border"],
            )
            style.configure("Treeview.Heading", background=colors["panel_2"], foreground=colors["muted"], padding=5)
            style.map("Treeview", background=[("selected", "#075985")], foreground=[("selected", "#e0f2fe")])
            style.configure("Vertical.TScrollbar", background=colors["panel"], troughcolor=colors["bg"], arrowcolor=colors["muted"])
            style.configure("Horizontal.TScrollbar", background=colors["panel"], troughcolor=colors["bg"], arrowcolor=colors["muted"])

        def _build_ui(self, tk: Any, ttk: Any, filedialog: Any, messagebox: Any) -> None:
            self.filedialog = filedialog
            self.messagebox = messagebox

            outer = ttk.Frame(self.root, padding=12, style="App.TFrame")
            outer.pack(fill=tk.BOTH, expand=True)
            outer.columnconfigure(0, weight=1)
            outer.rowconfigure(2, weight=1)

            header = ttk.Frame(outer, style="App.TFrame")
            header.grid(row=0, column=0, sticky="ew")
            header.columnconfigure(0, weight=1)
            ttk.Label(header, text="ShellGame Training Monitor", style="Title.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Label(
                header,
                text="Live training, resource, and matchup telemetry",
                style="Subtitle.TLabel",
            ).grid(row=1, column=0, sticky="w", pady=(2, 0))
            header_actions = ttk.Frame(header, style="App.TFrame")
            header_actions.grid(row=0, column=1, rowspan=2, sticky="e")
            ttk.Button(header_actions, text="Refresh", style="Accent.TButton", command=self.refresh_snapshot).grid(row=0, column=0, padx=(0, 8))
            ttk.Button(header_actions, text="Save JSON", command=self.save_json).grid(row=0, column=1, padx=(0, 8))
            ttk.Button(header_actions, text="Copy Target", command=self.copy_target).grid(row=0, column=2, padx=(0, 8))
            ttk.Button(header_actions, text="Copy Resume Cmd", command=self.copy_resume_command).grid(row=0, column=3, padx=(0, 8))
            ttk.Button(header_actions, text="Quit", command=self.root.destroy).grid(row=0, column=4)

            controls = ttk.Frame(outer, padding=10, style="Toolbar.TFrame")
            controls.grid(row=1, column=0, sticky="ew", pady=(12, 10))
            for col in range(12):
                controls.columnconfigure(col, weight=1 if col in (1, 5, 9) else 0)

            ttk.Label(controls, text="Target", style="Toolbar.TLabel").grid(row=0, column=0, sticky="w")
            ttk.Entry(controls, textvariable=self.target_var).grid(row=0, column=1, columnspan=7, sticky="ew", padx=(8, 10))
            ttk.Button(controls, text="Browse", command=self.browse_target).grid(row=0, column=8, sticky="ew", padx=(0, 8))
            ttk.Checkbutton(
                controls,
                text="Auto latest",
                style="Panel.TCheckbutton",
                variable=self.auto_latest_var,
                command=self.on_auto_latest_changed,
            ).grid(row=0, column=9, sticky="w", padx=(0, 8))
            ttk.Checkbutton(
                controls,
                text="Watch",
                style="Panel.TCheckbutton",
                variable=self.watch_var,
                command=self.on_watch_toggled,
            ).grid(row=0, column=10, sticky="w", padx=(0, 8))
            ttk.Checkbutton(
                controls,
                text="Auto-scroll",
                style="Panel.TCheckbutton",
                variable=self.auto_scroll_var,
            ).grid(row=0, column=11, sticky="w")

            compact_fields = (
                ("Roots", self.roots_var, 30),
                ("Resource roots", self.resource_roots_var, 18),
                ("Interval", self.interval_var, 7),
                ("Scan sec", self.run_scan_interval_var, 7),
                ("Tail", self.tail_var, 6),
                ("Recent", self.recent_samples_var, 7),
                ("Stale", self.stale_seconds_var, 7),
                ("Expected", self.expected_generations_var, 7),
            )
            row = 1
            col = 0
            for label, variable, width in compact_fields:
                span = 2 if label in {"Roots", "Resource roots"} else 1
                needed = 1 + span
                if col + needed > 12:
                    row += 1
                    col = 0
                ttk.Label(controls, text=label, style="Toolbar.TLabel").grid(row=row, column=col, sticky="w", pady=(10, 0))
                ttk.Entry(controls, textvariable=variable, width=width).grid(
                    row=row,
                    column=col + 1,
                    columnspan=span,
                    sticky="ew" if span > 1 else "w",
                    padx=(6, 12),
                    pady=(10, 0),
                )
                col += needed

            body = ttk.Frame(outer, style="App.TFrame")
            body.grid(row=2, column=0, sticky="nsew")
            body.columnconfigure(0, weight=0)
            body.columnconfigure(1, weight=1)
            body.rowconfigure(0, weight=1)

            runs_frame = ttk.Frame(body, padding=10, style="Panel.TFrame")
            runs_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
            runs_frame.grid_propagate(False)
            runs_frame.configure(width=360)
            runs_frame.rowconfigure(2, weight=1)
            runs_frame.columnconfigure(0, weight=1)
            ttk.Label(runs_frame, text="Runs", style="Panel.TLabel", font=("DejaVu Sans", 11, "bold")).grid(row=0, column=0, sticky="w")
            ttk.Button(runs_frame, text="Refresh Runs", command=lambda: self.refresh_runs(select_latest=False)).grid(
                row=1, column=0, sticky="ew", pady=(8, 10)
            )
            self.run_tree = ttk.Treeview(
                runs_frame,
                columns=("age", "gen", "found", "label"),
                show="headings",
                selectmode="browse",
            )
            for name, text, width, anchor in (
                ("age", "Age", 68, "e"),
                ("gen", "Gen", 48, "e"),
                ("found", "Found", 64, "e"),
                ("label", "Run", 160, "w"),
            ):
                self.run_tree.heading(name, text=text)
                self.run_tree.column(name, width=width, minwidth=40, anchor=anchor, stretch=(name == "label"))
            self.run_tree.grid(row=2, column=0, sticky="nsew")
            run_scroll = ttk.Scrollbar(runs_frame, orient=tk.VERTICAL, command=self.run_tree.yview)
            run_scroll.grid(row=2, column=1, sticky="ns")
            self.run_tree.configure(yscrollcommand=run_scroll.set)
            self.run_tree.bind("<<TreeviewSelect>>", self.on_run_selected)
            self._bind_vertical_scroll(self.run_tree)

            main = ttk.Frame(body, style="App.TFrame")
            main.grid(row=0, column=1, sticky="nsew")
            main.rowconfigure(0, weight=1)
            main.columnconfigure(0, weight=1)

            self.notebook = ttk.Notebook(main)
            self.notebook.grid(row=0, column=0, sticky="nsew")

            overview_frame = ttk.Frame(self.notebook, padding=10, style="App.TFrame")
            overview_frame.rowconfigure(1, weight=1)
            overview_frame.columnconfigure(0, weight=1)
            self.notebook.add(overview_frame, text="Overview")

            kpi_bar = ttk.Frame(overview_frame, style="App.TFrame")
            kpi_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            for col in range(6):
                kpi_bar.columnconfigure(col, weight=1)
            for idx, (key, title) in enumerate(
                (
                    ("generation", "Generation"),
                    ("found", "Found Rate"),
                    ("static_found", "Static Found"),
                    ("gpu", "GPU Recent"),
                    ("power", "Power"),
                    ("status", "Status"),
                )
            ):
                self._make_kpi_card(kpi_bar, idx, key, title, tk)

            self.dashboard = tk.Text(
                overview_frame,
                wrap=tk.NONE,
                font=("DejaVu Sans Mono", 10),
                undo=False,
                bg=self.COLORS["card_2"],
                fg=self.COLORS["text"],
                insertbackground=self.COLORS["text"],
                relief=tk.FLAT,
                highlightthickness=1,
                highlightbackground=self.COLORS["border"],
                padx=12,
                pady=10,
            )
            self.dashboard.grid(row=1, column=0, sticky="nsew")
            y_scroll = ttk.Scrollbar(overview_frame, orient=tk.VERTICAL, command=self.dashboard.yview)
            y_scroll.grid(row=1, column=1, sticky="ns")
            x_scroll = ttk.Scrollbar(overview_frame, orient=tk.HORIZONTAL, command=self.dashboard.xview)
            x_scroll.grid(row=2, column=0, sticky="ew")
            self.dashboard.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

            graphs_frame = ttk.Frame(self.notebook, padding=10, style="App.TFrame")
            graphs_frame.rowconfigure(1, weight=1)
            graphs_frame.columnconfigure(0, weight=1)
            self.notebook.add(graphs_frame, text="Graphs")

            graph_controls = ttk.Frame(graphs_frame, padding=8, style="Toolbar.TFrame")
            graph_controls.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
            graph_controls.columnconfigure(3, weight=1)
            ttk.Label(graph_controls, text="Group", style="Toolbar.TLabel").grid(row=0, column=0, sticky="w")
            group_box = ttk.Combobox(
                graph_controls,
                textvariable=self.graph_group_var,
                values=("Overview", "Training", "Derived", "Resource", "All"),
                state="readonly",
                width=12,
            )
            group_box.grid(row=0, column=1, sticky="w", padx=(6, 14))
            group_box.bind("<<ComboboxSelected>>", lambda _event: self.redraw_graphs())
            ttk.Label(graph_controls, text="Filter", style="Toolbar.TLabel").grid(row=0, column=2, sticky="w")
            filter_entry = ttk.Entry(graph_controls, textvariable=self.graph_filter_var, width=24)
            filter_entry.grid(row=0, column=3, sticky="ew", padx=(6, 14))
            filter_entry.bind("<KeyRelease>", lambda _event: self.redraw_graphs())

            for col, (label, variable, width) in enumerate(
                (
                    ("Max", self.chart_limit_var, 5),
                    ("Columns", self.chart_columns_var, 7),
                    ("Height", self.chart_height_var, 6),
                    ("Points", self.graph_points_var, 6),
                ),
                start=4,
            ):
                ttk.Label(graph_controls, text=label, style="Toolbar.TLabel").grid(row=0, column=col * 2 - 4, sticky="w", padx=(0, 6))
                entry = ttk.Entry(graph_controls, textvariable=variable, width=width)
                entry.grid(row=0, column=col * 2 - 3, sticky="w", padx=(0, 12))
                entry.bind("<Return>", lambda _event: self.redraw_graphs())
                entry.bind("<FocusOut>", lambda _event: self.redraw_graphs())
            ttk.Checkbutton(
                graph_controls,
                text="Freeze",
                style="Panel.TCheckbutton",
                variable=self.freeze_graphs_var,
            ).grid(row=0, column=12, sticky="w", padx=(0, 10))
            ttk.Button(graph_controls, text="Redraw", command=self.redraw_graphs).grid(row=0, column=13, sticky="e")

            self.graph_canvas = tk.Canvas(
                graphs_frame,
                background=self.COLORS["bg"],
                highlightthickness=1,
                highlightbackground=self.COLORS["border"],
            )
            self.graph_canvas.grid(row=1, column=0, sticky="nsew")
            graph_y_scroll = ttk.Scrollbar(graphs_frame, orient=tk.VERTICAL, command=self.graph_canvas.yview)
            graph_y_scroll.grid(row=1, column=1, sticky="ns")
            graph_x_scroll = ttk.Scrollbar(graphs_frame, orient=tk.HORIZONTAL, command=self.graph_canvas.xview)
            graph_x_scroll.grid(row=2, column=0, sticky="ew")
            self.graph_canvas.configure(yscrollcommand=graph_y_scroll.set, xscrollcommand=graph_x_scroll.set)
            self.graph_canvas.bind("<Configure>", self.on_graph_canvas_configure)
            self._bind_vertical_scroll(self.graph_canvas)

            logs_frame = ttk.Frame(self.notebook, padding=10, style="App.TFrame")
            logs_frame.rowconfigure(0, weight=1)
            logs_frame.columnconfigure(0, weight=1)
            self.notebook.add(logs_frame, text="Logs")
            self.log_text = tk.Text(
                logs_frame,
                wrap=tk.NONE,
                font=("DejaVu Sans Mono", 10),
                undo=False,
                bg=self.COLORS["card_2"],
                fg=self.COLORS["text"],
                insertbackground=self.COLORS["text"],
                relief=tk.FLAT,
                highlightthickness=1,
                highlightbackground=self.COLORS["border"],
                padx=12,
                pady=10,
            )
            self.log_text.grid(row=0, column=0, sticky="nsew")
            log_y = ttk.Scrollbar(logs_frame, orient=tk.VERTICAL, command=self.log_text.yview)
            log_y.grid(row=0, column=1, sticky="ns")
            log_x = ttk.Scrollbar(logs_frame, orient=tk.HORIZONTAL, command=self.log_text.xview)
            log_x.grid(row=1, column=0, sticky="ew")
            self.log_text.configure(yscrollcommand=log_y.set, xscrollcommand=log_x.set)

            status = ttk.Label(outer, textvariable=self.status_var, anchor="w", style="Muted.TLabel")
            status.grid(row=3, column=0, sticky="ew", pady=(8, 0))

        def _make_kpi_card(self, parent: Any, col: int, key: str, title: str, tk: Any) -> None:
            card = tk.Frame(
                parent,
                bg=self.COLORS["card"],
                highlightthickness=1,
                highlightbackground=self.COLORS["border"],
                padx=12,
                pady=9,
            )
            card.grid(row=0, column=col, sticky="ew", padx=(0, 8 if col < 5 else 0))
            title_label = tk.Label(card, text=title, bg=self.COLORS["card"], fg=self.COLORS["muted"], anchor="w", font=("DejaVu Sans", 9))
            title_label.pack(fill=tk.X)
            value_label = tk.Label(
                card,
                text="-",
                bg=self.COLORS["card"],
                fg=self.COLORS["text"],
                anchor="w",
                font=("DejaVu Sans", 15, "bold"),
            )
            value_label.pack(fill=tk.X, pady=(3, 0))
            self.kpi_labels[key] = title_label
            self.kpi_values[key] = value_label

        def _bind_vertical_scroll(self, widget: Any) -> None:
            def wheel(event: Any) -> str:
                if getattr(event, "num", None) == 4:
                    widget.yview_scroll(-3, "units")
                elif getattr(event, "num", None) == 5:
                    widget.yview_scroll(3, "units")
                else:
                    delta = getattr(event, "delta", 0)
                    widget.yview_scroll(int(-1 * (delta / 120)), "units")
                return "break"

            widget.bind("<MouseWheel>", wheel)
            widget.bind("<Button-4>", wheel)
            widget.bind("<Button-5>", wheel)

        def make_args(self) -> argparse.Namespace:
            target_text = self.target_var.get().strip()
            auto_latest = self.auto_latest_var.get()
            return argparse.Namespace(
                target=None if auto_latest or not target_text else Path(target_text),
                target_opt=None,
                roots=path_list_from_text(self.roots_var.get(), [Path("models")]),
                base=None,
                resource_roots=path_list_from_text(self.resource_roots_var.get(), [Path("resource_logs")]),
                watch=self.watch_var.get(),
                once=False,
                interval=max(0.1, float(self.interval_var.get().strip() or "2.0")),
                tail=max(0, int(self.tail_var.get().strip() or "12")),
                recent_samples=max(1, int(self.recent_samples_var.get().strip() or "300")),
                stale_seconds=max(1.0, float(self.stale_seconds_var.get().strip() or "180")),
                expected_generations=optional_int(self.expected_generations_var.get()),
                list=False,
                limit=20,
                no_clear=True,
                json=False,
                json_out=None,
                strict=False,
                scan_only=False,
            )

        def resolve_current_artifact(self) -> RunArtifacts | None:
            gui_args = self.make_args()
            return resolve_target(gui_args)

        def resolve_current_artifact_cached(self, gui_args: argparse.Namespace) -> RunArtifacts | None:
            if gui_args.target is not None or gui_args.target_opt is not None:
                self.current_artifact = resolve_target(gui_args)
                return self.current_artifact

            scan_interval = self.float_from_var(self.run_scan_interval_var, 15.0, minimum=1.0)
            now = time.monotonic()
            if not self.runs or now - self.last_run_scan_time >= scan_interval:
                self.refresh_runs(select_latest=True, update_status=False)
            if self.runs:
                self.current_artifact = collect_artifacts_for_dir(self.runs[0].run_dir, gui_args.resource_roots)
                self.target_var.set(str(self.current_artifact.run_dir))
                return self.current_artifact
            return None

        def refresh_runs(self, select_latest: bool, update_status: bool = True) -> None:
            try:
                gui_args = self.make_args()
                self.runs = discover_runs(roots_from_args(gui_args), gui_args.resource_roots)
                self.last_run_scan_time = time.monotonic()
            except Exception as exc:  # noqa: BLE001
                self.status_var.set(f"Run scan failed: {exc}")
                return

            previous_target = self.target_var.get().strip()
            self.updating_runs = True
            self.run_tree.delete(*self.run_tree.get_children())
            for run in self.runs[:200]:
                history, _error = load_history(run.history_path)
                record = latest_record(history)
                self.run_tree.insert(
                    "",
                    "end",
                    iid=str(len(self.run_tree.get_children())),
                    values=(
                        fmt_age(run.mtime),
                        str(record.get("generation", "-")),
                        fmt_float(record.get("found_rate"), 3),
                        run.label,
                    ),
                )

            if select_latest and self.runs:
                self.run_tree.selection_set("0")
                self.run_tree.focus("0")
                self.current_artifact = self.runs[0]
            elif previous_target:
                for idx, run in enumerate(self.runs[:200]):
                    if str(run.run_dir) == previous_target:
                        iid = str(idx)
                        self.run_tree.selection_set(iid)
                        self.run_tree.focus(iid)
                        break
            self.updating_runs = False
            if update_status:
                self.status_var.set(f"Discovered {len(self.runs)} run(s)")

        def refresh_snapshot(self) -> None:
            try:
                gui_args = self.make_args()
                artifact = self.resolve_current_artifact_cached(gui_args)
                if artifact is None:
                    self.write_dashboard("No training artifacts found.")
                    self.clear_graphs("No training artifacts found.")
                    self.write_logs({})
                    self.update_kpis({})
                    self.status_var.set("No training artifacts found")
                    return
                snapshot = make_snapshot(gui_args, artifact)
                self.last_snapshot = snapshot
                self.current_artifact = artifact
                self.write_dashboard(render_snapshot(snapshot, gui_args, clear=False))
                self.write_logs(snapshot)
                self.update_kpis(snapshot)
                self.draw_graphs(snapshot, force=False)
                target = snapshot["target"]["label"]
                gen = snapshot["state"].get("generation", "-")
                self.status_var.set(f"Updated {snapshot['generated_at']}  target={target}  generation={gen}")
            except Exception as exc:  # noqa: BLE001
                self.write_dashboard(f"Monitor update failed:\n{exc}")
                self.clear_graphs(f"Monitor update failed: {exc}")
                self.status_var.set(f"Update failed: {exc}")
            finally:
                self._schedule_if_watching()

        def write_dashboard(self, text: str) -> None:
            self.dashboard.configure(state=tk.NORMAL)
            self.dashboard.delete("1.0", tk.END)
            self.dashboard.insert(tk.END, text)
            self.dashboard.configure(state=tk.DISABLED)
            if self.auto_scroll_var.get():
                self.dashboard.see(tk.END)
                self.dashboard.yview_moveto(1.0)

        def write_logs(self, snapshot: dict[str, Any]) -> None:
            lines: list[str] = []
            if snapshot:
                train_tail = snapshot.get("logs", {}).get("train_tail", [])
                eval_tail = snapshot.get("logs", {}).get("eval_tail", [])
                if train_tail:
                    lines.append("Train Log Tail")
                    lines.append("=" * 90)
                    lines.extend(train_tail)
                    lines.append("")
                if eval_tail:
                    lines.append("Eval Log Tail")
                    lines.append("=" * 90)
                    lines.extend(eval_tail)
            if not lines:
                lines.append("No log tail available.")
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert(tk.END, "\n".join(lines))
            self.log_text.configure(state=tk.DISABLED)
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)
                self.log_text.yview_moveto(1.0)

        def update_kpis(self, snapshot: dict[str, Any]) -> None:
            def set_value(key: str, value: str, color: str | None = None) -> None:
                label = self.kpi_values.get(key)
                if label is not None:
                    label.configure(text=value, fg=color or self.COLORS["text"])

            if not snapshot:
                for key in self.kpi_values:
                    set_value(key, "-")
                return

            state = snapshot.get("state", {})
            record = snapshot.get("latest_record", {})
            resource = snapshot.get("resource", {})
            gen = state.get("generation")
            expected = state.get("expected_generations")
            generation_text = f"{gen}/{expected}" if gen is not None and expected else str(gen or "-")
            found = finite_float(record.get("found_rate"))
            static_found = finite_float(record.get("static_found_rate"))
            recent_gpu = finite_float(resource_value(resource, "recent_maxes", "gpu_util_pct"))
            current_power = finite_float(resource_value(resource, "last", "gpu_power_w"))
            severities = [alert.get("severity") for alert in snapshot.get("alerts", [])]
            status = "OK"
            status_color = self.COLORS["green"]
            if "failure" in severities:
                status = "FAIL"
                status_color = self.COLORS["red"]
            elif "warning" in severities:
                status = "WARN"
                status_color = self.COLORS["yellow"]

            set_value("generation", generation_text, self.COLORS["blue"])
            set_value("found", f"{found * 100:.1f}%" if found is not None else "-", self.COLORS["blue"])
            set_value("static_found", f"{static_found * 100:.1f}%" if static_found is not None else "-", self.COLORS["purple"])
            set_value("gpu", fmt_pct(recent_gpu), self.COLORS["teal"])
            set_value("power", f"{current_power:.1f}W" if current_power is not None else "-", self.COLORS["teal"])
            set_value("status", status, status_color)

        def clear_graphs(self, message: str) -> None:
            self.graph_canvas.delete("all")
            self.last_graph_signature = None
            self.graph_canvas.create_text(24, 24, text=message, anchor="nw", fill=self.COLORS["muted"])
            self.graph_canvas.configure(scrollregion=(0, 0, 600, 100))

        def on_graph_canvas_configure(self, _event: Any) -> None:
            if self.resize_after_id is not None:
                self.root.after_cancel(self.resize_after_id)
            self.resize_after_id = self.root.after(150, self.redraw_graphs)

        def redraw_graphs(self) -> None:
            self.last_graph_signature = None
            if self.last_snapshot is not None:
                self.draw_graphs(self.last_snapshot, force=True)

        def int_from_var(self, variable: Any, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
            try:
                value = int(str(variable.get()).strip() or str(default))
            except ValueError:
                value = default
            value = max(minimum, value)
            if maximum is not None:
                value = min(maximum, value)
            return value

        def float_from_var(self, variable: Any, default: float, *, minimum: float = 0.0) -> float:
            try:
                value = float(str(variable.get()).strip() or str(default))
            except ValueError:
                value = default
            return max(minimum, value)

        def selected_charts(self, snapshot: dict[str, Any]) -> list[tuple[str, list[float], str]]:
            charts = graph_series_from_snapshot(snapshot)
            group_filter = self.graph_group_var.get().strip() or "Overview"
            text_filter = self.graph_filter_var.get().strip().lower()
            point_limit = self.int_from_var(self.graph_points_var, 300, minimum=20, maximum=5000)
            chart_limit = self.int_from_var(self.chart_limit_var, 12, minimum=1, maximum=200)

            selected: list[tuple[str, list[float], str]] = []
            for label, values, group in charts:
                if group_filter == "Overview" and label not in self.OVERVIEW_CHARTS:
                    continue
                if group_filter not in {"Overview", "All"} and group != group_filter:
                    continue
                if text_filter and text_filter not in label.lower() and text_filter not in group.lower():
                    continue
                clean = [value for value in values if math.isfinite(value)]
                if len(clean) > point_limit:
                    stride = len(clean) / point_limit
                    clean = [clean[int(i * stride)] for i in range(point_limit)]
                if clean:
                    selected.append((label, clean, group))
            return selected[:chart_limit]

        def graph_signature(
            self,
            charts: list[tuple[str, list[float], str]],
            width: int,
            columns: int,
            chart_height: int,
        ) -> tuple[Any, ...]:
            data = []
            for label, values, group in charts:
                data.append(
                    (
                        label,
                        group,
                        len(values),
                        round(values[0], 6),
                        round(values[-1], 6),
                        round(min(values), 6),
                        round(max(values), 6),
                    )
                )
            return (
                width // 16,
                columns,
                chart_height,
                self.graph_group_var.get(),
                self.graph_filter_var.get(),
                self.chart_limit_var.get(),
                self.graph_points_var.get(),
                tuple(data),
            )

        def draw_graphs(self, snapshot: dict[str, Any], *, force: bool) -> None:
            if self.freeze_graphs_var.get() and not force:
                return
            canvas = self.graph_canvas
            charts = self.selected_charts(snapshot)
            if not charts:
                self.clear_graphs("No graphable metric data yet.")
                return

            width = max(canvas.winfo_width(), 880)
            margin = 16
            gap = 14
            raw_columns = self.chart_columns_var.get().strip().lower()
            if raw_columns == "auto" or not raw_columns:
                columns = 3 if width >= 1420 else 2 if width >= 920 else 1
            else:
                try:
                    columns = int(raw_columns)
                except ValueError:
                    columns = 2
                columns = max(1, min(columns, 4))
            chart_height = self.int_from_var(self.chart_height_var, 150, minimum=110, maximum=360)
            signature = self.graph_signature(charts, width, columns, chart_height)
            if not force and signature == self.last_graph_signature:
                return
            self.last_graph_signature = signature
            canvas.delete("all")

            chart_width = max(340, int((width - margin * 2 - gap * (columns - 1)) / columns))

            for idx, (label, values, group) in enumerate(charts):
                col = idx % columns
                row = idx // columns
                x = margin + col * (chart_width + gap)
                y = margin + row * (chart_height + gap)
                self.draw_chart(
                    x=x,
                    y=y,
                    width=chart_width,
                    height=chart_height,
                    title=label,
                    subtitle=group,
                    values=values,
                    color=self.GROUP_COLORS.get(group, self.COLORS["accent"]),
                )

            total_rows = math.ceil(len(charts) / columns)
            total_height = margin * 2 + total_rows * chart_height + max(0, total_rows - 1) * gap
            canvas.configure(scrollregion=(0, 0, width, total_height))

        def draw_chart(
            self,
            *,
            x: int,
            y: int,
            width: int,
            height: int,
            title: str,
            subtitle: str,
            values: list[float],
            color: str,
        ) -> None:
            canvas = self.graph_canvas
            clean = [value for value in values if math.isfinite(value)]
            canvas.create_rectangle(x, y, x + width, y + height, fill=self.COLORS["card"], outline=self.COLORS["border"])
            canvas.create_rectangle(x, y, x + 5, y + height, fill=color, outline=color)
            canvas.create_text(x + 16, y + 10, text=title, anchor="nw", fill=self.COLORS["text"], font=("DejaVu Sans", 10, "bold"))
            canvas.create_text(x + 16, y + 30, text=subtitle, anchor="nw", fill=self.COLORS["muted"], font=("DejaVu Sans", 8))

            if not clean:
                canvas.create_text(x + width / 2, y + height / 2, text="No data", fill=self.COLORS["muted"])
                return

            plot_left = x + 52
            plot_right = x + width - 14
            plot_top = y + 52
            plot_bottom = y + height - 26
            plot_width = max(1, plot_right - plot_left)
            plot_height = max(1, plot_bottom - plot_top)

            lo = min(clean)
            hi = max(clean)
            pad = (hi - lo) * 0.08
            if pad <= 1e-12:
                pad = max(abs(hi) * 0.05, 1.0)
            lo -= pad
            hi += pad

            canvas.create_rectangle(plot_left, plot_top, plot_right, plot_bottom, outline=self.COLORS["border"], fill=self.COLORS["card_2"])
            for fraction in (0.25, 0.5, 0.75):
                gy = plot_top + plot_height * fraction
                canvas.create_line(plot_left, gy, plot_right, gy, fill=self.COLORS["grid"])

            if lo < 0 < hi:
                zero_y = plot_bottom - ((0 - lo) / (hi - lo)) * plot_height
                canvas.create_line(plot_left, zero_y, plot_right, zero_y, fill=self.COLORS["subtle"], dash=(4, 3))

            max_points = max(2, int(plot_width))
            if len(clean) > max_points:
                stride = len(clean) / max_points
                sampled = [clean[int(i * stride)] for i in range(max_points)]
            else:
                sampled = clean

            coords: list[float] = []
            denom = max(1, len(sampled) - 1)
            for idx, value in enumerate(sampled):
                px = plot_left + (idx / denom) * plot_width
                py = plot_bottom - ((value - lo) / (hi - lo)) * plot_height
                coords.extend([px, py])

            if len(coords) >= 4:
                area = [plot_left, plot_bottom] + coords + [plot_right, plot_bottom]
                canvas.create_polygon(*area, fill="#102034", outline="")
                canvas.create_line(*coords, fill=color, width=2, smooth=False)
            else:
                canvas.create_oval(coords[0] - 3, coords[1] - 3, coords[0] + 3, coords[1] + 3, fill=color, outline=color)

            last = clean[-1]
            canvas.create_text(plot_left - 6, plot_top, text=fmt_float(max(clean), 2), anchor="e", fill=self.COLORS["muted"], font=("DejaVu Sans", 8))
            canvas.create_text(plot_left - 6, plot_bottom, text=fmt_float(min(clean), 2), anchor="e", fill=self.COLORS["muted"], font=("DejaVu Sans", 8))
            canvas.create_text(plot_right, y + 8, text=f"last {fmt_float(last, 3)}", anchor="ne", fill=color, font=("DejaVu Sans", 9))
            canvas.create_text(plot_left, y + height - 8, text=f"n={len(clean)}", anchor="sw", fill=self.COLORS["muted"], font=("DejaVu Sans", 8))

        def _schedule_if_watching(self) -> None:
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            if not self.watch_var.get():
                return
            try:
                interval_ms = max(100, int(float(self.interval_var.get().strip() or "2.0") * 1000))
            except ValueError:
                interval_ms = 2000
            self.after_id = self.root.after(interval_ms, self.refresh_snapshot)

        def on_watch_toggled(self) -> None:
            if self.watch_var.get():
                self.refresh_snapshot()
            else:
                if self.after_id is not None:
                    self.root.after_cancel(self.after_id)
                    self.after_id = None
                self.status_var.set("Watch paused")

        def on_auto_latest_changed(self) -> None:
            if self.auto_latest_var.get():
                self.refresh_runs(select_latest=True)
            self.refresh_snapshot()

        def on_run_selected(self, _event: Any) -> None:
            if self.updating_runs:
                return
            selection = self.run_tree.selection()
            if not selection:
                return
            index = int(selection[0])
            if index >= len(self.runs):
                return
            self.auto_latest_var.set(False)
            self.target_var.set(str(self.runs[index].run_dir))
            self.current_artifact = self.runs[index]
            self.refresh_snapshot()

        def browse_target(self) -> None:
            path = self.filedialog.askdirectory(initialdir=str(Path.cwd()), title="Choose Training Run Directory")
            if path:
                self.auto_latest_var.set(False)
                self.target_var.set(path)
                self.refresh_snapshot()

        def save_json(self) -> None:
            if self.last_snapshot is None:
                self.refresh_snapshot()
            if self.last_snapshot is None:
                self.messagebox.showerror("Save JSON", "No snapshot is available yet.")
                return
            path = self.filedialog.asksaveasfilename(
                title="Save Monitor Snapshot",
                defaultextension=".json",
                filetypes=(("JSON files", "*.json"), ("All files", "*")),
            )
            if not path:
                return
            write_json_snapshot(Path(path), self.last_snapshot)
            self.status_var.set(f"Saved JSON snapshot: {path}")

        def copy_target(self) -> None:
            target = self.target_var.get().strip()
            if not target and self.last_snapshot:
                target = self.last_snapshot["target"]["run_dir"]
            self.root.clipboard_clear()
            self.root.clipboard_append(target)
            self.status_var.set("Copied target path")

        def copy_resume_command(self) -> None:
            target = self.target_var.get().strip()
            if self.last_snapshot:
                target = self.last_snapshot["target"]["run_dir"]
            command = f"./train_iterate.sh {shlex.quote(target)} 1" if target else "./train_iterate.sh"
            self.root.clipboard_clear()
            self.root.clipboard_append(command)
            self.status_var.set("Copied resume command")

        def run(self) -> None:
            self.root.mainloop()

    MonitorWindow().run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", nargs="?", type=Path, help="checkpoint/run dir, training_history.json, or log")
    parser.add_argument("--target", dest="target_opt", type=Path, help="explicit checkpoint/run dir")
    parser.add_argument("--roots", type=Path, nargs="+", default=[Path("models")], help="roots to scan when no target is supplied")
    parser.add_argument("--base", type=Path, default=None, help="legacy alias for a single root")
    parser.add_argument("--resource-roots", type=Path, nargs="+", default=[Path("resource_logs")], help="resource CSV directories")
    parser.add_argument("--watch", "--follow", action="store_true", help="refresh until interrupted")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    parser.add_argument("--interval", type=float, default=2.0, help="watch refresh interval in seconds")
    parser.add_argument("--tail", type=int, default=12, help="log tail lines per section")
    parser.add_argument("--recent-samples", type=int, default=300, help="resource samples used for recent maxima")
    parser.add_argument("--stale-seconds", type=float, default=180.0, help="warn when active history is older than this")
    parser.add_argument("--expected-generations", type=int, default=None, help="override expected generation count")
    parser.add_argument("--list", action="store_true", help="list discovered runs and exit")
    parser.add_argument("--limit", type=int, default=20, help="max rows for --list")
    parser.add_argument("--no-clear", action="store_true", help="do not clear the terminal between watch frames")
    parser.add_argument("--json", action="store_true", help="print snapshot JSON")
    parser.add_argument("--json-out", type=Path, default=None, help="write snapshot JSON to this path")
    parser.add_argument("--strict", action="store_true", help="return non-zero for warnings in one-shot mode")
    parser.add_argument("--window", "--gui", "--show", dest="window", action="store_true", help="open the monitor control window")
    parser.add_argument("--no-window", dest="window", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(window=False)

    # Compatibility with the previous batch analyzer. These no longer control
    # plotting, but accepting them keeps old shell history from failing loudly.
    parser.add_argument("--scan-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--runs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expect-runs", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--static-generations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--coagent-generations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-generations", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-visuals", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plot-mode", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--plot-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--insights-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smooth", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-series", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-raw", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.scan_only:
        args.list = True
    if args.once:
        args.watch = False
    elif not args.watch and sys.stdout.isatty() and not args.json and args.json_out is None and not args.list:
        args.watch = True
    return args


def main() -> None:
    args = parse_args()
    if args.window:
        run_monitor_window(args)
        raise SystemExit(0)

    runs = discover_runs(roots_from_args(args), args.resource_roots)

    if args.list:
        print_run_list(runs, args.limit)
        raise SystemExit(0)

    artifact = resolve_target(args)
    if artifact is None:
        roots = ", ".join(to_display_path(root) for root in roots_from_args(args))
        raise SystemExit(f"No training artifacts found under: {roots}")

    last_snapshot: dict[str, Any] | None = None
    try:
        while True:
            snapshot = make_snapshot(args, artifact)
            last_snapshot = snapshot

            if args.json:
                print(json.dumps(snapshot, indent=2, default=str))
            else:
                clear = args.watch and not args.no_clear
                print(render_snapshot(snapshot, args, clear=clear), flush=True)

            if args.json_out is not None:
                write_json_snapshot(args.json_out, snapshot)

            if not args.watch:
                break

            # Re-resolve the default target each frame so a new checkpoint
            # created by train_iterate.sh becomes the monitored run naturally.
            if args.target is None and args.target_opt is None:
                refreshed = resolve_target(args)
                if refreshed is not None:
                    artifact = refreshed
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("\nmonitor stopped")

    if last_snapshot is None:
        raise SystemExit(1)
    if not args.watch:
        raise SystemExit(alert_exit_code(last_snapshot, strict=args.strict))


if __name__ == "__main__":
    main()
