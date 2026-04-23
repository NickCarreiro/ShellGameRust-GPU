#!/usr/bin/env python3
"""
Maintain persistent core Skoll/Hati models and promote only measured improvements.

The script keeps a central bundle in models/core/:
  - self_play_models.json
  - evader_model.json
  - searcher_model.json
  - model_metrics.json

Every discovered folder containing self_play_models.json also receives a
model_metrics.json sidecar. For run folders, that sidecar is derived from the
tail of training_history.json. For core, it is derived from fixed evaluation
seeds so future candidates are compared against a stable average.

Promotion is role-specific:
  - candidate evader + current core searcher must improve average_evader_reward
  - current core evader + candidate searcher must improve average_searcher_reward
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_BASES = (Path("models/fast_runs"), Path("models/pipeline_runs"))
CORE_DIR = Path("models/core")
METRICS_NAME = "model_metrics.json"
TAIL_WINDOW = 10
METRIC_KEYS = ("searcher_score", "evader_score", "found_rate", "avg_attempts")


@dataclass(frozen=True)
class EvalConfig:
    episodes: int
    seeds: tuple[int, ...]
    min_nodes: int
    max_nodes: int
    max_attempts_factor: int
    max_attempts_ratio: float | None
    max_attempts_cap: int | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def natural_key(path: Path) -> tuple[int, ...]:
    nums = [int(part) for part in re.findall(r"\d+", str(path))]
    return tuple(nums) if nums else (10**9,)


def discover_model_dirs(bases: list[Path], core_dir: Path) -> list[Path]:
    dirs: set[Path] = set()
    for base in bases:
        base = (ROOT / base).resolve() if not base.is_absolute() else base.resolve()
        if not base.exists():
            continue
        for bundle in base.rglob("self_play_models.json"):
            dirs.add(bundle.parent)
    core_abs = (ROOT / core_dir).resolve() if not core_dir.is_absolute() else core_dir.resolve()
    if (core_abs / "self_play_models.json").exists():
        dirs.add(core_abs)
    return sorted(dirs, key=natural_key)


def tail_metrics_from_history(model_dir: Path, tail_window: int) -> dict[str, Any] | None:
    history_path = model_dir / "training_history.json"
    if not history_path.exists():
        return None
    history = read_json(history_path)
    if not isinstance(history, list) or not history:
        return None
    tail = history[-min(tail_window, len(history)) :]
    averages = {
        key: mean(float(row[key]) for row in tail if key in row)
        for key in METRIC_KEYS
        if any(key in row for row in tail)
    }
    final = {key: history[-1].get(key) for key in METRIC_KEYS if key in history[-1]}
    for key in ("static_evader_score", "static_found_rate"):
        values = [row.get(key) for row in tail if row.get(key) is not None]
        if values:
            averages[key] = mean(float(value) for value in values)
            final[key] = history[-1].get(key)
    return {
        "schema_version": 1,
        "model_dir": rel(model_dir),
        "source": "training_history_tail",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "tail_window": len(tail),
        "records": len(history),
        "generation_range": [history[0].get("generation"), history[-1].get("generation")],
        "averages": averages,
        "final": final,
        "promotion": {
            "eligible": model_dir.name != "static_stage",
            "notes": "Derived from training history; promotion re-evaluates candidates against current core.",
        },
    }


def ensure_sidecar(model_dir: Path, tail_window: int, force: bool = False) -> dict[str, Any] | None:
    metrics_path = model_dir / METRICS_NAME
    if model_dir.name == "core" and metrics_path.exists():
        return read_json(metrics_path)
    if metrics_path.exists() and not force:
        return read_json(metrics_path)
    metrics = tail_metrics_from_history(model_dir, tail_window)
    if metrics is None:
        bundle = read_json(model_dir / "self_play_models.json")
        metrics = {
            "schema_version": 1,
            "model_dir": rel(model_dir),
            "source": "bundle_only",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "averages": {},
            "final": {},
            "model_roles": {
                "evader": bundle.get("evader", {}).get("role", "skoll"),
                "searcher": bundle.get("searcher", {}).get("role", "hati"),
            },
            "promotion": {
                "eligible": model_dir.name != "core",
                "notes": "No training history found; use promotion evaluation before comparing.",
            },
        }
    write_json(metrics_path, metrics)
    return metrics


def composite_score(metrics: dict[str, Any]) -> float:
    averages = metrics.get("averages", {})
    evader = float(averages.get("evader_score", float("-inf")))
    searcher = float(averages.get("searcher_score", 0.0))
    found = float(averages.get("found_rate", 1.0))
    if not math.isfinite(evader):
        return float("-inf")
    return evader + searcher - (found * 25.0)


def copy_bundle_to_core(source_dir: Path, core_dir: Path) -> None:
    core_dir.mkdir(parents=True, exist_ok=True)
    bundle = read_json(source_dir / "self_play_models.json")
    bundle["evader"]["role"] = "skoll"
    bundle["searcher"]["role"] = "hati"
    write_json(core_dir / "self_play_models.json", bundle)
    write_json(core_dir / "evader_model.json", bundle["evader"])
    write_json(core_dir / "searcher_model.json", bundle["searcher"])


def initialize_core(model_dirs: list[Path], core_dir: Path, tail_window: int, init_from: Path | None) -> Path:
    if (core_dir / "self_play_models.json").exists():
        return core_dir
    if init_from is not None:
        source = init_from if init_from.is_absolute() else ROOT / init_from
        if not (source / "self_play_models.json").exists():
            raise SystemExit(f"Missing model bundle in --init-from directory: {source}")
    else:
        candidates = []
        for model_dir in model_dirs:
            if model_dir.name == "static_stage":
                continue
            metrics = ensure_sidecar(model_dir, tail_window)
            if metrics is None:
                continue
            candidates.append((composite_score(metrics), model_dir))
        if not candidates:
            raise SystemExit("No candidate coagent model bundles found to initialize core.")
        source = max(candidates, key=lambda item: item[0])[1]
    source_metrics = ensure_sidecar(source, tail_window)
    copy_bundle_to_core(source, core_dir)
    write_json(core_dir / METRICS_NAME, {
        "schema_version": 1,
        "model_dir": rel(core_dir),
        "source": "initialized_from_candidate",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "initialized_from": rel(source),
        "averages": (source_metrics or {}).get("averages", {}),
        "final": (source_metrics or {}).get("final", {}),
        "promotion_history": [{
            "created_at": utc_now(),
            "event": "core_initialized",
            "source": rel(source),
        }],
    })
    print(f"Initialized core from {rel(source)}")
    return core_dir


def merged_bundle(evader_source: Path, searcher_source: Path, out_path: Path) -> None:
    evader_bundle = read_json(evader_source / "self_play_models.json")
    searcher_bundle = read_json(searcher_source / "self_play_models.json")
    merged = {
        "evader": evader_bundle["evader"],
        "searcher": searcher_bundle["searcher"],
    }
    merged["evader"]["role"] = "skoll"
    merged["searcher"]["role"] = "hati"
    write_json(out_path, merged)


def run_evaluate(bundle_path: Path, seed: int, config: EvalConfig, release: bool) -> dict[str, float]:
    cmd = ["cargo", "run"]
    if release:
        cmd.append("--release")
    cmd.extend([
        "--bin",
        "ml_self_play",
        "--",
        "evaluate",
        "--model-bundle",
        str(bundle_path),
        "--episodes",
        str(config.episodes),
        "--seed",
        str(seed),
        "--min-nodes",
        str(config.min_nodes),
        "--max-nodes",
        str(config.max_nodes),
        "--max-attempts-factor",
        str(config.max_attempts_factor),
    ])
    if config.max_attempts_ratio is not None:
        cmd.extend(["--max-attempts-ratio", str(config.max_attempts_ratio)])
    if config.max_attempts_cap is not None:
        cmd.extend(["--max-attempts-cap", str(config.max_attempts_cap)])
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    match = re.search(
        r"found_rate=([-+0-9.eE]+)\s+avg_attempts=([-+0-9.eE]+)\s+"
        r"searcher_reward=([-+0-9.eE]+)\s+evader_reward=([-+0-9.eE]+)",
        proc.stdout,
    )
    if not match:
        raise RuntimeError(f"Could not parse evaluator output:\n{proc.stdout}\n{proc.stderr}")
    return {
        "found_rate": float(match.group(1)),
        "avg_attempts": float(match.group(2)),
        "searcher_score": float(match.group(3)),
        "evader_score": float(match.group(4)),
    }


def evaluate_pair(evader_source: Path, searcher_source: Path, config: EvalConfig, release: bool) -> dict[str, Any]:
    (ROOT / "target").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="core_eval_", dir=ROOT / "target") as tmp:
        bundle_path = Path(tmp) / "self_play_models.json"
        merged_bundle(evader_source, searcher_source, bundle_path)
        per_seed = [run_evaluate(bundle_path, seed, config, release) for seed in config.seeds]
    averages = {key: mean(row[key] for row in per_seed) for key in METRIC_KEYS}
    return {
        "config": {
            "episodes": config.episodes,
            "seeds": list(config.seeds),
            "min_nodes": config.min_nodes,
            "max_nodes": config.max_nodes,
            "max_attempts_factor": config.max_attempts_factor,
            "max_attempts_ratio": config.max_attempts_ratio,
            "max_attempts_cap": config.max_attempts_cap,
        },
        "averages": averages,
        "per_seed": per_seed,
    }


def write_core_metrics(core_dir: Path, eval_result: dict[str, Any], history: list[dict[str, Any]]) -> None:
    payload = {
        "schema_version": 1,
        "model_dir": rel(core_dir),
        "source": "promotion_eval_average",
        "created_at": history[0]["created_at"] if history else utc_now(),
        "updated_at": utc_now(),
        "averages": eval_result["averages"],
        "evaluation": eval_result,
        "promotion_history": history,
    }
    write_json(core_dir / METRICS_NAME, payload)


def load_core_history(core_dir: Path) -> list[dict[str, Any]]:
    path = core_dir / METRICS_NAME
    if not path.exists():
        return []
    payload = read_json(path)
    history = payload.get("promotion_history", [])
    return history if isinstance(history, list) else []


def promote_role(core_dir: Path, candidate_dir: Path, role: str, eval_result: dict[str, Any]) -> None:
    core_bundle = read_json(core_dir / "self_play_models.json")
    candidate_bundle = read_json(candidate_dir / "self_play_models.json")
    if role == "evader":
        core_bundle["evader"] = candidate_bundle["evader"]
        core_bundle["evader"]["role"] = "skoll"
    elif role == "searcher":
        core_bundle["searcher"] = candidate_bundle["searcher"]
        core_bundle["searcher"]["role"] = "hati"
    else:
        raise ValueError(role)
    write_json(core_dir / "self_play_models.json", core_bundle)
    write_json(core_dir / "evader_model.json", core_bundle["evader"])
    write_json(core_dir / "searcher_model.json", core_bundle["searcher"])


def promote(args: argparse.Namespace) -> None:
    bases = [Path(base) for base in args.base]
    core_dir = (ROOT / args.core_dir).resolve() if not Path(args.core_dir).is_absolute() else Path(args.core_dir)
    model_dirs = discover_model_dirs(bases, core_dir)
    for model_dir in model_dirs:
        ensure_sidecar(model_dir, args.tail_window, force=args.refresh_sidecars)
    initialize_core(model_dirs, core_dir, args.tail_window, Path(args.init_from) if args.init_from else None)

    config = EvalConfig(
        episodes=args.episodes,
        seeds=tuple(args.seed),
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        max_attempts_factor=args.max_attempts_factor,
        max_attempts_ratio=args.max_attempts_ratio,
        max_attempts_cap=args.max_attempts_cap,
    )

    history = load_core_history(core_dir)
    if not history:
        history.append({
            "created_at": utc_now(),
            "event": "core_initialized",
            "core_dir": rel(core_dir),
        })

    core_eval = evaluate_pair(core_dir, core_dir, config, release=not args.debug_build)
    write_core_metrics(core_dir, core_eval, history)
    print(
        "Core baseline: "
        f"searcher={core_eval['averages']['searcher_score']:.3f} "
        f"evader={core_eval['averages']['evader_score']:.3f} "
        f"found={core_eval['averages']['found_rate']:.3f}"
    )

    candidates = [path for path in discover_model_dirs(bases, core_dir) if path != core_dir]
    candidates = [path for path in candidates if (path / "self_play_models.json").exists()]
    if args.coagent_only:
        candidates = [path for path in candidates if path.name == "coagent_stage"]
    promotions = 0

    for candidate_dir in sorted(candidates, key=natural_key):
        print(f"Checking {rel(candidate_dir)}")

        evader_eval = evaluate_pair(candidate_dir, core_dir, config, release=not args.debug_build)
        evader_score = evader_eval["averages"]["evader_score"]
        core_evader_score = core_eval["averages"]["evader_score"]
        if evader_score > core_evader_score + args.min_delta:
            print(f"  promote skoll: {evader_score:.3f} > {core_evader_score:.3f}")
            if not args.dry_run:
                promote_role(core_dir, candidate_dir, "evader", evader_eval)
                history.append({
                    "created_at": utc_now(),
                    "event": "promoted_evader",
                    "source": rel(candidate_dir),
                    "previous_score": core_evader_score,
                    "new_score": evader_score,
                    "evaluation": evader_eval,
                })
                core_eval = evaluate_pair(core_dir, core_dir, config, release=not args.debug_build)
                write_core_metrics(core_dir, core_eval, history)
            promotions += 1
        else:
            print(f"  keep skoll: {evader_score:.3f} <= {core_evader_score:.3f}")

        searcher_eval = evaluate_pair(core_dir, candidate_dir, config, release=not args.debug_build)
        searcher_score = searcher_eval["averages"]["searcher_score"]
        core_searcher_score = core_eval["averages"]["searcher_score"]
        if searcher_score > core_searcher_score + args.min_delta:
            print(f"  promote hati:  {searcher_score:.3f} > {core_searcher_score:.3f}")
            if not args.dry_run:
                promote_role(core_dir, candidate_dir, "searcher", searcher_eval)
                history.append({
                    "created_at": utc_now(),
                    "event": "promoted_searcher",
                    "source": rel(candidate_dir),
                    "previous_score": core_searcher_score,
                    "new_score": searcher_score,
                    "evaluation": searcher_eval,
                })
                core_eval = evaluate_pair(core_dir, core_dir, config, release=not args.debug_build)
                write_core_metrics(core_dir, core_eval, history)
            promotions += 1
        else:
            print(f"  keep hati:  {searcher_score:.3f} <= {core_searcher_score:.3f}")

    print(f"Promotion pass complete: {promotions} role promotion(s).")


def init_metrics(args: argparse.Namespace) -> None:
    bases = [Path(base) for base in args.base]
    core_dir = (ROOT / args.core_dir).resolve() if not Path(args.core_dir).is_absolute() else Path(args.core_dir)
    model_dirs = discover_model_dirs(bases, core_dir)
    for model_dir in model_dirs:
        ensure_sidecar(model_dir, args.tail_window, force=args.refresh_sidecars)
        print(f"metrics: {rel(model_dir / METRICS_NAME)}")
    if args.initialize_core:
        initialize_core(model_dirs, core_dir, args.tail_window, Path(args.init_from) if args.init_from else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--base", action="append", default=[], help="Model tree to scan; can be repeated.")
        p.add_argument("--core-dir", default=str(CORE_DIR))
        p.add_argument("--tail-window", type=int, default=TAIL_WINDOW)
        p.add_argument("--refresh-sidecars", action="store_true")
        p.add_argument("--init-from", default=None, help="Directory containing self_play_models.json for first core seed.")

    p_init = sub.add_parser("init-metrics", help="Create model_metrics.json sidecars.")
    add_common(p_init)
    p_init.add_argument("--initialize-core", action="store_true")
    p_init.set_defaults(func=init_metrics)

    p_promote = sub.add_parser("promote", help="Evaluate candidates and promote role improvements into core.")
    add_common(p_promote)
    p_promote.add_argument("--episodes", type=int, default=80)
    p_promote.add_argument("--seed", type=int, action="append", default=None)
    p_promote.add_argument("--min-nodes", type=int, default=11)
    p_promote.add_argument("--max-nodes", type=int, default=25)
    p_promote.add_argument("--max-attempts-factor", type=int, default=2)
    p_promote.add_argument("--max-attempts-ratio", type=float, default=0.40)
    p_promote.add_argument("--max-attempts-cap", type=int, default=10)
    p_promote.add_argument("--min-delta", type=float, default=0.0)
    p_promote.add_argument("--coagent-only", action="store_true", default=True)
    p_promote.add_argument("--include-static", dest="coagent_only", action="store_false")
    p_promote.add_argument("--dry-run", action="store_true")
    p_promote.add_argument("--debug-build", action="store_true")
    p_promote.set_defaults(func=promote)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.base:
        args.base = [str(base) for base in DEFAULT_BASES]
    if args.command == "promote" and args.seed is None:
        args.seed = [2026, 3023, 4020]
    args.func(args)


if __name__ == "__main__":
    main()
