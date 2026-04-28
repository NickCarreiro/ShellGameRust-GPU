#!/usr/bin/env python3
"""Shared training-history discovery for analysis and plotting scripts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


STAGED_STAGES = ("static_stage", "coagent_stage")
_RUN_RE = re.compile(r"^run_(\d+)$")
_VERSION_RE = re.compile(r"^v(\d+)$")


@dataclass(frozen=True)
class HistoryTarget:
    history_path: Path
    root: Path
    experiment: str
    group: str
    family: str
    stage: str
    structure: str


def _cwd() -> Path:
    return Path.cwd().resolve()


def to_display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_cwd()))
    except ValueError:
        return str(resolved)


def natural_label_key(label: str) -> tuple[tuple[int, ...], str]:
    nums = tuple(int(part) for part in re.findall(r"\d+", label))
    if not nums:
        nums = (10**9,)
    return nums, label


def _classify(history_path: Path) -> tuple[str, str, str, str, str]:
    parent = history_path.parent

    if parent.name in STAGED_STAGES and _RUN_RE.fullmatch(parent.parent.name):
        stage_dir = parent
        run_dir = stage_dir.parent
        family_dir = run_dir.parent.name
        experiment = f"{family_dir}/{run_dir.name}"
        return experiment, experiment, family_dir, stage_dir.name, "run_stage"

    if parent.parent.name == "checkpoints" and _VERSION_RE.fullmatch(parent.name):
        experiment = f"checkpoints/{parent.name}"
        return experiment, experiment, "checkpoints", "single_stage", "checkpoint"

    if parent.parent.name == "core":
        experiment = f"core/{parent.name}"
        return experiment, experiment, "core", "single_stage", "core"

    # Fallback for generic layouts.
    family = parent.parent.name if parent.parent != parent else parent.name
    experiment = to_display_path(parent)
    return experiment, experiment, family, "single_stage", "single"


def scan_histories(roots: Iterable[Path], recursive: bool = True) -> list[HistoryTarget]:
    discovered: dict[Path, HistoryTarget] = {}
    for root in roots:
        abs_root = root.expanduser().resolve()
        if not abs_root.exists():
            continue

        if abs_root.is_file():
            candidates = [abs_root] if abs_root.name == "training_history.json" else []
        else:
            pattern_iter = abs_root.rglob("training_history.json") if recursive else abs_root.glob("*/training_history.json")
            candidates = [path for path in pattern_iter if path.is_file()]

        for history_path in candidates:
            resolved = history_path.resolve()
            if resolved in discovered:
                continue
            experiment, group, family, stage, structure = _classify(resolved)
            discovered[resolved] = HistoryTarget(
                history_path=resolved,
                root=abs_root,
                experiment=experiment,
                group=group,
                family=family,
                stage=stage,
                structure=structure,
            )

    structure_rank = {
        "run_stage": 0,
        "checkpoint": 1,
        "core": 2,
        "single": 3,
    }

    return sorted(
        discovered.values(),
        key=lambda target: (
            structure_rank.get(target.structure, 99),
            natural_label_key(target.experiment),
            target.stage,
            to_display_path(target.history_path),
        ),
    )


def summarize_targets(targets: Iterable[HistoryTarget]) -> dict[str, dict[str, int]]:
    targets_list = list(targets)
    by_structure = Counter(target.structure for target in targets_list)
    by_stage = Counter(target.stage for target in targets_list)
    by_family = Counter(target.family for target in targets_list)

    return {
        "totals": {"histories": len(targets_list)},
        "by_structure": dict(by_structure),
        "by_stage": dict(by_stage),
        "by_family": dict(by_family),
    }
