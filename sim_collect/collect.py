#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CollectorSpec:
    name: str
    module: str
    description: str
    aliases: tuple[str, ...] = ()


COLLECTORS = (
    CollectorSpec(
        name="weight-sort",
        module="sim_collect.sim_env.weight_sort",
        description="Paired light/heavy PickPlaceCan force demonstrations.",
        aliases=("weight", "weight-sort-pick-place"),
    ),
    CollectorSpec(
        name="door",
        module="sim_collect.sim_env.door",
        description="Paired push/pull force-conditioned Door demonstrations.",
        aliases=("door-push-pull", "push-pull"),
    ),
    CollectorSpec(
        name="nut-assembly",
        module="sim_collect.sim_env.nut_assembly",
        description="Continuous NutAssemblySquare replay with live force sensing.",
        aliases=("nut", "square-nut", "nut-assembly-square"),
    ),
)


def _normalise(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _mapping() -> dict[str, CollectorSpec]:
    mapping: dict[str, CollectorSpec] = {}
    for spec in COLLECTORS:
        for name in (spec.name, *spec.aliases):
            key = _normalise(name)
            if key in mapping:
                raise RuntimeError(f"Duplicate collector alias: {key}")
            mapping[key] = spec
    return mapping


def _resolve(name: str) -> CollectorSpec:
    spec = _mapping().get(_normalise(name))
    if spec is None:
        choices = ", ".join(item.name for item in COLLECTORS)
        raise SystemExit(f"Unknown task {name!r}. Choose one of: {choices}")
    return spec


def _print_tasks() -> None:
    print("Available Force2Act simulation collectors:\n")
    width = max(len(spec.name) for spec in COLLECTORS)
    for spec in COLLECTORS:
        print(f"  {spec.name:<{width}}  {spec.description}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python -m sim_collect.collect <task> [task options]\n")
        _print_tasks()
        return 0

    if args[0] == "--list":
        _print_tasks()
        return 0

    task = _resolve(args[0])
    command = [sys.executable, "-m", task.module, *args[1:]]

    project_root = Path(__file__).resolve().parent.parent
    print(f"[collector] task={task.name}")
    print(f"[collector] module={task.module}")
    print()

    try:
        completed = subprocess.run(command, cwd=project_root, check=False)
    except KeyboardInterrupt:
        return 130

    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
