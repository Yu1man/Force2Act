from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.sim.common import (
    architecture_name,
    load_policy,
    seed_everything,
    training_seed,
)
from evaluation.sim.runner import evaluate_policy_rollouts, resolve_splits


DEFAULT_EVAL_SEEDS = [104729, 130363, 155921, 196613, 262147]

METRICS = (
    "success_rate",
    "light_success_rate",
    "heavy_success_rate",
    "push_success_rate",
    "pull_success_rate",
    "nut_standard_success_rate",
    "nut_reward_success_rate",
    "stack_order_correct_rate",
    "stack_touching_rate",
    "stack_light_released_rate",
    "stack_heavy_on_table_rate",
    "mean_episode_length",
    "mean_return",
    "mean_inference_time_ms",
    "mean_num_replans",
)


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run publication-style simulation ablations across trained checkpoints, "
            "execution horizons, and fixed evaluation seeds."
        )
    )
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--exec-horizons", nargs="+", type=int, required=True)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--eval-seeds",
        nargs="+",
        type=int,
        default=DEFAULT_EVAL_SEEDS,
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if any(value <= 0 for value in args.exec_horizons):
        raise ValueError("All --exec-horizons must be positive")
    if args.sampling_steps <= 0:
        raise ValueError("--sampling-steps must be positive")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if len(set(args.eval_seeds)) != len(args.eval_seeds):
        raise ValueError("--eval-seeds must be unique")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    expected_valid_names: list[str] | None = None
    expected_task: str | None = None

    for checkpoint_index, checkpoint_path in enumerate(args.checkpoints):
        policy, config, checkpoint = load_policy(checkpoint_path, device)
        architecture = architecture_name(config)
        train_seed = training_seed(config)
        _, valid_names = resolve_splits(config, checkpoint)

        if expected_valid_names is None:
            expected_valid_names = list(valid_names)
        elif list(valid_names) != expected_valid_names:
            raise ValueError(
                "Validation split differs across checkpoints. "
                "Do not compare these checkpoints in one ablation run."
            )

        action_horizon = int(config["dataset"]["action_horizon"])

        for exec_horizon in args.exec_horizons:
            if exec_horizon > action_horizon:
                raise ValueError(
                    f"exec_horizon={exec_horizon} exceeds action_horizon="
                    f"{action_horizon} for {checkpoint_path}"
                )

            config.setdefault("evaluation", {})["exec_horizon"] = exec_horizon
            config["evaluation"]["sampling_steps"] = args.sampling_steps
            config["evaluation"]["video_save_mode"] = "none"

            repeat_rows: list[dict[str, Any]] = []

            for repeat_index, eval_seed in enumerate(args.eval_seeds, start=1):
                seed_everything(eval_seed)
                metrics = evaluate_policy_rollouts(
                    policy=policy,
                    config=config,
                    validation_demos=valid_names,
                    device=device,
                    episodes=args.episodes,
                    video_path=None,
                    render=False,
                    return_episode_details=False,
                )

                task_type = str(metrics["task_type"])
                if expected_task is None:
                    expected_task = task_type
                elif task_type != expected_task:
                    raise ValueError(
                        f"Task mismatch across checkpoints: {expected_task} vs {task_type}"
                    )

                row: dict[str, Any] = {
                    "checkpoint": str(Path(checkpoint_path).resolve()),
                    "checkpoint_index": checkpoint_index,
                    "architecture": architecture,
                    "training_seed": train_seed,
                    "exec_horizon": exec_horizon,
                    "sampling_steps": args.sampling_steps,
                    "repeat": repeat_index,
                    "evaluation_seed": eval_seed,
                    "task_type": task_type,
                    "num_episodes": int(metrics["num_episodes"]),
                }
                for metric in METRICS:
                    if metric in metrics:
                        row[metric] = float(metrics[metric])

                repeat_rows.append(row)
                run_rows.append(row)
                print(
                    f"[{architecture} seed={train_seed} H={exec_horizon} "
                    f"eval={eval_seed}] success={metrics['success_rate']:.3f}"
                )

            checkpoint_row: dict[str, Any] = {
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "architecture": architecture,
                "training_seed": train_seed,
                "exec_horizon": exec_horizon,
                "sampling_steps": args.sampling_steps,
                "task_type": expected_task,
                "evaluation_repeats": len(repeat_rows),
            }
            for metric in METRICS:
                values = [
                    float(row[metric])
                    for row in repeat_rows
                    if metric in row
                ]
                if values:
                    checkpoint_row[f"{metric}_mean_over_eval_seeds"] = mean(values)
                    checkpoint_row[f"{metric}_std_over_eval_seeds"] = sample_std(values)
            checkpoint_rows.append(checkpoint_row)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in checkpoint_rows:
        grouped[(str(row["architecture"]), int(row["exec_horizon"]))].append(row)

    paper_rows: list[dict[str, Any]] = []
    for (architecture, exec_horizon), rows in sorted(grouped.items()):
        paper_row: dict[str, Any] = {
            "architecture": architecture,
            "exec_horizon": exec_horizon,
            "sampling_steps": args.sampling_steps,
            "task_type": expected_task,
            "num_training_seeds": len(rows),
            "training_seeds": ";".join(
                str(row["training_seed"]) for row in rows
            ),
            "evaluation_seeds": ";".join(str(seed) for seed in args.eval_seeds),
        }

        for metric in METRICS:
            key = f"{metric}_mean_over_eval_seeds"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                paper_row[f"{metric}_mean"] = mean(values)
                paper_row[f"{metric}_std_across_training_seeds"] = sample_std(values)

        paper_rows.append(paper_row)

    write_csv(output_dir / "runs.csv", run_rows)
    write_csv(output_dir / "checkpoint_summary.csv", checkpoint_rows)
    write_csv(output_dir / "paper_summary.csv", paper_rows)

    payload = {
        "evaluation_seeds": args.eval_seeds,
        "sampling_steps": args.sampling_steps,
        "episodes_per_repeat": args.episodes,
        "runs": run_rows,
        "checkpoint_summary": checkpoint_rows,
        "paper_summary": paper_rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nPAPER SUMMARY")
    for row in paper_rows:
        success_mean = row.get("success_rate_mean", float("nan"))
        success_std = row.get(
            "success_rate_std_across_training_seeds",
            float("nan"),
        )
        print(
            f"{row['architecture']:12s} H={row['exec_horizon']:2d} "
            f"success={success_mean:.3f} ± {success_std:.3f} "
            f"(training seeds={row['num_training_seeds']})"
        )

    print(f"\nSaved: {output_dir / 'paper_summary.csv'}")


if __name__ == "__main__":
    main()
