from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaluation.sim.common import (
    architecture_name,
    load_policy,
    seed_everything,
    training_seed,
)
from evaluation.sim.runner import evaluate_policy_rollouts, resolve_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a clean closed-loop simulation rollout evaluation."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--exec-horizon", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--video", default=None)
    parser.add_argument(
        "--video-mode",
        choices=("none", "first", "all", "success", "failure"),
        default="first",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.exec_horizon <= 0:
        raise ValueError("--exec-horizon must be positive")
    if args.sampling_steps <= 0:
        raise ValueError("--sampling-steps must be positive")
    if args.episodes is not None and args.episodes <= 0:
        raise ValueError("--episodes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, config, checkpoint = load_policy(args.checkpoint, device)

    action_horizon = int(config["dataset"]["action_horizon"])
    if args.exec_horizon > action_horizon:
        raise ValueError(
            f"exec_horizon={args.exec_horizon} exceeds trained "
            f"action_horizon={action_horizon}"
        )

    config.setdefault("evaluation", {})["exec_horizon"] = int(args.exec_horizon)
    config["evaluation"]["sampling_steps"] = int(args.sampling_steps)
    config["evaluation"]["video_save_mode"] = args.video_mode

    train_names, valid_names = resolve_splits(config, checkpoint)
    demo_names = train_names if args.split == "train" else valid_names

    seed_everything(args.seed)
    metrics = evaluate_policy_rollouts(
        policy=policy,
        config=config,
        validation_demos=demo_names,
        device=device,
        episodes=args.episodes,
        video_path=args.video,
        render=args.render,
        return_episode_details=True,
    )

    print("\nForce2Act simulation rollout")
    print(f"  architecture: {architecture_name(config)}")
    print(f"  training seed: {training_seed(config)}")
    print(f"  evaluation seed: {args.seed}")
    print(f"  task: {metrics['task_type']}")
    print(f"  exec horizon: {args.exec_horizon}")
    print(f"  sampling steps: {args.sampling_steps}")
    print(f"  episodes: {int(metrics['num_episodes'])}")
    print(f"  success rate: {100.0 * metrics['success_rate']:.1f}%")
    print(f"  mean inference: {metrics['mean_inference_time_ms']:.2f} ms/replan")

    if metrics["task_type"] == "door":
        print(f"  push: {100.0 * metrics['push_success_rate']:.1f}%")
        print(f"  pull: {100.0 * metrics['pull_success_rate']:.1f}%")
    elif metrics["task_type"] == "pick_place":
        print(f"  light: {100.0 * metrics['light_success_rate']:.1f}%")
        print(f"  heavy: {100.0 * metrics['heavy_success_rate']:.1f}%")
    elif metrics["task_type"] == "nut_assembly_square":
        print(
            f"  native task success: "
            f"{100.0 * metrics['nut_standard_success_rate']:.1f}%"
        )

    output_path = (
        Path(args.output_json)
        if args.output_json
        else Path(args.checkpoint).with_name(
            f"rollout_{args.split}_h{args.exec_horizon}.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
