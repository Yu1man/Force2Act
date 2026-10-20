from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from evaluation.sim.common import load_policy, seed_everything
from evaluation.sim.runner import evaluate_policy_rollouts, resolve_splits


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)[:120]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save one reproducible MP4 for each selected simulation demo."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--exec-horizon", type=int, default=16)
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=104729)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, config, checkpoint = load_policy(args.checkpoint, device)

    action_horizon = int(config["dataset"]["action_horizon"])
    if args.exec_horizon <= 0 or args.exec_horizon > action_horizon:
        raise ValueError(
            f"--exec-horizon must be in [1, {action_horizon}]"
        )
    if args.sampling_steps <= 0:
        raise ValueError("--sampling-steps must be positive")

    config.setdefault("evaluation", {})["exec_horizon"] = args.exec_horizon
    config["evaluation"]["sampling_steps"] = args.sampling_steps
    config["evaluation"]["video_save_mode"] = "first"

    train_names, valid_names = resolve_splits(config, checkpoint)
    names = train_names if args.split == "train" else valid_names
    if args.episodes is not None:
        names = names[: args.episodes]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, demo_name in enumerate(names):
        eval_seed = int(args.seed + index)
        seed_everything(eval_seed)
        video_path = output_dir / f"{index:03d}_{safe_name(demo_name)}.mp4"

        metrics = evaluate_policy_rollouts(
            policy=policy,
            config=config,
            validation_demos=[demo_name],
            device=device,
            episodes=1,
            video_path=video_path,
            render=False,
            return_episode_details=True,
        )
        success = bool(metrics["success_rate"] >= 0.5)
        final_path = video_path.with_name(
            f"{video_path.stem}_{'success' if success else 'failure'}.mp4"
        )
        if video_path.exists():
            video_path.rename(final_path)
        else:
            final_path = video_path

        rows.append(
            {
                "demo": demo_name,
                "evaluation_seed": eval_seed,
                "success": success,
                "video": str(final_path),
                "mean_inference_time_ms": float(
                    metrics.get("mean_inference_time_ms", 0.0)
                ),
            }
        )
        print(
            f"[{index + 1}/{len(names)}] {demo_name}: "
            f"{'success' if success else 'failure'} -> {final_path.name}"
        )

    (output_dir / "video_results.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(rows)} videos to {output_dir}")


if __name__ == "__main__":
    main()
