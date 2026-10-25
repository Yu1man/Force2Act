import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from evaluation.sim.common import EvaluationPolicy
from evaluation.sim.runner import evaluate_policy_rollouts
from force2act.data.split import get_train_val_names


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    cuda_states = None
    if torch.cuda.is_available():
        cuda_states = torch.cuda.get_rng_state_all()

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)

        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def run_sim_rollout_validation(
    model,
    normalizer,
    config,
    device,
    step,
    output_dir,
):
    dataset_cfg = config["dataset"]
    validation_cfg = config.get("validation", {})

    _, validation_demos = get_train_val_names(
        dataset_cfg["hdf5_path"]
    )

    rollout_policy = EvaluationPolicy(
        model=model,
        normalizer=normalizer,
        action_dim=int(dataset_cfg["action_dim"]),
        use_force=bool(dataset_cfg.get("use_force", False)),
    ).to(device)

    video_path = None

    if bool(validation_cfg.get("rollout_video", True)):
        rollout_dir = Path(output_dir) / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        video_path = (
            rollout_dir
            / f"rollout_step_{step:07d}.mp4"
        )

    was_training = model.training

    try:
        model.eval()

        with preserve_rng_state():
            seed = int(
                validation_cfg.get(
                    "rollout_seed",
                    104729,
                )
            )

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            metrics = evaluate_policy_rollouts(
                policy=rollout_policy,
                config=config,
                validation_demos=validation_demos,
                device=device,
                episodes=int(
                    validation_cfg.get(
                        "rollout_episodes",
                        5,
                    )
                ),
                video_path=video_path,
                render=False,
                return_episode_details=False,
            )

    finally:
        model.train(was_training)

    return metrics, video_path