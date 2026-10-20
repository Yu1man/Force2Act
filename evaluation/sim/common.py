from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from force2act.data.image_preprocessing import preprocess_images
from force2act.data.normalization import ModalityStats, StandardNormalizer
from force2act.models.factory import build_model
from force2act.models.policy.flow_matching import sample_action


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalizer_from_state_dict(state: dict[str, Any]) -> StandardNormalizer:
    stats: dict[str, ModalityStats] = {}
    for name, values in state.items():
        stats[name] = ModalityStats(
            mean=values["mean"].detach().clone(),
            std=values["std"].detach().clone(),
        )
    return StandardNormalizer(stats)


class EvaluationPolicy(nn.Module):
    """Thin inference wrapper around the current Force2Act model API."""

    def __init__(
        self,
        model: nn.Module,
        normalizer: StandardNormalizer,
        *,
        action_dim: int,
        use_force: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self.normalizer = normalizer
        self.action_dim = int(action_dim)
        self.use_force = bool(use_force)

    @torch.no_grad()
    def sample(
        self,
        *,
        state: torch.Tensor,
        images: torch.Tensor,
        action_horizon: int,
        force: torch.Tensor | None = None,
        sampling_steps: int = 20,
        clip_actions: bool = True,
        binarize_gripper: bool = True,
    ) -> torch.Tensor:
        device = next(self.model.parameters()).device

        state = state.to(device=device, dtype=torch.float32)
        state = self.normalizer.normalize(state, "state")

        images = images.to(device=device)
        images = preprocess_images(images)

        if self.use_force:
            if force is None:
                raise ValueError("Force-enabled policy received force=None")
            force = force.to(device=device, dtype=torch.float32)
            force = self.normalizer.normalize(force, "force")
        else:
            force = None

        normalized_action = sample_action(
            model=self.model,
            images=images,
            state=state,
            action_horizon=int(action_horizon),
            action_dim=self.action_dim,
            num_steps=int(sampling_steps),
            force=force,
        )

        action = self.normalizer.denormalize(
            normalized_action,
            "actions",
        )

        if clip_actions:
            action = action.clamp(-1.0, 1.0)

        if binarize_gripper:
            action[..., -1] = torch.where(
                action[..., -1] >= 0.0,
                torch.ones_like(action[..., -1]),
                -torch.ones_like(action[..., -1]),
            )

        return action


def load_policy(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[EvaluationPolicy, dict[str, Any], dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "config" not in checkpoint or "model" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain at least 'config' and 'model'."
        )
    if "normalizer" not in checkpoint:
        raise KeyError("Checkpoint is missing normalizer statistics.")

    config = copy.deepcopy(checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    normalizer = normalizer_from_state_dict(
        checkpoint["normalizer"]
    ).to(device)

    dataset_cfg = config["dataset"]
    policy = EvaluationPolicy(
        model=model,
        normalizer=normalizer,
        action_dim=int(dataset_cfg["action_dim"]),
        use_force=bool(dataset_cfg.get("use_force", False)),
    ).to(device)
    policy.eval()

    return policy, config, checkpoint


def architecture_name(config: dict[str, Any]) -> str:
    return str(config["model"].get("architecture", "cross_attn"))


def training_seed(config: dict[str, Any]) -> int:
    return int(config.get("training", {}).get("seed", 42))
