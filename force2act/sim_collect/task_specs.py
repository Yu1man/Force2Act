from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    state_keys: tuple[str, ...]
    camera_keys: tuple[str, ...]
    force_keys: tuple[str, ...]
    action_dim: int = 7


COMMON_CAMERAS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)

# Use Cartesian force only for the publication models (force_dim=3).
# Torque is still saved by the collectors and can be enabled later by adding
# robot0_eef_torque to force_keys in a training config.
TASK_SPECS: dict[str, TaskSpec] = {
    "weight-sort": TaskSpec(
        name="weight-sort",
        state_keys=(
            "robot0_eef_pos",
            "robot0_gripper_qpos",
        ),
        camera_keys=COMMON_CAMERAS,
        force_keys=("robot0_eef_force",),
    ),
    "door": TaskSpec(
        name="door",
        state_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        camera_keys=COMMON_CAMERAS,
        force_keys=("robot0_eef_force",),
    ),
    "nut-assembly": TaskSpec(
        name="nut-assembly",
        state_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        camera_keys=COMMON_CAMERAS,
        force_keys=("robot0_eef_force",),
    ),
}

ALIASES = {
    "weight": "weight-sort",
    "weight_sort": "weight-sort",
    "weight-sort-pick-place": "weight-sort",
    "push-pull": "door",
    "door-push-pull": "door",
    "square-nut": "nut-assembly",
    "nut": "nut-assembly",
    "nut-assembly-square": "nut-assembly",
}


def resolve_task_spec(name: str) -> TaskSpec:
    key = name.strip().lower().replace("_", "-")
    key = ALIASES.get(key, key)
    if key not in TASK_SPECS:
        raise KeyError(f"Unknown task {name!r}. Choices: {sorted(TASK_SPECS)}")
    return TASK_SPECS[key]
