from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from tqdm import tqdm

from sim_collect.common import (
    create_test_mask,
    get_gripper,
    get_model_xml,
    get_robot,
    get_state_vector,
    image_to_hwc_uint8,
    json_default,
    read_force_torque,
)



# Door opening needs orientation state because the demonstrations contain
# substantial wrist rotation before the grasp.
STATE_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
FORCE_KEYS = (
    "robot0_eef_force",
    "robot0_eef_torque",
)
CAMERA_NAMES = (
    "agentview",
    "robot0_eye_in_hand",
)
CAMERA_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
)
EXPECTED_ACTION_DIM = 7


class Stage(IntEnum):
    # Shared approach / grasp followed by a force probe and a direction-aware
    # push-or-pull opening motion.
    MOVE_ABOVE = 0
    ORIENT_AT_CLEARANCE = 1
    MOVE_PREGRASP = 2
    INSERT_TO_HANDLE = 3
    CLOSE_GRIPPER = 4
    VERIFY_GRASP = 5
    SEAT_GRASP = 6

    # Force-conditioned portion.
    FORCE_BASELINE = 7
    FORCE_PROBE = 8
    # Retained IDs keep old HDF5 stage numbering compatible.
    RETURN_TO_COMMON_POSE = 9
    PRECLASS_GUARD = 10
    # The selected direction is executed along the handle arc.
    ADAPTIVE_ARC_PULL = 11

    HOLD_OPEN = 12
    RELEASE = 13
    RETREAT = 14
    DONE = 15
    FAILED = 16


@dataclass(frozen=True)
class EpisodeSpec:
    pair_id: int
    true_class: str
    hinge_damping: float
    hinge_friction: float
    placement_seed: int
    resistance_score: float

    # Hidden physical properties used only by the simulator / success checker.
    # true_class is either "push" or "pull". The two members of a pair share
    # identical appearance, placement, damping, and friction; only the invisible
    # one-way hinge range differs.
    damage_force_limit_n: float
    hard_break_force_n: float
    damage_impulse_limit_ns: float
    max_safe_open_speed_rad_s: float


@dataclass
class GripperGeometry:
    left_body_id: int | None
    right_body_id: int | None
    left_geom_ids: tuple[int, ...]
    right_geom_ids: tuple[int, ...]
    left_name: str
    right_name: str


@dataclass
class DoorGeometry:
    hinge_name: str
    hinge_id: int
    handle_site_name: str
    handle_site_id: int
    handle_geom_ids: tuple[int, ...]
    primary_handle_geom_id: int | None
    primary_handle_name: str


@dataclass
class EpisodeResult:
    success: bool
    reason: str
    trajectory: dict[str, Any] | None
    predicted_class: str | None
    force_signal: float | None




def import_robosuite():
    try:
        import robosuite as suite
        try:
            from robosuite.controllers import load_composite_controller_config
        except ImportError:
            from robosuite import load_composite_controller_config
    except ImportError as exc:
        raise ImportError("robosuite is required") from exc
    return suite, load_composite_controller_config


def create_environment(args):
    suite, load_controller = import_robosuite()
    controller_config = load_controller(controller="BASIC")
    from robosuite.utils.placement_samplers import UniformRandomSampler

    np.random.seed(args.seed)

    # Robosuite's stock Door environment already randomises the complete door
    # assembly (and therefore the handle) on reset. Expose the ranges explicitly
    # so they are recorded and can be widened later without randomising the
    # handle independently from the door.
    placement_initializer = UniformRandomSampler(
        name="ForceDoorSampler",
        mujoco_objects=None,
        x_range=[args.door_x_min, args.door_x_max],
        y_range=[args.door_y_min, args.door_y_max],
        rotation=(
            np.deg2rad(args.door_yaw_min_deg),
            np.deg2rad(args.door_yaw_max_deg),
        ),
        rotation_axis="z",
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        # Robosuite Door samples these offsets relative to its table origin.
        # Keep the same stock reference so custom ranges do not move the door
        # to an unintended world location.
        reference_pos=(-0.2, -0.35, 0.8),
        z_offset=0.0,
    )

    env_kwargs = {
        "robots": "Panda",
        "controller_configs": controller_config,
        # This image-sequence FSM is for the fixed handle. A spring latch requires
        # an additional handle-turn stage and is deliberately not mixed into this
        # first reliable collector.
        "use_latch": False,
        "initialization_noise": None,
        "placement_initializer": placement_initializer,
        # Robosuite 1.5.1's Door constructor has no `seed` keyword. Pair-level
        # determinism is handled immediately before env.reset() by
        # reseed_environment_for_pair().
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "use_object_obs": True,
        "camera_names": list(CAMERA_NAMES),
        "camera_heights": args.image_size,
        "camera_widths": args.image_size,
        "camera_depths": False,
        "reward_shaping": False,
        "control_freq": args.control_freq,
        "horizon": args.max_episode_steps + args.settle_steps + 100,
        "ignore_done": True,
        "hard_reset": False,
    }

    env = suite.make("Door", **env_kwargs)

    low, high = env.action_spec
    if np.asarray(low).shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(
            "Expected 7D OSC action "
            "[dx,dy,dz,droll,dpitch,dyaw,grip], "
            f"got {np.asarray(low).shape}"
        )
    if np.asarray(high).shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(f"Unexpected action high shape: {np.asarray(high).shape}")

    serialisable_env_kwargs = dict(env_kwargs)
    # Robomimic later forwards env_kwargs directly into robosuite.make().
    # A JSON dictionary is not a valid placement sampler object and causes:
    #   AttributeError: 'dict' object has no attribute 'reset'
    # Keep the ranges in force_door_task metadata below, but omit the runtime
    # sampler object from the replayable env_kwargs.
    serialisable_env_kwargs.pop("placement_initializer", None)

    env_meta = {
        "env_name": "Door",
        "type": 1,
        "env_version": getattr(suite, "__version__", "unknown"),
        "env_kwargs": serialisable_env_kwargs,
        "force_door_task": {
            "fsm": (
                "move_above -> orient_at_clearance -> move_pregrasp -> "
                "insert_to_handle -> close -> verify -> force_baseline -> "
                "common_pull_probe -> force_guided_push_or_pull_arc"
            ),
            "classification_source": (
                "baseline-subtracted reaction force during a shared pull probe; "
                "high force means pull is blocked and the door must be pushed"
            ),
            "paired_scenes": True,
            "decision_mode": str(args.decision_mode),
            "collector_seed": int(args.seed),
            "door_placement_randomisation": {
                "x_range": [float(args.door_x_min), float(args.door_x_max)],
                "y_range": [float(args.door_y_min), float(args.door_y_max)],
                "yaw_range_deg": [
                    float(args.door_yaw_min_deg),
                    float(args.door_yaw_max_deg),
                ],
                "pair_shared": True,
                "handle_moves_rigidly_with_door": True,
            },
        },
    }
    return env, env_meta


def all_names(model, kind: str) -> list[str]:
    aliases = {
        "joint": ("njnt", "joint_id2name"),
        "jnt": ("njnt", "joint_id2name"),
        "site": ("nsite", "site_id2name"),
        "body": ("nbody", "body_id2name"),
        "geom": ("ngeom", "geom_id2name"),
    }
    if kind not in aliases:
        raise ValueError(f"Unsupported name kind: {kind}")

    count_attr, function_attr = aliases[kind]
    count = int(getattr(model, count_attr))
    function = getattr(model, function_attr)

    output = []
    for index in range(count):
        try:
            name = function(index)
        except Exception:
            name = None
        if name:
            output.append(str(name))
    return output


def choose_shortest(names: Iterable[str]) -> str | None:
    values = [str(name) for name in names]
    if not values:
        return None
    return sorted(values, key=lambda value: (len(value), value))[0]


def resolve_door_geometry(env) -> DoorGeometry:
    model = env.sim.model
    joint_names = all_names(model, "joint")
    site_names = all_names(model, "site")
    geom_names = all_names(model, "geom")

    hinge_candidates = [
        name
        for name in joint_names
        if "hinge" in name.lower()
        and "handle" not in name.lower()
        and "latch" not in name.lower()
    ]
    if not hinge_candidates:
        hinge_candidates = [
            name
            for name in joint_names
            if "door" in name.lower()
            and "handle" not in name.lower()
            and "latch" not in name.lower()
        ]
    hinge_name = choose_shortest(hinge_candidates)
    if hinge_name is None:
        raise RuntimeError(f"Could not resolve door hinge. Joints: {joint_names}")

    handle_sites = [
        name
        for name in site_names
        if "handle" in name.lower() or "latch" in name.lower()
    ]
    handle_site_name = choose_shortest(handle_sites)
    if handle_site_name is None:
        raise RuntimeError(f"Could not resolve handle site. Sites: {site_names}")

    hinge_id = int(model.joint_name2id(hinge_name))
    handle_site_id = int(model.site_name2id(handle_site_name))

    handle_geom_ids = []
    for name in geom_names:
        lower = name.lower()
        if "handle" not in lower and "latch" not in lower:
            continue
        geom_id = int(model.geom_name2id(name))
        # Prefer collision geoms. Visual-only geoms are normally group 1.
        group = int(model.geom_group[geom_id])
        if group == 0:
            handle_geom_ids.append(geom_id)

    if not handle_geom_ids:
        for name in geom_names:
            lower = name.lower()
            if "handle" in lower or "latch" in lower:
                handle_geom_ids.append(int(model.geom_name2id(name)))

    site_position = np.asarray(
        env.sim.data.site_xpos[handle_site_id],
        dtype=np.float64,
    )

    primary_handle_geom_id = None
    primary_handle_name = handle_site_name
    if handle_geom_ids:
        primary_handle_geom_id = min(
            handle_geom_ids,
            key=lambda geom_id: float(
                np.linalg.norm(
                    np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
                    - site_position
                )
            ),
        )
        name = model.geom_id2name(primary_handle_geom_id)
        primary_handle_name = (
            str(name) if name is not None else f"geom_{primary_handle_geom_id}"
        )

    return DoorGeometry(
        hinge_name=hinge_name,
        hinge_id=hinge_id,
        handle_site_name=handle_site_name,
        handle_site_id=handle_site_id,
        handle_geom_ids=tuple(handle_geom_ids),
        primary_handle_geom_id=primary_handle_geom_id,
        primary_handle_name=primary_handle_name,
    )


def _rank_finger_names(
    names: list[str],
    side: str,
) -> list[str]:
    """Rank only names whose *finger token* belongs to the requested side.

    Robosuite Panda names contain the arm side as a prefix, for example:

        gripper0_right_leftfinger
        gripper0_right_rightfinger

    A naive check such as ``"right" in name`` therefore matches BOTH fingers.
    That was the reason the previous collector printed the left finger for both
    references and generated an arbitrary wrist rotation.
    """
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported finger side: {side!r}")

    opposite = "right" if side == "left" else "left"

    # Match the side immediately attached to the finger / pad token, not the
    # robot-arm prefix. Covers common Robosuite naming variants.
    wanted_patterns = (
        rf"{side}[_\-]?finger",
        rf"finger[_\-]?{side}",
        rf"{side}[_\-]?pad",
        rf"pad[_\-]?{side}",
    )
    opposite_patterns = (
        rf"{opposite}[_\-]?finger",
        rf"finger[_\-]?{opposite}",
        rf"{opposite}[_\-]?pad",
        rf"pad[_\-]?{opposite}",
    )

    ranked: list[tuple[int, int, str]] = []

    for name in names:
        lower = name.lower()

        if not any(re.search(pattern, lower) for pattern in wanted_patterns):
            continue
        if any(re.search(pattern, lower) for pattern in opposite_patterns):
            continue

        score = 0

        # Prefer physical collision references over visual-only references.
        if "visual" in lower:
            score += 100
        if "collision" in lower or "coll" in lower:
            score -= 20
        if "fingerpad" in lower or "finger_pad" in lower:
            score -= 10

        # Prefer the actual finger body over very small helper / tip objects.
        if "tip" in lower:
            score += 5

        ranked.append((score, len(name), name))

    ranked.sort()
    return [name for _, _, name in ranked]


def resolve_gripper_geometry(env) -> GripperGeometry:
    model = env.sim.model
    body_names = all_names(model, "body")
    geom_names = all_names(model, "geom")

    left_body_names = _rank_finger_names(body_names, "left")
    right_body_names = _rank_finger_names(body_names, "right")

    left_body_id = (
        int(model.body_name2id(left_body_names[0]))
        if left_body_names
        else None
    )
    right_body_id = (
        int(model.body_name2id(right_body_names[0]))
        if right_body_names
        else None
    )

    left_geom_names = _rank_finger_names(geom_names, "left")
    right_geom_names = _rank_finger_names(geom_names, "right")

    left_geom_ids = tuple(
        int(model.geom_name2id(name))
        for name in left_geom_names
        if int(model.geom_group[int(model.geom_name2id(name))]) == 0
    )
    right_geom_ids = tuple(
        int(model.geom_name2id(name))
        for name in right_geom_names
        if int(model.geom_group[int(model.geom_name2id(name))]) == 0
    )

    if not left_geom_ids:
        left_geom_ids = tuple(
            int(model.geom_name2id(name))
            for name in left_geom_names
        )
    if not right_geom_ids:
        right_geom_ids = tuple(
            int(model.geom_name2id(name))
            for name in right_geom_names
        )

    if (
        (left_body_id is None and not left_geom_ids)
        or (right_body_id is None and not right_geom_ids)
    ):
        raise RuntimeError(
            "Could not resolve both Panda fingers. "
            f"Left bodies={left_body_names[:5]}, "
            f"right bodies={right_body_names[:5]}, "
            f"left geoms={left_geom_names[:5]}, "
            f"right geoms={right_geom_names[:5]}"
        )

    if (
        left_body_id is not None
        and right_body_id is not None
        and left_body_id == right_body_id
    ):
        raise RuntimeError(
            "Resolved the same body for both fingers. "
            f"left={left_body_names[:3]}, right={right_body_names[:3]}"
        )

    if left_geom_ids and right_geom_ids:
        overlap = set(left_geom_ids).intersection(right_geom_ids)
        if overlap:
            overlapping_names = [
                str(model.geom_id2name(geom_id))
                for geom_id in sorted(overlap)
            ]
            raise RuntimeError(
                "Resolved overlapping collision geoms for left and right "
                f"fingers: {overlapping_names}"
            )

    return GripperGeometry(
        left_body_id=left_body_id,
        right_body_id=right_body_id,
        left_geom_ids=left_geom_ids,
        right_geom_ids=right_geom_ids,
        left_name=(
            left_body_names[0]
            if left_body_names
            else left_geom_names[0]
        ),
        right_name=(
            right_body_names[0]
            if right_body_names
            else right_geom_names[0]
        ),
    )






def get_eef_site_id(env) -> int:
    site_id = get_robot(env).eef_site_id
    if isinstance(site_id, dict):
        if "right" in site_id:
            site_id = site_id["right"]
        else:
            site_id = next(iter(site_id.values()))
    return int(site_id)


def get_eef_position(env, obs: dict | None = None) -> np.ndarray:
    if obs is not None and "robot0_eef_pos" in obs:
        return np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        ).reshape(3)

    return np.asarray(
        env.sim.data.site_xpos[get_eef_site_id(env)],
        dtype=np.float64,
    ).copy()


def get_eef_rotation(env) -> np.ndarray:
    return np.asarray(
        env.sim.data.site_xmat[get_eef_site_id(env)],
        dtype=np.float64,
    ).reshape(3, 3).copy()


def get_finger_centre(
    env,
    body_id: int | None,
    geom_ids: tuple[int, ...],
) -> np.ndarray:
    if body_id is not None:
        return np.asarray(
            env.sim.data.body_xpos[body_id],
            dtype=np.float64,
        ).copy()

    if not geom_ids:
        raise RuntimeError("No body or geom reference for finger")

    points = [
        np.asarray(env.sim.data.geom_xpos[geom_id], dtype=np.float64)
        for geom_id in geom_ids
    ]
    return np.mean(np.stack(points), axis=0)


def get_handle_position(env, geometry: DoorGeometry) -> np.ndarray:
    if geometry.primary_handle_geom_id is not None:
        return np.asarray(
            env.sim.data.geom_xpos[geometry.primary_handle_geom_id],
            dtype=np.float64,
        ).copy()

    return np.asarray(
        env.sim.data.site_xpos[geometry.handle_site_id],
        dtype=np.float64,
    ).copy()




def joint_qpos(env, joint_id: int) -> float:
    qpos_address = int(env.sim.model.jnt_qposadr[joint_id])
    return float(env.sim.data.qpos[qpos_address])


def set_hinge_resistance(
    env,
    hinge_id: int,
    damping: float,
    friction: float,
) -> None:
    dof_id = int(env.sim.model.jnt_dofadr[hinge_id])
    env.sim.model.dof_damping[dof_id] = float(damping)
    env.sim.model.dof_frictionloss[dof_id] = float(friction)
    env.sim.forward()


def reseed_environment_for_pair(env, seed: int) -> None:
    """Give both members of a light/heavy pair the same placement random state.

    Robosuite 1.5.1 UniformRandomSampler uses NumPy's global RNG. Some newer
    builds also expose environment / sampler Generator objects, so those are
    reseeded when present. Calling this immediately before env.reset() makes
    both members of a matched pair receive the same door pose.
    """
    seed = int(seed)
    np.random.seed(seed)
    generator = np.random.default_rng(seed)

    if hasattr(env, "rng"):
        env.rng = generator

    sampler = getattr(env, "placement_initializer", None)
    if sampler is not None and hasattr(sampler, "rng"):
        sampler.rng = generator


def resistance_derived_safety_limits(
    hinge_damping: float,
    hinge_friction: float,
    args: argparse.Namespace,
) -> dict[str, float]:
    """Derive door strength from hinge resistance with one shared rule.

    This is a task-design proxy, not a material-science law. Crucially, the
    evaluator does not branch on the light/heavy label. Every door uses the
    same deterministic mapping from its hidden damping and friction values to
    its allowed force, accumulated overload, and opening speed.
    """
    damping = max(float(hinge_damping), 0.0)
    friction = max(float(hinge_friction), 0.0)

    resistance_score = (
        damping
        + args.resistance_friction_weight * friction
    )

    damage_force_limit_n = (
        args.damage_force_base_n
        + args.damage_force_per_resistance_n * resistance_score
    )
    damage_force_limit_n = float(
        np.clip(
            damage_force_limit_n,
            args.damage_force_min_n,
            args.damage_force_max_n,
        )
    )

    hard_break_force_n = float(
        damage_force_limit_n * args.hard_break_force_factor
    )
    damage_impulse_limit_ns = float(
        damage_force_limit_n * args.damage_impulse_time_s
    )

    max_safe_open_speed_rad_s = float(
        np.clip(
            args.safe_speed_base_rad_s
            + args.safe_speed_per_resistance_rad_s * resistance_score,
            args.safe_speed_min_rad_s,
            args.safe_speed_max_rad_s,
        )
    )

    return {
        "resistance_score": float(resistance_score),
        "damage_force_limit_n": damage_force_limit_n,
        "hard_break_force_n": hard_break_force_n,
        "damage_impulse_limit_ns": damage_impulse_limit_ns,
        "max_safe_open_speed_rad_s": max_safe_open_speed_rad_s,
    }


def hinge_axis_world(env, hinge_id: int) -> np.ndarray:
    model = env.sim.model
    body_id = int(model.jnt_bodyid[hinge_id])
    local_axis = np.asarray(
        model.jnt_axis[hinge_id],
        dtype=np.float64,
    )
    body_rotation = np.asarray(
        env.sim.data.body_xmat[body_id],
        dtype=np.float64,
    ).reshape(3, 3)

    axis = body_rotation @ local_axis
    return axis / max(float(np.linalg.norm(axis)), 1e-8)


def hinge_anchor_world(env, hinge_id: int) -> np.ndarray:
    model = env.sim.model
    body_id = int(model.jnt_bodyid[hinge_id])
    body_position = np.asarray(
        env.sim.data.body_xpos[body_id],
        dtype=np.float64,
    )
    body_rotation = np.asarray(
        env.sim.data.body_xmat[body_id],
        dtype=np.float64,
    ).reshape(3, 3)

    return (
        body_position
        + body_rotation
        @ np.asarray(model.jnt_pos[hinge_id], dtype=np.float64)
    )


def door_frame(
    env,
    geometry: DoorGeometry,
    *,
    reference_tangent: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Returns:
        radius: hinge -> handle inside the door plane
        tangent: signed opening direction
        axis: hinge axis
        opening_sign: sign used for signed door-angle progress
    """
    handle_position = get_handle_position(env, geometry)
    anchor = hinge_anchor_world(env, geometry.hinge_id)
    axis = hinge_axis_world(env, geometry.hinge_id)

    radius = handle_position - anchor
    radius = radius - axis * float(np.dot(radius, axis))
    radius = radius / max(float(np.linalg.norm(radius)), 1e-8)

    positive_tangent = np.cross(axis, radius)
    positive_tangent = positive_tangent / max(
        float(np.linalg.norm(positive_tangent)),
        1e-8,
    )

    if reference_tangent is not None:
        tangent = positive_tangent
        if float(np.dot(tangent, reference_tangent)) < 0:
            tangent *= -1.0

        # Determine which q direction generated this tangent.
        opening_sign = (
            1.0
            if float(np.dot(tangent, positive_tangent)) >= 0
            else -1.0
        )
        return radius, tangent, axis, opening_sign

    q0 = joint_qpos(env, geometry.hinge_id)
    joint_range = np.asarray(
        env.sim.model.jnt_range[geometry.hinge_id],
        dtype=np.float64,
    )

    positive_room = float(joint_range[1] - q0)
    negative_room = float(q0 - joint_range[0])
    opening_sign = 1.0 if positive_room >= negative_room else -1.0

    tangent = positive_tangent * opening_sign
    return radius, tangent, axis, opening_sign


def robot_facing_surface_normal(
    handle_position: np.ndarray,
    eef_position: np.ndarray,
    radius: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    # The panel normal is perpendicular to its radial direction and hinge axis.
    normal = np.cross(axis, radius)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)

    if float(np.dot(eef_position - handle_position, normal)) < 0:
        normal *= -1.0
    return normal


def axis_angle_matrix(
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    x, y, z = axis

    cross_matrix = np.asarray(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)

    return (
        identity
        + np.sin(angle_rad) * cross_matrix
        + (1.0 - np.cos(angle_rad))
        * (cross_matrix @ cross_matrix)
    )


def rotation_error_vector(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> np.ndarray:
    error_rotation = (
        np.asarray(target_rotation)
        @ np.asarray(current_rotation).T
    )

    cosine = np.clip(
        (np.trace(error_rotation) - 1.0) * 0.5,
        -1.0,
        1.0,
    )
    angle = float(np.arccos(cosine))

    if angle < 1e-7:
        return np.zeros(3, dtype=np.float64)

    if np.pi - angle < 1e-5:
        diagonal = np.maximum(
            (np.diag(error_rotation) + 1.0) * 0.5,
            0.0,
        )
        axis = np.sqrt(diagonal)
        axis[0] = np.copysign(
            axis[0],
            error_rotation[2, 1] - error_rotation[1, 2],
        )
        axis[1] = np.copysign(
            axis[1],
            error_rotation[0, 2] - error_rotation[2, 0],
        )
        axis[2] = np.copysign(
            axis[2],
            error_rotation[1, 0] - error_rotation[0, 1],
        )
        axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    else:
        axis = np.asarray(
            [
                error_rotation[2, 1] - error_rotation[1, 2],
                error_rotation[0, 2] - error_rotation[2, 0],
                error_rotation[1, 0] - error_rotation[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * np.sin(angle))

    return axis * angle


def rotation_angle(
    current_rotation: np.ndarray,
    target_rotation: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(
            rotation_error_vector(
                current_rotation,
                target_rotation,
            )
        )
    )


def calibrated_grasp_rotation(
    env,
    gripper_geometry: GripperGeometry,
    desired_forward_world: np.ndarray,
    desired_finger_axis_world: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calibrate from the actual simulated Panda geometry instead of assuming
    which EEF matrix column is palm-forward or finger-closing.

    The local direction from EEF to the finger midpoint is treated as the tool
    forward direction. The direction from left finger to right finger is the
    finger-closing axis. We then map those measured local directions onto the
    desired world directions.
    """
    current_rotation = get_eef_rotation(env)
    eef_position = get_eef_position(env)

    left_position = get_finger_centre(
        env,
        gripper_geometry.left_body_id,
        gripper_geometry.left_geom_ids,
    )
    right_position = get_finger_centre(
        env,
        gripper_geometry.right_body_id,
        gripper_geometry.right_geom_ids,
    )
    finger_midpoint = 0.5 * (left_position + right_position)

    finger_axis_world = right_position - left_position
    finger_separation = float(np.linalg.norm(finger_axis_world))
    if not 0.005 <= finger_separation <= 0.20:
        raise RuntimeError(
            "Invalid Panda finger geometry calibration. "
            f"left={left_position}, right={right_position}, "
            f"separation={finger_separation:.6f} m. "
            "The left and right finger references were probably resolved "
            "incorrectly."
        )
    finger_axis_world = finger_axis_world / finger_separation

    # IMPORTANT:
    # Robosuite's Panda EEF site is near the gripper / finger-tip region, while
    # the resolved finger BODY centres sit farther back toward the wrist.
    # Therefore (finger_midpoint - eef_position) points BACKWARDS, not toward
    # the handle. Using it as tool-forward caused the wrist to rotate right and
    # present the back of the hand to the door.
    #
    # The true approach direction of the fingers is the opposite vector.
    tool_forward_world = eef_position - finger_midpoint
    tool_forward_world = (
        tool_forward_world
        - finger_axis_world
        * float(np.dot(tool_forward_world, finger_axis_world))
    )
    tool_reach = float(np.linalg.norm(tool_forward_world))
    if tool_reach < 0.015:
        raise RuntimeError(
            "Could not infer Panda tool-forward direction from finger geometry; "
            f"measured reach={tool_reach:.5f} m"
        )
    tool_forward_world = tool_forward_world / tool_reach

    finger_local = current_rotation.T @ finger_axis_world
    forward_local = current_rotation.T @ tool_forward_world

    finger_local = finger_local / max(
        float(np.linalg.norm(finger_local)),
        1e-8,
    )
    forward_local = (
        forward_local
        - finger_local * float(np.dot(forward_local, finger_local))
    )
    forward_local = forward_local / max(
        float(np.linalg.norm(forward_local)),
        1e-8,
    )
    side_local = np.cross(finger_local, forward_local)
    side_local = side_local / max(
        float(np.linalg.norm(side_local)),
        1e-8,
    )

    local_basis = np.column_stack(
        [side_local, finger_local, forward_local]
    )

    desired_forward = np.asarray(
        desired_forward_world,
        dtype=np.float64,
    )
    desired_forward = desired_forward / max(
        float(np.linalg.norm(desired_forward)),
        1e-8,
    )

    def build_candidate(finger_sign: float) -> np.ndarray:
        desired_finger = (
            np.asarray(desired_finger_axis_world, dtype=np.float64)
            * finger_sign
        )
        desired_finger = (
            desired_finger
            - desired_forward
            * float(np.dot(desired_finger, desired_forward))
        )
        desired_finger = desired_finger / max(
            float(np.linalg.norm(desired_finger)),
            1e-8,
        )
        desired_side = np.cross(desired_finger, desired_forward)
        desired_side = desired_side / max(
            float(np.linalg.norm(desired_side)),
            1e-8,
        )
        desired_basis = np.column_stack(
            [desired_side, desired_finger, desired_forward]
        )
        candidate = desired_basis @ local_basis.T

        # Numerical repair to a proper rotation.
        u, _, vt = np.linalg.svd(candidate)
        candidate = u @ vt
        if np.linalg.det(candidate) < 0:
            u[:, -1] *= -1
            candidate = u @ vt
        return candidate

    candidate_a = build_candidate(+1.0)
    candidate_b = build_candidate(-1.0)

    target_rotation = min(
        (candidate_a, candidate_b),
        key=lambda candidate: rotation_angle(
            current_rotation,
            candidate,
        ),
    )

    return (
        target_rotation,
        tool_reach,
        finger_midpoint,
        left_position,
        right_position,
        finger_separation,
    )


def pose_action(
    current_position: np.ndarray,
    target_position: np.ndarray,
    gripper_command: float,
    translation_step: float,
    *,
    current_rotation: np.ndarray | None = None,
    target_rotation: np.ndarray | None = None,
    orientation_step: float = 0.20,
) -> np.ndarray:
    action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)

    action[:3] = np.clip(
        (
            np.asarray(target_position, dtype=np.float64)
            - np.asarray(current_position, dtype=np.float64)
        )
        / max(float(translation_step), 1e-6),
        -1.0,
        1.0,
    ).astype(np.float32)

    if current_rotation is not None and target_rotation is not None:
        error = rotation_error_vector(
            current_rotation,
            target_rotation,
        )
        action[3:6] = np.clip(
            error / max(float(orientation_step), 1e-6),
            -1.0,
            1.0,
        ).astype(np.float32)

    action[6] = float(np.clip(gripper_command, -1.0, 1.0))
    return action


def limit_pose_action(
    action: np.ndarray,
    *,
    max_translation_norm: float,
    max_rotation_norm: float,
) -> np.ndarray:
    """Limit Cartesian command magnitude without changing its direction."""
    output = np.asarray(action, dtype=np.float32).copy()

    translation_norm = float(np.linalg.norm(output[:3]))
    if translation_norm > max_translation_norm > 0.0:
        output[:3] *= max_translation_norm / translation_norm

    rotation_norm = float(np.linalg.norm(output[3:6]))
    if rotation_norm > max_rotation_norm > 0.0:
        output[3:6] *= max_rotation_norm / rotation_norm

    return output


def hold_action(gripper_command: float) -> np.ndarray:
    action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
    action[6] = float(np.clip(gripper_command, -1.0, 1.0))
    return action








def make_observation(
    env,
    observation: dict[str, Any],
    geometry: DoorGeometry,
    stage: Stage,
    allow_missing_force: bool,
) -> dict[str, Any]:
    force, torque = read_force_torque(
        env,
        allow_missing_force,
    )

    output = dict(observation)
    output[FORCE_KEYS[0]] = force
    output[FORCE_KEYS[1]] = torque
    output["door_angle"] = np.asarray(
        [joint_qpos(env, geometry.hinge_id)],
        dtype=np.float32,
    )
    output["handle_pos"] = get_handle_position(
        env,
        geometry,
    ).astype(np.float32)
    output["fsm_stage"] = np.asarray(
        [int(stage)],
        dtype=np.int32,
    )
    return output


def observations_to_arrays(
    observations: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    required = STATE_KEYS + FORCE_KEYS + CAMERA_KEYS
    extras = (
        "door_angle",
        "handle_pos",
        "fsm_stage",
    )

    arrays = {}
    for key in required + extras:
        values = []
        for observation in observations:
            if key not in observation:
                raise KeyError(
                    f"Missing observation key {key!r}. "
                    f"Available={sorted(observation)}"
                )

            if key in CAMERA_KEYS:
                value = image_to_hwc_uint8(
                    observation[key],
                    key,
                )
            else:
                dtype = (
                    np.int32
                    if key == "fsm_stage"
                    else np.float32
                )
                value = np.asarray(
                    observation[key],
                    dtype=dtype,
                ).reshape(-1)

            values.append(value)

        arrays[key] = np.stack(values, axis=0)

    return arrays


def finger_handle_contacts(
    env,
    gripper_geometry: GripperGeometry,
    door_geometry: DoorGeometry,
) -> tuple[bool, bool, list[tuple[str, str]]]:
    model = env.sim.model
    left_ids = set(gripper_geometry.left_geom_ids)
    right_ids = set(gripper_geometry.right_geom_ids)
    handle_ids = set(door_geometry.handle_geom_ids)

    if not handle_ids:
        return False, False, []

    left_contact = False
    right_contact = False
    pairs = []

    for contact_index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[contact_index]
        geom_a = int(contact.geom1)
        geom_b = int(contact.geom2)

        names = (
            model.geom_id2name(geom_a),
            model.geom_id2name(geom_b),
        )

        if (
            geom_a in left_ids
            and geom_b in handle_ids
            or geom_b in left_ids
            and geom_a in handle_ids
        ):
            left_contact = True
            pairs.append((str(names[0]), str(names[1])))

        if (
            geom_a in right_ids
            and geom_b in handle_ids
            or geom_b in right_ids
            and geom_a in handle_ids
        ):
            right_contact = True
            pairs.append((str(names[0]), str(names[1])))

    return left_contact, right_contact, pairs


def show_preview(
    observation: dict[str, Any],
    stage: Stage,
    spec: EpisodeSpec,
    predicted_class: str | None,
    force_signal: float | None,
    door_progress: float,
    delay_ms: int,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "--render requires opencv-python"
        ) from exc

    agent = image_to_hwc_uint8(
        observation[CAMERA_KEYS[0]],
        CAMERA_KEYS[0],
    )
    hand = image_to_hwc_uint8(
        observation[CAMERA_KEYS[1]],
        CAMERA_KEYS[1],
    )
    if hand.shape[:2] != agent.shape[:2]:
        hand = cv2.resize(
            hand,
            (agent.shape[1], agent.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    preview = np.concatenate([agent, hand], axis=1)
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    lines = (
        f"stage={stage.name}",
        (
            f"true={spec.true_class} "
            f"pred={predicted_class or '-'} "
            f"damping={spec.hinge_damping:.2f}"
        ),
        (
            f"signal={0.0 if force_signal is None else force_signal:.2f}N "
            f"door={door_progress:.3f}rad"
        ),
    )

    y = 20
    for line in lines:
        cv2.putText(
            preview,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y += 18

    cv2.imshow("Force-guided Push/Pull Door FSM", preview)
    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    if key in (ord("q"), 27):
        raise KeyboardInterrupt


def reconstruct_door_relative_target(
    env,
    geometry: DoorGeometry,
    reference_tangent: np.ndarray,
    offset_components: np.ndarray,
    *,
    tangent_lead: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    radius, tangent, axis, _ = door_frame(
        env,
        geometry,
        reference_tangent=reference_tangent,
    )
    handle_position = get_handle_position(env, geometry)

    target = (
        handle_position
        + radius * float(offset_components[0])
        + tangent * float(offset_components[1])
        + axis * float(offset_components[2])
        + tangent * float(tangent_lead)
    )
    return target, radius, tangent, axis


def reconstruct_grasp_target_at_progress(
    *,
    initial_hinge_anchor: np.ndarray,
    initial_handle_position: np.ndarray,
    initial_radius: np.ndarray,
    initial_tangent: np.ndarray,
    hinge_axis: np.ndarray,
    opening_sign: float,
    desired_progress: float,
    offset_components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict the handle / grasp pose at a future door angle.

    The old pull target was:

        live_handle + live_tangent * tangent_lead

    That pulls the gripper outward along the handle cylinder. With the previous
    5.5-11 cm lead, the fingers slide off the handle at roughly 30 degrees.

    This function instead rotates the complete handle pose around the hinge and
    preserves the EEF-to-handle offset captured after grasping. The gripper
    therefore follows the handle's circular arc without sliding along its axis.
    """
    desired_progress = max(float(desired_progress), 0.0)
    delta_q = float(opening_sign) * desired_progress
    rotation = axis_angle_matrix(hinge_axis, delta_q)

    handle_relative = (
        np.asarray(initial_handle_position, dtype=np.float64)
        - np.asarray(initial_hinge_anchor, dtype=np.float64)
    )
    desired_handle = (
        np.asarray(initial_hinge_anchor, dtype=np.float64)
        + rotation @ handle_relative
    )

    desired_radius = rotation @ np.asarray(
        initial_radius,
        dtype=np.float64,
    )
    desired_tangent = rotation @ np.asarray(
        initial_tangent,
        dtype=np.float64,
    )
    desired_axis = np.asarray(hinge_axis, dtype=np.float64)

    desired_radius /= max(
        float(np.linalg.norm(desired_radius)),
        1e-8,
    )
    desired_tangent /= max(
        float(np.linalg.norm(desired_tangent)),
        1e-8,
    )
    desired_axis /= max(
        float(np.linalg.norm(desired_axis)),
        1e-8,
    )

    target = (
        desired_handle
        + desired_radius * float(offset_components[0])
        + desired_tangent * float(offset_components[1])
        + desired_axis * float(offset_components[2])
    )
    return target, desired_handle, desired_tangent


def run_episode(
    env,
    spec: EpisodeSpec,
    door_geometry: DoorGeometry,
    gripper_geometry: GripperGeometry,
    args: argparse.Namespace,
) -> EpisodeResult:
    reseed_environment_for_pair(env, spec.placement_seed)
    observation = env.reset()

    set_hinge_resistance(
        env,
        door_geometry.hinge_id,
        damping=spec.hinge_damping,
        friction=spec.hinge_friction,
    )

    for _ in range(args.settle_steps):
        observation, _, _, _ = env.step(
            hold_action(gripper_command=-1.0)
        )

    initial_door_angle = joint_qpos(
        env,
        door_geometry.hinge_id,
    )
    initial_eef_position = get_eef_position(
        env,
        observation,
    )
    initial_eef_rotation = get_eef_rotation(env)

    # Build an invisible one-way hinge. Both members of a matched pair start
    # from the same pose and use the same visual geometry. A push-only door has
    # usable joint range only in the direction moving the panel away from the
    # robot; a pull-only door has usable range only toward the robot.
    original_hinge_range = np.asarray(
        env.sim.model.jnt_range[door_geometry.hinge_id],
        dtype=np.float64,
    ).copy()

    # Temporarily expose equal range on both sides so the geometric positive
    # tangent can be resolved independently of the hidden direction class.
    symmetric_range = max(float(args.door_opening_range), 0.90)
    env.sim.model.jnt_range[door_geometry.hinge_id, 0] = (
        initial_door_angle - symmetric_range
    )
    env.sim.model.jnt_range[door_geometry.hinge_id, 1] = (
        initial_door_angle + symmetric_range
    )
    env.sim.forward()

    radius0, positive_tangent0, hinge_axis0, _ = door_frame(
        env,
        door_geometry,
    )
    handle0 = get_handle_position(env, door_geometry)
    hinge_anchor0 = hinge_anchor_world(
        env,
        door_geometry.hinge_id,
    )

    surface_normal = robot_facing_surface_normal(
        handle_position=handle0,
        eef_position=initial_eef_position,
        radius=radius0,
        axis=hinge_axis0,
    )

    # surface_normal points from the panel toward the robot. Therefore moving
    # along +surface_normal pulls the door toward the robot, while moving along
    # -surface_normal pushes it away. Convert those physical directions into
    # the hinge joint's positive / negative q direction.
    pull_tangent0 = surface_normal.copy()
    push_tangent0 = -surface_normal.copy()
    push_sign = (
        1.0
        if float(np.dot(positive_tangent0, push_tangent0)) >= 0.0
        else -1.0
    )
    pull_sign = -push_sign

    if spec.true_class not in ("push", "pull"):
        raise ValueError(
            f"Push/pull collector expected true_class push or pull, got "
            f"{spec.true_class!r}"
        )

    allowed_direction_sign = (
        push_sign if spec.true_class == "push" else pull_sign
    )
    opening_sign = float(allowed_direction_sign)
    tangent0 = positive_tangent0 * opening_sign

    # Every paired episode must use the same pre-probe geometry and motion.
    # The common active-sensing action is always a small PULL, so use the pull
    # tangent when seating the grasp and defining the gripper-to-handle offset.
    # The class-specific tangent above is used only after the force decision.
    common_preprobe_tangent0 = pull_tangent0.copy()

    # Keep BOTH members of a matched pair under the same tiny two-sided lock
    # throughout approach, grasp, seating, and force-baseline collection. The
    # hidden push-only / pull-only constraint is activated only when the common
    # probe begins. Applying the class-specific range earlier lets contact during
    # grasp move the two doors differently, which leaks the hidden class and
    # causes every pair-alignment check to fail.
    wrong_slack = max(float(args.wrong_direction_slack_rad), 0.0)
    common_preprobe_hinge_range = np.asarray(
        [
            initial_door_angle - wrong_slack,
            initial_door_angle + wrong_slack,
        ],
        dtype=np.float64,
    )

    if allowed_direction_sign > 0.0:
        expanded_hinge_range = np.asarray(
            [
                initial_door_angle - wrong_slack,
                initial_door_angle + args.door_opening_range,
            ],
            dtype=np.float64,
        )
    else:
        expanded_hinge_range = np.asarray(
            [
                initial_door_angle - args.door_opening_range,
                initial_door_angle + wrong_slack,
            ],
            dtype=np.float64,
        )

    env.sim.model.jnt_range[door_geometry.hinge_id] = (
        common_preprobe_hinge_range
    )
    env.sim.forward()
    one_way_lock_activated = False

    # Fingers move toward the door along this direction.
    desired_tool_forward = -surface_normal

    # One finger above and one below the protruding handle, matching the
    # demonstrated image sequence.
    desired_finger_axis = hinge_axis0

    (
        grasp_rotation0,
        tool_reach,
        initial_finger_midpoint,
        initial_left_finger_position,
        initial_right_finger_position,
        initial_finger_separation,
    ) = calibrated_grasp_rotation(
        env,
        gripper_geometry,
        desired_forward_world=desired_tool_forward,
        desired_finger_axis_world=desired_finger_axis,
    )

    # Positioning correction:
    #
    # ``tool_reach`` was measured from the EEF site to the FINGER BODY centres.
    # Those body centres are behind the actual open-finger gap in this Panda
    # model. Subtracting tool_reach therefore stopped the EEF ~4.5 cm in front
    # of the handle, exactly as seen in the rendered rollout.
    #
    # Robosuite's robot0_eef_pos / grip site is already the useful grasp control
    # point. Put that point directly on the handle centre, with only a small
    # configurable depth / height correction.
    grasp_target = (
        handle0
        + desired_tool_forward * args.grasp_depth_adjust
        + hinge_axis0 * args.grasp_height_adjust
    )
    pregrasp_target = (
        grasp_target
        - desired_tool_forward * args.pregrasp_distance
    )
    above_target = (
        pregrasp_target
        + hinge_axis0 * args.above_handle_height
        - desired_tool_forward * args.above_handle_backoff
    )

    retreat_target = (
        above_target
        + hinge_axis0 * args.retreat_extra_height
    )

    # The live model currently carries the tiny common pre-probe lock. Use the
    # class-specific range that will be activated at FORCE_PROBE to calculate
    # the real opening target.
    hinge_range = np.asarray(
        expanded_hinge_range,
        dtype=np.float64,
    )
    if opening_sign > 0:
        available_opening = float(hinge_range[1] - initial_door_angle)
    else:
        available_opening = float(initial_door_angle - hinge_range[0])
    available_opening = max(available_opening, 0.0)

    target_open_angle = (
        float(args.open_angle)
        if args.open_angle > 0.0
        else args.open_fraction * available_opening
    )
    target_open_angle = min(
        target_open_angle,
        max(0.0, available_opening - args.open_range_margin),
    )
    if target_open_angle <= 0.0:
        raise RuntimeError(
            "Door hinge has no usable opening range. "
            f"initial={initial_door_angle:.4f}, range={hinge_range.tolist()}"
        )

    if args.debug_geometry:
        print("\n" + "=" * 100)
        print(
            f"[scene] pair={spec.pair_id} true={spec.true_class} "
            f"seed={spec.placement_seed}"
        )
        print(
            f"[door] hinge={door_geometry.hinge_name} "
            f"handle_site={door_geometry.handle_site_name} "
            f"handle_geom={door_geometry.primary_handle_name}"
        )
        print(
            f"[gripper] left={gripper_geometry.left_name} "
            f"right={gripper_geometry.right_name}"
        )
        print(
            f"[frame] radius={np.array2string(radius0, precision=3)} "
            f"tangent={np.array2string(tangent0, precision=3)} "
            f"axis={np.array2string(hinge_axis0, precision=3)} "
            f"normal={np.array2string(surface_normal, precision=3)}"
        )
        print(
            f"[calibration] tool_reach={tool_reach:.4f} "
            f"finger_separation={initial_finger_separation:.4f} "
            f"initial_finger_mid="
            f"{np.array2string(initial_finger_midpoint, precision=3)}"
        )
        print(
            f"[calibration] left_finger="
            f"{np.array2string(initial_left_finger_position, precision=3)} "
            f"right_finger="
            f"{np.array2string(initial_right_finger_position, precision=3)}"
        )
        print(
            f"[targets] above={np.array2string(above_target, precision=3)} "
            f"pregrasp={np.array2string(pregrasp_target, precision=3)} "
            f"grasp={np.array2string(grasp_target, precision=3)}"
        )
        print(
            f"[grasp-positioning] mode=direct-eef-to-handle "
            f"depth_adjust={args.grasp_depth_adjust:.4f} "
            f"contact_seek={args.contact_seek_depth:.4f} "
            f"body_tool_reach_diagnostic={tool_reach:.4f}"
        )
        print(
            f"[rotation] calibration=EEF-minus-finger-midpoint "
            f"required="
            f"{np.rad2deg(rotation_angle(initial_eef_rotation, grasp_rotation0)):.1f}deg"
        )
        print(
            f"[hinge-range] original="
            f"{np.array2string(original_hinge_range, precision=4)} "
            f"expanded="
            f"{np.array2string(expanded_hinge_range, precision=4)} "
            f"requested_opening={args.door_opening_range:.4f}rad"
        )
        print(
            f"[opening-target] available={available_opening:.4f}rad "
            f"target={target_open_angle:.4f}rad "
            f"fraction={args.open_fraction:.2f}"
        )
        print(
            f"[resistance-derived-safety] "
            f"score={spec.resistance_score:.3f} "
            f"damage_force={spec.damage_force_limit_n:.2f}N "
            f"hard_break={spec.hard_break_force_n:.2f}N "
            f"damage_impulse={spec.damage_impulse_limit_ns:.2f}Ns "
            f"max_speed={spec.max_safe_open_speed_rad_s:.3f}rad_s"
        )
        print(
            f"[push-pull-check] allowed={spec.true_class} "
            f"push_sign={push_sign:+.0f} pull_sign={pull_sign:+.0f} "
            f"wrong_slack={args.wrong_direction_slack_rad:.6f}rad "
            f"one_way_lock=activated_at_probe "
            f"probe=pull probe_steps={args.direction_probe_steps} "
            f"threshold={args.force_threshold:.3f}N "
            f"max_opening_steps={args.pull_timeout}"
        )

    # Rotate at the pose the controller actually reaches, rather than fighting
    # the remaining Cartesian tracking error while also turning the wrist.
    orientation_hold_target = above_target.copy()

    stage = Stage.MOVE_ABOVE
    stage_steps = 0
    stable_steps = 0
    contact_streak = 0
    success_streak = 0

    baseline_samples: list[np.ndarray] = []
    baseline_force_world_samples: list[np.ndarray] = []
    probe_samples: list[np.ndarray] = []
    probe_force_world_samples: list[np.ndarray] = []
    probe_tangential_force_samples: list[float] = []
    probe_delta_norm_samples: list[float] = []
    probe_progress_samples: list[float] = []
    probe_speed_samples: list[float] = []
    sensing_eef_positions: list[np.ndarray] = []
    guard_eef_positions: list[np.ndarray] = []
    guard_eef_quaternions: list[np.ndarray] = []
    guard_gripper_qpos: list[np.ndarray] = []
    guard_door_progress: list[float] = []

    predicted_class = None
    force_predicted_class = None
    force_signal = None
    force_baseline_mean = None
    force_probe_mean = None
    force_delta_mean = None
    force_baseline_world = None
    force_probe_world = None
    probe_tangential_median_n = None
    probe_tangential_trimmed_mean_n = None
    probe_tangential_impulse_ns = None
    probe_delta_norm_median_n = None
    probe_actual_progress_rad = None
    probe_actual_speed_median_rad_s = None

    # Online force adaptation during the real opening pull. There is no
    # separate probe movement and no reset. The first few pull commands are a
    # common gentle warm-up; their force response is then used to smoothly
    # interpolate between the gentle and strong strategies.
    online_force_raw_samples: list[np.ndarray] = []
    online_force_world_samples: list[np.ndarray] = []
    online_tangential_force_samples: list[float] = []
    online_delta_norm_samples: list[float] = []
    online_door_speed_samples: list[float] = []
    online_force_signal_history: list[float] = []
    online_strategy_blend = 0.0
    force_decision_progress_rad = 0.0
    force_decision_step = -1

    # Push/pull direction state. Every episode uses the same small PULL probe.
    # A pull-only hinge yields and produces a low reaction; a push-only hinge is
    # at its hard stop and produces a high reaction.
    probe_direction_sign = float(pull_sign)
    probe_tangent0 = pull_tangent0.copy()
    selected_direction_sign: float | None = None
    allowed_direction_peak_progress_rad = 0.0
    wrong_direction_peak_progress_rad = 0.0
    direction_lock_violation = False

    # Safety / damage state.
    #
    # IMPORTANT: "pulling force" is the baseline-subtracted force projected
    # onto the live door-opening tangent. Using the full 3-D norm here mixes
    # tangential pull with harmless radial / vertical grasp forces and caused
    # false break events near the end of otherwise-correct trajectories.
    interaction_force_ema_n = 0.0
    interaction_force_norm_ema_n = 0.0
    peak_interaction_force_n = 0.0
    peak_interaction_force_norm_n = 0.0
    damage_impulse_ns = 0.0
    door_speed_ema_rad_s = 0.0
    peak_door_speed_rad_s = 0.0
    overspeed_streak = 0
    previous_pull_progress = 0.0
    previous_pull_target_progress = 0.0

    # Global task counters. Opening time starts at the first real
    # ADAPTIVE_ARC_PULL action.
    opening_steps_total = 0
    breaking_force_streak = 0

    # Safety-limit exceedances remain diagnostics even when enforcement is off.
    hard_force_limit_exceeded = False
    sustained_force_limit_exceeded = False
    damage_impulse_limit_exceeded = False
    speed_limit_exceeded = False

    door_broken = False
    insufficient_progress = False

    reason = "max episode steps reached"

    grasp_offset_components = None
    seated_offset_components = None
    grasp_rotation_reference = None
    grasp_door_angle = None

    probe_base_world_target = None
    probe_world_target = None
    probe_reference_rotation = None
    probe_reference_gripper_qpos = None
    probe_start_progress = 0.0
    probe_previous_progress = 0.0
    return_start_progress = 0.0
    return_previous_eef_position = None
    return_previous_progress = 0.0
    sensing_pull_end_progress = 0.0

    observations: list[dict[str, Any]] = []
    next_observations: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []

    model_xml = get_model_xml(env)

    def signed_progress() -> float:
        return float(
            opening_sign
            * (
                joint_qpos(env, door_geometry.hinge_id)
                - initial_door_angle
            )
        )

    def transition(new_stage: Stage) -> None:
        nonlocal stage, stage_steps, stable_steps, contact_streak
        if args.debug_geometry:
            print(
                f"[fsm] {stage.name} -> {new_stage.name} "
                f"after {stage_steps} steps "
                f"door={signed_progress():.4f}"
            )
        stage = new_stage
        stage_steps = 0
        stable_steps = 0
        contact_streak = 0

    for _ in range(args.max_episode_steps):
        stage_for_action = stage
        if stage_for_action == Stage.DONE:
            reason = "fsm completed"
            break
        if stage_for_action == Stage.FAILED:
            break

        observation = make_observation(
            env,
            observation,
            geometry=door_geometry,
            stage=stage_for_action,
            allow_missing_force=args.allow_missing_force,
        )

        current_position = get_eef_position(
            env,
            observation,
        )
        current_rotation = get_eef_rotation(env)
        current_force = np.asarray(
            observation[FORCE_KEYS[0]],
            dtype=np.float64,
        )
        if args.force_sensor_frame == "eef":
            current_force_world = (
                current_rotation @ current_force
            )
        else:
            current_force_world = current_force.copy()

        # ---------------------------------------------------------------
        # GLOBAL TASK RULE
        # ---------------------------------------------------------------
        # Every door is judged by the same three conditions:
        #   1. reach the required opening angle;
        #   2. do so within --max-opening-steps;
        #   3. never exceed that door's resistance-derived breaking force.
        #
        # The rule never branches on the light/heavy label. Only the hidden
        # damping and friction determine the physical breaking-force limit.
        opening_stage = (
            stage_for_action == Stage.ADAPTIVE_ARC_PULL
        )
        required_open_progress = max(
            0.0,
            target_open_angle - args.open_success_tolerance,
        )

        if opening_stage:
            progress_now = max(0.0, signed_progress())

            # opening_steps_total counts opening actions already executed.
            if (
                opening_steps_total >= args.pull_timeout
                and progress_now < required_open_progress
            ):
                insufficient_progress = True
                transition(Stage.FAILED)
                reason = (
                    "global opening-step limit exceeded; "
                    f"progress={progress_now:.4f}rad < "
                    f"required={required_open_progress:.4f}rad after "
                    f"{opening_steps_total}/{args.pull_timeout} opening steps"
                )
                continue

            if (
                force_baseline_world is not None
                and np.isfinite(force_baseline_world).all()
            ):
                # Project onto the live opening tangent. This is the force that
                # actually pulls the door open and is the quantity used by the
                # online force classifier. Keep the full delta-force norm only
                # as a diagnostic, not as the breaking criterion.
                _, live_tangent_safety, _, _ = door_frame(
                    env,
                    door_geometry,
                    reference_tangent=tangent0,
                )
                delta_force_world_safety = (
                    current_force_world
                    - np.asarray(
                        force_baseline_world,
                        dtype=np.float64,
                    )
                )
                interaction_force_n = float(
                    abs(
                        np.dot(
                            delta_force_world_safety,
                            live_tangent_safety,
                        )
                    )
                )
                interaction_force_norm_n = float(
                    np.linalg.norm(delta_force_world_safety)
                )

                alpha = args.damage_force_ema_alpha
                interaction_force_ema_n = (
                    alpha * interaction_force_n
                    + (1.0 - alpha) * interaction_force_ema_n
                )
                interaction_force_norm_ema_n = (
                    alpha * interaction_force_norm_n
                    + (1.0 - alpha)
                    * interaction_force_norm_ema_n
                )
                peak_interaction_force_n = max(
                    peak_interaction_force_n,
                    interaction_force_ema_n,
                )
                peak_interaction_force_norm_n = max(
                    peak_interaction_force_norm_n,
                    interaction_force_norm_ema_n,
                )

                progress_delta = max(
                    0.0,
                    progress_now - previous_pull_progress,
                )
                door_speed_rad_s = progress_delta * args.control_freq
                previous_pull_progress = progress_now
                speed_alpha = args.door_speed_ema_alpha
                door_speed_ema_rad_s = (
                    speed_alpha * door_speed_rad_s
                    + (1.0 - speed_alpha) * door_speed_ema_rad_s
                )
                peak_door_speed_rad_s = max(
                    peak_door_speed_rad_s,
                    door_speed_ema_rad_s,
                )

                if opening_steps_total >= args.damage_grace_steps:
                    overload_n = max(
                        0.0,
                        interaction_force_ema_n
                        - spec.damage_force_limit_n,
                    )
                    damage_impulse_ns += (
                        overload_n / float(args.control_freq)
                    )

                    if (
                        interaction_force_ema_n
                        > spec.damage_force_limit_n
                    ):
                        breaking_force_streak += 1
                    else:
                        breaking_force_streak = 0

                    if (
                        door_speed_ema_rad_s
                        > spec.max_safe_open_speed_rad_s
                    ):
                        overspeed_streak += 1
                    else:
                        overspeed_streak = 0
                else:
                    breaking_force_streak = 0
                    overspeed_streak = 0

                # Record every artificial safety-limit exceedance. For this
                # push-vs-pull dataset they are diagnostics by default; only the
                # optional --enforce-safety-limits mode turns them into terminal
                # failures.
                if interaction_force_ema_n > spec.hard_break_force_n:
                    hard_force_limit_exceeded = True
                    if args.enforce_safety_limits:
                        door_broken = True
                        transition(Stage.FAILED)
                        reason = (
                            "hard tangential break force exceeded; "
                            f"pull_force_ema={interaction_force_ema_n:.3f}N > "
                            f"hard_limit={spec.hard_break_force_n:.3f}N; "
                            f"resistance_score={spec.resistance_score:.3f}"
                        )
                        continue

                if (
                    breaking_force_streak
                    >= args.breaking_force_hold_steps
                ):
                    sustained_force_limit_exceeded = True
                    if args.enforce_safety_limits:
                        door_broken = True
                        transition(Stage.FAILED)
                        reason = (
                            "sustained tangential breaking force exceeded; "
                            f"pull_force_ema={interaction_force_ema_n:.3f}N > "
                            f"limit={spec.damage_force_limit_n:.3f}N for "
                            f"{breaking_force_streak} consecutive steps; "
                            f"resistance_score={spec.resistance_score:.3f}"
                        )
                        continue

                if damage_impulse_ns > spec.damage_impulse_limit_ns:
                    damage_impulse_limit_exceeded = True
                    if args.enforce_safety_limits:
                        door_broken = True
                        transition(Stage.FAILED)
                        reason = (
                            "accumulated tangential overload exceeded; "
                            f"overload_impulse={damage_impulse_ns:.3f}Ns > "
                            f"limit={spec.damage_impulse_limit_ns:.3f}Ns; "
                            f"resistance_score={spec.resistance_score:.3f}"
                        )
                        continue

                if overspeed_streak >= args.overspeed_hold_steps:
                    speed_limit_exceeded = True
                    if args.enforce_safety_limits:
                        door_broken = True
                        transition(Stage.FAILED)
                        reason = (
                            "resistance-derived safe opening speed exceeded; "
                            f"speed_ema={door_speed_ema_rad_s:.3f}rad/s > "
                            f"limit={spec.max_safe_open_speed_rad_s:.3f}rad/s "
                            f"for {overspeed_streak} consecutive steps; "
                            f"resistance_score={spec.resistance_score:.3f}"
                        )
                        continue

        door_delta_q = (
            joint_qpos(env, door_geometry.hinge_id)
            - initial_door_angle
        )
        moving_grasp_rotation = (
            axis_angle_matrix(hinge_axis0, door_delta_q)
            @ grasp_rotation0
        )

        if stage_for_action == Stage.MOVE_ABOVE:
            # Image 1: first move to a safe point above and outside the door.
            action = pose_action(
                current_position,
                above_target,
                gripper_command=-1.0,
                translation_step=args.controller_step,
                current_rotation=current_rotation,
                target_rotation=initial_eef_rotation,
                orientation_step=args.orientation_controller_step,
            )
            position_error = float(
                np.linalg.norm(current_position - above_target)
            )
            stable_steps = (
                stable_steps + 1
                if position_error <= args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                orientation_hold_target = current_position.copy()
                transition(Stage.ORIENT_AT_CLEARANCE)
            elif stage_steps + 1 >= args.move_above_timeout:
                transition(Stage.FAILED)
                reason = (
                    "failed to reach safe point above handle; "
                    f"error={position_error:.4f}"
                )

        elif stage_for_action == Stage.ORIENT_AT_CLEARANCE:
            # Image 2: rotate LEFT at a safe clearance. With the corrected Panda
            # tool axis, the shortest valid orientation points the open fingers
            # toward the handle instead of showing the back of the hand.
            action = pose_action(
                current_position,
                orientation_hold_target,
                gripper_command=-1.0,
                translation_step=args.controller_step,
                current_rotation=current_rotation,
                target_rotation=grasp_rotation0,
                orientation_step=args.orientation_controller_step,
            )
            position_error = float(
                np.linalg.norm(current_position - orientation_hold_target)
            )
            orientation_error = rotation_angle(
                current_rotation,
                grasp_rotation0,
            )
            aligned = (
                position_error <= args.orient_position_tolerance
                and orientation_error
                <= np.deg2rad(args.orientation_tolerance_deg)
            )
            stable_steps = stable_steps + 1 if aligned else 0
            if stable_steps >= args.stable_steps:
                # Re-aim at the current handle position. Even a few milliradians
                # of door motion during wrist rotation changes its Cartesian pose.
                live_handle = get_handle_position(env, door_geometry)
                handle_shift = live_handle - handle0
                above_target = above_target + handle_shift
                pregrasp_target = pregrasp_target + handle_shift
                grasp_target = grasp_target + handle_shift
                retreat_target = retreat_target + handle_shift
                transition(Stage.MOVE_PREGRASP)
            elif stage_steps + 1 >= args.orient_timeout:
                transition(Stage.FAILED)
                reason = (
                    "failed to orient at safe clearance; "
                    f"position_error={position_error:.4f}, "
                    f"orientation_error_deg="
                    f"{np.rad2deg(orientation_error):.2f}"
                )

        elif stage_for_action == Stage.MOVE_PREGRASP:
            # Image 2 -> 3: descend and approach while orientation is fixed.
            action = pose_action(
                current_position,
                pregrasp_target,
                gripper_command=-1.0,
                translation_step=args.controller_step,
                current_rotation=current_rotation,
                target_rotation=grasp_rotation0,
                orientation_step=args.orientation_controller_step,
            )
            position_error = float(
                np.linalg.norm(current_position - pregrasp_target)
            )
            orientation_error = rotation_angle(
                current_rotation,
                grasp_rotation0,
            )
            aligned = (
                position_error <= args.pregrasp_tolerance
                and orientation_error
                <= np.deg2rad(args.orientation_tolerance_deg)
            )
            stable_steps = stable_steps + 1 if aligned else 0
            if stable_steps >= args.stable_steps:
                transition(Stage.INSERT_TO_HANDLE)
            elif stage_steps + 1 >= args.pregrasp_timeout:
                transition(Stage.FAILED)
                reason = (
                    "failed to reach pregrasp; "
                    f"position_error={position_error:.4f}, "
                    f"orientation_error_deg="
                    f"{np.rad2deg(orientation_error):.2f}"
                )

        elif stage_for_action == Stage.INSERT_TO_HANDLE:
            # Image 3: straight insertion only. The wrist has already finished
            # rotating, so the palm does not sweep sideways into the door.
            #
            # Do NOT gate this transition on the resolved finger BODY centres.
            # In this Panda model those centres remain about 8.9 cm behind the
            # actual open-finger gap even when the visually correct EEF target has
            # been reached. The old gate therefore stopped forever at the handle.
            action = pose_action(
                current_position,
                grasp_target,
                gripper_command=-1.0,
                translation_step=args.insert_controller_step,
                current_rotation=current_rotation,
                target_rotation=grasp_rotation0,
                orientation_step=args.orientation_hold_step,
            )
            action = limit_pose_action(
                action,
                max_translation_norm=args.insert_max_translation_action,
                max_rotation_norm=args.insert_max_rotation_action,
            )

            if float(np.linalg.norm(current_force)) >= args.grasp_impact_force_limit_n:
                transition(Stage.FAILED)
                reason = (
                    "handle impact force exceeded during insertion; "
                    f"|F|={np.linalg.norm(current_force):.3f}N >= "
                    f"{args.grasp_impact_force_limit_n:.3f}N"
                )
                continue

            position_error = float(
                np.linalg.norm(current_position - grasp_target)
            )
            orientation_error = rotation_angle(
                current_rotation,
                grasp_rotation0,
            )

            # Keep this value only for debugging. It is not the finger-gap centre.
            finger_body_midpoint = 0.5 * (
                get_finger_centre(
                    env,
                    gripper_geometry.left_body_id,
                    gripper_geometry.left_geom_ids,
                )
                + get_finger_centre(
                    env,
                    gripper_geometry.right_body_id,
                    gripper_geometry.right_geom_ids,
                )
            )
            body_midpoint_diagnostic = float(
                np.linalg.norm(
                    finger_body_midpoint
                    - get_handle_position(env, door_geometry)
                )
            )

            ready = (
                stage_steps + 1 >= args.insert_min_steps
                and position_error <= args.insert_position_tolerance
                and orientation_error
                <= np.deg2rad(args.insert_orientation_tolerance_deg)
            )
            stable_steps = stable_steps + 1 if ready else 0

            if stable_steps >= args.stable_steps:
                if args.debug_geometry:
                    print(
                        f"[insert] grasp pose reached; "
                        f"eef_error={position_error:.4f}, "
                        f"orientation_error_deg="
                        f"{np.rad2deg(orientation_error):.2f}, "
                        f"body_midpoint_diagnostic="
                        f"{body_midpoint_diagnostic:.4f}"
                    )
                transition(Stage.CLOSE_GRIPPER)

            elif stage_steps + 1 >= args.insert_timeout:
                pose_is_valid = (
                    position_error
                    <= max(
                        args.insert_position_tolerance,
                        args.insert_timeout_position_tolerance,
                    )
                    and orientation_error
                    <= np.deg2rad(
                        args.insert_timeout_orientation_tolerance_deg
                    )
                )

                if pose_is_valid:
                    if args.debug_geometry:
                        print(
                            f"[insert-timeout] closing from valid EEF pose; "
                            f"eef_error={position_error:.4f}, "
                            f"orientation_error_deg="
                            f"{np.rad2deg(orientation_error):.2f}, "
                            f"body_midpoint_diagnostic="
                            f"{body_midpoint_diagnostic:.4f}"
                        )
                    transition(Stage.CLOSE_GRIPPER)
                else:
                    transition(Stage.FAILED)
                    reason = (
                        "failed to reach handle insertion pose; "
                        f"eef_error={position_error:.4f}, "
                        f"orientation_error_deg="
                        f"{np.rad2deg(orientation_error):.2f}, "
                        f"body_midpoint_diagnostic="
                        f"{body_midpoint_diagnostic:.4f}"
                    )

        elif stage_for_action == Stage.CLOSE_GRIPPER:
            action = pose_action(
                current_position,
                grasp_target,
                gripper_command=1.0,
                translation_step=args.insert_controller_step,
                current_rotation=current_rotation,
                target_rotation=grasp_rotation0,
                orientation_step=args.orientation_hold_step,
            )
            action = limit_pose_action(
                action,
                max_translation_norm=args.close_max_translation_action,
                max_rotation_norm=args.insert_max_rotation_action,
            )
            if stage_steps + 1 >= args.close_steps:
                transition(Stage.VERIFY_GRASP)

        elif stage_for_action == Stage.VERIFY_GRASP:
            # After closing, gently move a few millimetres deeper toward the
            # handle. This seats the bar in the finger gap without the large
            # palm-first collision caused by the earlier approach.
            seek_fraction = min(
                1.0,
                (stage_steps + 1)
                / max(float(args.grasp_verify_timeout), 1.0),
            )
            verify_target = (
                grasp_target
                + desired_tool_forward
                * args.contact_seek_depth
                * seek_fraction
            )
            action = pose_action(
                current_position,
                verify_target,
                gripper_command=1.0,
                translation_step=args.contact_seek_controller_step,
                current_rotation=current_rotation,
                target_rotation=grasp_rotation0,
                orientation_step=args.orientation_hold_step,
            )
            action = limit_pose_action(
                action,
                max_translation_norm=args.contact_seek_max_translation_action,
                max_rotation_norm=args.insert_max_rotation_action,
            )

            if float(np.linalg.norm(current_force)) >= args.grasp_impact_force_limit_n:
                transition(Stage.FAILED)
                reason = (
                    "handle impact force exceeded during grasp verification; "
                    f"|F|={np.linalg.norm(current_force):.3f}N >= "
                    f"{args.grasp_impact_force_limit_n:.3f}N"
                )
                continue

            left_contact, right_contact, pairs = finger_handle_contacts(
                env,
                gripper_geometry,
                door_geometry,
            )
            handle_position = get_handle_position(
                env,
                door_geometry,
            )
            finger_midpoint = 0.5 * (
                get_finger_centre(
                    env,
                    gripper_geometry.left_body_id,
                    gripper_geometry.left_geom_ids,
                )
                + get_finger_centre(
                    env,
                    gripper_geometry.right_body_id,
                    gripper_geometry.right_geom_ids,
                )
            )
            midpoint_error = float(
                np.linalg.norm(finger_midpoint - handle_position)
            )

            contact_ok = (
                left_contact and right_contact
                or (
                    args.allow_one_sided_grasp
                    and (left_contact or right_contact)
                )
            )
            contact_streak = (
                contact_streak + 1
                if contact_ok
                else 0
            )

            verified_by_contact = (
                contact_streak >= args.grasp_contact_hold_steps
            )
            verified_by_closed_pose = (
                stage_steps + 1 >= args.grasp_verify_timeout
                and not args.strict_grasp_contacts
                and np.linalg.norm(current_position - verify_target)
                <= args.grasp_fallback_pose_tolerance
                and rotation_angle(
                    current_rotation,
                    grasp_rotation0,
                )
                <= np.deg2rad(
                    args.grasp_fallback_orientation_tolerance_deg
                )
            )

            # MuJoCo contact-name matching differs across Robosuite versions.
            # Prefer real finger-handle contacts, but by default allow the FSM to
            # continue from the stable closed pose. A false grasp will still be
            # rejected by the final door-opening success condition.
            if verified_by_contact or verified_by_closed_pose:
                radius, tangent, axis, _ = door_frame(
                    env,
                    door_geometry,
                    reference_tangent=common_preprobe_tangent0,
                )
                offset = current_position - handle_position
                grasp_offset_components = np.asarray(
                    [
                        np.dot(offset, radius),
                        np.dot(offset, tangent),
                        np.dot(offset, axis),
                    ],
                    dtype=np.float64,
                )
                grasp_rotation_reference = current_rotation.copy()
                grasp_door_angle = joint_qpos(
                    env,
                    door_geometry.hinge_id,
                )

                if args.debug_geometry:
                    verification_mode = (
                        "contact"
                        if verified_by_contact
                        else "closed-pose fallback"
                    )
                    print(
                        f"[grasp] verified_by={verification_mode} "
                        f"contacts L={left_contact} R={right_contact} "
                        f"pairs={pairs[:4]} "
                        f"body_midpoint_diagnostic={midpoint_error:.4f}"
                    )
                    print(
                        f"[grasp] offset_components="
                        f"{np.array2string(grasp_offset_components, precision=4)}"
                    )

                transition(Stage.SEAT_GRASP)

            elif stage_steps + 1 >= args.grasp_verify_timeout:
                transition(Stage.FAILED)
                reason = (
                    "strict grasp verification failed; "
                    f"left_contact={left_contact}, "
                    f"right_contact={right_contact}, "
                    f"body_midpoint_diagnostic={midpoint_error:.4f}, "
                    f"contact_pairs={pairs[:6]}"
                )

        elif stage_for_action == Stage.SEAT_GRASP:
            assert grasp_offset_components is not None

            seat_target, _, live_tangent, _ = (
                reconstruct_door_relative_target(
                    env,
                    door_geometry,
                    reference_tangent=common_preprobe_tangent0,
                    offset_components=grasp_offset_components,
                    tangent_lead=-args.seat_tangent_backoff,
                )
            )
            action = pose_action(
                current_position,
                seat_target,
                gripper_command=1.0,
                translation_step=args.seat_controller_step,
                current_rotation=current_rotation,
                target_rotation=moving_grasp_rotation,
                orientation_step=args.orientation_hold_step,
            )
            action = limit_pose_action(
                action,
                max_translation_norm=args.seat_max_translation_action,
                max_rotation_norm=args.insert_max_rotation_action,
            )
            error = float(
                np.linalg.norm(current_position - seat_target)
            )
            stable_steps = (
                stable_steps + 1
                if error <= args.seat_tolerance
                else 0
            )
            if (
                stable_steps >= args.stable_steps
                or stage_steps + 1 >= args.seat_steps
            ):
                handle_position = get_handle_position(
                    env,
                    door_geometry,
                )
                radius, tangent, axis, _ = door_frame(
                    env,
                    door_geometry,
                    reference_tangent=common_preprobe_tangent0,
                )
                offset = current_position - handle_position
                seated_offset_components = np.asarray(
                    [
                        np.dot(offset, radius),
                        np.dot(offset, tangent),
                        np.dot(offset, axis),
                    ],
                    dtype=np.float64,
                )

                # Prepare one continuous opening pull. The stationary baseline
                # only estimates sensor bias; it does not move the door. Once
                # the baseline is ready, the very first door movement is the
                # real task motion.
                probe_base_world_target = current_position.copy()
                probe_reference_rotation = current_rotation.copy()
                probe_reference_gripper_qpos = np.asarray(
                    observation["robot0_gripper_qpos"],
                    dtype=np.float64,
                ).reshape(-1).copy()
                probe_start_progress = max(0.0, signed_progress())
                previous_pull_progress = probe_start_progress
                previous_pull_target_progress = probe_start_progress

                if args.debug_geometry:
                    print(
                        f"[continuous-pull] start={probe_start_progress:.4f}rad "
                        f"separate_probe=False "
                        f"common_warmup={args.common_pull_warmup_steps}steps "
                        f"online_force_window={args.online_force_window}steps "
                        f"force_frame={args.force_sensor_frame}"
                    )

                transition(Stage.FORCE_BASELINE)

        elif stage_for_action == Stage.FORCE_BASELINE:
            assert probe_base_world_target is not None
            assert probe_reference_rotation is not None
            action = pose_action(
                current_position,
                probe_base_world_target,
                gripper_command=1.0,
                translation_step=args.probe_controller_step,
                current_rotation=current_rotation,
                target_rotation=probe_reference_rotation,
                orientation_step=args.orientation_hold_step,
            )
            if stage_steps >= args.baseline_settle_steps:
                baseline_samples.append(current_force.copy())
                baseline_force_world_samples.append(
                    current_force_world.copy()
                )

            if len(baseline_samples) >= args.baseline_samples:
                baseline_window = np.stack(
                    baseline_samples[-args.baseline_stat_window:]
                )
                baseline_world_window = np.stack(
                    baseline_force_world_samples[
                        -args.baseline_stat_window:
                    ]
                )
                force_baseline_mean = np.median(
                    baseline_window, axis=0
                )
                force_baseline_world = np.median(
                    baseline_world_window, axis=0
                )
                probe_start_progress = float(
                    probe_direction_sign
                    * (
                        joint_qpos(env, door_geometry.hinge_id)
                        - initial_door_angle
                    )
                )
                probe_previous_progress = probe_start_progress
                transition(Stage.FORCE_PROBE)

        elif stage_for_action == Stage.FORCE_PROBE:
            assert seated_offset_components is not None
            assert force_baseline_world is not None

            # Activate the hidden one-way hinge only now, after the paired
            # episodes have reached the same pre-probe state and the common
            # force baseline has been measured. The first probe response is
            # therefore the first class-dependent physical observation.
            if not one_way_lock_activated:
                env.sim.model.jnt_range[door_geometry.hinge_id] = (
                    expanded_hinge_range
                )
                env.sim.forward()
                one_way_lock_activated = True

            current_probe_q = joint_qpos(
                env, door_geometry.hinge_id
            )
            probe_progress = float(
                probe_direction_sign
                * (current_probe_q - initial_door_angle)
            )
            current_allowed_progress = max(
                0.0,
                float(
                    allowed_direction_sign
                    * (current_probe_q - initial_door_angle)
                ),
            )
            current_wrong_progress = max(
                0.0,
                float(
                    -allowed_direction_sign
                    * (current_probe_q - initial_door_angle)
                ),
            )
            allowed_direction_peak_progress_rad = max(
                allowed_direction_peak_progress_rad,
                current_allowed_progress,
            )
            wrong_direction_peak_progress_rad = max(
                wrong_direction_peak_progress_rad,
                current_wrong_progress,
            )
            if (
                current_wrong_progress
                > args.wrong_direction_slack_rad
                + args.direction_lock_tolerance_rad
            ):
                direction_lock_violation = True
                transition(Stage.FAILED)
                reason = (
                    "one-way hinge violation during probe; "
                    f"wrong_progress={current_wrong_progress:.6f}rad"
                )
                continue
            probe_progress_samples.append(probe_progress)
            sensing_eef_positions.append(current_position.copy())

            _, probe_live_tangent, _, _ = door_frame(
                env,
                door_geometry,
                reference_tangent=probe_tangent0,
            )
            delta_force_world = (
                current_force_world
                - np.asarray(force_baseline_world, dtype=np.float64)
            )
            reaction_force_n = float(
                abs(np.dot(delta_force_world, probe_live_tangent))
            )
            delta_norm_n = float(np.linalg.norm(delta_force_world))

            progress_delta = abs(probe_progress - probe_previous_progress)
            probe_previous_progress = probe_progress
            probe_speed = progress_delta * float(args.control_freq)

            if stage_steps >= args.direction_probe_ignore_steps:
                probe_samples.append(current_force.copy())
                probe_force_world_samples.append(
                    current_force_world.copy()
                )
                probe_tangential_force_samples.append(
                    reaction_force_n
                )
                probe_delta_norm_samples.append(delta_norm_n)
                probe_speed_samples.append(probe_speed)

            desired_probe_progress = min(
                float(args.direction_probe_angle_rad),
                max(probe_progress, 0.0)
                + float(args.direction_probe_angle_step_rad),
            )
            probe_target, _, _ = reconstruct_grasp_target_at_progress(
                initial_hinge_anchor=hinge_anchor0,
                initial_handle_position=handle0,
                initial_radius=radius0,
                initial_tangent=common_preprobe_tangent0,
                hinge_axis=hinge_axis0,
                opening_sign=probe_direction_sign,
                desired_progress=desired_probe_progress,
                offset_components=seated_offset_components,
            )
            action = pose_action(
                current_position,
                probe_target,
                gripper_command=1.0,
                translation_step=args.direction_probe_controller_step,
                current_rotation=current_rotation,
                target_rotation=moving_grasp_rotation,
                orientation_step=args.orientation_hold_step,
            )
            action = limit_pose_action(
                action,
                max_translation_norm=(
                    args.direction_probe_max_translation_action
                ),
                max_rotation_norm=args.probe_max_rotation_action,
            )

            if stage_steps >= args.direction_probe_steps:
                if not probe_tangential_force_samples:
                    transition(Stage.FAILED)
                    reason = "direction probe produced no usable force samples"
                    action = hold_action(gripper_command=1.0)
                    continue

                force_signal = float(
                    np.median(
                        np.asarray(
                            probe_tangential_force_samples,
                            dtype=np.float64,
                        )
                    )
                )
                force_predicted_class = (
                    "push"
                    if force_signal >= args.force_threshold
                    else "pull"
                )
                force_decision_step = int(stage_steps)
                force_decision_progress_rad = abs(probe_progress)
                sensing_pull_end_progress = abs(probe_progress)

                if args.decision_mode == "force":
                    predicted_class = force_predicted_class
                elif args.decision_mode == "oracle":
                    predicted_class = spec.true_class
                elif args.decision_mode == "always_push":
                    predicted_class = "push"
                elif args.decision_mode == "always_pull":
                    predicted_class = "pull"
                else:
                    raise RuntimeError(
                        f"Unsupported decision mode: {args.decision_mode}"
                    )

                selected_direction_sign = (
                    push_sign
                    if predicted_class == "push"
                    else pull_sign
                )

                if args.debug_geometry:
                    print(
                        "[direction-decision] "
                        f"true={spec.true_class} "
                        f"force_pred={force_predicted_class} "
                        f"decision={predicted_class} "
                        f"signal={force_signal:.3f}N "
                        f"threshold={args.force_threshold:.3f}N "
                        f"probe_progress={probe_progress:.5f}rad"
                    )

                # Stop probing immediately. The next stage opens in the selected
                # direction. The one-way hinge itself prevents a wrong decision
                # from rotating the door.
                action = hold_action(gripper_command=1.0)
                previous_pull_progress = max(
                    0.0,
                    float(
                        selected_direction_sign
                        * (
                            joint_qpos(env, door_geometry.hinge_id)
                            - initial_door_angle
                        )
                    ),
                )
                previous_pull_target_progress = previous_pull_progress
                transition(Stage.ADAPTIVE_ARC_PULL)

        elif stage_for_action in (
            Stage.RETURN_TO_COMMON_POSE,
            Stage.PRECLASS_GUARD,
        ):
            raise RuntimeError(
                "Unused compatibility stage entered in push/pull collector"
            )

        elif stage_for_action == Stage.ADAPTIVE_ARC_PULL:
            assert seated_offset_components is not None
            assert force_baseline_world is not None
            assert selected_direction_sign is not None

            current_q = joint_qpos(env, door_geometry.hinge_id)
            selected_progress = max(
                0.0,
                float(
                    selected_direction_sign
                    * (current_q - initial_door_angle)
                ),
            )
            allowed_progress = max(0.0, signed_progress())
            wrong_progress = max(
                0.0,
                float(
                    -allowed_direction_sign
                    * (current_q - initial_door_angle)
                ),
            )
            allowed_direction_peak_progress_rad = max(
                allowed_direction_peak_progress_rad,
                allowed_progress,
            )
            wrong_direction_peak_progress_rad = max(
                wrong_direction_peak_progress_rad,
                wrong_progress,
            )
            if (
                wrong_progress
                > args.wrong_direction_slack_rad
                + args.direction_lock_tolerance_rad
            ):
                direction_lock_violation = True
                transition(Stage.FAILED)
                reason = (
                    "one-way hinge violation; "
                    f"wrong_progress={wrong_progress:.6f}rad > "
                    f"allowed={args.wrong_direction_slack_rad + args.direction_lock_tolerance_rad:.6f}rad"
                )
                continue

            maximum_target_increment = (
                args.direction_open_target_rate_rad_s
                / float(args.control_freq)
            )
            desired_goal = min(
                target_open_angle,
                selected_progress + args.direction_open_angle_lead,
            )
            desired_progress = min(
                desired_goal,
                previous_pull_target_progress + maximum_target_increment,
            )
            desired_progress = min(
                target_open_angle,
                max(selected_progress, desired_progress),
            )
            previous_pull_target_progress = desired_progress

            pull_target, desired_handle, _ = (
                reconstruct_grasp_target_at_progress(
                    initial_hinge_anchor=hinge_anchor0,
                    initial_handle_position=handle0,
                    initial_radius=radius0,
                    initial_tangent=common_preprobe_tangent0,
                    hinge_axis=hinge_axis0,
                    opening_sign=selected_direction_sign,
                    desired_progress=desired_progress,
                    offset_components=seated_offset_components,
                )
            )

            action = pose_action(
                current_position,
                pull_target,
                gripper_command=1.0,
                translation_step=args.direction_open_controller_step,
                current_rotation=current_rotation,
                target_rotation=moving_grasp_rotation,
                orientation_step=args.orientation_hold_step,
            )
            action = limit_pose_action(
                action,
                max_translation_norm=(
                    args.direction_open_max_translation_action
                ),
                max_rotation_norm=args.pull_max_rotation_action,
            )

            left_contact, right_contact, contact_pairs = (
                finger_handle_contacts(
                    env, gripper_geometry, door_geometry
                )
            )
            contact_ok = (
                left_contact and right_contact
                or (
                    args.allow_one_sided_grasp
                    and (left_contact or right_contact)
                )
            )
            contact_streak = 0 if contact_ok else contact_streak + 1

            if (
                args.require_pull_contacts
                and contact_streak >= args.pull_contact_loss_steps
            ):
                transition(Stage.FAILED)
                reason = (
                    "lost handle contact during direction-aware opening; "
                    f"selected={predicted_class}, "
                    f"allowed={spec.true_class}, "
                    f"contacts={contact_pairs[:4]}"
                )
                continue

            if args.debug_geometry and (
                (stage_steps + 1) % args.pull_debug_interval == 0
            ):
                print(
                    "[push-pull-open] "
                    f"allowed={spec.true_class} "
                    f"selected={predicted_class} "
                    f"selected_progress={selected_progress:.4f}rad "
                    f"allowed_progress={allowed_progress:.4f}/"
                    f"{target_open_angle:.4f}rad "
                    f"wrong_progress={wrong_progress:.6f}rad "
                    f"target={desired_progress:.4f}rad "
                    f"contacts=L{left_contact}/R{right_contact}"
                )

            if (
                allowed_progress
                >= target_open_angle - args.open_success_tolerance
            ):
                success_streak += 1
            else:
                success_streak = 0

            if success_streak >= args.open_hold_detection_steps:
                transition(Stage.HOLD_OPEN)

        elif stage_for_action == Stage.HOLD_OPEN:
            assert seated_offset_components is not None

            hold_target, _, _ = reconstruct_grasp_target_at_progress(
                initial_hinge_anchor=hinge_anchor0,
                initial_handle_position=handle0,
                initial_radius=radius0,
                initial_tangent=common_preprobe_tangent0,
                hinge_axis=hinge_axis0,
                opening_sign=opening_sign,
                desired_progress=target_open_angle,
                offset_components=seated_offset_components,
            )
            action = pose_action(
                current_position,
                hold_target,
                gripper_command=1.0,
                translation_step=args.hold_open_controller_step,
                current_rotation=current_rotation,
                target_rotation=moving_grasp_rotation,
                orientation_step=args.orientation_hold_step,
            )
            if stage_steps + 1 >= args.hold_open_steps:
                if args.release_after_open:
                    transition(Stage.RELEASE)
                else:
                    transition(Stage.DONE)

        elif stage_for_action == Stage.RELEASE:
            action = hold_action(gripper_command=-1.0)
            if stage_steps + 1 >= args.release_steps:
                transition(Stage.RETREAT)

        elif stage_for_action == Stage.RETREAT:
            action = pose_action(
                current_position,
                retreat_target,
                gripper_command=-1.0,
                translation_step=args.controller_step,
            )
            error = float(
                np.linalg.norm(current_position - retreat_target)
            )
            stable_steps = (
                stable_steps + 1
                if error <= args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                transition(Stage.DONE)
            elif stage_steps + 1 >= args.retreat_timeout:
                transition(Stage.DONE)

        else:
            raise RuntimeError(
                f"Unhandled FSM stage: {stage_for_action}"
            )

        observations.append(deepcopy(observation))
        states.append(get_state_vector(env))
        actions.append(action.copy())
        stages.append(int(stage_for_action))

        next_observation, _, _, _ = env.step(action)
        next_observation = make_observation(
            env,
            next_observation,
            geometry=door_geometry,
            stage=stage,
            allow_missing_force=args.allow_missing_force,
        )

        next_observations.append(
            deepcopy(next_observation)
        )
        next_states.append(get_state_vector(env))
        observation = next_observation

        if stage_for_action == Stage.ADAPTIVE_ARC_PULL:
            opening_steps_total += 1

        if stage == stage_for_action:
            stage_steps += 1

        if (
            args.debug_geometry
            and stage == stage_for_action
            and stage_steps > 0
            and stage_steps % args.debug_interval == 0
        ):
            force_norm = float(np.linalg.norm(current_force))
            print(
                f"[debug] stage={stage_for_action.name} "
                f"step={stage_steps} "
                f"door={signed_progress():.4f} "
                f"|F|={force_norm:.3f}"
            )

        if args.render:
            show_preview(
                observation,
                stage=stage,
                spec=spec,
                predicted_class=predicted_class,
                force_signal=force_signal,
                door_progress=signed_progress(),
                delay_ms=args.render_delay_ms,
            )

    # Finalise the shared pull-probe diagnostics. A high reaction means the
    # common pull probe hit the forbidden hard stop, so the correct direction is
    # push; a low reaction means pulling is physically allowed.
    if probe_tangential_force_samples:
        probe_tangent_array = np.asarray(
            probe_tangential_force_samples, dtype=np.float64
        )
        probe_delta_array = np.asarray(
            probe_delta_norm_samples, dtype=np.float64
        )
        probe_speed_array = np.asarray(
            probe_speed_samples, dtype=np.float64
        )
        probe_raw_array = np.stack(probe_samples, axis=0)
        probe_world_array = np.stack(
            probe_force_world_samples, axis=0
        )

        force_signal = float(np.median(probe_tangent_array))
        force_predicted_class = (
            "push" if force_signal >= args.force_threshold else "pull"
        )
        if predicted_class is None:
            if args.decision_mode == "force":
                predicted_class = force_predicted_class
            elif args.decision_mode == "oracle":
                predicted_class = spec.true_class
            elif args.decision_mode == "always_push":
                predicted_class = "push"
            elif args.decision_mode == "always_pull":
                predicted_class = "pull"

        force_probe_mean = np.median(probe_raw_array, axis=0)
        force_probe_world = np.median(probe_world_array, axis=0)
        force_delta_mean = (
            force_probe_world
            - np.asarray(force_baseline_world, dtype=np.float64)
        )
        probe_tangential_median_n = force_signal
        probe_tangential_trimmed_mean_n = float(
            np.mean(probe_tangent_array)
        )
        probe_tangential_impulse_ns = float(
            np.sum(probe_tangent_array) / args.control_freq
        )
        probe_delta_norm_median_n = float(
            np.median(probe_delta_array)
        )
        probe_actual_progress_rad = float(
            max(probe_progress_samples, default=0.0)
        )
        probe_actual_speed_median_rad_s = float(
            np.median(probe_speed_array)
            if probe_speed_array.size
            else 0.0
        )
    else:
        force_signal = 0.0
        force_predicted_class = "pull"
        if predicted_class is None:
            predicted_class = (
                spec.true_class
                if args.decision_mode == "oracle"
                else "push"
                if args.decision_mode == "always_push"
                else "pull"
            )

    final_door_angle = joint_qpos(
        env,
        door_geometry.hinge_id,
    )
    final_progress = float(
        opening_sign
        * (final_door_angle - initial_door_angle)
    )
    required_open_progress = max(
        0.0,
        target_open_angle - args.open_success_tolerance,
    )
    opened = bool(
        final_progress >= required_open_progress
    )
    completed_within_step_limit = bool(
        opening_steps_total <= args.pull_timeout
    )
    damage_limit_exceeded = bool(
        hard_force_limit_exceeded
        or sustained_force_limit_exceeded
        or damage_impulse_limit_exceeded
        or speed_limit_exceeded
    )
    stayed_below_breaking_force = bool(
        not damage_limit_exceeded
    )
    safety_success = bool(
        stayed_below_breaking_force
        or not args.enforce_safety_limits
    )

    # Push/pull task success is direction-aware opening within the shared
    # deadline while respecting the physical one-way lock. Artificial force /
    # speed limits are diagnostic unless explicitly enabled.
    physical_success = bool(
        stage == Stage.DONE
        and opened
        and completed_within_step_limit
        and safety_success
        and not direction_lock_violation
    )
    class_correct = bool(
        predicted_class == spec.true_class
    )
    force_class_correct = bool(
        force_predicted_class == spec.true_class
    )

    # A demonstration is physically successful when the door reaches the
    # opening target and the FSM completes. Force classification is recorded as
    # a separate diagnostic, so a threshold that is not calibrated yet does not
    # incorrectly reject a visibly successful door-opening trajectory.
    success = bool(
        physical_success
        and (
            class_correct
            or not args.require_correct_force_classification
        )
    )

    if not actions:
        return EpisodeResult(
            success=False,
            reason="no actions generated",
            trajectory=None,
            predicted_class=predicted_class,
            force_signal=force_signal,
        )

    if reason == "fsm completed":
        if physical_success and not class_correct:
            reason = (
                "fsm completed; physical_success=True; "
                "classification_correct=False"
            )
        elif not success:
            reason = (
                f"physical_success={physical_success}, "
                f"opened={opened}, "
                f"class_correct={class_correct}, "
                f"require_correct_classification="
                f"{args.require_correct_force_classification}"
            )

    if args.debug_geometry:
        print(
            f"[episode] final_stage={stage.name} "
            f"success={success} "
            f"physical_success={physical_success} "
            f"opened={opened} "
            f"class_correct={class_correct} "
            f"door_progress={final_progress:.4f} "
            f"target={target_open_angle:.4f} "
            f"tolerance={args.open_success_tolerance:.4f} "
            f"true={spec.true_class} "
            f"force_pred={force_predicted_class} "
            f"decision={predicted_class} "
            f"mode={args.decision_mode} "
            f"signal={force_signal} "
            f"enforce_safety={args.enforce_safety_limits} "
            f"damage_limit_exceeded={damage_limit_exceeded} "
            f"door_broken={door_broken} "
            f"damage={damage_impulse_ns:.3f}Ns "
            f"peak_tangential_force="
            f"{peak_interaction_force_n:.3f}N "
            f"peak_delta_force_norm="
            f"{peak_interaction_force_norm_n:.3f}N "
            f"peak_speed={peak_door_speed_rad_s:.3f}rad_s "
            f"reason={reason!r}"
        )

    rewards = np.zeros(len(actions), dtype=np.float32)
    dones = np.zeros(len(actions), dtype=np.int64)
    if success:
        rewards[-1] = 1.0
        dones[-1] = 1

    trajectory = {
        "states": np.stack(states, axis=0),
        "next_states": np.stack(next_states, axis=0),
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "rewards": rewards,
        "dones": dones,
        "obs": observations_to_arrays(observations),
        "next_obs": observations_to_arrays(next_observations),
        "fsm_stage": np.asarray(stages, dtype=np.int32),
        "model_file": model_xml,
        "initial_door_angle": float(initial_door_angle),
        "final_door_angle": float(final_door_angle),
        "final_door_progress": float(final_progress),
        "target_open_angle": float(target_open_angle),
        "open_success_tolerance": float(
            args.open_success_tolerance
        ),
        "required_open_progress": float(required_open_progress),
        "opening_steps_total": int(opening_steps_total),
        "max_opening_steps": int(args.pull_timeout),
        "completed_within_step_limit": bool(
            completed_within_step_limit
        ),
        "enforce_safety_limits": bool(
            args.enforce_safety_limits
        ),
        "damage_limit_exceeded": bool(
            damage_limit_exceeded
        ),
        "hard_force_limit_exceeded": bool(
            hard_force_limit_exceeded
        ),
        "sustained_force_limit_exceeded": bool(
            sustained_force_limit_exceeded
        ),
        "damage_impulse_limit_exceeded": bool(
            damage_impulse_limit_exceeded
        ),
        "speed_limit_exceeded": bool(
            speed_limit_exceeded
        ),
        "stayed_below_breaking_force": bool(
            stayed_below_breaking_force
        ),
        "opened": bool(opened),
        "physical_success": bool(physical_success),
        "classification_correct": bool(class_correct),
        "collector_success": bool(success),
        "door_broken": bool(door_broken),
        "insufficient_progress": bool(insufficient_progress),
        "damage_impulse_ns": float(damage_impulse_ns),
        "peak_interaction_force_n": float(
            peak_interaction_force_n
        ),
        "peak_interaction_force_norm_n": float(
            peak_interaction_force_norm_n
        ),
        "peak_door_speed_rad_s": float(
            peak_door_speed_rad_s
        ),
        "resistance_score": float(spec.resistance_score),
        "damage_force_limit_n": float(
            spec.damage_force_limit_n
        ),
        "hard_break_force_n": float(
            spec.hard_break_force_n
        ),
        "damage_impulse_limit_ns": float(
            spec.damage_impulse_limit_ns
        ),
        "max_safe_open_speed_rad_s": float(
            spec.max_safe_open_speed_rad_s
        ),
        "available_opening": float(available_opening),
        "original_hinge_range": original_hinge_range.copy(),
        "expanded_hinge_range": expanded_hinge_range.copy(),
        "door_opening_range": float(args.door_opening_range),
        "decision_mode": str(args.decision_mode),
        "force_measurement_mode": "common_pull_probe",
        "reset_after_force_measurement": False,
        "sensing_pull_end_progress_rad": float(
            force_decision_progress_rad
        ),
        "force_decision_progress_rad": float(
            force_decision_progress_rad
        ),
        "force_decision_step": int(force_decision_step),
        "online_force_window": int(args.online_force_window),
        "online_strategy_blend_final": 0.0,
        "decision_class": (
            str(predicted_class)
            if predicted_class is not None
            else "none"
        ),
        "force_predicted_class": (
            str(force_predicted_class)
            if force_predicted_class is not None
            else "none"
        ),
        "force_class_correct": bool(force_class_correct),
        "opening_sign": float(opening_sign),
        "allowed_direction_sign": float(allowed_direction_sign),
        "push_sign": float(push_sign),
        "pull_sign": float(pull_sign),
        "selected_direction_sign": float(
            selected_direction_sign
            if selected_direction_sign is not None
            else 0.0
        ),
        "wrong_direction_slack_rad": float(
            args.wrong_direction_slack_rad
        ),
        "wrong_direction_peak_progress_rad": float(
            wrong_direction_peak_progress_rad
        ),
        "allowed_direction_peak_progress_rad": float(
            allowed_direction_peak_progress_rad
        ),
        "direction_lock_violation": bool(direction_lock_violation),
        "probe_direction": "pull",
        "opening_tangent": tangent0.copy(),
        "surface_normal": surface_normal.copy(),
        "grasp_rotation": grasp_rotation0.copy(),
        "tool_reach": float(tool_reach),
        "force_baseline_mean": (
            np.asarray(force_baseline_mean, dtype=np.float64)
            if force_baseline_mean is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        "force_probe_mean": (
            np.asarray(force_probe_mean, dtype=np.float64)
            if force_probe_mean is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        "force_delta_mean": (
            np.asarray(force_delta_mean, dtype=np.float64)
            if force_delta_mean is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        "force_baseline_world": (
            np.asarray(force_baseline_world, dtype=np.float64)
            if force_baseline_world is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        "force_probe_world": (
            np.asarray(force_probe_world, dtype=np.float64)
            if force_probe_world is not None
            else np.full(3, np.nan, dtype=np.float64)
        ),
        "probe_tangential_median_n": float(
            probe_tangential_median_n
            if probe_tangential_median_n is not None
            else np.nan
        ),
        "probe_tangential_trimmed_mean_n": float(
            probe_tangential_trimmed_mean_n
            if probe_tangential_trimmed_mean_n is not None
            else np.nan
        ),
        "probe_tangential_impulse_ns": float(
            probe_tangential_impulse_ns
            if probe_tangential_impulse_ns is not None
            else np.nan
        ),
        "probe_delta_norm_median_n": float(
            probe_delta_norm_median_n
            if probe_delta_norm_median_n is not None
            else np.nan
        ),
        "probe_actual_progress_rad": float(
            probe_actual_progress_rad
            if probe_actual_progress_rad is not None
            else np.nan
        ),
        "probe_actual_speed_median_rad_s": float(
            probe_actual_speed_median_rad_s
            if probe_actual_speed_median_rad_s is not None
            else np.nan
        ),
        "sensing_eef_positions": (
            np.stack(
                sensing_eef_positions,
                axis=0,
            )
            if sensing_eef_positions
            else np.empty(
                (0, 3),
                dtype=np.float64,
            )
        ),
        "sensing_door_progress": np.asarray(
            probe_progress_samples,
            dtype=np.float64,
        ),
        "guard_eef_positions": (
            np.stack(guard_eef_positions, axis=0)
            if guard_eef_positions
            else np.empty((0, 3), dtype=np.float64)
        ),
        "guard_eef_quaternions": (
            np.stack(guard_eef_quaternions, axis=0)
            if guard_eef_quaternions
            else np.empty((0, 4), dtype=np.float64)
        ),
        "guard_gripper_qpos": (
            np.stack(guard_gripper_qpos, axis=0)
            if guard_gripper_qpos
            else np.empty((0, 0), dtype=np.float64)
        ),
        "guard_door_progress": np.asarray(
            guard_door_progress,
            dtype=np.float64,
        ),
    }

    return EpisodeResult(
        success=success,
        reason=reason,
        trajectory=trajectory,
        predicted_class=predicted_class,
        force_signal=force_signal,
    )


def pair_alignment_metrics(
    result_a: EpisodeResult,
    result_b: EpisodeResult,
) -> dict[str, float]:
    """Compare matched push/pull episodes immediately before the shared probe.

    The collector no longer tries to erase information after interaction. This
    check only verifies that both members of a pair start the opening pull from
    the same robot pose, gripper state, door angle, and camera scene. Any later
    visual / state difference is a genuine physical response to resistance.
    """
    assert result_a.trajectory is not None
    assert result_b.trajectory is not None

    trajectory_a = result_a.trajectory
    trajectory_b = result_b.trajectory

    def final_baseline_index(trajectory: dict[str, Any]) -> int:
        stages = np.asarray(
            trajectory["fsm_stage"],
            dtype=np.int64,
        ).reshape(-1)
        indices = np.flatnonzero(
            stages == int(Stage.FORCE_BASELINE)
        )
        if not indices.size:
            raise ValueError(
                "FORCE_BASELINE frame missing from trajectory"
            )
        return int(indices[-1])

    index_a = final_baseline_index(trajectory_a)
    index_b = final_baseline_index(trajectory_b)
    obs_a = trajectory_a["obs"]
    obs_b = trajectory_b["obs"]

    eef_a = np.asarray(
        obs_a["robot0_eef_pos"][index_a],
        dtype=np.float64,
    ).reshape(-1)
    eef_b = np.asarray(
        obs_b["robot0_eef_pos"][index_b],
        dtype=np.float64,
    ).reshape(-1)
    quat_a = np.asarray(
        obs_a["robot0_eef_quat"][index_a],
        dtype=np.float64,
    ).reshape(-1)
    quat_b = np.asarray(
        obs_b["robot0_eef_quat"][index_b],
        dtype=np.float64,
    ).reshape(-1)
    gripper_a = np.asarray(
        obs_a["robot0_gripper_qpos"][index_a],
        dtype=np.float64,
    ).reshape(-1)
    gripper_b = np.asarray(
        obs_b["robot0_gripper_qpos"][index_b],
        dtype=np.float64,
    ).reshape(-1)
    door_a = float(
        np.asarray(obs_a["door_angle"][index_a]).reshape(-1)[0]
    )
    door_b = float(
        np.asarray(obs_b["door_angle"][index_b]).reshape(-1)[0]
    )

    quat_a = quat_a / max(float(np.linalg.norm(quat_a)), 1e-12)
    quat_b = quat_b / max(float(np.linalg.norm(quat_b)), 1e-12)
    cosine = float(
        np.clip(abs(np.dot(quat_a, quat_b)), 0.0, 1.0)
    )
    rotation_gap_deg = float(
        np.degrees(2.0 * np.arccos(cosine))
    )

    agent_a = np.asarray(
        obs_a["agentview_image"][index_a],
        dtype=np.float64,
    )
    agent_b = np.asarray(
        obs_b["agentview_image"][index_b],
        dtype=np.float64,
    )
    hand_a = np.asarray(
        obs_a["robot0_eye_in_hand_image"][index_a],
        dtype=np.float64,
    )
    hand_b = np.asarray(
        obs_b["robot0_eye_in_hand_image"][index_b],
        dtype=np.float64,
    )

    return {
        "eef_gap_m": float(np.linalg.norm(eef_a - eef_b)),
        "rotation_gap_deg": rotation_gap_deg,
        "gripper_gap": float(
            np.linalg.norm(gripper_a - gripper_b)
        ),
        "door_gap_rad": float(abs(door_a - door_b)),
        "agentview_mae_uint8": float(
            np.mean(np.abs(agent_a - agent_b))
        ),
        "eye_in_hand_mae_uint8": float(
            np.mean(np.abs(hand_a - hand_b))
        ),
    }

def write_episode(
    data_group: h5py.Group,
    demo_name: str,
    result: EpisodeResult,
    spec: EpisodeSpec,
) -> int:
    assert result.trajectory is not None
    trajectory = result.trajectory

    demo = data_group.create_group(demo_name)

    for key in (
        "states",
        "next_states",
        "actions",
        "rewards",
        "dones",
        "fsm_stage",
    ):
        demo.create_dataset(
            key,
            data=trajectory[key],
        )

    observation_group = demo.create_group("obs")
    next_observation_group = demo.create_group("next_obs")

    for key, value in trajectory["obs"].items():
        observation_group.create_dataset(
            key,
            data=value,
            compression="gzip" if key in CAMERA_KEYS else None,
        )

    for key, value in trajectory["next_obs"].items():
        next_observation_group.create_dataset(
            key,
            data=value,
            compression="gzip" if key in CAMERA_KEYS else None,
        )

    if trajectory.get("model_file"):
        demo.attrs["model_file"] = trajectory["model_file"]

    demo.attrs["num_samples"] = int(
        trajectory["actions"].shape[0]
    )
    demo.attrs["task"] = "force_essential_door_push_pull"
    demo.attrs["collection_mode"] = (
        "paired_force_essential_door_push_pull"
    )
    demo.attrs["pair_id"] = int(spec.pair_id)
    demo.attrs["placement_seed"] = int(spec.placement_seed)
    # Keep the old attribute as a compatibility alias for generic loaders.
    demo.attrs["true_resistance_class"] = spec.true_class
    demo.attrs["true_direction_class"] = spec.true_class
    demo.attrs["predicted_resistance_class"] = str(
        result.predicted_class
    )
    demo.attrs["predicted_direction_class"] = str(
        result.predicted_class
    )
    demo.attrs["decision_mode"] = str(
        trajectory["decision_mode"]
    )
    demo.attrs["force_measurement_mode"] = str(
        trajectory["force_measurement_mode"]
    )
    demo.attrs["reset_after_force_measurement"] = bool(
        trajectory["reset_after_force_measurement"]
    )
    demo.attrs["sensing_pull_end_progress_rad"] = float(
        trajectory["sensing_pull_end_progress_rad"]
    )
    demo.attrs["force_decision_progress_rad"] = float(
        trajectory["force_decision_progress_rad"]
    )
    demo.attrs["force_decision_step"] = int(
        trajectory["force_decision_step"]
    )
    demo.attrs["online_force_window"] = int(
        trajectory["online_force_window"]
    )
    demo.attrs["online_strategy_blend_final"] = float(
        trajectory["online_strategy_blend_final"]
    )
    demo.attrs["force_predicted_resistance_class"] = str(
        trajectory["force_predicted_class"]
    )
    demo.attrs["hinge_damping"] = float(spec.hinge_damping)
    demo.attrs["hinge_friction"] = float(spec.hinge_friction)
    demo.attrs["force_signal_n"] = float(result.force_signal)
    demo.attrs["force_baseline_mean"] = trajectory["force_baseline_mean"]
    demo.attrs["force_probe_mean"] = trajectory["force_probe_mean"]
    demo.attrs["force_delta_mean"] = trajectory["force_delta_mean"]
    demo.attrs["force_baseline_world"] = trajectory[
        "force_baseline_world"
    ]
    demo.attrs["force_probe_world"] = trajectory[
        "force_probe_world"
    ]
    demo.attrs["probe_tangential_median_n"] = float(
        trajectory["probe_tangential_median_n"]
    )
    demo.attrs["probe_tangential_trimmed_mean_n"] = float(
        trajectory[
            "probe_tangential_trimmed_mean_n"
        ]
    )
    demo.attrs["probe_tangential_impulse_ns"] = float(
        trajectory["probe_tangential_impulse_ns"]
    )
    demo.attrs["probe_delta_norm_median_n"] = float(
        trajectory["probe_delta_norm_median_n"]
    )
    demo.attrs["probe_actual_progress_rad"] = float(
        trajectory["probe_actual_progress_rad"]
    )
    demo.attrs[
        "probe_actual_speed_median_rad_s"
    ] = float(
        trajectory[
            "probe_actual_speed_median_rad_s"
        ]
    )
    demo.attrs["classification_correct"] = bool(
        trajectory["classification_correct"]
    )
    demo.attrs["force_classification_correct"] = bool(
        trajectory["force_class_correct"]
    )
    demo.attrs["opened"] = bool(
        trajectory["opened"]
    )
    demo.attrs["physical_success"] = bool(
        trajectory["physical_success"]
    )
    demo.attrs["collector_success"] = bool(
        trajectory["collector_success"]
    )
    demo.attrs["enforce_safety_limits"] = bool(
        trajectory["enforce_safety_limits"]
    )
    demo.attrs["damage_limit_exceeded"] = bool(
        trajectory["damage_limit_exceeded"]
    )
    demo.attrs["hard_force_limit_exceeded"] = bool(
        trajectory["hard_force_limit_exceeded"]
    )
    demo.attrs["sustained_force_limit_exceeded"] = bool(
        trajectory["sustained_force_limit_exceeded"]
    )
    demo.attrs["damage_impulse_limit_exceeded"] = bool(
        trajectory["damage_impulse_limit_exceeded"]
    )
    demo.attrs["speed_limit_exceeded"] = bool(
        trajectory["speed_limit_exceeded"]
    )
    demo.attrs["door_broken"] = bool(
        trajectory["door_broken"]
    )
    demo.attrs["insufficient_progress"] = bool(
        trajectory["insufficient_progress"]
    )
    demo.attrs["damage_impulse_ns"] = float(
        trajectory["damage_impulse_ns"]
    )
    demo.attrs["peak_interaction_force_n"] = float(
        trajectory["peak_interaction_force_n"]
    )
    demo.attrs["peak_interaction_force_norm_n"] = float(
        trajectory["peak_interaction_force_norm_n"]
    )
    demo.attrs["peak_door_speed_rad_s"] = float(
        trajectory["peak_door_speed_rad_s"]
    )
    demo.attrs["resistance_score"] = float(
        trajectory["resistance_score"]
    )
    demo.attrs["damage_force_limit_n"] = float(
        trajectory["damage_force_limit_n"]
    )
    demo.attrs["hard_break_force_n"] = float(
        trajectory["hard_break_force_n"]
    )
    demo.attrs["damage_impulse_limit_ns"] = float(
        trajectory["damage_impulse_limit_ns"]
    )
    demo.attrs["max_safe_open_speed_rad_s"] = float(
        trajectory["max_safe_open_speed_rad_s"]
    )
    demo.attrs["open_success_tolerance"] = float(
        trajectory["open_success_tolerance"]
    )
    demo.attrs["required_open_progress"] = float(
        trajectory["required_open_progress"]
    )
    demo.attrs["opening_steps_total"] = int(
        trajectory["opening_steps_total"]
    )
    demo.attrs["allowed_direction_sign"] = float(
        trajectory["allowed_direction_sign"]
    )
    demo.attrs["push_sign"] = float(trajectory["push_sign"])
    demo.attrs["pull_sign"] = float(trajectory["pull_sign"])
    demo.attrs["selected_direction_sign"] = float(
        trajectory["selected_direction_sign"]
    )
    demo.attrs["wrong_direction_slack_rad"] = float(
        trajectory["wrong_direction_slack_rad"]
    )
    demo.attrs["wrong_direction_peak_progress_rad"] = float(
        trajectory["wrong_direction_peak_progress_rad"]
    )
    demo.attrs["direction_lock_violation"] = bool(
        trajectory["direction_lock_violation"]
    )
    demo.attrs["max_opening_steps"] = int(
        trajectory["max_opening_steps"]
    )
    demo.attrs["completed_within_step_limit"] = bool(
        trajectory["completed_within_step_limit"]
    )
    demo.attrs["stayed_below_breaking_force"] = bool(
        trajectory["stayed_below_breaking_force"]
    )
    demo.attrs["initial_door_angle"] = float(
        trajectory["initial_door_angle"]
    )
    demo.attrs["final_door_angle"] = float(
        trajectory["final_door_angle"]
    )
    demo.attrs["final_door_progress"] = float(
        trajectory["final_door_progress"]
    )
    demo.attrs["success"] = bool(
        trajectory["physical_success"]
    )

    demo.attrs["ep_meta"] = json.dumps(
        {
            "pair_id": int(spec.pair_id),
            "placement_seed": int(spec.placement_seed),
            "true_class": spec.true_class,
            "true_direction_class": spec.true_class,
            "predicted_class": result.predicted_class,
            "predicted_direction_class": result.predicted_class,
            "decision_mode": trajectory["decision_mode"],
            "force_measurement_mode": trajectory[
                "force_measurement_mode"
            ],
            "reset_after_force_measurement": trajectory[
                "reset_after_force_measurement"
            ],
            "sensing_pull_end_progress_rad": trajectory[
                "sensing_pull_end_progress_rad"
            ],
            "force_decision_progress_rad": trajectory[
                "force_decision_progress_rad"
            ],
            "force_decision_step": trajectory[
                "force_decision_step"
            ],
            "online_force_window": trajectory[
                "online_force_window"
            ],
            "online_strategy_blend_final": trajectory[
                "online_strategy_blend_final"
            ],
            "force_predicted_class": trajectory[
                "force_predicted_class"
            ],
            "force_class_correct": trajectory[
                "force_class_correct"
            ],
            "classification_correct": trajectory[
                "classification_correct"
            ],
            "opened": trajectory["opened"],
            "physical_success": trajectory[
                "physical_success"
            ],
            "collector_success": trajectory[
                "collector_success"
            ],
            "enforce_safety_limits": trajectory[
                "enforce_safety_limits"
            ],
            "damage_limit_exceeded": trajectory[
                "damage_limit_exceeded"
            ],
            "hard_force_limit_exceeded": trajectory[
                "hard_force_limit_exceeded"
            ],
            "sustained_force_limit_exceeded": trajectory[
                "sustained_force_limit_exceeded"
            ],
            "damage_impulse_limit_exceeded": trajectory[
                "damage_impulse_limit_exceeded"
            ],
            "speed_limit_exceeded": trajectory[
                "speed_limit_exceeded"
            ],
            "door_broken": trajectory["door_broken"],
            "insufficient_progress": trajectory[
                "insufficient_progress"
            ],
            "damage_impulse_ns": trajectory[
                "damage_impulse_ns"
            ],
            "peak_interaction_force_n": trajectory[
                "peak_interaction_force_n"
            ],
            "peak_interaction_force_norm_n": trajectory[
                "peak_interaction_force_norm_n"
            ],
            "peak_door_speed_rad_s": trajectory[
                "peak_door_speed_rad_s"
            ],
            "resistance_score": trajectory["resistance_score"],
            "damage_force_limit_n": trajectory[
                "damage_force_limit_n"
            ],
            "hard_break_force_n": trajectory[
                "hard_break_force_n"
            ],
            "damage_impulse_limit_ns": trajectory[
                "damage_impulse_limit_ns"
            ],
            "max_safe_open_speed_rad_s": trajectory[
                "max_safe_open_speed_rad_s"
            ],
            "open_success_tolerance": trajectory[
                "open_success_tolerance"
            ],
            "hinge_damping": float(spec.hinge_damping),
            "hinge_friction": float(spec.hinge_friction),
            "force_signal_n": float(result.force_signal),
            "force_baseline_mean": trajectory["force_baseline_mean"],
            "force_probe_mean": trajectory["force_probe_mean"],
            "force_delta_mean": trajectory["force_delta_mean"],
            "force_baseline_world": trajectory[
                "force_baseline_world"
            ],
            "force_probe_world": trajectory[
                "force_probe_world"
            ],
            "probe_tangential_median_n": trajectory[
                "probe_tangential_median_n"
            ],
            "probe_tangential_trimmed_mean_n": trajectory[
                "probe_tangential_trimmed_mean_n"
            ],
            "probe_tangential_impulse_ns": trajectory[
                "probe_tangential_impulse_ns"
            ],
            "probe_delta_norm_median_n": trajectory[
                "probe_delta_norm_median_n"
            ],
            "probe_actual_progress_rad": trajectory[
                "probe_actual_progress_rad"
            ],
            "probe_actual_speed_median_rad_s": trajectory[
                "probe_actual_speed_median_rad_s"
            ],
            "final_door_progress": float(
                trajectory["final_door_progress"]
            ),
            "target_open_angle": float(
                trajectory["target_open_angle"]
            ),
            "available_opening": float(
                trajectory["available_opening"]
            ),
            "opening_sign": float(
                trajectory["opening_sign"]
            ),
            "allowed_direction_sign": float(
                trajectory["allowed_direction_sign"]
            ),
            "push_sign": float(trajectory["push_sign"]),
            "pull_sign": float(trajectory["pull_sign"]),
            "selected_direction_sign": float(
                trajectory["selected_direction_sign"]
            ),
            "wrong_direction_slack_rad": float(
                trajectory["wrong_direction_slack_rad"]
            ),
            "wrong_direction_peak_progress_rad": float(
                trajectory["wrong_direction_peak_progress_rad"]
            ),
            "direction_lock_violation": bool(
                trajectory["direction_lock_violation"]
            ),
            "opening_tangent": trajectory["opening_tangent"],
            "surface_normal": trajectory["surface_normal"],
            "tool_reach": float(trajectory["tool_reach"]),
        },
        default=json_default,
    )

    return int(trajectory["actions"].shape[0])



def load_excluded_door_placement_seeds(
    paths: list[str] | None,
) -> set[int]:
    """Load every previously used Door placement_seed from old train/val HDF5s."""
    excluded: set[int] = set()
    for raw_path in paths or []:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--exclude-hdf5 does not exist: {path}")
        with h5py.File(path, "r") as hdf5_file:
            if "data" not in hdf5_file:
                raise KeyError(f"{path} has no /data group")
            for demo in hdf5_file["data"].values():
                if "placement_seed" in demo.attrs:
                    excluded.add(int(demo.attrs["placement_seed"]))
                    continue
                ep_meta_raw = demo.attrs.get("ep_meta")
                if ep_meta_raw is not None:
                    if isinstance(ep_meta_raw, bytes):
                        ep_meta_raw = ep_meta_raw.decode("utf-8")
                    try:
                        ep_meta = json.loads(str(ep_meta_raw))
                    except Exception:
                        ep_meta = {}
                    if "placement_seed" in ep_meta:
                        excluded.add(int(ep_meta["placement_seed"]))
    return excluded




def create_pairwise_masks(
    output: h5py.File,
    demo_names: list[str],
    pair_ids: dict[str, int],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    pairs: dict[int, list[str]] = defaultdict(list)
    for demo_name in demo_names:
        pairs[int(pair_ids[demo_name])].append(demo_name)

    pair_numbers = sorted(pairs)
    rng = np.random.default_rng(seed)
    rng.shuffle(pair_numbers)

    if len(pair_numbers) <= 1:
        validation_pair_count = 0
    else:
        validation_pair_count = max(
            1,
            int(round(len(pair_numbers) * val_ratio)),
        )

    validation_pairs = set(
        pair_numbers[:validation_pair_count]
    )

    train_names = sorted(
        demo_name
        for pair_id, names in pairs.items()
        if pair_id not in validation_pairs
        for demo_name in names
    )
    validation_names = sorted(
        demo_name
        for pair_id, names in pairs.items()
        if pair_id in validation_pairs
        for demo_name in names
    )

    mask = output.create_group("mask")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    mask.create_dataset(
        "train",
        data=np.asarray(train_names, dtype=object),
        dtype=string_dtype,
    )
    mask.create_dataset(
        "valid",
        data=np.asarray(validation_names, dtype=object),
        dtype=string_dtype,
    )
    return train_names, validation_names


def sample_pair_specs(
    pair_id: int,
    placement_seed: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> list[EpisodeSpec]:
    # Both pair members share all continuous physics. The ONLY difference is
    # whether the invisible one-way joint range permits pushing or pulling.
    hinge_damping = float(
        rng.uniform(
            args.direction_damping_min,
            args.direction_damping_max,
        )
    )
    hinge_friction = float(
        rng.uniform(
            args.direction_friction_min,
            args.direction_friction_max,
        )
    )
    limits = resistance_derived_safety_limits(
        hinge_damping, hinge_friction, args
    )

    def make_spec(direction: str) -> EpisodeSpec:
        return EpisodeSpec(
            pair_id=pair_id,
            true_class=direction,
            hinge_damping=hinge_damping,
            hinge_friction=hinge_friction,
            placement_seed=placement_seed,
            resistance_score=limits["resistance_score"],
            damage_force_limit_n=limits["damage_force_limit_n"],
            hard_break_force_n=limits["hard_break_force_n"],
            damage_impulse_limit_ns=limits[
                "damage_impulse_limit_ns"
            ],
            max_safe_open_speed_rad_s=limits[
                "max_safe_open_speed_rad_s"
            ],
        )

    specs = [make_spec("push"), make_spec("pull")]
    rng.shuffle(specs)
    return specs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect paired Robosuite Door demonstrations in which an invisible "
            "one-way hinge permits only PUSH or only PULL. The robot executes "
            "one shared pull probe and uses reaction force to choose direction."
        )
    )

    parser.add_argument(
        "--output",
        default="datasets/force_door_push_pull.hdf5",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Total demonstrations. Must be even because scenes are paired.",
    )
    parser.add_argument("--max-pair-attempts", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--test-only",
        action="store_true",
        help=(
            "Create a final held-out dataset with /mask/test instead of "
            "train/valid masks."
        ),
    )
    parser.add_argument(
        "--exclude-hdf5",
        nargs="*",
        default=None,
        help=(
            "Existing Door train/validation HDF5 files. Every placement_seed "
            "found in these files is forbidden in the new test collection."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Pair-shared nuisance randomisation. Defaults reproduce Robosuite's stock
    # Door placement distribution. The handle is rigidly attached to the door,
    # so its world position changes automatically with the sampled door pose.
    parser.add_argument("--door-x-min", type=float, default=0.07)
    parser.add_argument("--door-x-max", type=float, default=0.09)
    parser.add_argument("--door-y-min", type=float, default=-0.01)
    parser.add_argument("--door-y-max", type=float, default=0.01)
    parser.add_argument(
        "--door-yaw-min-deg",
        type=float,
        default=-104.32394487827058,
    )
    parser.add_argument(
        "--door-yaw-max-deg",
        type=float,
        default=-90.0,
    )

    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=1200)
    parser.add_argument("--settle-steps", type=int, default=20)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay-ms", type=int, default=15)
    parser.add_argument("--debug-geometry", action="store_true")
    parser.add_argument("--debug-interval", type=int, default=20)
    parser.add_argument("--allow-missing-force", action="store_true")

    # General closed-loop control.
    parser.add_argument("--controller-step", type=float, default=0.035)
    parser.add_argument("--insert-controller-step", type=float, default=0.035)
    parser.add_argument("--seat-controller-step", type=float, default=0.030)
    parser.add_argument("--probe-controller-step", type=float, default=0.025)
    parser.add_argument("--pull-controller-step", type=float, default=0.020)
    parser.add_argument(
        "--orientation-controller-step",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--orient-position-tolerance",
        type=float,
        default=0.050,
        help=(
            "Allowed Cartesian drift while rotating at the reached safe pose. "
            "Orientation is the important gate in this stage."
        ),
    )
    parser.add_argument(
        "--orientation-hold-step",
        type=float,
        default=0.12,
    )

    # Command limits prevent abrupt handle impacts. These cap the norm of the
    # normalised OSC command rather than clipping each axis independently.
    parser.add_argument(
        "--insert-max-translation-action",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--insert-max-rotation-action",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--close-max-translation-action",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--contact-seek-max-translation-action",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--seat-max-translation-action",
        type=float,
        default=0.14,
    )
    parser.add_argument(
        "--probe-max-translation-action",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--probe-max-rotation-action",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--grasp-impact-force-limit-n",
        type=float,
        default=30.0,
        help=(
            "Reject a demonstration when insertion / grasping creates a large "
            "raw force spike. This prevents collision-heavy data from being "
            "accepted as expert behaviour."
        ),
    )

    parser.add_argument("--stable-steps", type=int, default=4)
    parser.add_argument("--position-tolerance", type=float, default=0.020)
    parser.add_argument("--pregrasp-tolerance", type=float, default=0.018)
    parser.add_argument(
        "--orientation-tolerance-deg",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--insert-orientation-tolerance-deg",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--insert-position-tolerance",
        type=float,
        default=0.012,
    )
    parser.add_argument(
        "--finger-midpoint-tolerance",
        type=float,
        default=0.030,
        help=(
            "Deprecated diagnostic value. Finger BODY centres are not used "
            "to decide when to close because they sit behind the finger gap."
        ),
    )
    parser.add_argument(
        "--insert-timeout-position-tolerance",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--insert-timeout-orientation-tolerance-deg",
        type=float,
        default=15.0,
    )

    # Image-sequence geometry.
    parser.add_argument(
        "--above-handle-height",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--above-handle-backoff",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--pregrasp-distance",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--grasp-depth-adjust",
        type=float,
        default=0.004,
        help=(
            "EEF offset from the handle centre along the approach direction. "
            "Positive moves a little deeper toward the door. The old 4.5 cm "
            "finger-body offset has been removed."
        ),
    )
    parser.add_argument(
        "--grasp-height-adjust",
        type=float,
        default=0.0,
    )

    parser.add_argument("--move-above-timeout", type=int, default=140)
    parser.add_argument("--orient-timeout", type=int, default=140)
    parser.add_argument("--pregrasp-timeout", type=int, default=140)
    parser.add_argument("--insert-timeout", type=int, default=120)
    parser.add_argument(
        "--insert-min-steps",
        type=int,
        default=12,
        help="Minimum number of slow straight-in steps before closing.",
    )

    # Grasp.
    parser.add_argument("--close-steps", type=int, default=3)
    parser.add_argument(
        "--contact-seek-depth",
        type=float,
        default=0.003,
        help=(
            "Additional slow forward motion after closing, used to seat the "
            "handle inside the finger gap."
        ),
    )
    parser.add_argument(
        "--contact-seek-controller-step",
        type=float,
        default=0.025,
    )
    parser.add_argument("--grasp-verify-timeout", type=int, default=45)
    parser.add_argument("--grasp-contact-hold-steps", type=int, default=3)
    parser.add_argument(
        "--grasp-verify-distance",
        type=float,
        default=0.035,
    )
    parser.add_argument(
        "--allow-one-sided-grasp",
        dest="allow_one_sided_grasp",
        action="store_true",
        help="Accept a stable contact from either finger.",
    )
    parser.add_argument(
        "--strict-two-sided-grasp",
        dest="allow_one_sided_grasp",
        action="store_false",
        help="Require simultaneous contact from both fingers.",
    )
    parser.set_defaults(allow_one_sided_grasp=True)
    parser.add_argument(
        "--strict-grasp-contacts",
        action="store_true",
        help=(
            "Require matched MuJoCo finger-handle contacts before pulling. "
            "Default behaviour may continue from a stable closed grasp pose."
        ),
    )
    parser.add_argument(
        "--grasp-fallback-pose-tolerance",
        type=float,
        default=0.030,
    )
    parser.add_argument(
        "--grasp-fallback-orientation-tolerance-deg",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--seat-tangent-backoff",
        type=float,
        default=0.004,
    )
    parser.add_argument("--seat-tolerance", type=float, default=0.018)
    parser.add_argument("--seat-steps", type=int, default=3)

    # Force baseline and continuous online adaptation. The legacy probe / return
    # options below are retained only for command-line compatibility and are not
    # used by the continuous-pull FSM.
    parser.add_argument("--baseline-settle-steps", type=int, default=0)
    parser.add_argument("--baseline-samples", type=int, default=3)
    parser.add_argument(
        "--baseline-stat-window",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--dynamic-probe-angle",
        type=float,
        default=0.040,
        help=(
            "Small door-arc displacement used by the constant-rate dynamic "
            "probe, in radians."
        ),
    )
    parser.add_argument(
        "--dynamic-probe-steps",
        type=int,
        default=8,
        help=(
            "Number of control steps used to execute the dynamic probe. "
            "With 20 Hz and 0.04 rad this targets about 0.08 rad/s."
        ),
    )
    parser.add_argument(
        "--dynamic-probe-ignore-steps",
        type=int,
        default=1,
        help=(
            "Initial dynamic-probe frames ignored to reject contact-onset "
            "transients."
        ),
    )
    parser.add_argument(
        "--dynamic-probe-trim-fraction",
        type=float,
        default=0.15,
        help=(
            "Fraction trimmed from each end when reporting the robust mean."
        ),
    )
    parser.add_argument(
        "--force-sensor-frame",
        choices=("eef", "world"),
        default="eef",
        help=(
            "Frame of robot0_eef_force. Robosuite force sensors normally "
            "report in the EEF/site frame, so the default rotates them into "
            "world coordinates before projecting onto the opening tangent."
        ),
    )
    parser.add_argument(
        "--probe-max-door-angle",
        type=float,
        default=0.060,
        help=(
            "Safety cap on actual door motion during the dynamic probe."
        ),
    )
    parser.add_argument(
        "--force-threshold",
        type=float,
        default=3.6,
        help=(
            "Reaction-force threshold for the shared PULL probe. A value above "
            "the threshold means pulling is blocked, so the robot must PUSH. "
            "Calibrate this once on development pairs, then freeze it."
        ),
    )
    parser.add_argument(
        "--wrong-direction-slack-rad",
        type=float,
        default=0.0005,
        help=(
            "Tiny numerical joint-range slack on the physically forbidden side. "
            "The default is about 0.03 degrees."
        ),
    )
    parser.add_argument(
        "--direction-lock-tolerance-rad",
        type=float,
        default=0.0025,
        help="Extra numerical tolerance when verifying that the blocked side did not move.",
    )
    parser.add_argument(
        "--direction-probe-steps", type=int, default=10
    )
    parser.add_argument(
        "--direction-probe-ignore-steps", type=int, default=2
    )
    parser.add_argument(
        "--direction-probe-angle-rad", type=float, default=0.015
    )
    parser.add_argument(
        "--direction-probe-angle-step-rad", type=float, default=0.003
    )
    parser.add_argument(
        "--direction-probe-controller-step", type=float, default=0.020
    )
    parser.add_argument(
        "--direction-probe-max-translation-action",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--direction-open-angle-lead", type=float, default=0.060
    )
    parser.add_argument(
        "--direction-open-controller-step", type=float, default=0.035
    )
    parser.add_argument(
        "--direction-open-target-rate-rad-s",
        type=float,
        default=0.26,
    )
    parser.add_argument(
        "--direction-open-max-translation-action",
        type=float,
        default=0.46,
    )
    parser.add_argument(
        "--direction-damping-min", type=float, default=0.40
    )
    parser.add_argument(
        "--direction-damping-max", type=float, default=0.80
    )
    parser.add_argument(
        "--direction-friction-min", type=float, default=0.02
    )
    parser.add_argument(
        "--direction-friction-max", type=float, default=0.08
    )
    parser.add_argument(
        "--online-force-window",
        type=int,
        default=3,
        help=(
            "Rolling median window for tangential force during the real pull. "
            "Three samples gives a fast response at 20 Hz."
        ),
    )
    parser.add_argument(
        "--online-force-ignore-steps",
        type=int,
        default=1,
        help=(
            "Initial real-pull response frames ignored before building the "
            "online force estimate."
        ),
    )
    parser.add_argument(
        "--common-pull-warmup-steps",
        type=int,
        default=3,
        help=(
            "Initial real-pull steps that use the common gentle strategy. "
            "The door keeps moving forward; this is not a separate probe."
        ),
    )
    parser.add_argument(
        "--force-blend-band-n",
        type=float,
        default=1.0,
        help=(
            "Half-width around --force-threshold used to smoothly blend from "
            "the gentle to the strong strategy in force mode."
        ),
    )
    parser.add_argument(
        "--force-control-ema-alpha",
        type=float,
        default=0.5,
        help=(
            "EMA coefficient for smoothing the online strategy strength."
        ),
    )

    parser.add_argument(
        "--decision-mode",
        choices=(
            "force",
            "oracle",
            "always_push",
            "always_pull",
        ),
        default="force",
        help=(
            "force uses the shared pull-probe reaction, oracle reads the hidden "
            "direction for diagnostics, and always_push / always_pull are fixed "
            "no-force baselines. The hinge physically blocks the wrong side."
        ),
    )

    # Retained only so old command lines do not fail; it is unused here.
    parser.add_argument(
        "--fixed-strategy-blend",
        type=float,
        default=0.5,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--return-motion-steps",
        type=int,
        default=16,
        help=(
            "Minimum number of control steps used to return from the dynamic "
            "probe to the exact pre-probe decision state."
        ),
    )
    parser.add_argument(
        "--return-controller-step",
        type=float,
        default=0.040,
        help=(
            "Cartesian scaling for the smooth common-pose return. A larger "
            "value produces a gentler normalised OSC command."
        ),
    )
    parser.add_argument(
        "--return-max-translation-action",
        type=float,
        default=0.16,
    )
    parser.add_argument(
        "--return-debug-interval",
        type=int,
        default=4,
    )
    parser.add_argument("--return-timeout", type=int, default=140)
    parser.add_argument(
        "--common-pose-stable-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--common-pose-eef-speed-tolerance",
        type=float,
        default=0.006,
        help="Maximum EEF speed at the shared decision state, in m/s.",
    )
    parser.add_argument(
        "--common-pose-door-speed-tolerance",
        type=float,
        default=0.006,
        help="Maximum door speed at the shared decision state, in rad/s.",
    )
    parser.add_argument(
        "--preclass-pose-tolerance",
        type=float,
        default=0.0010,
    )
    parser.add_argument(
        "--preclass-rotation-tolerance-deg",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--preclass-gripper-tolerance",
        type=float,
        default=0.0015,
    )
    parser.add_argument(
        "--preclass-door-angle-tolerance",
        type=float,
        default=0.0010,
    )
    parser.add_argument(
        "--preclass-guard-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--preclass-guard-timeout",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--pair-eef-gap-tolerance",
        type=float,
        default=0.00075,
    )
    parser.add_argument(
        "--pair-rotation-gap-tolerance-deg",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--pair-gripper-gap-tolerance",
        type=float,
        default=0.0010,
    )
    parser.add_argument(
        "--pair-door-gap-tolerance",
        type=float,
        default=0.00075,
    )
    parser.add_argument(
        "--pair-agent-image-mae-tolerance",
        type=float,
        default=-1.0,
        help=(
            "Optional pair rejection threshold on agent-view uint8 MAE. "
            "Negative disables image-based rejection."
        ),
    )
    parser.add_argument(
        "--pair-hand-image-mae-tolerance",
        type=float,
        default=-1.0,
        help=(
            "Optional pair rejection threshold on eye-in-hand uint8 MAE. "
            "Negative disables image-based rejection."
        ),
    )

    # Force-conditioned arc pull.
    parser.add_argument(
        "--light-pull-angle-lead",
        type=float,
        default=0.030,
        help=(
            "How far ahead of the live door angle the light-door arc target "
            "is placed, in radians."
        ),
    )
    parser.add_argument(
        "--heavy-pull-angle-lead",
        type=float,
        default=0.075,
        help=(
            "How far ahead of the live door angle the heavy-door arc target "
            "is placed, in radians."
        ),
    )
    parser.add_argument(
        "--light-pull-controller-step",
        type=float,
        default=0.055,
        help=(
            "Larger value gives a gentler Cartesian command for light doors."
        ),
    )
    parser.add_argument(
        "--heavy-pull-controller-step",
        type=float,
        default=0.030,
        help=(
            "Smaller value gives a stronger Cartesian command for heavy doors."
        ),
    )
    parser.add_argument(
        "--light-pull-target-rate-rad-s",
        type=float,
        default=0.16,
        help="Maximum rate of the virtual arc target for the gentle strategy.",
    )
    parser.add_argument(
        "--heavy-pull-target-rate-rad-s",
        type=float,
        default=0.30,
        help="Maximum rate of the virtual arc target for the strong strategy.",
    )
    parser.add_argument(
        "--light-pull-max-translation-action",
        type=float,
        default=0.26,
    )
    parser.add_argument(
        "--heavy-pull-max-translation-action",
        type=float,
        default=0.52,
    )
    parser.add_argument(
        "--pull-max-rotation-action",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--strategy-ramp-steps",
        type=int,
        default=15,
        help=(
            "Smoothly introduce the selected / force-derived strong strategy "
            "after the common initial real-pull steps."
        ),
    )
    parser.add_argument(
        "--hold-open-controller-step",
        type=float,
        default=0.020,
    )
    parser.add_argument(
        "--pull-debug-interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-opening-steps",
        "--pull-timeout",
        dest="pull_timeout",
        type=int,
        default=120,
        help=(
            "Global maximum number of real ADAPTIVE_ARC_PULL actions. The same "
            "deadline is applied to every door, regardless of class."
        ),
    )
    parser.add_argument(
        "--progress-check-steps",
        "--heavy-progress-check-steps",
        dest="progress_check_steps",
        type=int,
        default=80,
        help=(
            "Deprecated compatibility option. The strict global success rule "
            "uses only final angle, total opening-step budget, and breaking "
            "force; this checkpoint is not used to accept demonstrations."
        ),
    )
    parser.add_argument(
        "--minimum-progress-at-check",
        "--heavy-minimum-progress-at-check",
        dest="minimum_progress_at_check",
        type=float,
        default=0.45,
        help=(
            "Deprecated compatibility option paired with "
            "--progress-check-steps; not used by strict success."
        ),
    )
    parser.add_argument(
        "--door-opening-range",
        type=float,
        default=0.90,
        help=(
            "Physical opening range made available from the closed pose, in "
            "radians. 0 keeps the stock Robosuite joint range. Default 0.90 "
            "rad is about 51.6 degrees."
        ),
    )
    parser.add_argument(
        "--open-angle",
        type=float,
        default=0.0,
        help=(
            "Explicit opening angle in radians. 0 uses --open-fraction of "
            "the available hinge range."
        ),
    )
    parser.add_argument(
        "--open-fraction",
        type=float,
        default=0.90,
        help=(
            "Fraction of the available range used as the success target. "
            "With the default 0.90 rad range this targets about 46 degrees."
        ),
    )
    parser.add_argument(
        "--open-range-margin",
        type=float,
        default=0.02,
        help="Safety margin from the hinge limit in radians.",
    )
    parser.add_argument(
        "--open-success-tolerance",
        type=float,
        default=0.10,
        help=(
            "Door-angle tolerance in radians. With target 0.810 rad and "
            "the validated 0.100 rad tolerance, success begins at 0.710 rad."
        ),
    )
    parser.add_argument(
        "--require-correct-force-classification",
        action="store_true",
        help=(
            "Also require the selected push/pull direction to match the true "
            "one-way hinge direction before accepting a demo. Off by default: physical opening "
            "success and force-classification accuracy are recorded separately."
        ),
    )
    parser.add_argument(
        "--open-hold-detection-steps",
        type=int,
        default=3,
    )
    parser.add_argument("--hold-open-steps", type=int, default=25)
    parser.add_argument(
        "--release-after-open",
        action="store_true",
        help=(
            "Release and retreat after opening. Off by default because the "
            "passive door closes again after release."
        ),
    )
    parser.add_argument("--release-steps", type=int, default=20)
    parser.add_argument("--retreat-extra-height", type=float, default=0.08)
    parser.add_argument("--retreat-timeout", type=int, default=100)

    parser.add_argument(
        "--require-pull-contacts",
        dest="require_pull_contacts",
        action="store_true",
        help="Fail a rollout when handle contact is lost for too long.",
    )
    parser.add_argument(
        "--ignore-pull-contact-loss",
        dest="require_pull_contacts",
        action="store_false",
        help="Continue pulling even when contact matching reports no contact.",
    )
    parser.set_defaults(require_pull_contacts=True)
    parser.add_argument(
        "--pull-contact-loss-steps",
        type=int,
        default=5,
    )

    # Force / speed safety diagnostics. These limits are recorded for every
    # rollout, but are non-fatal by default because the push-vs-pull task is
    # defined by direction inference, opening success, timeout, and the physical
    # one-way lock. Enable enforcement only for a separate safe-manipulation
    # experiment.
    parser.add_argument(
        "--enforce-safety-limits",
        "--enforce-damage-limits",
        dest="enforce_safety_limits",
        action="store_true",
        help=(
            "Reject a rollout when force, overload impulse, or opening-speed "
            "limits are exceeded. Off by default."
        ),
    )
    parser.add_argument(
        "--ignore-safety-limits",
        "--ignore-damage-limits",
        dest="enforce_safety_limits",
        action="store_false",
        help=(
            "Record safety-limit exceedances but do not reject the rollout "
            "(default)."
        ),
    )
    parser.set_defaults(enforce_safety_limits=False)

    # Force / speed diagnostic model.
    parser.add_argument(
        "--damage-force-ema-alpha",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--door-speed-ema-alpha",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--damage-grace-steps",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--breaking-force-hold-steps",
        type=int,
        default=4,
        help=(
            "Consecutive tangential-force EMA samples above the derived soft "
            "breaking limit required to fail. A separate hard limit still "
            "fails immediately, while this value rejects sustained overload "
            "without reacting to a single contact transient."
        ),
    )
    parser.add_argument(
        "--overspeed-hold-steps",
        type=int,
        default=4,
    )

    # One deterministic physical-strength mapping for every door. The class
    # label is never used by this rule. Low-resistance doors are intentionally
    # more fragile, while high-resistance doors can safely tolerate the stronger
    # strategy. The mapping is a task-design proxy, not a material-science law.
    parser.add_argument(
        "--resistance-friction-weight",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--damage-force-base-n",
        type=float,
        default=35.0,
        help=(
            "Base term of the shared breaking-force formula. Lower-resistance "
            "doors therefore remain fragile without using the class label."
        ),
    )
    parser.add_argument(
        "--damage-force-per-resistance-n",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--damage-force-min-n",
        type=float,
        default=35.0,
    )
    parser.add_argument(
        "--damage-force-max-n",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--hard-break-force-factor",
        type=float,
        default=1.75,
    )
    parser.add_argument(
        "--damage-impulse-time-s",
        type=float,
        default=0.15,
        help=(
            "Allowed accumulated overload equals damage_force_limit multiplied "
            "by this time constant."
        ),
    )
    parser.add_argument(
        "--safe-speed-base-rad-s",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--safe-speed-per-resistance-rad-s",
        type=float,
        default=0.020,
    )
    parser.add_argument(
        "--safe-speed-min-rad-s",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--safe-speed-max-rad-s",
        type=float,
        default=1.20,
    )

    # Resistance distributions.
    parser.add_argument("--light-damping-min", type=float, default=0.05)
    parser.add_argument("--light-damping-max", type=float, default=0.15)
    parser.add_argument("--light-friction-min", type=float, default=0.00)
    parser.add_argument("--light-friction-max", type=float, default=0.02)

    parser.add_argument("--heavy-damping-min", type=float, default=8.0)
    parser.add_argument("--heavy-damping-max", type=float, default=12.0)
    parser.add_argument("--heavy-friction-min", type=float, default=1.50)
    parser.add_argument("--heavy-friction-max", type=float, default=2.50)

    args = parser.parse_args()

    if args.episodes <= 0 or args.episodes % 2 != 0:
        raise ValueError("--episodes must be a positive even number")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.door_x_min > args.door_x_max:
        raise ValueError("--door-x-min must be <= --door-x-max")
    if args.door_y_min > args.door_y_max:
        raise ValueError("--door-y-min must be <= --door-y-max")
    if args.door_yaw_min_deg > args.door_yaw_max_deg:
        raise ValueError(
            "--door-yaw-min-deg must be <= --door-yaw-max-deg"
        )
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")
    if args.preclass_guard_steps <= 0:
        raise ValueError("--preclass-guard-steps must be positive")
    if args.preclass_guard_timeout < args.preclass_guard_steps:
        raise ValueError(
            "--preclass-guard-timeout must be >= "
            "--preclass-guard-steps"
        )
    if args.common_pose_stable_steps <= 0:
        raise ValueError(
            "--common-pose-stable-steps must be positive"
        )
    if args.preclass_pose_tolerance <= 0.0:
        raise ValueError(
            "--preclass-pose-tolerance must be positive"
        )
    if args.preclass_rotation_tolerance_deg <= 0.0:
        raise ValueError(
            "--preclass-rotation-tolerance-deg must be positive"
        )
    if args.preclass_gripper_tolerance <= 0.0:
        raise ValueError(
            "--preclass-gripper-tolerance must be positive"
        )
    if args.preclass_door_angle_tolerance <= 0.0:
        raise ValueError(
            "--preclass-door-angle-tolerance must be positive"
        )
    if args.pair_eef_gap_tolerance <= 0.0:
        raise ValueError(
            "--pair-eef-gap-tolerance must be positive"
        )
    if args.pair_rotation_gap_tolerance_deg <= 0.0:
        raise ValueError(
            "--pair-rotation-gap-tolerance-deg must be positive"
        )
    if args.pair_gripper_gap_tolerance <= 0.0:
        raise ValueError(
            "--pair-gripper-gap-tolerance must be positive"
        )
    if args.pair_door_gap_tolerance <= 0.0:
        raise ValueError(
            "--pair-door-gap-tolerance must be positive"
        )
    if args.light_damping_max >= args.heavy_damping_min:
        raise ValueError(
            "Light and heavy damping ranges must not overlap"
        )
    if args.force_threshold <= 0:
        raise ValueError("--force-threshold must be positive")
    if not 0.0 <= args.fixed_strategy_blend <= 1.0:
        raise ValueError(
            "--fixed-strategy-blend must be in [0, 1]"
        )
    if args.online_force_window <= 0:
        raise ValueError("--online-force-window must be positive")
    if args.online_force_ignore_steps < 0:
        raise ValueError(
            "--online-force-ignore-steps must be non-negative"
        )
    minimum_warmup = (
        args.online_force_ignore_steps
        + args.online_force_window
        - 1
    )
    if args.common_pull_warmup_steps < minimum_warmup:
        raise ValueError(
            "--common-pull-warmup-steps must be at least "
            "online_force_ignore_steps + online_force_window - 1"
        )
    if args.force_blend_band_n <= 0.0:
        raise ValueError("--force-blend-band-n must be positive")
    if not 0.0 < args.force_control_ema_alpha <= 1.0:
        raise ValueError(
            "--force-control-ema-alpha must be in (0, 1]"
        )
    if args.insert_min_steps < 0:
        raise ValueError("--insert-min-steps must be non-negative")
    if args.light_pull_angle_lead <= 0:
        raise ValueError("--light-pull-angle-lead must be positive")
    if args.heavy_pull_angle_lead <= 0:
        raise ValueError("--heavy-pull-angle-lead must be positive")
    if args.pull_debug_interval <= 0:
        raise ValueError("--pull-debug-interval must be positive")
    if args.strategy_ramp_steps <= 0:
        raise ValueError("--strategy-ramp-steps must be positive")
    if args.progress_check_steps <= 0:
        raise ValueError(
            "--progress-check-steps must be positive"
        )
    if not 0.0 < args.minimum_progress_at_check:
        raise ValueError(
            "--minimum-progress-at-check must be positive"
        )
    if not 0.0 < args.open_fraction <= 1.0:
        raise ValueError("--open-fraction must be in (0, 1]")
    if args.open_angle < 0.0:
        raise ValueError("--open-angle must be >= 0")
    if args.open_success_tolerance < 0.0:
        raise ValueError(
            "--open-success-tolerance must be >= 0"
        )
    if args.door_opening_range < 0.0:
        raise ValueError("--door-opening-range must be >= 0")
    if args.probe_max_door_angle <= 0.0:
        raise ValueError("--probe-max-door-angle must be positive")
    if args.return_motion_steps <= 0:
        raise ValueError("--return-motion-steps must be positive")
    if args.return_controller_step <= 0.0:
        raise ValueError(
            "--return-controller-step must be positive"
        )
    if args.return_max_translation_action <= 0.0:
        raise ValueError(
            "--return-max-translation-action must be positive"
        )
    if args.common_pose_eef_speed_tolerance <= 0.0:
        raise ValueError(
            "--common-pose-eef-speed-tolerance must be positive"
        )
    if args.common_pose_door_speed_tolerance <= 0.0:
        raise ValueError(
            "--common-pose-door-speed-tolerance must be positive"
        )
    if args.return_debug_interval <= 0:
        raise ValueError(
            "--return-debug-interval must be positive"
        )
    if args.return_timeout <= args.return_motion_steps:
        raise ValueError(
            "--return-timeout must be greater than "
            "--return-motion-steps"
        )
    if args.dynamic_probe_angle <= 0.0:
        raise ValueError("--dynamic-probe-angle must be positive")
    if args.dynamic_probe_steps <= 2:
        raise ValueError(
            "--dynamic-probe-steps must be greater than 2"
        )
    if (
        args.dynamic_probe_ignore_steps < 0
        or args.dynamic_probe_ignore_steps
        >= args.dynamic_probe_steps
    ):
        raise ValueError(
            "--dynamic-probe-ignore-steps must be in "
            "[0, dynamic-probe-steps)"
        )
    if not 0.0 <= args.dynamic_probe_trim_fraction < 0.5:
        raise ValueError(
            "--dynamic-probe-trim-fraction must be in [0, 0.5)"
        )
    if args.dynamic_probe_angle > args.probe_max_door_angle:
        raise ValueError(
            "--dynamic-probe-angle must not exceed "
            "--probe-max-door-angle"
        )
    if args.baseline_stat_window <= 0:
        raise ValueError("--baseline-stat-window must be positive")
    if not 0.0 < args.damage_force_ema_alpha <= 1.0:
        raise ValueError("--damage-force-ema-alpha must be in (0, 1]")
    if args.breaking_force_hold_steps <= 0:
        raise ValueError(
            "--breaking-force-hold-steps must be positive"
        )
    if not 0.0 < args.door_speed_ema_alpha <= 1.0:
        raise ValueError("--door-speed-ema-alpha must be in (0, 1]")
    if args.resistance_friction_weight < 0.0:
        raise ValueError("--resistance-friction-weight must be non-negative")
    if args.damage_force_base_n <= 0.0:
        raise ValueError("--damage-force-base-n must be positive")
    if args.damage_force_per_resistance_n < 0.0:
        raise ValueError(
            "--damage-force-per-resistance-n must be non-negative"
        )
    if not 0.0 < args.damage_force_min_n < args.damage_force_max_n:
        raise ValueError("Invalid damage-force clipping range")
    if args.hard_break_force_factor <= 1.0:
        raise ValueError("--hard-break-force-factor must be greater than 1")
    if args.damage_impulse_time_s <= 0.0:
        raise ValueError("--damage-impulse-time-s must be positive")
    if not 0.0 < args.safe_speed_min_rad_s <= args.safe_speed_max_rad_s:
        raise ValueError("Invalid safe-speed clipping range")
    if args.safe_speed_base_rad_s <= 0.0:
        raise ValueError("--safe-speed-base-rad-s must be positive")
    if args.safe_speed_per_resistance_rad_s < 0.0:
        raise ValueError(
            "--safe-speed-per-resistance-rad-s must be non-negative"
        )
    for name in (
        "insert_max_translation_action",
        "insert_max_rotation_action",
        "close_max_translation_action",
        "contact_seek_max_translation_action",
        "seat_max_translation_action",
        "probe_max_translation_action",
        "probe_max_rotation_action",
        "light_pull_max_translation_action",
        "heavy_pull_max_translation_action",
        "pull_max_rotation_action",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.grasp_impact_force_limit_n <= 0.0:
        raise ValueError("--grasp-impact-force-limit-n must be positive")
    if args.light_pull_target_rate_rad_s <= 0.0:
        raise ValueError("--light-pull-target-rate-rad-s must be positive")
    if args.heavy_pull_target_rate_rad_s <= 0.0:
        raise ValueError("--heavy-pull-target-rate-rad-s must be positive")

    for name in (
        "direction_probe_steps",
        "direction_open_target_rate_rad_s",
        "direction_open_angle_lead",
        "direction_open_controller_step",
        "direction_open_max_translation_action",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be positive"
            )
    if args.direction_probe_ignore_steps < 0:
        raise ValueError(
            "--direction-probe-ignore-steps must be non-negative"
        )
    if args.wrong_direction_slack_rad < 0.0:
        raise ValueError(
            "--wrong-direction-slack-rad must be non-negative"
        )
    if args.direction_lock_tolerance_rad < 0.0:
        raise ValueError(
            "--direction-lock-tolerance-rad must be non-negative"
        )
    if args.direction_damping_min > args.direction_damping_max:
        raise ValueError("Invalid direction damping range")
    if args.direction_friction_min > args.direction_friction_max:
        raise ValueError("Invalid direction friction range")

    return args


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Add --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    excluded_placement_seeds = load_excluded_door_placement_seeds(
        args.exclude_hdf5
    )
    used_test_placement_seeds: set[int] = set()
    if args.test_only:
        print(
            f"FINAL TEST MODE: excluding "
            f"{len(excluded_placement_seeds)} previously used placement seeds."
        )

    env, env_meta = create_environment(args)
    try:
        # Populate model references after the first reset.
        env.reset()
        door_geometry = resolve_door_geometry(env)
        gripper_geometry = resolve_gripper_geometry(env)
        # run_episode edits the MuJoCo joint range in-place. With hard_reset=False
        # that model mutation survives env.reset(), so restore the stock range
        # before every episode to make paired resets genuinely identical.
        canonical_hinge_range = np.asarray(
            env.sim.model.jnt_range[door_geometry.hinge_id],
            dtype=np.float64,
        ).copy()

        print(
            f"Door hinge: {door_geometry.hinge_name}"
        )
        print(
            f"Handle site: {door_geometry.handle_site_name}"
        )
        print(
            f"Primary handle collision geom: "
            f"{door_geometry.primary_handle_name}"
        )
        print(
            f"Finger references: "
            f"left={gripper_geometry.left_name}, "
            f"right={gripper_geometry.right_name}"
        )
        if gripper_geometry.left_name == gripper_geometry.right_name:
            raise RuntimeError(
                "Left and right finger names are identical; refusing to run "
                "because wrist calibration would be invalid."
            )
        configured_target = (
            args.open_angle
            if args.open_angle > 0
            else args.open_fraction * args.door_opening_range
        )
        required_progress = max(
            0.0,
            configured_target - args.open_success_tolerance,
        )
        print(
            "GLOBAL TASK SUCCESS = angle reached AND opening-step budget "
            "respected AND resistance-derived breaking force not exceeded."
        )
        print(
            f"Angle: target≈{configured_target:.3f}rad, "
            f"required={required_progress:.3f}rad. "
            f"Time: max_opening_steps={args.pull_timeout}, counted from the "
            "first ADAPTIVE_ARC_PULL action."
        )
        print(
            "Safety formula: resistance_score=damping+"
            f"{args.resistance_friction_weight:.2f}*friction; "
            f"damage_force=clip({args.damage_force_base_n:.1f}+"
            f"{args.damage_force_per_resistance_n:.2f}*score, "
            f"{args.damage_force_min_n:.1f}, {args.damage_force_max_n:.1f})N"
        )

        print(
            "FSM: MOVE_ABOVE -> ORIENT_AT_CLEARANCE -> "
            "MOVE_PREGRASP -> INSERT_TO_HANDLE -> CLOSE_GRIPPER -> "
            "VERIFY_GRASP -> SEAT_GRASP -> FORCE_BASELINE -> "
            "ADAPTIVE_ARC_PULL -> HOLD_OPEN"
        )

        pair_target = args.episodes // 2
        max_pair_attempts = (
            args.max_pair_attempts
            if args.max_pair_attempts is not None
            else max(pair_target * 20, pair_target)
        )

        accepted_demo_names: list[str] = []
        accepted_pair_ids: dict[str, int] = {}
        counts = defaultdict(int)
        total_samples = 0

        with h5py.File(output_path, "w") as output:
            data_group = output.create_group("data")
            data_group.attrs["env"] = "Door"
            data_group.attrs["dataset_role"] = (
                "final_test" if args.test_only else "train_validation"
            )
            data_group.attrs["non_repeat_test_enforced"] = bool(args.test_only)
            data_group.attrs["excluded_placement_seed_count"] = int(
                len(excluded_placement_seeds)
            )
            data_group.attrs["env_args"] = json.dumps(
                env_meta,
                default=json_default,
            )
            data_group.attrs["force_threshold_n"] = float(
                args.force_threshold
            )
            data_group.attrs["hinge_joint_name"] = (
                door_geometry.hinge_name
            )
            data_group.attrs["handle_site_name"] = (
                door_geometry.handle_site_name
            )
            data_group.attrs["handle_geom_name"] = (
                door_geometry.primary_handle_name
            )
            data_group.attrs["preclass_guard_steps"] = 0
            data_group.attrs["common_pose_return_enabled"] = False
            data_group.attrs["common_pose_stable_steps"] = int(
                args.common_pose_stable_steps
            )
            data_group.attrs["preclass_pose_tolerance_m"] = float(
                args.preclass_pose_tolerance
            )
            data_group.attrs["preclass_rotation_tolerance_deg"] = float(
                args.preclass_rotation_tolerance_deg
            )
            data_group.attrs["preclass_gripper_tolerance"] = float(
                args.preclass_gripper_tolerance
            )
            data_group.attrs["preclass_door_tolerance_rad"] = float(
                args.preclass_door_angle_tolerance
            )
            data_group.attrs["pair_eef_gap_tolerance_m"] = float(
                args.pair_eef_gap_tolerance
            )
            data_group.attrs["pair_rotation_gap_tolerance_deg"] = float(
                args.pair_rotation_gap_tolerance_deg
            )
            data_group.attrs["pair_gripper_gap_tolerance"] = float(
                args.pair_gripper_gap_tolerance
            )
            data_group.attrs["pair_door_gap_tolerance_rad"] = float(
                args.pair_door_gap_tolerance
            )
            data_group.attrs["door_x_range"] = np.asarray(
                [args.door_x_min, args.door_x_max],
                dtype=np.float64,
            )
            data_group.attrs["door_y_range"] = np.asarray(
                [args.door_y_min, args.door_y_max],
                dtype=np.float64,
            )
            data_group.attrs["door_yaw_range_deg"] = np.asarray(
                [
                    args.door_yaw_min_deg,
                    args.door_yaw_max_deg,
                ],
                dtype=np.float64,
            )
            data_group.attrs["pair_shared_placement_seed"] = True
            data_group.attrs["probe_force_summary_observation"] = False
            data_group.attrs["force_measurement_mode"] = (
                "common_pull_probe"
            )
            data_group.attrs["separate_force_probe"] = True
            data_group.attrs["reset_after_force_measurement"] = False
            data_group.attrs["online_force_window"] = int(
                args.online_force_window
            )
            data_group.attrs["common_pull_warmup_steps"] = int(
                args.common_pull_warmup_steps
            )
            data_group.attrs["force_blend_band_n"] = float(
                args.force_blend_band_n
            )
            data_group.attrs["paired_scenes"] = True
            data_group.attrs["direction_classes"] = json.dumps(["push", "pull"])
            data_group.attrs["common_probe_direction"] = "pull"
            data_group.attrs["wrong_direction_slack_rad"] = float(
                args.wrong_direction_slack_rad
            )
            data_group.attrs["direction_lock_is_physical_joint_range"] = True
            data_group.attrs["success_target_open_angle_rad"] = float(
                args.open_angle if args.open_angle > 0.0
                else args.open_fraction * args.door_opening_range
            )
            data_group.attrs["success_tolerance_rad"] = float(
                args.open_success_tolerance
            )
            data_group.attrs["legacy_progress_check_steps_unused"] = int(
                args.progress_check_steps
            )
            data_group.attrs[
                "legacy_minimum_progress_at_check_rad_unused"
            ] = float(args.minimum_progress_at_check)
            data_group.attrs["global_pull_timeout_steps"] = int(
                args.pull_timeout
            )
            data_group.attrs["global_max_opening_steps"] = int(
                args.pull_timeout
            )
            data_group.attrs["opening_step_count_starts_at"] = (
                "first ADAPTIVE_ARC_PULL action"
            )
            data_group.attrs["task_design_version"] = (
                "one_way_force_guided_push_pull_v2_direction_only"
            )
            data_group.attrs["fixed_force_threshold_n"] = float(
                args.force_threshold
            )
            data_group.attrs["threshold_calibration_required"] = False
            data_group.attrs["enforce_safety_limits"] = bool(
                args.enforce_safety_limits
            )
            data_group.attrs["global_success_rule"] = (
                "open in the hidden physically allowed push/pull direction, reach "
                "required_open_progress within one shared deadline, and never "
                "violate the one-way joint lock; force/speed safety limits are "
                + (
                    "terminal"
                    if args.enforce_safety_limits
                    else "diagnostic only"
                )
            )
            data_group.attrs["safety_rule"] = (
                "force, overload impulse, and speed exceedances are always "
                "recorded; rejection is controlled by enforce_safety_limits"
            )
            data_group.attrs["resistance_friction_weight"] = float(
                args.resistance_friction_weight
            )
            data_group.attrs["damage_force_base_n"] = float(
                args.damage_force_base_n
            )
            data_group.attrs["damage_force_per_resistance_n"] = float(
                args.damage_force_per_resistance_n
            )
            data_group.attrs["hard_break_force_factor"] = float(
                args.hard_break_force_factor
            )
            data_group.attrs["damage_impulse_time_s"] = float(
                args.damage_impulse_time_s
            )
            data_group.attrs["safe_speed_base_rad_s"] = float(
                args.safe_speed_base_rad_s
            )
            data_group.attrs["safe_speed_per_resistance_rad_s"] = float(
                args.safe_speed_per_resistance_rad_s
            )

            progress = tqdm(
                total=args.episodes,
                desc="valid paired door demos",
            )

            accepted_pair_count = 0
            pair_attempt = 0

            while (
                accepted_pair_count < pair_target
                and pair_attempt < max_pair_attempts
            ):
                pair_attempt += 1
                pair_id = accepted_pair_count
                placement_seed = int(
                    rng.integers(0, 2**31 - 1)
                )
                if args.test_only:
                    while (
                        placement_seed in excluded_placement_seeds
                        or placement_seed in used_test_placement_seeds
                    ):
                        placement_seed = int(
                            rng.integers(0, 2**31 - 1)
                        )

                specs = sample_pair_specs(
                    pair_id=pair_id,
                    placement_seed=placement_seed,
                    rng=rng,
                    args=args,
                )

                pair_results: list[
                    tuple[EpisodeSpec, EpisodeResult]
                ] = []
                pair_failed = False

                for spec in specs:
                    try:
                        env.sim.model.jnt_range[door_geometry.hinge_id] = (
                            canonical_hinge_range
                        )
                        env.sim.forward()
                        result = run_episode(
                            env,
                            spec,
                            door_geometry,
                            gripper_geometry,
                            args,
                        )
                    except KeyboardInterrupt:
                        raise
                    except Exception as error:
                        print(
                            f"[pair reject] attempt={pair_attempt} "
                            f"class={spec.true_class} error={error}"
                        )
                        pair_failed = True
                        break

                    if (
                        not result.success
                        or result.trajectory is None
                    ):
                        print(
                            f"[pair reject] attempt={pair_attempt} "
                            f"class={spec.true_class} "
                            f"pred={result.predicted_class} "
                            f"signal={result.force_signal} "
                            f"reason={result.reason}"
                        )
                        pair_failed = True
                        break

                    pair_results.append((spec, result))

                if pair_failed or len(pair_results) != 2:
                    continue

                alignment = pair_alignment_metrics(
                    pair_results[0][1],
                    pair_results[1][1],
                )
                pair_alignment_failed = (
                    alignment["eef_gap_m"]
                    > args.pair_eef_gap_tolerance
                    or alignment["rotation_gap_deg"]
                    > args.pair_rotation_gap_tolerance_deg
                    or alignment["gripper_gap"]
                    > args.pair_gripper_gap_tolerance
                    or alignment["door_gap_rad"]
                    > args.pair_door_gap_tolerance
                    or (
                        args.pair_agent_image_mae_tolerance >= 0.0
                        and alignment["agentview_mae_uint8"]
                        > args.pair_agent_image_mae_tolerance
                    )
                    or (
                        args.pair_hand_image_mae_tolerance >= 0.0
                        and alignment["eye_in_hand_mae_uint8"]
                        > args.pair_hand_image_mae_tolerance
                    )
                )
                if pair_alignment_failed:
                    print(
                        f"[pair reject] attempt={pair_attempt} "
                        f"decision_eef_gap="
                        f"{alignment['eef_gap_m']:.6f}m "
                        f"decision_rotation_gap="
                        f"{alignment['rotation_gap_deg']:.3f}deg "
                        f"decision_gripper_gap="
                        f"{alignment['gripper_gap']:.6f} "
                        f"decision_door_gap="
                        f"{alignment['door_gap_rad']:.6f}rad "
                        f"agent_mae="
                        f"{alignment['agentview_mae_uint8']:.3f} "
                        f"hand_mae="
                        f"{alignment['eye_in_hand_mae_uint8']:.3f}"
                    )
                    continue

                # Save both pair members only after both succeed and start the
                # real pull from a matched pre-interaction state.
                for spec, result in sorted(
                    pair_results,
                    key=lambda item: (0 if item[0].true_class == "push" else 1),
                ):
                    demo_name = (
                        f"demo_{len(accepted_demo_names)}"
                    )
                    total_samples += write_episode(
                        data_group,
                        demo_name,
                        result,
                        spec,
                    )
                    accepted_demo_names.append(demo_name)
                    accepted_pair_ids[demo_name] = spec.pair_id
                    counts[spec.true_class] += 1
                    progress.update(1)

                if args.test_only:
                    used_test_placement_seeds.add(int(placement_seed))

                accepted_pair_count += 1
                progress.set_postfix(
                    push=counts["push"],
                    pull=counts["pull"],
                    pair_attempt=pair_attempt,
                )

            progress.close()

            if accepted_pair_count < pair_target:
                raise RuntimeError(
                    f"Collected only {accepted_pair_count}/{pair_target} "
                    f"valid pairs after {pair_attempt} attempts. "
                    "Inspect the preceding [pair reject] reason. It may be a "
                    "force classification, safety, deadline, or geometry "
                    "failure."
                )

            data_group.attrs["total"] = int(total_samples)
            if args.test_only:
                test_names = create_test_mask(
                    output,
                    accepted_demo_names,
                )
                train_names: list[str] = []
                validation_names: list[str] = []
                data_group.attrs["num_test_demos"] = int(len(test_names))
            else:
                train_names, validation_names = create_pairwise_masks(
                    output,
                    accepted_demo_names,
                    accepted_pair_ids,
                    val_ratio=args.val_ratio,
                    seed=args.split_seed,
                )

        print(
            f"Saved {len(accepted_demo_names)} demonstrations "
            f"({accepted_pair_count} pairs) to {output_path}"
        )
        split_summary = (
            f"test={len(test_names)}"
            if args.test_only
            else f"train={len(train_names)}, valid={len(validation_names)}"
        )
        print(
            f"Direction counts: {dict(counts)}; {split_summary}"
        )

    finally:
        if args.render:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
