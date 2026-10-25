from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import h5py
import numpy as np
import torch

from evaluation.sim.common import EvaluationPolicy
from force2act.data.normalization import StandardNormalizer
from force2act.data.split import get_train_val_names
from sim_collect.common import (
    add_force_to_observation,
    check_success,
    create_data_processing_env,
    image_to_hwc_uint8,
    initialize_obs_utils,
    make_initial_state,
    reset_to,
    step_env,
)


def decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")

    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")

    return str(value)


def resolve_splits(
    config: dict,
    checkpoint: dict | None = None,
) -> tuple[list[str], list[str]]:
    del checkpoint
    return get_train_val_names(config["dataset"]["hdf5_path"])


def build_history_tensors(
    observation_history: Sequence[dict[str, Any]],
    *,
    state_keys: Sequence[str],
    camera_keys: Sequence[str],
    force_keys: Sequence[str] = (),
    flip_images: bool,
) -> dict[str, torch.Tensor]:
    """Build one model input window in the same layout as EpisodeDataset.

    Images stay uint8 here; the Force2Act image preprocessor owns scaling and
    ImageNet normalization, exactly as in training.
    """
    states = []
    for observation in observation_history:
        parts = [
            np.asarray(observation[key], dtype=np.float32).reshape(-1)
            for key in state_keys
        ]
        states.append(np.concatenate(parts, axis=-1))

    state = torch.from_numpy(np.stack(states, axis=0)).unsqueeze(0)

    cameras = []
    for key in camera_keys:
        frames = []
        for observation in observation_history:
            frame = image_to_hwc_uint8(
                observation[key],
                key,
                flip_vertical=flip_images,
            )
            frame = torch.from_numpy(frame).permute(2, 0, 1).contiguous()
            frames.append(frame)
        cameras.append(torch.stack(frames, dim=0))

    images = torch.stack(cameras, dim=0).unsqueeze(0)
    result: dict[str, torch.Tensor] = {
        "state": state,
        "images": images,
    }

    if force_keys:
        forces = []
        for observation in observation_history:
            parts = [
                np.asarray(observation[key], dtype=np.float32).reshape(-1)
                for key in force_keys
            ]
            forces.append(np.concatenate(parts, axis=-1))
        result["force"] = torch.from_numpy(
            np.stack(forces, axis=0)
        ).unsqueeze(0)

    return result


def unwrap_env(env: Any) -> Any:
    current = env
    visited: set[int] = set()

    for _ in range(12):
        if current is None or id(current) in visited:
            break

        visited.add(id(current))

        next_env = None

        for attr in ("env", "wrapped_env", "base_env"):
            candidate = getattr(current, attr, None)

            if candidate is not None and candidate is not current:
                next_env = candidate
                break

        if next_env is None:
            break

        current = next_env

    return current


def get_sim(env: Any) -> Any:
    current = env
    visited: set[int] = set()

    for _ in range(12):
        if current is None or id(current) in visited:
            break

        visited.add(id(current))

        if hasattr(current, "sim"):
            return current.sim

        next_env = None

        for attr in ("env", "wrapped_env", "base_env"):
            candidate = getattr(current, attr, None)

            if candidate is not None and candidate is not current:
                next_env = candidate
                break

        current = next_env

    raise RuntimeError(
        "Could not access the underlying MuJoCo simulation."
    )


def find_object_body_name(env: Any) -> str:
    """Resolve the active PickPlaceCan object body, not objects[0].

    PickPlaceCan may keep inactive Milk / Bread / Cereal bodies far away.
    Using objects[0] can therefore select Milk_main and produce positions
    around [10, 10, ...]. This mirrors the collector's active_object logic.
    """
    root_env = unwrap_env(env)
    sim = get_sim(env)

    object_name = getattr(root_env, "obj_to_use", None) or "Can"
    object_model = None

    objects = getattr(root_env, "objects", None) or []
    for obj in objects:
        name = str(getattr(obj, "name", ""))
        if name == object_name or name.lower() == str(object_name).lower():
            object_model = obj
            break

    if object_model is None:
        for obj in objects:
            name = str(getattr(obj, "name", ""))
            if "can" in name.lower():
                object_model = obj
                break

    obj_body_id = getattr(root_env, "obj_body_id", None)
    if object_model is not None and obj_body_id is not None:
        model_name = str(getattr(object_model, "name", object_name))
        if model_name in obj_body_id:
            body_id = int(obj_body_id[model_name])
            body_name = sim.model.body_id2name(body_id)
            if body_name:
                return str(body_name)

    if object_model is not None:
        for attr in ("root_body", "body_name"):
            value = getattr(object_model, attr, None)
            if value:
                try:
                    sim.model.body_name2id(str(value))
                    return str(value)
                except Exception:
                    pass

    for name in ("Can_main", "Can", "can_main", "can"):
        try:
            sim.model.body_name2id(name)
            return name
        except Exception:
            pass

    available_names = []
    for body_id in range(int(sim.model.nbody)):
        try:
            name = sim.model.body_id2name(body_id)
        except Exception:
            name = None
        if name:
            available_names.append(str(name))

    likely_names = [
        name
        for name in available_names
        if "can" in name.lower() and "visual" not in name.lower()
    ]
    if likely_names:
        return likely_names[0]

    raise RuntimeError(
        "Could not identify the active Can body. "
        f"obj_to_use={object_name!r}; available={available_names[:40]}"
    )


def restore_object_mass(
    env: Any,
    body_name: str,
    target_mass_kg: float,
) -> dict[str, float]:
    sim = get_sim(env)

    body_id = int(
        sim.model.body_name2id(body_name)
    )

    previous_mass = float(
        sim.model.body_mass[body_id]
    )

    previous_inertia = np.asarray(
        sim.model.body_inertia[body_id],
        dtype=np.float64,
    ).copy()

    if target_mass_kg <= 0:
        raise ValueError(
            f"Invalid target mass: {target_mass_kg}"
        )

    scale = target_mass_kg / previous_mass

    sim.model.body_mass[body_id] = target_mass_kg
    sim.model.body_inertia[body_id] = (
        previous_inertia * scale
    )

    sim.forward()

    return {
        "previous_mass": previous_mass,
        "restored_mass": float(
            sim.model.body_mass[body_id]
        ),
        "inertia_scale": scale,
    }


def get_object_position(
    env: Any,
    body_name: str,
) -> np.ndarray:
    sim = get_sim(env)

    body_id = int(
        sim.model.body_name2id(body_name)
    )

    return np.asarray(
        sim.data.body_xpos[body_id],
        dtype=np.float64,
    ).copy()




def normalize_stack_cube_name(value: Any) -> str:
    """Normalize collector cube labels while rejecting ambiguous metadata."""
    text = decode_attr(value).strip().lower()
    mapping = {
        "cubea": "cubeA",
        "cube_a": "cubeA",
        "a": "cubeA",
        "cubeb": "cubeB",
        "cube_b": "cubeB",
        "b": "cubeB",
    }
    if text not in mapping:
        raise ValueError(
            f"Expected a Stack cube name (cubeA or cubeB), got {value!r}."
        )
    return mapping[text]


def resolve_stack_geometries(env: Any) -> dict[str, dict[str, Any]]:
    """Resolve the stock robosuite Stack cube references after every reset."""
    root_env = unwrap_env(env)
    sim = get_sim(env)
    output: dict[str, dict[str, Any]] = {}

    for name in ("cubeA", "cubeB"):
        object_model = getattr(root_env, name, None)
        body_id = getattr(root_env, f"{name}_body_id", None)

        if body_id is None and object_model is not None:
            for attr in ("root_body", "body_name"):
                body_name = getattr(object_model, attr, None)
                if body_name:
                    try:
                        body_id = int(sim.model.body_name2id(str(body_name)))
                        break
                    except Exception:
                        pass

        if body_id is None:
            available = []
            for candidate_id in range(int(sim.model.nbody)):
                try:
                    candidate = sim.model.body_id2name(candidate_id)
                except Exception:
                    candidate = None
                if candidate:
                    available.append(str(candidate))
            likely = [
                candidate
                for candidate in available
                if name.lower() in candidate.lower()
                and "visual" not in candidate.lower()
            ]
            if not likely:
                raise RuntimeError(
                    f"Could not resolve {name} body; available={available[:60]}"
                )
            body_id = int(sim.model.body_name2id(likely[0]))

        body_id = int(body_id)
        body_name = sim.model.body_id2name(body_id)

        if object_model is None:
            raise RuntimeError(
                f"The underlying Stack environment does not expose {name}."
            )

        top = np.asarray(object_model.top_offset, dtype=np.float64).reshape(3)
        bottom = np.asarray(
            object_model.bottom_offset, dtype=np.float64
        ).reshape(3)
        half_height = 0.5 * abs(float(top[2] - bottom[2]))
        if not np.isfinite(half_height) or half_height <= 0.0:
            raise RuntimeError(
                f"Could not resolve a positive half-height for {name}."
            )

        output[name] = {
            "name": name,
            "body_id": body_id,
            "body_name": str(body_name) if body_name else name,
            "object_model": object_model,
            "half_height": float(half_height),
        }

    return output


def restore_stack_masses(
    env: Any,
    geometries: dict[str, dict[str, Any]],
    cube_a_mass_kg: float,
    cube_b_mass_kg: float,
) -> dict[str, dict[str, float]]:
    """Restore the hidden per-demo cube masses and scale inertia consistently."""
    sim = get_sim(env)
    target_masses = {
        "cubeA": float(cube_a_mass_kg),
        "cubeB": float(cube_b_mass_kg),
    }
    information: dict[str, dict[str, float]] = {}

    for name, target_mass in target_masses.items():
        if not np.isfinite(target_mass) or target_mass <= 0.0:
            raise ValueError(f"Invalid {name} mass: {target_mass}")

        body_id = int(geometries[name]["body_id"])
        previous_mass = float(sim.model.body_mass[body_id])
        if not np.isfinite(previous_mass) or previous_mass <= 0.0:
            raise ValueError(
                f"Invalid current {name} mass before restore: {previous_mass}"
            )

        scale = target_mass / previous_mass
        sim.model.body_mass[body_id] = target_mass
        sim.model.body_inertia[body_id] = (
            np.asarray(sim.model.body_inertia[body_id], dtype=np.float64)
            * scale
        )
        information[name] = {
            "previous_mass": previous_mass,
            "restored_mass": target_mass,
            "inertia_scale": float(scale),
        }

    sim.forward()
    return information


def get_stack_cube_position(
    env: Any,
    geometry: dict[str, Any],
) -> np.ndarray:
    sim = get_sim(env)
    return np.asarray(
        sim.data.body_xpos[int(geometry["body_id"])],
        dtype=np.float64,
    ).copy()


def get_stack_gripper(root_env: Any) -> Any:
    robots = getattr(root_env, "robots", None)
    if not robots:
        raise RuntimeError("Stack environment contains no robot.")
    gripper = robots[0].gripper
    if not isinstance(gripper, dict):
        return gripper
    if "right" in gripper:
        return gripper["right"]
    if len(gripper) == 1:
        return next(iter(gripper.values()))
    raise RuntimeError(f"Could not select Stack gripper from {list(gripper)}")


def stack_cube_is_grasped(
    env: Any,
    geometry: dict[str, Any],
) -> bool:
    root_env = unwrap_env(env)
    gripper = get_stack_gripper(root_env)
    object_model = geometry["object_model"]
    try:
        return bool(
            root_env._check_grasp(
                gripper=gripper,
                object_geoms=object_model,
            )
        )
    except TypeError:
        return bool(root_env._check_grasp(gripper, object_model))


def stack_cubes_touch(
    env: Any,
    first: dict[str, Any],
    second: dict[str, Any],
) -> bool:
    root_env = unwrap_env(env)
    try:
        return bool(
            root_env.check_contact(
                first["object_model"],
                second["object_model"],
            )
        )
    except Exception:
        sim = get_sim(env)
        geom_sets: list[set[int]] = []
        for geometry in (first, second):
            body_id = int(geometry["body_id"])
            first_geom = int(sim.model.body_geomadr[body_id])
            count = int(sim.model.body_geomnum[body_id])
            geom_sets.append(set(range(first_geom, first_geom + count)))

        for contact_index in range(int(sim.data.ncon)):
            contact = sim.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 in geom_sets[0]
                and geom2 in geom_sets[1]
                or geom2 in geom_sets[0]
                and geom1 in geom_sets[1]
            ):
                return True
        return False


def check_stack_success(
    env: Any,
    geometries: dict[str, dict[str, Any]],
    true_heavy_name: str,
    true_light_name: str,
    *,
    xy_tolerance_m: float,
    z_tolerance_m: float,
    bottom_table_tolerance_m: float,
) -> dict[str, Any]:
    """Apply the collector's exact heavy-bottom / light-top success rule."""
    root_env = unwrap_env(env)
    heavy = geometries[true_heavy_name]
    light = geometries[true_light_name]

    heavy_position = get_stack_cube_position(env, heavy)
    light_position = get_stack_cube_position(env, light)
    horizontal_error = float(
        np.linalg.norm(light_position[:2] - heavy_position[:2])
    )
    expected_separation = float(
        heavy["half_height"] + light["half_height"]
    )
    actual_separation = float(light_position[2] - heavy_position[2])
    vertical_error = abs(actual_separation - expected_separation)
    touching = stack_cubes_touch(env, heavy, light)
    light_released = not stack_cube_is_grasped(env, light)

    table_offset = np.asarray(
        getattr(root_env, "table_offset", (0.0, 0.0, 0.0)),
        dtype=np.float64,
    ).reshape(-1)
    if table_offset.size < 3:
        raise RuntimeError(
            f"Invalid Stack table_offset: {table_offset}"
        )
    heavy_on_table = bool(
        heavy_position[2]
        <= table_offset[2]
        + float(heavy["half_height"])
        + float(bottom_table_tolerance_m)
    )
    order_correct = bool(actual_separation > 0.0)

    correct_stack = bool(
        touching
        and light_released
        and heavy_on_table
        and order_correct
        and horizontal_error <= float(xy_tolerance_m)
        and vertical_error <= float(z_tolerance_m)
    )

    return {
        "correct_stack": correct_stack,
        "touching": bool(touching),
        "light_released": bool(light_released),
        "heavy_on_table": bool(heavy_on_table),
        "order_correct": bool(order_correct),
        "horizontal_error": horizontal_error,
        "vertical_error": float(vertical_error),
        "expected_vertical_separation": expected_separation,
        "actual_vertical_separation": actual_separation,
        "heavy_position": heavy_position,
        "light_position": light_position,
    }


def read_first_observation(
    demo: h5py.Group,
    key: str,
) -> np.ndarray:
    if "obs" not in demo or key not in demo["obs"]:
        raise KeyError(
            f"{demo.name} is missing obs/{key}"
        )

    values = np.asarray(
        demo["obs"][key],
        dtype=np.float64,
    )

    return values.reshape(values.shape[0], -1)[0]


def get_force(
    observation: dict,
) -> tuple[np.ndarray, float, float]:
    force = np.asarray(
        observation.get(
            "robot0_eef_force",
            np.zeros(3),
        ),
        dtype=np.float64,
    ).reshape(-1)

    if force.size < 3:
        force = np.pad(
            force,
            (0, 3 - force.size),
        )

    force = force[:3]

    return (
        force,
        float(np.linalg.norm(force)),
        float(abs(force[2])),
    )


def get_normalized_force(
    normalizer: StandardNormalizer,
    force: np.ndarray,
) -> np.ndarray | None:
    if "force" not in normalizer.stats:
        return None

    force_tensor = torch.as_tensor(
        force,
        dtype=torch.float32,
        device=normalizer.stats["force"].mean.device,
    ).reshape(1, 1, -1)
    result = normalizer.normalize(force_tensor, "force")
    return result.detach().cpu().numpy().reshape(-1)


def check_custom_success(
    env: Any,
    object_position: np.ndarray,
    target_position: np.ndarray,
    xy_fraction: float,
    *,
    max_z_above_target: float = 0.035,
    object_speed_m_s: float = 0.0,
    max_settle_speed_m_s: float = 0.05,
    gripper_action: float | None = None,
    require_release: bool = True,
    open_action_threshold: float = -0.5,
) -> tuple[bool, float, float, float, float]:
    """Strict PickPlace success: inside XY, lowered, released, and settled.

    The previous evaluator used ``bin2_pos[2] + 0.25`` as its only vertical
    condition. That volume extends far above the bin, so an object carried over
    the target could be counted as successful before placement.

    ``target_position`` comes from the demonstration's light/heavy target
    observation. The object must now be no more than ``max_z_above_target``
    above that target. By default, the latest policy action must also command
    the gripper open and the object must be moving slowly. Consecutive
    ``success_hold_steps`` in the rollout loop then confirm that it remains
    placed instead of merely passing through the valid region.
    """
    root_env = unwrap_env(env)
    bin_size = np.asarray(root_env.bin_size, dtype=np.float64)

    if max_z_above_target < 0:
        raise ValueError(
            "max_z_above_target must be non-negative, got "
            f"{max_z_above_target}"
        )
    if max_settle_speed_m_s < 0:
        raise ValueError(
            "max_settle_speed_m_s must be non-negative, got "
            f"{max_settle_speed_m_s}"
        )

    half_x = float(bin_size[0] * 0.5 * xy_fraction)
    half_y = float(bin_size[1] * 0.5 * xy_fraction)

    dx = float(abs(object_position[0] - target_position[0]))
    dy = float(abs(object_position[1] - target_position[1]))
    z_limit = float(target_position[2] + max_z_above_target)

    inside_xy = bool(dx <= half_x and dy <= half_y)
    lowered_into_target = bool(float(object_position[2]) <= z_limit)
    settled = bool(float(object_speed_m_s) <= max_settle_speed_m_s)

    if require_release:
        released = bool(
            gripper_action is not None
            and float(gripper_action) <= open_action_threshold
        )
    else:
        released = True

    success = bool(
        inside_xy
        and lowered_into_target
        and released
        and settled
    )
    return success, dx, dy, half_x, half_y



def read_demo_ep_meta(demo: h5py.Group) -> dict[str, Any]:
    """Read optional JSON episode metadata without failing on old datasets."""
    raw = demo.attrs.get("ep_meta")
    if raw is None:
        return {}

    try:
        decoded = decode_attr(raw)
        parsed = json.loads(decoded)
    except Exception:
        return {}

    return parsed if isinstance(parsed, dict) else {}


def demo_scalar(
    demo: h5py.Group,
    name: str,
    default: Any,
    *,
    ep_meta: dict[str, Any] | None = None,
) -> Any:
    """Read a scalar from direct attrs first, then ep_meta."""
    if name in demo.attrs:
        value = demo.attrs[name]
        if isinstance(value, np.generic):
            return value.item()
        return value

    if ep_meta is None:
        ep_meta = read_demo_ep_meta(demo)

    return ep_meta.get(name, default)


DOOR_FORCE_PROBE_STAGE = 8
DOOR_ADAPTIVE_OPEN_STAGE = 11


def first_demo_stage_index(
    demo: h5py.Group,
    stage_id: int,
) -> int | None:
    """Return the first action index assigned to a collector FSM stage."""
    arrays = []
    if "fsm_stage" in demo:
        arrays.append(np.asarray(demo["fsm_stage"], dtype=np.int64).reshape(-1))
    if "obs" in demo and "fsm_stage" in demo["obs"]:
        arrays.append(
            np.asarray(demo["obs"]["fsm_stage"], dtype=np.int64).reshape(-1)
        )

    for values in arrays:
        indices = np.flatnonzero(values == int(stage_id))
        if indices.size:
            return int(indices[0])

    return None


def detect_rollout_task(
    env: Any,
    env_metadata: dict[str, Any],
) -> str:
    """Return the supported rollout task family for the restored environment."""
    names = [
        str(env_metadata.get("env_name", "")),
        str(getattr(unwrap_env(env), "__class__", type(env)).__name__),
    ]
    joined = " ".join(names).lower()

    if "door" in joined:
        return "door"
    if "stack" in joined:
        return "stack"
    if (
        "nutassemblysquare" in joined
        or "nut_assembly_square" in joined
        or "nutassembly" in joined
        or "nut_assembly" in joined
    ):
        return "nut_assembly_square"
    if "pickplace" in joined or "pick_place" in joined:
        return "pick_place"

    raise RuntimeError(
        "Unsupported rollout environment. Expected Door, Stack, "
        "NutAssemblySquare, or PickPlace, "
        f"got names={names}."
    )


def available_joint_names(sim: Any) -> list[str]:
    names: list[str] = []
    for joint_id in range(int(sim.model.njnt)):
        try:
            name = sim.model.joint_id2name(joint_id)
        except Exception:
            name = None
        if name:
            names.append(str(name))
    return names


def find_door_hinge_joint(env: Any) -> tuple[str, int]:
    """Find the active Door hinge joint robustly across wrappers / versions."""
    sim = get_sim(env)
    candidates = (
        "Door_hinge",
        "door_hinge",
        "hinge",
    )

    for name in candidates:
        try:
            return name, int(sim.model.joint_name2id(name))
        except Exception:
            pass

    names = available_joint_names(sim)
    likely = [
        name
        for name in names
        if "door" in name.lower() and "hinge" in name.lower()
    ]
    if not likely:
        likely = [
            name
            for name in names
            if "hinge" in name.lower()
        ]

    if likely:
        name = likely[0]
        return name, int(sim.model.joint_name2id(name))

    raise RuntimeError(
        "Could not identify the Door hinge joint. "
        f"Available joints={names}"
    )


def get_joint_qpos(env: Any, joint_id: int) -> float:
    sim = get_sim(env)
    address = int(sim.model.jnt_qposadr[int(joint_id)])
    return float(sim.data.qpos[address])


def _unit_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm < 1e-8:
        raise ValueError(f"Invalid {name}: {vector}")
    return vector / norm


def get_hinge_axis_world(env: Any, hinge_id: int) -> np.ndarray:
    sim = get_sim(env)
    body_id = int(sim.model.jnt_bodyid[int(hinge_id)])
    body_rotation = np.asarray(
        sim.data.body_xmat[body_id],
        dtype=np.float64,
    ).reshape(3, 3)
    local_axis = np.asarray(
        sim.model.jnt_axis[int(hinge_id)],
        dtype=np.float64,
    )
    return _unit_vector(body_rotation @ local_axis, "door hinge axis")


def rotate_vector_about_axis(
    vector: Any,
    axis: Any,
    angle_rad: float,
) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    axis = _unit_vector(axis, "rotation axis")
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    rotated = (
        vector * cosine
        + np.cross(axis, vector) * sine
        + axis * float(np.dot(axis, vector)) * (1.0 - cosine)
    )
    return _unit_vector(rotated, "live door-opening tangent")


def resolve_initial_opening_tangent(
    demo: h5py.Group,
    ep_meta: dict[str, Any],
    env: Any,
    hinge_id: int,
    opening_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the same opening tangent used by the collector.

    New datasets store the initial tangent in ep_meta. The geometric fallback
    keeps older datasets usable without reverting to a full 3-D force norm.
    """
    hinge_axis = get_hinge_axis_world(env, hinge_id)
    stored = demo_scalar(
        demo,
        "opening_tangent",
        None,
        ep_meta=ep_meta,
    )
    if stored is not None:
        return _unit_vector(stored, "stored opening tangent"), hinge_axis

    sim = get_sim(env)
    body_id = int(sim.model.jnt_bodyid[int(hinge_id)])
    body_position = np.asarray(
        sim.data.body_xpos[body_id], dtype=np.float64
    )
    body_rotation = np.asarray(
        sim.data.body_xmat[body_id], dtype=np.float64
    ).reshape(3, 3)
    hinge_anchor = body_position + body_rotation @ np.asarray(
        sim.model.jnt_pos[int(hinge_id)], dtype=np.float64
    )

    handle_position = None
    candidates: list[tuple[int, np.ndarray]] = []
    for site_id in range(int(sim.model.nsite)):
        name = sim.model.site_id2name(site_id)
        if name and ("handle" in name.lower() or "latch" in name.lower()):
            candidates.append((len(str(name)), np.asarray(
                sim.data.site_xpos[site_id], dtype=np.float64
            )))
    if candidates:
        handle_position = min(candidates, key=lambda item: item[0])[1]
    if handle_position is None:
        raise RuntimeError(
            "Could not resolve Door handle position for tangential-force "
            "evaluation, and the demo has no opening_tangent metadata."
        )

    radius = handle_position - hinge_anchor
    radius -= hinge_axis * float(np.dot(radius, hinge_axis))
    radius = _unit_vector(radius, "hinge-to-handle radius")
    positive_tangent = _unit_vector(
        np.cross(hinge_axis, radius),
        "positive door tangent",
    )
    return positive_tangent * (1.0 if opening_sign >= 0 else -1.0), hinge_axis


def restore_door_episode_physics(
    env: Any,
    *,
    hinge_id: int,
    damping: float,
    friction: float,
    initial_angle: float,
    wrong_direction_slack: float,
) -> None:
    """Restore resistance and the collector's class-neutral pre-probe lock."""
    sim = get_sim(env)
    dof_id = int(sim.model.jnt_dofadr[int(hinge_id)])

    if np.isfinite(damping):
        sim.model.dof_damping[dof_id] = float(damping)
    if np.isfinite(friction):
        sim.model.dof_frictionloss[dof_id] = float(friction)

    slack = max(float(wrong_direction_slack), 0.0)
    sim.model.jnt_range[int(hinge_id), 0] = float(initial_angle - slack)
    sim.model.jnt_range[int(hinge_id), 1] = float(initial_angle + slack)
    sim.forward()


def activate_door_direction_lock(
    env: Any,
    *,
    hinge_id: int,
    initial_angle: float,
    opening_sign: float,
    wrong_direction_slack: float,
    opening_range: float,
) -> None:
    """Activate the hidden one-way range at the collector's FORCE_PROBE stage."""
    sim = get_sim(env)
    slack = max(float(wrong_direction_slack), 0.0)
    opening_range = max(float(opening_range), slack + 1e-6)

    if opening_sign >= 0:
        lower = initial_angle - slack
        upper = initial_angle + opening_range
    else:
        lower = initial_angle - opening_range
        upper = initial_angle + slack

    sim.model.jnt_range[int(hinge_id), 0] = float(lower)
    sim.model.jnt_range[int(hinge_id), 1] = float(upper)
    sim.forward()


def resistance_derived_safety_limits(
    damping: float,
    friction: float,
    *,
    friction_weight: float,
    damage_force_base_n: float,
    damage_force_per_resistance_n: float,
    damage_force_min_n: float,
    damage_force_max_n: float,
    hard_break_force_factor: float,
    damage_impulse_time_s: float,
    safe_speed_base_rad_s: float,
    safe_speed_per_resistance_rad_s: float,
    safe_speed_min_rad_s: float,
    safe_speed_max_rad_s: float,
) -> dict[str, float]:
    """Apply the same resistance-to-strength mapping to every door."""
    resistance_score = max(float(damping), 0.0) + (
        friction_weight * max(float(friction), 0.0)
    )
    damage_force_limit_n = float(
        np.clip(
            damage_force_base_n
            + damage_force_per_resistance_n * resistance_score,
            damage_force_min_n,
            damage_force_max_n,
        )
    )
    return {
        "resistance_score": float(resistance_score),
        "damage_force_limit_n": damage_force_limit_n,
        "hard_break_force_n": float(
            damage_force_limit_n * hard_break_force_factor
        ),
        "damage_impulse_limit_ns": float(
            damage_force_limit_n * damage_impulse_time_s
        ),
        "max_safe_open_speed_rad_s": float(
            np.clip(
                safe_speed_base_rad_s
                + safe_speed_per_resistance_rad_s * resistance_score,
                safe_speed_min_rad_s,
                safe_speed_max_rad_s,
            )
        ),
    }


def _first_numeric_id(value: Any) -> int | None:
    if isinstance(value, dict):
        for preferred in ("right", "robot0_right", "right_arm"):
            if preferred in value:
                return _first_numeric_id(value[preferred])
        for item in value.values():
            result = _first_numeric_id(item)
            if result is not None:
                return result
        return None

    if isinstance(value, (tuple, list, np.ndarray)):
        for item in value:
            result = _first_numeric_id(item)
            if result is not None:
                return result
        return None

    if isinstance(value, (int, np.integer)):
        return int(value)

    return None


def get_eef_rotation_world(env: Any) -> np.ndarray:
    """World rotation of the force-sensor / EEF frame."""
    root_env = unwrap_env(env)
    sim = get_sim(env)

    robots = getattr(root_env, "robots", None) or []
    if robots:
        robot = robots[0]
        for attr in ("eef_site_id", "eef_site_ids"):
            site_id = _first_numeric_id(getattr(robot, attr, None))
            if site_id is not None:
                return np.asarray(
                    sim.data.site_xmat[site_id],
                    dtype=np.float64,
                ).reshape(3, 3).copy()

    for site_name in (
        "gripper0_right_eef",
        "robot0_right_eef",
        "gripper0_eef",
    ):
        try:
            site_id = int(sim.model.site_name2id(site_name))
            return np.asarray(
                sim.data.site_xmat[site_id],
                dtype=np.float64,
            ).reshape(3, 3).copy()
        except Exception:
            pass

    # Some models expose the EEF as a body rather than a site.
    for body_name in (
        "gripper0_right_eef",
        "robot0_right_eef",
    ):
        try:
            body_id = int(sim.model.body_name2id(body_name))
            return np.asarray(
                sim.data.body_xmat[body_id],
                dtype=np.float64,
            ).reshape(3, 3).copy()
        except Exception:
            pass

    raise RuntimeError(
        "Could not resolve the EEF rotation needed to transform force "
        "measurements into the world frame."
    )


def get_force_world(
    env: Any,
    observation: dict,
    *,
    sensor_frame: str = "eef",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return local force, world force, and local norm."""
    local_force, force_norm, _ = get_force(observation)

    if sensor_frame == "world":
        world_force = local_force.copy()
    elif sensor_frame == "eef":
        world_force = get_eef_rotation_world(env) @ local_force
    else:
        raise ValueError(
            f"Unsupported force sensor frame: {sensor_frame!r}"
        )

    return local_force, world_force, force_norm



def render_observation(
    observation: dict,
    camera_keys: Sequence[str],
    lines: Sequence[str],
    scale: float,
) -> bool:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Rendering requires opencv-python."
        ) from exc

    frames = []

    for camera_key in camera_keys:
        if camera_key not in observation:
            continue

        frame = image_to_hwc_uint8(
            observation[camera_key],
            camera_key,
            flip_vertical=False,
        )

        frames.append(frame)

    if not frames:
        return False

    preview_rgb = np.concatenate(
        frames,
        axis=1,
    )

    preview_bgr = cv2.cvtColor(
        preview_rgb,
        cv2.COLOR_RGB2BGR,
    )

    y = 22

    for line in lines:
        cv2.putText(
            preview_bgr,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            preview_bgr,
            line,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        y += 20

    if scale != 1.0:
        preview_bgr = cv2.resize(
            preview_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )

    cv2.imshow(
        "Force2Act rollout debug",
        preview_bgr,
    )

    key = cv2.waitKey(1) & 0xFF

    return key in (ord("q"), 27)



def sanitize_video_component(value: Any) -> str:
    """Make demo / label text safe for use in an output filename."""
    text = str(value).strip()
    safe = []
    for char in text:
        if char.isalnum() or char in ("-", "_"):
            safe.append(char)
        else:
            safe.append("_")
    result = "".join(safe).strip("_")
    return result or "episode"


def save_video_frames(
    frames: Sequence[np.ndarray],
    output_path: str | Path,
    fps: int,
) -> Path:
    """Write one rollout video and return its resolved output path."""
    if not frames:
        raise ValueError("Cannot save a rollout video with zero frames.")

    import imageio.v2 as imageio

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), fps=int(fps))
    return path


def should_save_episode_video(
    mode: str,
    *,
    episode_index: int,
    success: bool,
) -> bool:
    """Return whether this episode should be written for the selected mode."""
    if mode == "none":
        return False
    if mode == "first":
        return episode_index == 0
    if mode == "all":
        return True
    if mode == "success":
        return bool(success)
    if mode == "failure":
        return not bool(success)
    raise ValueError(f"Unsupported video_save_mode: {mode!r}")


@torch.no_grad()
def evaluate_policy_rollouts(
    *,
    policy: EvaluationPolicy,
    config: dict,
    validation_demos: Sequence[str],
    device: torch.device,
    episodes: int | None = None,
    video_path: str | Path | None = None,
    render: bool = False,
    render_scale: float = 1.0,
    return_episode_details: bool = False,
) -> dict[str, Any]:
    """Run exact-scene rollouts for Door, Stack, or PickPlaceCan datasets.

    Stack evaluation restores both hidden cube masses and applies the collector's heavy-bottom / light-top contact, release, table, and geometry rules.
    Door evaluation restores each demo's hinge damping / friction, delays the
    hidden one-way lock until the recorded FORCE_PROBE stage, measures opening
    progress from Door_hinge, and uses the dataset's direction-only success rule.
    Force / speed limits are diagnostic unless the dataset explicitly says they
    were enforced during collection.
    """
    normalizer = policy.normalizer
    dataset_cfg = config["dataset"]
    evaluation_cfg = config.get("evaluation", {})

    selected_demos = list(validation_demos)
    if episodes is not None:
        selected_demos = selected_demos[:episodes]

    state_keys = tuple(dataset_cfg["state_keys"])
    camera_keys = tuple(dataset_cfg["camera_keys"])

    force_enabled = bool(dataset_cfg.get("use_force", False))
    force_keys = (
        tuple(dataset_cfg.get("force_keys", []))
        if force_enabled
        else ()
    )

    camera_names = tuple(
        key[:-len("_image")]
        if key.endswith("_image")
        else key
        for key in camera_keys
    )

    initialize_obs_utils(
        state_keys,
        camera_keys,
        force_keys,
    )

    env, env_metadata = create_data_processing_env(
        dataset_path=dataset_cfg["hdf5_path"],
        camera_names=camera_names,
        image_size=int(dataset_cfg["image_size"]),
        expected_env_name=dataset_cfg.get(
            "robosuite_env_name"
        ),
    )

    task_type = detect_rollout_task(env, env_metadata)
    print(
        f"[ROLLOUT TASK] type={task_type} "
        f"env={env_metadata.get('env_name', 'unknown')}"
    )

    object_body_name: str | None = None
    if task_type == "pick_place":
        object_body_name = find_object_body_name(env)
        print(f"[ACTIVE OBJECT] body={object_body_name}")

    max_steps = int(
        evaluation_cfg.get(
            "max_episode_steps",
            500,
        )
    )
    door_max_steps = int(
        evaluation_cfg.get("door_max_episode_steps", 1200)
    )
    sampling_steps = int(
        evaluation_cfg.get(
            "sampling_steps",
            20,
        )
    )
    exec_horizon = int(
        evaluation_cfg.get(
            "exec_horizon",
            4,
        )
    )
    success_hold_steps = int(
        evaluation_cfg.get(
            "success_hold_steps",
            5,
        )
    )
    force_print_interval = int(
        evaluation_cfg.get(
            "force_debug_interval",
            10,
        )
    )
    require_model_file = bool(
        evaluation_cfg.get(
            "require_model_file",
            True,
        )
    )

    # PickPlace-only success settings.
    #
    # The old vertical condition was bin2_pos[2] + 0.25, which could count a
    # heavy object as successful while it was still being carried above the
    # target. Success now requires a tighter XY region, a low object height,
    # an open-gripper command, and low object speed for success_hold_steps.
    target_xy_fraction = float(
        evaluation_cfg.get(
            "target_xy_fraction",
            0.35,
        )
    )
    pick_place_max_z_above_target = float(
        evaluation_cfg.get(
            "pick_place_max_z_above_target",
            0.035,
        )
    )
    pick_place_require_release = bool(
        evaluation_cfg.get(
            "pick_place_require_release",
            True,
        )
    )
    pick_place_open_action_threshold = float(
        evaluation_cfg.get(
            "pick_place_open_action_threshold",
            -0.5,
        )
    )
    pick_place_max_settle_speed = float(
        evaluation_cfg.get(
            "pick_place_max_settle_speed_m_s",
            0.05,
        )
    )
    root_control_freq = float(
        getattr(unwrap_env(env), "control_freq", 20.0)
    )
    pick_place_control_freq = float(
        evaluation_cfg.get(
            "pick_place_control_freq",
            root_control_freq,
        )
    )

    # Stack evaluator settings. Defaults match the collector. The generic
    # success_hold_steps remains a supported fallback for old configs.
    stack_xy_tolerance_m = float(
        evaluation_cfg.get("stack_xy_tolerance_m", 0.030)
    )
    stack_z_tolerance_m = float(
        evaluation_cfg.get("stack_z_tolerance_m", 0.018)
    )
    stack_bottom_table_tolerance_m = float(
        evaluation_cfg.get("stack_bottom_table_tolerance_m", 0.015)
    )
    stack_success_hold_steps = int(
        evaluation_cfg.get(
            "stack_success_hold_steps",
            evaluation_cfg.get("success_hold_steps", 8),
        )
    )
    if stack_xy_tolerance_m <= 0.0:
        raise ValueError("evaluation.stack_xy_tolerance_m must be positive.")
    if stack_z_tolerance_m <= 0.0:
        raise ValueError("evaluation.stack_z_tolerance_m must be positive.")
    if stack_bottom_table_tolerance_m <= 0.0:
        raise ValueError(
            "evaluation.stack_bottom_table_tolerance_m must be positive."
        )
    if stack_success_hold_steps <= 0:
        raise ValueError(
            "evaluation.stack_success_hold_steps must be positive."
        )

    # NutAssemblySquare evaluator settings.
    #
    # The replay collector keeps only successful square-nut demonstrations.
    # Online evaluation therefore uses the restored Robosuite task's own
    # success signal. A configurable consecutive-success hold prevents a
    # one-frame contact from being counted accidentally.
    nut_success_hold_steps = int(
        evaluation_cfg.get(
            "nut_success_hold_steps",
            evaluation_cfg.get("success_hold_steps", 1),
        )
    )
    nut_success_reward_threshold = float(
        evaluation_cfg.get(
            "nut_success_reward_threshold",
            evaluation_cfg.get("success_reward_threshold", 0.999),
        )
    )
    if nut_success_hold_steps <= 0:
        raise ValueError(
            "evaluation.nut_success_hold_steps must be positive."
        )

    # Door evaluator settings. By default, the validated physical task rules
    # stored by the collector in each demo take priority over stale checkpoint
    # config values. Set door_use_dataset_task_metadata=False only for an
    # intentional counterfactual evaluation.
    door_use_dataset_task_metadata = bool(
        evaluation_cfg.get("door_use_dataset_task_metadata", True)
    )
    door_sensor_frame = str(
        evaluation_cfg.get("door_force_sensor_frame", "eef")
    )
    door_control_freq = float(
        evaluation_cfg.get(
            "door_control_freq",
            getattr(unwrap_env(env), "control_freq", 20.0),
        )
    )
    configured_door_pull_start_progress = (
        float(evaluation_cfg["door_pull_start_progress"])
        if "door_pull_start_progress" in evaluation_cfg
        else None
    )
    door_baseline_freeze_progress = float(
        evaluation_cfg.get("door_baseline_freeze_progress", 0.001)
    )
    door_baseline_window = int(
        evaluation_cfg.get("door_baseline_window", 3)
    )
    door_force_ema_alpha = float(
        evaluation_cfg.get("door_damage_force_ema_alpha", 0.35)
    )
    door_speed_ema_alpha = float(
        evaluation_cfg.get("door_speed_ema_alpha", 0.35)
    )
    door_damage_grace_steps = int(
        evaluation_cfg.get("door_damage_grace_steps", 5)
    )
    door_breaking_force_hold_steps = int(
        evaluation_cfg.get("door_breaking_force_hold_steps", 4)
    )
    door_overspeed_hold_steps = int(
        evaluation_cfg.get("door_overspeed_hold_steps", 4)
    )

    # Retained only for old config compatibility. The collector does not use
    # this intermediate checkpoint as a success / failure condition.
    global_progress_check_steps = int(
        evaluation_cfg.get(
            "door_progress_check_steps",
            evaluation_cfg.get("door_global_progress_check_steps", 80),
        )
    )
    global_minimum_progress = float(
        evaluation_cfg.get(
            "door_minimum_progress_at_check",
            evaluation_cfg.get(
                "door_global_minimum_progress_at_check", 0.45
            ),
        )
    )
    global_pull_timeout_steps = int(
        evaluation_cfg.get(
            "door_pull_timeout_steps",
            evaluation_cfg.get("door_global_pull_timeout_steps", 120),
        )
    )

    # Exact fallback values from the validated collector. New datasets also
    # store the derived per-door limits, which take priority below.
    resistance_friction_weight = float(
        evaluation_cfg.get("door_resistance_friction_weight", 4.0)
    )
    damage_force_base_n = float(
        evaluation_cfg.get("door_damage_force_base_n", 35.0)
    )
    damage_force_per_resistance_n = float(
        evaluation_cfg.get(
            "door_damage_force_per_resistance_n", 1.5
        )
    )
    damage_force_min_n = float(
        evaluation_cfg.get("door_damage_force_min_n", 35.0)
    )
    damage_force_max_n = float(
        evaluation_cfg.get("door_damage_force_max_n", 100.0)
    )
    hard_break_force_factor = float(
        evaluation_cfg.get("door_hard_break_force_factor", 1.75)
    )
    damage_impulse_time_s = float(
        evaluation_cfg.get("door_damage_impulse_time_s", 0.15)
    )
    safe_speed_base_rad_s = float(
        evaluation_cfg.get("door_safe_speed_base_rad_s", 0.18)
    )
    safe_speed_per_resistance_rad_s = float(
        evaluation_cfg.get(
            "door_safe_speed_per_resistance_rad_s", 0.028
        )
    )
    safe_speed_min_rad_s = float(
        evaluation_cfg.get("door_safe_speed_min_rad_s", 0.18)
    )
    safe_speed_max_rad_s = float(
        evaluation_cfg.get("door_safe_speed_max_rad_s", 0.80)
    )

    if door_baseline_freeze_progress < 0.0:
        raise ValueError(
            "evaluation.door_baseline_freeze_progress must be non-negative."
        )
    if door_baseline_window <= 0:
        raise ValueError("evaluation.door_baseline_window must be positive.")
    if global_progress_check_steps <= 0:
        raise ValueError(
            "evaluation.door_progress_check_steps must be positive."
        )
    if global_minimum_progress <= 0.0:
        raise ValueError(
            "evaluation.door_minimum_progress_at_check must be positive."
        )
    if global_pull_timeout_steps <= 0:
        raise ValueError(
            "evaluation.door_pull_timeout_steps must be positive."
        )
    if door_damage_grace_steps < 0:
        raise ValueError(
            "evaluation.door_damage_grace_steps must be non-negative."
        )
    if door_breaking_force_hold_steps <= 0:
        raise ValueError(
            "evaluation.door_breaking_force_hold_steps must be positive."
        )
    if door_overspeed_hold_steps <= 0:
        raise ValueError(
            "evaluation.door_overspeed_hold_steps must be positive."
        )
    if resistance_friction_weight < 0.0:
        raise ValueError(
            "evaluation.door_resistance_friction_weight must be non-negative."
        )
    if damage_force_base_n <= 0.0:
        raise ValueError(
            "evaluation.door_damage_force_base_n must be positive."
        )
    if damage_force_per_resistance_n < 0.0:
        raise ValueError(
            "evaluation.door_damage_force_per_resistance_n must be non-negative."
        )
    if not 0.0 < damage_force_min_n < damage_force_max_n:
        raise ValueError("Invalid Door damage-force clipping range.")
    if hard_break_force_factor <= 1.0:
        raise ValueError(
            "evaluation.door_hard_break_force_factor must be greater than 1."
        )
    if damage_impulse_time_s <= 0.0:
        raise ValueError(
            "evaluation.door_damage_impulse_time_s must be positive."
        )
    if not 0.0 < safe_speed_min_rad_s <= safe_speed_max_rad_s:
        raise ValueError("Invalid Door safe-speed clipping range.")

    configured_door_target = float(
        evaluation_cfg.get("door_success_progress", 0.81)
    )
    configured_door_tolerance = float(
        evaluation_cfg.get("door_success_tolerance", 0.10)
    )
    if configured_door_tolerance < 0.0:
        raise ValueError(
            "evaluation.door_success_tolerance must be non-negative."
        )
    respect_env_done_for_door = bool(
        evaluation_cfg.get("door_respect_env_done", False)
    )
    door_direction_lock_tolerance = float(
        evaluation_cfg.get("door_direction_lock_tolerance_rad", 0.0025)
    )
    door_opening_range = float(
        evaluation_cfg.get("door_opening_range_rad", 0.90)
    )
    door_use_demo_stage_schedule = bool(
        evaluation_cfg.get("door_use_demo_stage_schedule", True)
    )
    if door_direction_lock_tolerance < 0.0:
        raise ValueError(
            "evaluation.door_direction_lock_tolerance_rad must be non-negative."
        )
    if door_opening_range <= 0.0:
        raise ValueError("evaluation.door_opening_range_rad must be positive.")

    video_save_mode = str(
        evaluation_cfg.get("video_save_mode", "first")
    ).strip().lower()
    valid_video_modes = {
        "none",
        "first",
        "all",
        "success",
        "failure",
    }
    if video_save_mode not in valid_video_modes:
        raise ValueError(
            "evaluation.video_save_mode must be one of "
            f"{sorted(valid_video_modes)}, got {video_save_mode!r}."
        )

    video_fps = int(evaluation_cfg.get("video_fps", 20))
    video_camera_key = str(
        evaluation_cfg.get("video_camera_key", camera_keys[0])
    )
    if video_camera_key not in camera_keys:
        raise ValueError(
            f"video_camera_key={video_camera_key!r} is not in "
            f"dataset.camera_keys={list(camera_keys)}."
        )

    saved_video_count = 0
    primary_video_written = False
    episode_results: list[dict[str, Any]] = []

    previous_training_state = policy.training
    policy.eval()
    stop_requested = False

    try:
        with h5py.File(
            dataset_cfg["hdf5_path"],
            "r",
        ) as hdf5_file:
            data_group = hdf5_file["data"]

            for episode_index, demo_name in enumerate(
                selected_demos
            ):
                if stop_requested:
                    break

                demo = data_group[demo_name]
                ep_meta = read_demo_ep_meta(demo)
                episode_video_frames: list[np.ndarray] = []

                observation = reset_to(
                    env,
                    make_initial_state(
                        demo,
                        require_model_file=require_model_file,
                    ),
                )

                # reset_to may replace the underlying MuJoCo model, so all IDs
                # must be resolved after each reset.
                if task_type == "door":
                    truth_label = decode_attr(
                        demo.attrs[
                            "true_resistance_class"
                        ]
                    ).lower()

                    hinge_name, hinge_id = (
                        find_door_hinge_joint(env)
                    )
                    initial_door_angle = float(
                        demo_scalar(
                            demo,
                            "initial_door_angle",
                            get_joint_qpos(env, hinge_id),
                            ep_meta=ep_meta,
                        )
                    )
                    opening_sign = float(
                        demo_scalar(
                            demo,
                            "opening_sign",
                            1.0,
                            ep_meta=ep_meta,
                        )
                    )
                    if opening_sign == 0:
                        opening_sign = 1.0

                    episode_wrong_direction_slack = float(
                        demo_scalar(
                            demo,
                            "wrong_direction_slack_rad",
                            data_group.attrs.get(
                                "wrong_direction_slack_rad", 0.0005
                            ),
                            ep_meta=ep_meta,
                        )
                    )
                    episode_enforce_safety_limits = bool(
                        demo_scalar(
                            demo,
                            "enforce_safety_limits",
                            data_group.attrs.get(
                                "enforce_safety_limits", False
                            ),
                            ep_meta=ep_meta,
                        )
                    )
                    force_probe_start_step = (
                        first_demo_stage_index(
                            demo, DOOR_FORCE_PROBE_STAGE
                        )
                        if door_use_demo_stage_schedule
                        else None
                    )
                    adaptive_open_start_step = (
                        first_demo_stage_index(
                            demo, DOOR_ADAPTIVE_OPEN_STAGE
                        )
                        if door_use_demo_stage_schedule
                        else None
                    )

                    # New paired datasets always contain the stage schedule.
                    # For old files, activate immediately to preserve legacy
                    # behavior rather than leaving the door locked both ways.
                    if force_probe_start_step is None:
                        force_probe_start_step = 0

                    # Use the exact global task stored by the validated
                    # collector. All demos should contain the same target and
                    # tolerance; reading metadata prevents stale checkpoint
                    # config from silently changing the rollout task.
                    metadata_tolerance = float(
                        demo_scalar(
                            demo,
                            "open_success_tolerance",
                            configured_door_tolerance,
                            ep_meta=ep_meta,
                        )
                    )
                    metadata_target = float(
                        demo_scalar(
                            demo,
                            "target_open_angle",
                            float("nan"),
                            ep_meta=ep_meta,
                        )
                    )
                    if not np.isfinite(metadata_target):
                        required_progress_meta = float(
                            demo_scalar(
                                demo,
                                "required_open_progress",
                                float("nan"),
                                ep_meta=ep_meta,
                            )
                        )
                        if np.isfinite(required_progress_meta):
                            metadata_target = (
                                required_progress_meta + metadata_tolerance
                            )
                    target_progress = (
                        metadata_target
                        if (
                            door_use_dataset_task_metadata
                            and np.isfinite(metadata_target)
                        )
                        else configured_door_target
                    )
                    success_tolerance = (
                        metadata_tolerance
                        if door_use_dataset_task_metadata
                        else configured_door_tolerance
                    )
                    episode_pull_timeout_steps = int(
                        demo_scalar(
                            demo,
                            "max_opening_steps",
                            global_pull_timeout_steps,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else global_pull_timeout_steps
                    )
                    episode_max_steps = int(
                        demo_scalar(
                            demo,
                            "max_episode_steps",
                            door_max_steps,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_max_steps
                    )
                    episode_door_control_freq = float(
                        demo_scalar(
                            demo,
                            "control_freq",
                            door_control_freq,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_control_freq
                    )
                    episode_force_sensor_frame = str(
                        demo_scalar(
                            demo,
                            "force_sensor_frame",
                            door_sensor_frame,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_sensor_frame
                    )
                    episode_damage_grace_steps = int(
                        demo_scalar(
                            demo,
                            "damage_grace_steps",
                            door_damage_grace_steps,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_damage_grace_steps
                    )
                    episode_breaking_force_hold_steps = int(
                        demo_scalar(
                            demo,
                            "breaking_force_hold_steps",
                            door_breaking_force_hold_steps,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_breaking_force_hold_steps
                    )
                    episode_overspeed_hold_steps = int(
                        demo_scalar(
                            demo,
                            "overspeed_hold_steps",
                            door_overspeed_hold_steps,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_overspeed_hold_steps
                    )
                    episode_baseline_window = int(
                        demo_scalar(
                            demo,
                            "baseline_stat_window",
                            door_baseline_window,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else door_baseline_window
                    )
                    episode_success_hold_steps = int(
                        demo_scalar(
                            demo,
                            "open_hold_detection_steps",
                            3,
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else int(
                            evaluation_cfg.get(
                                "door_success_hold_steps", 3
                            )
                        )
                    )
                    door_pull_start_progress = (
                        configured_door_pull_start_progress
                        if configured_door_pull_start_progress is not None
                        else door_baseline_freeze_progress
                    )
                    damping = float(
                        demo_scalar(
                            demo,
                            "hinge_damping",
                            np.nan,
                            ep_meta=ep_meta,
                        )
                    )
                    friction = float(
                        demo_scalar(
                            demo,
                            "hinge_friction",
                            np.nan,
                            ep_meta=ep_meta,
                        )
                    )
                    # Apply one shared formula to every door. The light/heavy
                    # label is not used; only the restored hinge resistance is.
                    safety_limits = resistance_derived_safety_limits(
                        damping,
                        friction,
                        friction_weight=resistance_friction_weight,
                        damage_force_base_n=damage_force_base_n,
                        damage_force_per_resistance_n=(
                            damage_force_per_resistance_n
                        ),
                        damage_force_min_n=damage_force_min_n,
                        damage_force_max_n=damage_force_max_n,
                        hard_break_force_factor=(
                            hard_break_force_factor
                        ),
                        damage_impulse_time_s=damage_impulse_time_s,
                        safe_speed_base_rad_s=safe_speed_base_rad_s,
                        safe_speed_per_resistance_rad_s=(
                            safe_speed_per_resistance_rad_s
                        ),
                        safe_speed_min_rad_s=safe_speed_min_rad_s,
                        safe_speed_max_rad_s=safe_speed_max_rad_s,
                    )
                    resistance_score = float(
                        demo_scalar(
                            demo,
                            "resistance_score",
                            safety_limits["resistance_score"],
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else safety_limits["resistance_score"]
                    )
                    damage_force_limit = float(
                        demo_scalar(
                            demo,
                            "damage_force_limit_n",
                            safety_limits["damage_force_limit_n"],
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else safety_limits["damage_force_limit_n"]
                    )
                    hard_break_force = float(
                        demo_scalar(
                            demo,
                            "hard_break_force_n",
                            safety_limits["hard_break_force_n"],
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else safety_limits["hard_break_force_n"]
                    )
                    damage_impulse_limit = float(
                        demo_scalar(
                            demo,
                            "damage_impulse_limit_ns",
                            safety_limits["damage_impulse_limit_ns"],
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else safety_limits["damage_impulse_limit_ns"]
                    )
                    max_safe_speed = float(
                        demo_scalar(
                            demo,
                            "max_safe_open_speed_rad_s",
                            safety_limits["max_safe_open_speed_rad_s"],
                            ep_meta=ep_meta,
                        )
                        if door_use_dataset_task_metadata
                        else safety_limits["max_safe_open_speed_rad_s"]
                    )
                    (
                        initial_opening_tangent,
                        hinge_axis_world,
                    ) = resolve_initial_opening_tangent(
                        demo,
                        ep_meta,
                        env,
                        hinge_id,
                        opening_sign,
                    )

                    episode_opening_range = max(
                        door_opening_range,
                        target_progress + success_tolerance,
                    )
                    restore_door_episode_physics(
                        env,
                        hinge_id=hinge_id,
                        damping=damping,
                        friction=friction,
                        initial_angle=initial_door_angle,
                        wrong_direction_slack=(
                            episode_wrong_direction_slack
                        ),
                    )

                    target_position = None
                    mass_kg = None
                    mass_information = None
                    stack_geometries = None
                    true_heavy_name = None
                    true_light_name = None
                    cube_a_mass_kg = None
                    cube_b_mass_kg = None
                    stack_mass_information = None
                elif task_type == "stack":
                    stack_geometries = resolve_stack_geometries(env)
                    true_heavy_name = normalize_stack_cube_name(
                        demo_scalar(
                            demo,
                            "true_heavy_name",
                            None,
                            ep_meta=ep_meta,
                        )
                    )
                    true_light_name = normalize_stack_cube_name(
                        demo_scalar(
                            demo,
                            "true_light_name",
                            None,
                            ep_meta=ep_meta,
                        )
                    )
                    if true_heavy_name == true_light_name:
                        raise ValueError(
                            f"Invalid Stack labels in {demo_name}: "
                            f"heavy={true_heavy_name}, light={true_light_name}"
                        )
                    truth_label = f"heavy_{true_heavy_name}"
                    cube_a_mass_kg = float(
                        demo_scalar(
                            demo,
                            "cubeA_mass_kg",
                            float("nan"),
                            ep_meta=ep_meta,
                        )
                    )
                    cube_b_mass_kg = float(
                        demo_scalar(
                            demo,
                            "cubeB_mass_kg",
                            float("nan"),
                            ep_meta=ep_meta,
                        )
                    )
                    stack_mass_information = restore_stack_masses(
                        env,
                        stack_geometries,
                        cube_a_mass_kg,
                        cube_b_mass_kg,
                    )
                    episode_stack_xy_tolerance = float(
                        demo_scalar(
                            demo,
                            "stack_xy_tolerance_m",
                            stack_xy_tolerance_m,
                            ep_meta=ep_meta,
                        )
                    )
                    episode_stack_z_tolerance = float(
                        demo_scalar(
                            demo,
                            "stack_z_tolerance_m",
                            stack_z_tolerance_m,
                            ep_meta=ep_meta,
                        )
                    )
                    episode_stack_bottom_tolerance = float(
                        demo_scalar(
                            demo,
                            "bottom_table_tolerance_m",
                            stack_bottom_table_tolerance_m,
                            ep_meta=ep_meta,
                        )
                    )
                    episode_stack_hold_steps = int(
                        demo_scalar(
                            demo,
                            "stack_success_hold_steps",
                            stack_success_hold_steps,
                            ep_meta=ep_meta,
                        )
                    )
                    episode_max_steps = int(
                        demo_scalar(
                            demo,
                            "max_episode_steps",
                            max_steps,
                            ep_meta=ep_meta,
                        )
                    )
                    target_position = None
                    mass_kg = None
                    mass_information = None
                    episode_baseline_window = door_baseline_window
                elif task_type == "nut_assembly_square":
                    # The source HDF5 model_file / simulator state already
                    # restores the exact NutAssemblySquare scene. There is no
                    # hidden mass or target-class metadata to restore.
                    truth_label = "square_nut"
                    episode_max_steps = int(
                        demo_scalar(
                            demo,
                            "max_episode_steps",
                            max_steps,
                            ep_meta=ep_meta,
                        )
                    )
                    episode_baseline_window = door_baseline_window
                    target_position = None
                    mass_kg = None
                    mass_information = None
                    stack_geometries = None
                    true_heavy_name = None
                    true_light_name = None
                    cube_a_mass_kg = None
                    cube_b_mass_kg = None
                    stack_mass_information = None
                else:
                    assert object_body_name is not None
                    truth_label = decode_attr(
                        demo.attrs["true_weight_class"]
                    ).lower()
                    mass_kg = float(
                        demo.attrs["object_mass_kg"]
                    )
                    target_key = (
                        "heavy_target_pos"
                        if truth_label == "heavy"
                        else "light_target_pos"
                    )
                    target_position = read_first_observation(
                        demo,
                        target_key,
                    )
                    mass_information = restore_object_mass(
                        env,
                        object_body_name,
                        mass_kg,
                    )
                    episode_max_steps = max_steps
                    episode_baseline_window = door_baseline_window
                    stack_geometries = None
                    true_heavy_name = None
                    true_light_name = None
                    cube_a_mass_kg = None
                    cube_b_mass_kg = None
                    stack_mass_information = None

                observation = add_force_to_observation(
                    env,
                    dict(observation),
                )

                history = deque(
                    maxlen=int(dataset_cfg["history"])
                )
                for _ in range(history.maxlen):
                    history.append(deepcopy(observation))

                print("\n" + "=" * 90)

                if task_type == "door":
                    print(
                        f"[START] demo={demo_name} "
                        f"truth={truth_label} "
                        f"hinge={hinge_name} "
                        f"damping={damping:.4f} "
                        f"friction={friction:.4f} "
                        f"target={target_progress:.4f}rad "
                        f"tolerance={success_tolerance:.4f}rad "
                        f"threshold="
                        f"{target_progress - success_tolerance:.4f}rad "
                        f"model_uses_force={force_enabled} "
                        f"lock_step={force_probe_start_step} "
                        f"open_budget_step={adaptive_open_start_step}"
                    )
                    print(
                        "[SAFETY DIAGNOSTICS] "
                        f"score={resistance_score:.3f} "
                        f"damage_force={damage_force_limit:.2f}N "
                        f"hard_break={hard_break_force:.2f}N "
                        f"damage_impulse={damage_impulse_limit:.2f}Ns "
                        f"max_speed={max_safe_speed:.3f}rad/s "
                        f"deprecated_progress_check={global_minimum_progress:.3f}rad/"
                        f"{global_progress_check_steps}steps "
                        f"pull_timeout={episode_pull_timeout_steps} "
                        f"enforced={episode_enforce_safety_limits}"
                    )
                elif task_type == "stack":
                    assert stack_mass_information is not None
                    assert true_heavy_name is not None
                    assert true_light_name is not None
                    print(
                        f"[START] demo={demo_name} "
                        f"truth_heavy={true_heavy_name} "
                        f"truth_light={true_light_name} "
                        f"cubeA_mass={cube_a_mass_kg:.4f}kg "
                        f"cubeB_mass={cube_b_mass_kg:.4f}kg "
                        f"model_uses_force={force_enabled}"
                    )
                    print(
                        "[STACK SUCCESS] "
                        f"xy_tol={episode_stack_xy_tolerance:.4f}m "
                        f"z_tol={episode_stack_z_tolerance:.4f}m "
                        f"bottom_tol={episode_stack_bottom_tolerance:.4f}m "
                        f"hold_steps={episode_stack_hold_steps}"
                    )
                    print(
                        "[MASS RESTORE] "
                        f"A={stack_mass_information['cubeA']['previous_mass']:.4f}"
                        f"->{stack_mass_information['cubeA']['restored_mass']:.4f}kg "
                        f"B={stack_mass_information['cubeB']['previous_mass']:.4f}"
                        f"->{stack_mass_information['cubeB']['restored_mass']:.4f}kg"
                    )
                elif task_type == "nut_assembly_square":
                    print(
                        f"[START] demo={demo_name} "
                        f"task=NutAssemblySquare "
                        f"model_uses_force={force_enabled} "
                        f"success_hold={nut_success_hold_steps} "
                        f"reward_threshold={nut_success_reward_threshold:.3f}"
                    )
                else:
                    print(
                        f"[START] demo={demo_name} "
                        f"truth={truth_label} "
                        f"mass={mass_kg:.4f} kg "
                        f"model_uses_force={force_enabled}"
                    )
                    print(
                        f"[MASS] body={object_body_name} "
                        f"previous="
                        f"{mass_information['previous_mass']:.4f} "
                        f"restored="
                        f"{mass_information['restored_mass']:.4f}"
                    )
                    print(
                        f"[TARGET] "
                        f"{np.array2string(target_position, precision=4)} "
                        f"xy_fraction={target_xy_fraction:.3f} "
                        f"max_z_above="
                        f"{pick_place_max_z_above_target:.3f}m "
                        f"require_release={pick_place_require_release} "
                        f"max_settle_speed="
                        f"{pick_place_max_settle_speed:.3f}m/s"
                    )

                steps = 0
                episode_return = 0.0
                success_streak = 0
                episode_success = False
                failure_reason = ""

                # Stack metrics.
                last_stack_metrics: dict[str, Any] = {
                    "correct_stack": False,
                    "touching": False,
                    "light_released": False,
                    "heavy_on_table": False,
                    "order_correct": False,
                    "horizontal_error": float("inf"),
                    "vertical_error": float("inf"),
                    "expected_vertical_separation": float("nan"),
                    "actual_vertical_separation": float("nan"),
                    "heavy_position": np.full(3, np.nan),
                    "light_position": np.full(3, np.nan),
                }

                # PickPlace metrics.
                last_custom_success = False
                last_standard_success = False
                last_xy_distance = float("inf")
                last_z_error = float("inf")
                last_dx = float("inf")
                last_dy = float("inf")
                last_half_x = 0.0
                last_half_y = 0.0
                last_object_speed_m_s = float("inf")
                last_gripper_action = float("nan")
                last_pick_place_released = False
                last_pick_place_settled = False
                previous_object_position: np.ndarray | None = None
                if task_type == "pick_place":
                    assert object_body_name is not None
                    previous_object_position = get_object_position(
                        env,
                        object_body_name,
                    )

                # NutAssemblySquare metrics.
                last_nut_standard_success = False
                last_nut_reward_success = False
                last_nut_reward = 0.0

                # Door metrics.
                last_door_progress = 0.0
                door_broken = False
                insufficient_progress = False
                direction_lock_violation = False
                wrong_direction_peak_progress_rad = 0.0
                one_way_lock_activated = False
                hard_force_limit_exceeded = False
                sustained_force_limit_exceeded = False
                damage_impulse_limit_exceeded = False
                speed_limit_exceeded = False
                damage_impulse_ns = 0.0
                interaction_force_ema_n = 0.0
                door_speed_ema_rad_s = 0.0
                peak_interaction_force_n = 0.0
                peak_delta_force_norm_n = 0.0
                peak_door_speed_rad_s = 0.0
                breaking_force_streak = 0
                overspeed_streak = 0
                pull_started = False
                baseline_frozen = False
                pull_steps = 0
                previous_progress = 0.0
                baseline_force_world_samples: deque[np.ndarray] = deque(
                    maxlen=episode_baseline_window
                )
                baseline_force_world: np.ndarray | None = None

                inference_times_ms: list[float] = []
                num_replans = 0

                if task_type == "door":
                    initial_local_force, initial_world_force, _ = (
                        get_force_world(
                            env,
                            observation,
                            sensor_frame=episode_force_sensor_frame,
                        )
                    )
                    del initial_local_force
                    baseline_force_world_samples.append(
                        initial_world_force
                    )

                while (
                    steps < episode_max_steps
                    and not episode_success
                    and not door_broken
                    and not insufficient_progress
                    and not direction_lock_violation
                    and not stop_requested
                ):
                    inputs = build_history_tensors(
                        history,
                        state_keys=state_keys,
                        camera_keys=camera_keys,
                        force_keys=force_keys,
                        flip_images=False,
                    )
                    inputs = {
                        key: value.to(device)
                        for key, value in inputs.items()
                    }

                    inference_start = perf_counter()
                    action_chunk = policy.sample(
                        state=inputs["state"],
                        images=inputs["images"],
                        force=inputs.get("force"),
                        action_horizon=int(
                            dataset_cfg["action_horizon"]
                        ),
                        sampling_steps=sampling_steps,
                        clip_actions=True,
                        binarize_gripper=True,
                    )[0].detach().cpu().numpy()
                    inference_times_ms.append(
                        (perf_counter() - inference_start) * 1000.0
                    )
                    num_replans += 1

                    for action in action_chunk[:exec_horizon]:
                        if task_type == "door":
                            action_index = int(steps)

                            if (
                                not one_way_lock_activated
                                and action_index >= force_probe_start_step
                            ):
                                if baseline_force_world is None:
                                    baseline_force_world = np.median(
                                        np.stack(
                                            baseline_force_world_samples,
                                            axis=0,
                                        ),
                                        axis=0,
                                    )
                                baseline_frozen = True
                                activate_door_direction_lock(
                                    env,
                                    hinge_id=hinge_id,
                                    initial_angle=initial_door_angle,
                                    opening_sign=opening_sign,
                                    wrong_direction_slack=(
                                        episode_wrong_direction_slack
                                    ),
                                    opening_range=episode_opening_range,
                                )
                                one_way_lock_activated = True
                                print(
                                    "[DOOR ONE-WAY LOCK ACTIVATED] "
                                    f"step={action_index} "
                                    f"allowed_sign={opening_sign:+.0f}"
                                )

                            if (
                                adaptive_open_start_step is not None
                                and not pull_started
                                and action_index
                                >= adaptive_open_start_step
                            ):
                                pull_started = True
                                previous_progress = last_door_progress
                                pull_steps = 0
                                print(
                                    "[DOOR OPENING BUDGET START] "
                                    f"step={action_index} "
                                    f"source=demo_stage_schedule"
                                )

                        observation, reward, done, _ = (
                            step_env(env, action)
                        )
                        observation = add_force_to_observation(
                            env,
                            dict(observation),
                        )
                        history.append(deepcopy(observation))

                        steps += 1
                        episode_return += float(reward)

                        force, force_norm, force_z = (
                            get_force(observation)
                        )
                        normalized_force = (
                            get_normalized_force(
                                normalizer,
                                force,
                            )
                        )

                        if task_type == "door":
                            current_angle = get_joint_qpos(
                                env,
                                hinge_id,
                            )
                            last_door_progress = max(
                                0.0,
                                opening_sign
                                * (
                                    current_angle
                                    - initial_door_angle
                                ),
                            )
                            if one_way_lock_activated:
                                wrong_progress = max(
                                    0.0,
                                    -opening_sign
                                    * (
                                        current_angle
                                        - initial_door_angle
                                    ),
                                )
                                wrong_direction_peak_progress_rad = max(
                                    wrong_direction_peak_progress_rad,
                                    wrong_progress,
                                )
                                lock_limit = (
                                    episode_wrong_direction_slack
                                    + door_direction_lock_tolerance
                                )
                                if wrong_progress > lock_limit + 1e-9:
                                    direction_lock_violation = True
                                    failure_reason = (
                                        "one-way hinge violation; "
                                        f"wrong_progress={wrong_progress:.6f}rad "
                                        f"> allowed={lock_limit:.6f}rad"
                                    )

                            (
                                _,
                                current_force_world,
                                _,
                            ) = get_force_world(
                                env,
                                observation,
                                sensor_frame=episode_force_sensor_frame,
                            )

                            if not baseline_frozen:
                                if (
                                    door_use_demo_stage_schedule
                                    and steps <= force_probe_start_step
                                ):
                                    baseline_force_world_samples.append(
                                        current_force_world
                                    )
                                elif (
                                    not door_use_demo_stage_schedule
                                    and last_door_progress
                                    < door_baseline_freeze_progress
                                ):
                                    baseline_force_world_samples.append(
                                        current_force_world
                                    )
                                else:
                                    baseline_force_world = np.median(
                                        np.stack(
                                            baseline_force_world_samples,
                                            axis=0,
                                        ),
                                        axis=0,
                                    )
                                    baseline_frozen = True

                            # Legacy fallback for datasets without an FSM stage
                            # schedule. New Door data starts the budget at the
                            # recorded ADAPTIVE_ARC_PULL action above.
                            if (
                                adaptive_open_start_step is None
                                and not pull_started
                                and last_door_progress
                                >= door_pull_start_progress
                            ):
                                if baseline_force_world is None:
                                    baseline_force_world = np.median(
                                        np.stack(
                                            baseline_force_world_samples,
                                            axis=0,
                                        ),
                                        axis=0,
                                    )
                                    baseline_frozen = True

                                pull_started = True
                                previous_progress = last_door_progress
                                pull_steps = 0
                                print(
                                    f"[DOOR OPENING BUDGET START] "
                                    f"step={steps} "
                                    f"progress="
                                    f"{last_door_progress:.4f}rad "
                                    f"source=progress_fallback"
                                )

                            if pull_started:
                                pull_steps += 1
                                assert baseline_force_world is not None

                                delta_force_world = (
                                    current_force_world
                                    - baseline_force_world
                                )
                                live_opening_tangent = rotate_vector_about_axis(
                                    initial_opening_tangent,
                                    hinge_axis_world,
                                    current_angle - initial_door_angle,
                                )
                                interaction_force = float(
                                    abs(
                                        np.dot(
                                            delta_force_world,
                                            live_opening_tangent,
                                        )
                                    )
                                )
                                delta_force_norm = float(
                                    np.linalg.norm(delta_force_world)
                                )
                                interaction_force_ema_n = (
                                    door_force_ema_alpha
                                    * interaction_force
                                    + (1.0 - door_force_ema_alpha)
                                    * interaction_force_ema_n
                                )
                                peak_interaction_force_n = max(
                                    peak_interaction_force_n,
                                    interaction_force_ema_n,
                                )
                                peak_delta_force_norm_n = max(
                                    peak_delta_force_norm_n,
                                    delta_force_norm,
                                )

                                door_speed = max(
                                    0.0,
                                    (
                                        last_door_progress
                                        - previous_progress
                                    )
                                    * episode_door_control_freq,
                                )
                                previous_progress = last_door_progress
                                door_speed_ema_rad_s = (
                                    door_speed_ema_alpha * door_speed
                                    + (1.0 - door_speed_ema_alpha)
                                    * door_speed_ema_rad_s
                                )
                                peak_door_speed_rad_s = max(
                                    peak_door_speed_rad_s,
                                    door_speed_ema_rad_s,
                                )

                                if pull_steps >= episode_damage_grace_steps:
                                    excess_force = max(
                                        0.0,
                                        interaction_force_ema_n
                                        - damage_force_limit,
                                    )
                                    damage_impulse_ns += (
                                        excess_force / episode_door_control_freq
                                    )

                                    if (
                                        interaction_force_ema_n
                                        > damage_force_limit
                                    ):
                                        breaking_force_streak += 1
                                    else:
                                        breaking_force_streak = 0

                                    if door_speed_ema_rad_s > max_safe_speed:
                                        overspeed_streak += 1
                                    else:
                                        overspeed_streak = 0
                                else:
                                    breaking_force_streak = 0
                                    overspeed_streak = 0

                                if interaction_force_ema_n > hard_break_force:
                                    hard_force_limit_exceeded = True
                                    if episode_enforce_safety_limits:
                                        door_broken = True
                                        failure_reason = (
                                            "hard tangential break force exceeded; "
                                            f"pull_force_ema={interaction_force_ema_n:.3f}N "
                                            f"> hard_limit={hard_break_force:.3f}N"
                                        )

                                if (
                                    breaking_force_streak
                                    >= episode_breaking_force_hold_steps
                                ):
                                    sustained_force_limit_exceeded = True
                                    if episode_enforce_safety_limits:
                                        door_broken = True
                                        failure_reason = (
                                            "sustained tangential breaking force exceeded; "
                                            f"pull_force_ema={interaction_force_ema_n:.3f}N "
                                            f"> limit={damage_force_limit:.3f}N for "
                                            f"{breaking_force_streak} steps"
                                        )

                                if damage_impulse_ns > damage_impulse_limit:
                                    damage_impulse_limit_exceeded = True
                                    if episode_enforce_safety_limits:
                                        door_broken = True
                                        failure_reason = (
                                            "accumulated tangential overload exceeded; "
                                            f"impulse={damage_impulse_ns:.3f}Ns > "
                                            f"limit={damage_impulse_limit:.3f}Ns"
                                        )

                                if (
                                    overspeed_streak
                                    >= episode_overspeed_hold_steps
                                ):
                                    speed_limit_exceeded = True
                                    if episode_enforce_safety_limits:
                                        door_broken = True
                                        failure_reason = (
                                            "resistance-derived safe opening speed "
                                            "exceeded; "
                                            f"speed_ema={door_speed_ema_rad_s:.3f}rad/s "
                                            f"> limit={max_safe_speed:.3f}rad/s for "
                                            f"{overspeed_streak} steps"
                                        )

                                # The old 80-step progress checkpoint is
                                # deliberately not enforced: the validated
                                # collector marks it deprecated and judges every
                                # door only by final angle, the common opening
                                # budget, and the shared safety rules.
                                if (
                                    pull_steps >= episode_pull_timeout_steps
                                    and last_door_progress
                                    < target_progress - success_tolerance
                                ):
                                    insufficient_progress = True
                                    failure_reason = (
                                        "global opening-step limit exceeded; "
                                        f"progress={last_door_progress:.4f}rad < "
                                        f"required="
                                        f"{target_progress - success_tolerance:.4f}rad "
                                        f"after {pull_steps}/"
                                        f"{episode_pull_timeout_steps} pull steps"
                                    )

                            opened = (
                                last_door_progress
                                >= target_progress
                                - success_tolerance
                            )
                            damage_limit_exceeded = bool(
                                hard_force_limit_exceeded
                                or sustained_force_limit_exceeded
                                or damage_impulse_limit_exceeded
                                or speed_limit_exceeded
                            )
                            safety_success = bool(
                                not damage_limit_exceeded
                                or not episode_enforce_safety_limits
                            )
                            safe_opened = bool(
                                opened
                                and not door_broken
                                and not insufficient_progress
                                and not direction_lock_violation
                                and safety_success
                            )
                            success_streak = (
                                success_streak + 1
                                if safe_opened
                                else 0
                            )
                            episode_success = (
                                success_streak
                                >= episode_success_hold_steps
                            )

                            should_print = (
                                steps == 1
                                or steps % force_print_interval == 0
                                or episode_success
                                or door_broken
                                or insufficient_progress
                            )

                            if should_print:
                                print(
                                    f"[STEP {steps:04d}] "
                                    f"truth={truth_label} "
                                    f"progress="
                                    f"{last_door_progress:.4f}/"
                                    f"{target_progress:.4f}rad "
                                    f"F={np.array2string(force, precision=3)} "
                                    f"|F|={force_norm:.3f} "
                                    f"interaction_ema="
                                    f"{interaction_force_ema_n:.3f}N "
                                    f"speed_ema="
                                    f"{door_speed_ema_rad_s:.3f}rad/s "
                                    f"damage="
                                    f"{damage_impulse_ns:.3f}Ns "
                                    f"safety_exceeded="
                                    f"{damage_limit_exceeded} "
                                    f"safety_enforced="
                                    f"{episode_enforce_safety_limits} "
                                    f"lock_violation="
                                    f"{direction_lock_violation} "
                                    f"timeout={insufficient_progress} "
                                    f"success={episode_success}"
                                )
                                if normalized_force is not None:
                                    print(
                                        "    normalized_force="
                                        f"{np.array2string(normalized_force, precision=4)}"
                                    )

                            if render:
                                stop_requested = (
                                    render_observation(
                                        observation,
                                        camera_keys,
                                        (
                                            f"door truth={truth_label}",
                                            (
                                                f"step={steps} "
                                                f"progress="
                                                f"{last_door_progress:.3f}/"
                                                f"{target_progress:.3f}"
                                            ),
                                            (
                                                f"|F|={force_norm:.2f} "
                                                f"interaction="
                                                f"{interaction_force_ema_n:.2f}"
                                            ),
                                            (
                                                f"safety="
                                                f"{damage_limit_exceeded}/"
                                                f"{episode_enforce_safety_limits} "
                                                f"lock="
                                                f"{direction_lock_violation} "
                                                f"timeout="
                                                f"{insufficient_progress} "
                                                f"success="
                                                f"{episode_success}"
                                            ),
                                        ),
                                        render_scale,
                                    )
                                )

                            effective_done = bool(
                                done
                                and respect_env_done_for_door
                            )
                        elif task_type == "stack":
                            assert stack_geometries is not None
                            assert true_heavy_name is not None
                            assert true_light_name is not None

                            last_stack_metrics = check_stack_success(
                                env,
                                stack_geometries,
                                true_heavy_name,
                                true_light_name,
                                xy_tolerance_m=(
                                    episode_stack_xy_tolerance
                                ),
                                z_tolerance_m=(
                                    episode_stack_z_tolerance
                                ),
                                bottom_table_tolerance_m=(
                                    episode_stack_bottom_tolerance
                                ),
                            )
                            last_standard_success = bool(
                                check_success(env)
                            )
                            success_streak = (
                                success_streak + 1
                                if last_stack_metrics["correct_stack"]
                                else 0
                            )
                            episode_success = bool(
                                success_streak
                                >= episode_stack_hold_steps
                            )

                            should_print = (
                                steps == 1
                                or steps % force_print_interval == 0
                                or done
                                or last_stack_metrics["correct_stack"]
                            )
                            if should_print:
                                print(
                                    f"[STEP {steps:04d}] "
                                    f"truth_heavy={true_heavy_name} "
                                    f"|F|={force_norm:.3f} "
                                    f"heavy_pos="
                                    f"{np.array2string(last_stack_metrics['heavy_position'], precision=3)} "
                                    f"light_pos="
                                    f"{np.array2string(last_stack_metrics['light_position'], precision=3)} "
                                    f"xy={last_stack_metrics['horizontal_error']:.4f}/"
                                    f"{episode_stack_xy_tolerance:.4f} "
                                    f"z={last_stack_metrics['vertical_error']:.4f}/"
                                    f"{episode_stack_z_tolerance:.4f} "
                                    f"touch={last_stack_metrics['touching']} "
                                    f"released={last_stack_metrics['light_released']} "
                                    f"heavy_table={last_stack_metrics['heavy_on_table']} "
                                    f"order={last_stack_metrics['order_correct']} "
                                    f"success={last_stack_metrics['correct_stack']} "
                                    f"streak={success_streak}/"
                                    f"{episode_stack_hold_steps}"
                                )
                                if normalized_force is not None:
                                    print(
                                        "    normalized_force="
                                        f"{np.array2string(normalized_force, precision=4)}"
                                    )

                            if render:
                                stop_requested = render_observation(
                                    observation,
                                    camera_keys,
                                    (
                                        (
                                            f"heavy={true_heavy_name} "
                                            f"light={true_light_name}"
                                        ),
                                        (
                                            f"step={steps} |F|={force_norm:.2f}"
                                        ),
                                        (
                                            f"xy={last_stack_metrics['horizontal_error']:.3f}/"
                                            f"{episode_stack_xy_tolerance:.3f} "
                                            f"z={last_stack_metrics['vertical_error']:.3f}/"
                                            f"{episode_stack_z_tolerance:.3f}"
                                        ),
                                        (
                                            f"touch={last_stack_metrics['touching']} "
                                            f"release={last_stack_metrics['light_released']} "
                                            f"table={last_stack_metrics['heavy_on_table']}"
                                        ),
                                        (
                                            f"success={last_stack_metrics['correct_stack']} "
                                            f"streak={success_streak}/"
                                            f"{episode_stack_hold_steps}"
                                        ),
                                    ),
                                    render_scale,
                                )

                            effective_done = bool(done)
                        elif task_type == "nut_assembly_square":
                            # Use the task's native success check, while also
                            # recording sparse reward success for diagnostics.
                            last_nut_standard_success = bool(
                                check_success(env)
                            )
                            last_nut_reward = float(reward)
                            last_nut_reward_success = bool(
                                float(reward)
                                >= nut_success_reward_threshold
                            )

                            # Native task success is authoritative. The reward
                            # signal is retained so replay/evaluation behaviour
                            # can be inspected if Robosuite versions differ.
                            success_streak = (
                                success_streak + 1
                                if last_nut_standard_success
                                else 0
                            )
                            episode_success = bool(
                                success_streak
                                >= nut_success_hold_steps
                            )

                            should_print = (
                                steps == 1
                                or steps % force_print_interval == 0
                                or done
                                or last_nut_standard_success
                            )
                            if should_print:
                                print(
                                    f"[STEP {steps:04d}] "
                                    f"task=NutAssemblySquare "
                                    f"|F|={force_norm:.3f} "
                                    f"reward={last_nut_reward:.3f} "
                                    f"env_success={last_nut_standard_success} "
                                    f"reward_success={last_nut_reward_success} "
                                    f"streak={success_streak}/"
                                    f"{nut_success_hold_steps}"
                                )
                                if normalized_force is not None:
                                    print(
                                        "    normalized_force="
                                        f"{np.array2string(normalized_force, precision=4)}"
                                    )

                            if render:
                                stop_requested = render_observation(
                                    observation,
                                    camera_keys,
                                    (
                                        "NutAssemblySquare",
                                        (
                                            f"step={steps} "
                                            f"|F|={force_norm:.2f}"
                                        ),
                                        (
                                            f"reward={last_nut_reward:.3f} "
                                            f"env_success="
                                            f"{last_nut_standard_success}"
                                        ),
                                        (
                                            f"streak={success_streak}/"
                                            f"{nut_success_hold_steps}"
                                        ),
                                    ),
                                    render_scale,
                                )

                            effective_done = bool(done)
                        else:
                            assert object_body_name is not None
                            assert target_position is not None

                            object_position = (
                                get_object_position(
                                    env,
                                    object_body_name,
                                )
                            )

                            if previous_object_position is None:
                                last_object_speed_m_s = float("inf")
                            else:
                                last_object_speed_m_s = float(
                                    np.linalg.norm(
                                        object_position
                                        - previous_object_position
                                    )
                                    * pick_place_control_freq
                                )
                            previous_object_position = object_position.copy()

                            last_gripper_action = (
                                float(action[-1])
                                if np.asarray(action).size > 0
                                else float("nan")
                            )
                            last_pick_place_released = bool(
                                not pick_place_require_release
                                or (
                                    np.isfinite(last_gripper_action)
                                    and last_gripper_action
                                    <= pick_place_open_action_threshold
                                )
                            )
                            last_pick_place_settled = bool(
                                last_object_speed_m_s
                                <= pick_place_max_settle_speed
                            )

                            (
                                last_custom_success,
                                last_dx,
                                last_dy,
                                last_half_x,
                                last_half_y,
                            ) = check_custom_success(
                                env,
                                object_position,
                                target_position,
                                target_xy_fraction,
                                max_z_above_target=(
                                    pick_place_max_z_above_target
                                ),
                                object_speed_m_s=last_object_speed_m_s,
                                max_settle_speed_m_s=(
                                    pick_place_max_settle_speed
                                ),
                                gripper_action=last_gripper_action,
                                require_release=(
                                    pick_place_require_release
                                ),
                                open_action_threshold=(
                                    pick_place_open_action_threshold
                                ),
                            )
                            last_xy_distance = float(
                                np.linalg.norm(
                                    object_position[:2]
                                    - target_position[:2]
                                )
                            )
                            last_z_error = float(
                                object_position[2]
                                - target_position[2]
                            )
                            last_standard_success = bool(
                                check_success(env)
                            )
                            success_streak = (
                                success_streak + 1
                                if last_custom_success
                                else 0
                            )
                            episode_success = (
                                success_streak
                                >= success_hold_steps
                            )

                            should_print = (
                                steps == 1
                                or steps % force_print_interval == 0
                                or done
                                or last_custom_success
                            )
                            if should_print:
                                print(
                                    f"[STEP {steps:04d}] "
                                    f"truth={truth_label} "
                                    f"F="
                                    f"{np.array2string(force, precision=3)} "
                                    f"|F|={force_norm:.3f} "
                                    f"|Fz|={force_z:.3f} "
                                    f"object="
                                    f"{np.array2string(object_position, precision=3)} "
                                    f"xy_distance="
                                    f"{last_xy_distance:.3f} "
                                    f"dx={last_dx:.3f}/"
                                    f"{last_half_x:.3f} "
                                    f"dy={last_dy:.3f}/"
                                    f"{last_half_y:.3f} "
                                    f"z_error="
                                    f"{last_z_error:+.3f}/"
                                    f"+{pick_place_max_z_above_target:.3f} "
                                    f"speed="
                                    f"{last_object_speed_m_s:.3f}/"
                                    f"{pick_place_max_settle_speed:.3f}m/s "
                                    f"grip_action="
                                    f"{last_gripper_action:+.1f} "
                                    f"released="
                                    f"{last_pick_place_released} "
                                    f"settled="
                                    f"{last_pick_place_settled} "
                                    f"custom="
                                    f"{last_custom_success} "
                                    f"standard="
                                    f"{last_standard_success}"
                                )
                                if normalized_force is not None:
                                    print(
                                        "    normalized_force="
                                        f"{np.array2string(normalized_force, precision=4)}"
                                    )

                            if render:
                                stop_requested = (
                                    render_observation(
                                        observation,
                                        camera_keys,
                                        (
                                            (
                                                f"truth={truth_label} "
                                                f"mass={mass_kg:.3f}kg"
                                            ),
                                            (
                                                f"step={steps} "
                                                f"|F|={force_norm:.2f} "
                                                f"Fz={force[2]:+.2f}"
                                            ),
                                            (
                                                f"xy="
                                                f"{last_xy_distance:.3f} "
                                                f"z={last_z_error:+.3f}/"
                                                f"+{pick_place_max_z_above_target:.3f}"
                                            ),
                                            (
                                                f"speed="
                                                f"{last_object_speed_m_s:.3f} "
                                                f"released="
                                                f"{last_pick_place_released}"
                                            ),
                                            (
                                                f"custom="
                                                f"{last_custom_success} "
                                                f"standard="
                                                f"{last_standard_success}"
                                            ),
                                        ),
                                        render_scale,
                                    )
                                )

                            effective_done = bool(done)

                        if (
                            video_path is not None
                            and video_save_mode != "none"
                            and video_camera_key in observation
                        ):
                            episode_video_frames.append(
                                image_to_hwc_uint8(
                                    observation[video_camera_key],
                                    video_camera_key,
                                    flip_vertical=False,
                                )
                            )

                        if (
                            effective_done
                            or episode_success
                            or door_broken
                            or insufficient_progress
                            or direction_lock_violation
                            or stop_requested
                            or steps >= episode_max_steps
                        ):
                            break

                if task_type == "door":
                    result = {
                        "demo": demo_name,
                        "truth_label": truth_label,
                        "task_type": task_type,
                        "success": bool(episode_success),
                        "enforce_safety_limits": bool(
                            episode_enforce_safety_limits
                        ),
                        "damage_limit_exceeded": bool(
                            hard_force_limit_exceeded
                            or sustained_force_limit_exceeded
                            or damage_impulse_limit_exceeded
                            or speed_limit_exceeded
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
                        "door_broken": bool(door_broken),
                        "direction_lock_violation": bool(
                            direction_lock_violation
                        ),
                        "wrong_direction_peak_progress_rad": float(
                            wrong_direction_peak_progress_rad
                        ),
                        "wrong_direction_limit_rad": float(
                            episode_wrong_direction_slack
                            + door_direction_lock_tolerance
                        ),
                        "one_way_lock_activated": bool(
                            one_way_lock_activated
                        ),
                        "force_probe_start_step": int(
                            force_probe_start_step
                        ),
                        "adaptive_open_start_step": (
                            int(adaptive_open_start_step)
                            if adaptive_open_start_step is not None
                            else None
                        ),
                        "insufficient_progress": bool(
                            insufficient_progress
                        ),
                        "failure_reason": failure_reason,
                        "hinge_name": hinge_name,
                        "hinge_damping": float(damping),
                        "hinge_friction": float(friction),
                        "initial_door_angle": float(
                            initial_door_angle
                        ),
                        "opening_sign": float(opening_sign),
                        "target_door_progress": float(
                            target_progress
                        ),
                        "door_success_tolerance": float(
                            success_tolerance
                        ),
                        "resistance_score": float(resistance_score),
                        "derived_damage_force_limit_n": float(
                            damage_force_limit
                        ),
                        "derived_hard_break_force_limit_n": float(
                            hard_break_force
                        ),
                        "derived_damage_impulse_limit_ns": float(
                            damage_impulse_limit
                        ),
                        "derived_max_safe_open_speed_rad_s": float(
                            max_safe_speed
                        ),
                        "progress_check_steps": int(
                            global_progress_check_steps
                        ),
                        "minimum_progress_at_check": float(
                            global_minimum_progress
                        ),
                        "pull_timeout_steps": int(
                            episode_pull_timeout_steps
                        ),
                        "max_episode_steps": int(episode_max_steps),
                        "control_freq": float(
                            episode_door_control_freq
                        ),
                        "breaking_force_hold_steps": int(
                            episode_breaking_force_hold_steps
                        ),
                        "overspeed_hold_steps": int(
                            episode_overspeed_hold_steps
                        ),
                        "success_hold_steps": int(
                            episode_success_hold_steps
                        ),
                        "final_door_progress": float(
                            last_door_progress
                        ),
                        "damage_impulse_ns": float(
                            damage_impulse_ns
                        ),
                        "peak_interaction_force_n": float(
                            peak_interaction_force_n
                        ),
                        "peak_delta_force_norm_n": float(
                            peak_delta_force_norm_n
                        ),
                        "peak_door_speed_rad_s": float(
                            peak_door_speed_rad_s
                        ),
                        "pull_steps": int(pull_steps),
                        "steps": int(steps),
                        "return": float(episode_return),
                        "num_replans": int(num_replans),
                        "mean_inference_time_ms": (
                            float(np.mean(inference_times_ms))
                            if inference_times_ms
                            else 0.0
                        ),
                    }
                    print(
                        f"[END] demo={demo_name} "
                        f"truth={truth_label} "
                        f"success={episode_success} "
                        f"progress={last_door_progress:.4f}/"
                        f"{target_progress:.4f} "
                        f"safety_exceeded="
                        f"{result['damage_limit_exceeded']} "
                        f"safety_enforced="
                        f"{episode_enforce_safety_limits} "
                        f"lock_violation="
                        f"{direction_lock_violation} "
                        f"timeout={insufficient_progress} "
                        f"reason={failure_reason!r}"
                    )
                elif task_type == "stack":
                    assert true_heavy_name is not None
                    assert true_light_name is not None
                    result = {
                        "demo": demo_name,
                        "truth_label": truth_label,
                        "task_type": task_type,
                        "true_heavy_name": true_heavy_name,
                        "true_light_name": true_light_name,
                        "cubeA_mass_kg": float(cube_a_mass_kg),
                        "cubeB_mass_kg": float(cube_b_mass_kg),
                        "success": bool(episode_success),
                        "correct_stack_last": bool(
                            last_stack_metrics["correct_stack"]
                        ),
                        "order_correct_last": bool(
                            last_stack_metrics["order_correct"]
                        ),
                        "touching_last": bool(
                            last_stack_metrics["touching"]
                        ),
                        "light_released_last": bool(
                            last_stack_metrics["light_released"]
                        ),
                        "heavy_on_table_last": bool(
                            last_stack_metrics["heavy_on_table"]
                        ),
                        "final_horizontal_error_m": float(
                            last_stack_metrics["horizontal_error"]
                        ),
                        "final_vertical_error_m": float(
                            last_stack_metrics["vertical_error"]
                        ),
                        "expected_vertical_separation_m": float(
                            last_stack_metrics[
                                "expected_vertical_separation"
                            ]
                        ),
                        "actual_vertical_separation_m": float(
                            last_stack_metrics[
                                "actual_vertical_separation"
                            ]
                        ),
                        "final_heavy_position": (
                            np.asarray(
                                last_stack_metrics["heavy_position"]
                            ).tolist()
                        ),
                        "final_light_position": (
                            np.asarray(
                                last_stack_metrics["light_position"]
                            ).tolist()
                        ),
                        "stack_xy_tolerance_m": float(
                            episode_stack_xy_tolerance
                        ),
                        "stack_z_tolerance_m": float(
                            episode_stack_z_tolerance
                        ),
                        "bottom_table_tolerance_m": float(
                            episode_stack_bottom_tolerance
                        ),
                        "success_hold_steps": int(
                            episode_stack_hold_steps
                        ),
                        "standard_success_last": bool(
                            last_standard_success
                        ),
                        "steps": int(steps),
                        "return": float(episode_return),
                        "num_replans": int(num_replans),
                        "mean_inference_time_ms": (
                            float(np.mean(inference_times_ms))
                            if inference_times_ms
                            else 0.0
                        ),
                    }
                    print(
                        f"[END] demo={demo_name} "
                        f"truth_heavy={true_heavy_name} "
                        f"success={episode_success} "
                        f"correct_stack_last="
                        f"{last_stack_metrics['correct_stack']} "
                        f"order={last_stack_metrics['order_correct']} "
                        f"touch={last_stack_metrics['touching']} "
                        f"released={last_stack_metrics['light_released']} "
                        f"heavy_table={last_stack_metrics['heavy_on_table']} "
                        f"xy={last_stack_metrics['horizontal_error']:.4f} "
                        f"z={last_stack_metrics['vertical_error']:.4f}"
                    )
                elif task_type == "nut_assembly_square":
                    result = {
                        "demo": demo_name,
                        "truth_label": truth_label,
                        "task_type": task_type,
                        "success": bool(episode_success),
                        "standard_success_last": bool(
                            last_nut_standard_success
                        ),
                        "reward_success_last": bool(
                            last_nut_reward_success
                        ),
                        "final_reward": float(last_nut_reward),
                        "success_hold_steps": int(
                            nut_success_hold_steps
                        ),
                        "success_reward_threshold": float(
                            nut_success_reward_threshold
                        ),
                        "steps": int(steps),
                        "return": float(episode_return),
                        "num_replans": int(num_replans),
                        "mean_inference_time_ms": (
                            float(np.mean(inference_times_ms))
                            if inference_times_ms
                            else 0.0
                        ),
                    }
                    print(
                        f"[END] demo={demo_name} "
                        f"task=NutAssemblySquare "
                        f"success={episode_success} "
                        f"env_success={last_nut_standard_success} "
                        f"reward={last_nut_reward:.3f} "
                        f"steps={steps}"
                    )
                else:
                    assert object_body_name is not None
                    assert target_position is not None
                    result = {
                        "demo": demo_name,
                        "truth_label": truth_label,
                        "task_type": task_type,
                        "mass_kg": mass_kg,
                        "active_object_body": object_body_name,
                        "success": bool(episode_success),
                        "custom_success_last": bool(
                            last_custom_success
                        ),
                        "standard_success_last": bool(
                            last_standard_success
                        ),
                        "final_object_position": (
                            get_object_position(
                                env,
                                object_body_name,
                            ).tolist()
                        ),
                        "target_position": (
                            target_position.tolist()
                        ),
                        "final_xy_distance": float(
                            last_xy_distance
                        ),
                        "final_dx": float(last_dx),
                        "final_dy": float(last_dy),
                        "allowed_half_x": float(last_half_x),
                        "allowed_half_y": float(last_half_y),
                        "final_z_error": float(last_z_error),
                        "max_z_above_target": float(
                            pick_place_max_z_above_target
                        ),
                        "final_object_speed_m_s": float(
                            last_object_speed_m_s
                        ),
                        "max_settle_speed_m_s": float(
                            pick_place_max_settle_speed
                        ),
                        "final_gripper_action": float(
                            last_gripper_action
                        ),
                        "require_release": bool(
                            pick_place_require_release
                        ),
                        "released_last": bool(
                            last_pick_place_released
                        ),
                        "settled_last": bool(
                            last_pick_place_settled
                        ),
                        "steps": int(steps),
                        "return": float(episode_return),
                        "num_replans": int(num_replans),
                        "mean_inference_time_ms": (
                            float(np.mean(inference_times_ms))
                            if inference_times_ms
                            else 0.0
                        ),
                    }
                    print(
                        f"[END] demo={demo_name} "
                        f"truth={truth_label} "
                        f"success={episode_success} "
                        f"custom_last={last_custom_success} "
                        f"standard_last={last_standard_success} "
                        f"xy_distance={last_xy_distance:.4f} "
                        f"z_error={last_z_error:+.4f} "
                        f"speed={last_object_speed_m_s:.4f}m/s "
                        f"released={last_pick_place_released} "
                        f"settled={last_pick_place_settled}"
                    )

                if (
                    video_path is not None
                    and episode_video_frames
                    and should_save_episode_video(
                        video_save_mode,
                        episode_index=episode_index,
                        success=bool(episode_success),
                    )
                ):
                    requested_video = Path(video_path)

                    if video_save_mode == "first":
                        episode_video_path = requested_video
                    else:
                        video_directory = requested_video.with_suffix("")
                        status = (
                            "success" if episode_success else "failure"
                        )
                        safe_demo = sanitize_video_component(demo_name)
                        safe_truth = sanitize_video_component(truth_label)
                        extension = requested_video.suffix or ".mp4"
                        episode_video_path = (
                            video_directory
                            / (
                                f"episode_{episode_index:03d}_"
                                f"{safe_demo}_{safe_truth}_{status}"
                                f"{extension}"
                            )
                        )

                    written_path = save_video_frames(
                        episode_video_frames,
                        episode_video_path,
                        video_fps,
                    )
                    result["video_path"] = str(written_path)
                    saved_video_count += 1
                    print(f"[VIDEO] saved={written_path}")

                    # train.py expects the exact requested path to exist so it
                    # can upload one representative video to W&B. In multi-video
                    # modes, keep the first selected episode at that primary path
                    # as well; all episode-specific videos remain in the folder.
                    if (
                        video_save_mode != "first"
                        and not primary_video_written
                    ):
                        save_video_frames(
                            episode_video_frames,
                            requested_video,
                            video_fps,
                        )
                        primary_video_written = True

                episode_results.append(result)

    finally:
        policy.train(previous_training_state)

        if render:
            try:
                import cv2
                cv2.destroyAllWindows()
            except ImportError:
                pass

        if hasattr(env, "close"):
            env.close()

    success_values = [
        float(result["success"])
        for result in episode_results
    ]
    light_results = [
        result
        for result in episode_results
        if result["truth_label"] == "light"
    ]
    heavy_results = [
        result
        for result in episode_results
        if result["truth_label"] == "heavy"
    ]
    heavy_cube_a_results = [
        result
        for result in episode_results
        if result["truth_label"] == "heavy_cubeA"
    ]
    heavy_cube_b_results = [
        result
        for result in episode_results
        if result["truth_label"] == "heavy_cubeB"
    ]
    stack_results = [
        result
        for result in episode_results
        if result.get("task_type") == "stack"
    ]
    push_results = [
        result
        for result in episode_results
        if result["truth_label"] == "push"
    ]
    pull_results = [
        result
        for result in episode_results
        if result["truth_label"] == "pull"
    ]

    metrics: dict[str, Any] = {
        "task_type": task_type,
        "success_rate": (
            float(np.mean(success_values))
            if success_values
            else 0.0
        ),
        "light_success_rate": (
            float(
                np.mean(
                    [
                        float(result["success"])
                        for result in light_results
                    ]
                )
            )
            if light_results
            else 0.0
        ),
        "heavy_success_rate": (
            float(
                np.mean(
                    [
                        float(result["success"])
                        for result in heavy_results
                    ]
                )
            )
            if heavy_results
            else 0.0
        ),
        "heavy_cubeA_success_rate": (
            float(
                np.mean(
                    [
                        float(result["success"])
                        for result in heavy_cube_a_results
                    ]
                )
            )
            if heavy_cube_a_results
            else 0.0
        ),
        "heavy_cubeB_success_rate": (
            float(
                np.mean(
                    [
                        float(result["success"])
                        for result in heavy_cube_b_results
                    ]
                )
            )
            if heavy_cube_b_results
            else 0.0
        ),
        "stack_order_correct_rate": (
            float(
                np.mean(
                    [
                        float(result["order_correct_last"])
                        for result in stack_results
                    ]
                )
            )
            if stack_results
            else 0.0
        ),
        "stack_touching_rate": (
            float(
                np.mean(
                    [
                        float(result["touching_last"])
                        for result in stack_results
                    ]
                )
            )
            if stack_results
            else 0.0
        ),
        "stack_light_released_rate": (
            float(
                np.mean(
                    [
                        float(result["light_released_last"])
                        for result in stack_results
                    ]
                )
            )
            if stack_results
            else 0.0
        ),
        "stack_heavy_on_table_rate": (
            float(
                np.mean(
                    [
                        float(result["heavy_on_table_last"])
                        for result in stack_results
                    ]
                )
            )
            if stack_results
            else 0.0
        ),
        "push_success_rate": (
            float(
                np.mean(
                    [
                        float(result["success"])
                        for result in push_results
                    ]
                )
            )
            if push_results
            else 0.0
        ),
        "pull_success_rate": (
            float(
                np.mean(
                    [
                        float(result["success"])
                        for result in pull_results
                    ]
                )
            )
            if pull_results
            else 0.0
        ),
        "mean_episode_length": (
            float(
                np.mean(
                    [
                        result["steps"]
                        for result in episode_results
                    ]
                )
            )
            if episode_results
            else 0.0
        ),
        "mean_return": (
            float(
                np.mean(
                    [
                        result["return"]
                        for result in episode_results
                    ]
                )
            )
            if episode_results
            else 0.0
        ),
        "num_episodes": float(len(episode_results)),
        "saved_video_count": float(saved_video_count),
        "mean_inference_time_ms": (
            float(np.mean([
                result.get("mean_inference_time_ms", 0.0)
                for result in episode_results
            ]))
            if episode_results
            else 0.0
        ),
        "mean_num_replans": (
            float(np.mean([
                result.get("num_replans", 0)
                for result in episode_results
            ]))
            if episode_results
            else 0.0
        ),
        "env_name": str(
            env_metadata.get(
                "env_name",
                "unknown",
            )
        ),
    }

    if task_type == "nut_assembly_square":
        metrics.update(
            {
                "nut_standard_success_rate": (
                    float(
                        np.mean(
                            [
                                float(
                                    result["standard_success_last"]
                                )
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "nut_reward_success_rate": (
                    float(
                        np.mean(
                            [
                                float(
                                    result["reward_success_last"]
                                )
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
            }
        )

    if task_type == "door":
        metrics.update(
            {
                "door_broken_rate": (
                    float(
                        np.mean(
                            [
                                float(result["door_broken"])
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "safety_limit_exceed_rate": (
                    float(
                        np.mean(
                            [
                                float(
                                    result["damage_limit_exceeded"]
                                )
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "direction_lock_violation_rate": (
                    float(
                        np.mean(
                            [
                                float(
                                    result["direction_lock_violation"]
                                )
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "max_wrong_direction_progress_rad": (
                    max(
                        [
                            float(
                                result[
                                    "wrong_direction_peak_progress_rad"
                                ]
                            )
                            for result in episode_results
                        ],
                        default=0.0,
                    )
                ),
                "insufficient_progress_rate": (
                    float(
                        np.mean(
                            [
                                float(
                                    result[
                                        "insufficient_progress"
                                    ]
                                )
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "mean_final_door_progress": (
                    float(
                        np.mean(
                            [
                                result[
                                    "final_door_progress"
                                ]
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "mean_peak_interaction_force_n": (
                    float(
                        np.mean(
                            [
                                result[
                                    "peak_interaction_force_n"
                                ]
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
                "mean_peak_door_speed_rad_s": (
                    float(
                        np.mean(
                            [
                                result[
                                    "peak_door_speed_rad_s"
                                ]
                                for result in episode_results
                            ]
                        )
                    )
                    if episode_results
                    else 0.0
                ),
            }
        )

    if return_episode_details:
        metrics["episodes"] = episode_results

    return metrics

