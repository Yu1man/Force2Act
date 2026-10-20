from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch


DEFAULT_FORCE_KEY = "robot0_eef_force"
DEFAULT_TORQUE_KEY = "robot0_eef_torque"


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def import_robosuite():
    try:
        import robosuite as suite
        try:
            from robosuite.controllers import load_composite_controller_config
        except ImportError:
            from robosuite import load_composite_controller_config
    except ImportError as exc:
        raise ImportError("robosuite is required for simulation collection") from exc
    return suite, load_composite_controller_config


def import_robomimic():
    try:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
    except ImportError as exc:
        raise ImportError(
            "robomimic is required for replay-based simulation collection"
        ) from exc
    return EnvUtils, FileUtils, ObsUtils


def initialize_obs_utils(
    state_keys: Sequence[str],
    camera_keys: Sequence[str],
    force_keys: Sequence[str] = (),
) -> None:
    _, _, ObsUtils = import_robomimic()
    spec = {
        "obs": {
            "low_dim": list(state_keys) + list(force_keys),
            "rgb": list(camera_keys),
            "depth": [],
            "scan": [],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=spec)


def create_data_processing_env(
    dataset_path: str,
    camera_names: Sequence[str],
    image_size: int,
    expected_env_name: str | None = None,
):
    EnvUtils, FileUtils, _ = import_robomimic()
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    if not isinstance(env_meta, dict):
        raise TypeError(f"Expected environment metadata dict, got {type(env_meta)!r}")

    actual_env_name = str(env_meta.get("env_name", ""))
    if not actual_env_name:
        raise ValueError("Dataset environment metadata has no env_name")

    if expected_env_name is not None and actual_env_name != expected_env_name:
        raise ValueError(
            "Config / dataset task mismatch: "
            f"expected {expected_env_name!r}, dataset says {actual_env_name!r}"
        )

    env = EnvUtils.create_env_for_data_processing(
        env_meta=deepcopy(env_meta),
        camera_names=list(camera_names),
        camera_height=image_size,
        camera_width=image_size,
        reward_shaping=False,
    )
    return env, env_meta


def sorted_demo_names(data_group: h5py.Group) -> list[str]:
    def key(name: str):
        if name.startswith("demo_"):
            suffix = name[len("demo_") :]
            if suffix.isdigit():
                return (0, int(suffix))
        return (1, name)

    return sorted(
        [name for name in data_group.keys() if name.startswith("demo_")],
        key=key,
    )


def copy_hdf5_attributes(
    source: h5py.AttributeManager,
    target: h5py.AttributeManager,
) -> None:
    for key, value in source.items():
        target[key] = value


def make_initial_state(
    demo_group: h5py.Group,
    require_model_file: bool = False,
) -> dict[str, Any]:
    if "states" not in demo_group:
        raise KeyError(f"{demo_group.name} has no states dataset")

    states = np.asarray(demo_group["states"])
    if states.shape[0] == 0:
        raise ValueError(f"{demo_group.name} has no simulator states")

    initial_state: dict[str, Any] = {"states": states[0]}

    if "model_file" in demo_group.attrs:
        initial_state["model"] = demo_group.attrs["model_file"]
    elif require_model_file:
        raise KeyError(
            f"{demo_group.name} has no model_file attribute; exact scene restoration "
            "cannot be guaranteed"
        )

    if "ep_meta" in demo_group.attrs:
        initial_state["ep_meta"] = demo_group.attrs["ep_meta"]

    return initial_state


def reset_to(env: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    observation = env.reset_to(dict(state))
    if observation is None:
        if hasattr(env, "get_observation"):
            observation = env.get_observation()
        else:
            raise RuntimeError(
                "env.reset_to returned None and the environment has no get_observation"
            )
    return observation


def reset_to_initial_state(env: Any, demo_group: h5py.Group) -> dict[str, Any]:
    return reset_to(env, make_initial_state(demo_group))


def step_env(env: Any, action: np.ndarray):
    result = env.step(action)

    if len(result) == 4:
        observation, reward, done, info = result
        return observation, float(reward), bool(done), info

    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), info

    raise RuntimeError(f"Unexpected env.step return length: {len(result)}")


def check_success(env: Any) -> bool:
    try:
        result = env.is_success()
        if isinstance(result, dict):
            return bool(result.get("task", False))
        return bool(result)
    except Exception:
        return False


def unwrap_robosuite_env(env: Any) -> Any:
    current = env
    for _ in range(12):
        if hasattr(current, "sim") and hasattr(current, "robots"):
            return current
        if hasattr(current, "env"):
            current = current.env
        else:
            break
    raise RuntimeError("Could not unwrap environment to raw Robosuite env")


def get_robot(env: Any) -> Any:
    raw_env = unwrap_robosuite_env(env)
    if not getattr(raw_env, "robots", None):
        raise RuntimeError("Robosuite environment contains no robot")
    return raw_env.robots[0]


def get_gripper(robot: Any) -> Any:
    gripper = robot.gripper
    if not isinstance(gripper, dict):
        return gripper
    if "right" in gripper:
        return gripper["right"]
    if len(gripper) == 1:
        return next(iter(gripper.values()))
    raise RuntimeError(f"Could not choose gripper from keys: {list(gripper)}")


def read_force_torque(
    env: Any,
    allow_missing_force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    raw_env = unwrap_robosuite_env(env)
    raw_env.sim.forward()

    robot = get_robot(raw_env)
    gripper = get_gripper(robot)

    try:
        sensors = getattr(gripper, "important_sensors")
        force = np.asarray(
            robot.get_sensor_measurement(sensors["force_ee"]),
            dtype=np.float32,
        ).reshape(-1)
        torque = np.asarray(
            robot.get_sensor_measurement(sensors["torque_ee"]),
            dtype=np.float32,
        ).reshape(-1)
    except Exception as sensor_error:
        force_attr = getattr(robot, "ee_force", None)
        torque_attr = getattr(robot, "ee_torque", None)

        if force_attr is not None and torque_attr is not None:
            force = np.asarray(force_attr, dtype=np.float32).reshape(-1)
            torque = np.asarray(torque_attr, dtype=np.float32).reshape(-1)
        elif allow_missing_force:
            force = np.zeros(3, dtype=np.float32)
            torque = np.zeros(3, dtype=np.float32)
        else:
            raise RuntimeError(
                "Could not read live end-effector force / torque. "
                "Do not silently collect zero-force training data."
            ) from sensor_error

    if force.shape != (3,) or torque.shape != (3,):
        raise ValueError(
            "Expected force and torque shapes (3,), got "
            f"force={force.shape}, torque={torque.shape}"
        )

    if not np.isfinite(force).all() or not np.isfinite(torque).all():
        raise ValueError(f"Non-finite force / torque: force={force}, torque={torque}")

    return force, torque


def add_force_to_observation(
    env: Any,
    observation: dict[str, Any],
    allow_missing_force: bool = False,
    force_key: str = DEFAULT_FORCE_KEY,
    torque_key: str = DEFAULT_TORQUE_KEY,
) -> dict[str, Any]:
    output = dict(observation)
    force, torque = read_force_torque(env, allow_missing_force=allow_missing_force)
    output[force_key] = force
    output[torque_key] = torque
    return output


# Compatibility names used by the task collectors.
add_live_force = add_force_to_observation
add_live_force_to_observation = add_force_to_observation


def get_state_vector(env: Any) -> np.ndarray:
    # Direct Robosuite collectors historically stored MuJoCo sim state. Preserve
    # that behaviour instead of preferring a higher-level env.get_state method.
    if hasattr(env, "sim") and hasattr(env, "robots"):
        state = env.sim.get_state()
        if hasattr(state, "flatten"):
            return np.asarray(state.flatten(), dtype=np.float64).reshape(-1)
        return np.asarray(state, dtype=np.float64).reshape(-1)

    if hasattr(env, "get_state"):
        try:
            state = env.get_state()
            if isinstance(state, dict):
                for key in ("states", "state"):
                    if key in state:
                        return np.asarray(state[key], dtype=np.float64).reshape(-1)
            elif state is not None:
                return np.asarray(state, dtype=np.float64).reshape(-1)
        except Exception:
            pass

    raw_env = unwrap_robosuite_env(env)
    state = raw_env.sim.get_state()
    if hasattr(state, "flatten"):
        return np.asarray(state.flatten(), dtype=np.float64).reshape(-1)
    return np.asarray(state, dtype=np.float64).reshape(-1)


def get_model_xml(env: Any) -> str | None:
    raw_env = unwrap_robosuite_env(env)
    candidates = (
        getattr(raw_env.sim.model, "get_xml", None),
        getattr(raw_env.sim.model, "get_model", None),
        getattr(raw_env.model, "get_xml", None),
    )

    for fn in candidates:
        if not callable(fn):
            continue
        try:
            value = fn()
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if isinstance(value, str) and value.strip():
                return value
        except Exception:
            continue

    return None


def image_to_hwc_uint8(
    image: Any,
    key: str,
    *,
    flip_vertical: bool = True,
) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image for {key}, got {image.shape}")

    if image.shape[-1] in (1, 3, 4):
        output = image
    elif image.shape[0] in (1, 3, 4):
        output = image.transpose(1, 2, 0)
    else:
        raise ValueError(f"Cannot infer image layout for {key}: {image.shape}")

    if output.shape[-1] == 4:
        output = output[..., :3]
    elif output.shape[-1] == 1:
        output = np.repeat(output, 3, axis=-1)

    if output.dtype != np.uint8:
        output = output.astype(np.float32, copy=False)
        if output.size and float(output.max()) <= 2.0:
            output = output * 255.0
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)

    if flip_vertical:
        output = np.flip(output, axis=0)

    return np.ascontiguousarray(output)


def image_to_chw_float(
    image: Any,
    key: str,
    *,
    flip_vertical: bool = False,
) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected single image for {key}, got {image.shape}")

    if image.shape[-1] in (1, 3, 4):
        output = image.transpose(2, 0, 1)
    elif image.shape[0] in (1, 3, 4):
        output = image
    else:
        raise ValueError(f"Cannot infer image layout for {key}: {image.shape}")

    if output.shape[0] == 4:
        output = output[:3]
    elif output.shape[0] == 1:
        output = np.repeat(output, 3, axis=0)

    if flip_vertical:
        output = np.flip(output, axis=1)

    output = np.ascontiguousarray(output).astype(np.float32, copy=False)
    if output.size and float(output.max()) > 2.0:
        output = output / 255.0

    return torch.from_numpy(output)


def observations_to_arrays(
    observations: Sequence[Mapping[str, Any]],
    *,
    required_keys: Sequence[str],
    camera_keys: Sequence[str],
    extras: Sequence[str] = (),
    int_keys: Sequence[str] = (),
    flip_images: bool = True,
) -> dict[str, np.ndarray]:
    camera_key_set = set(camera_keys)
    int_key_set = set(int_keys)
    arrays: dict[str, np.ndarray] = {}

    for key in tuple(required_keys) + tuple(extras):
        values = []
        for observation in observations:
            if key not in observation:
                raise KeyError(
                    f"Missing observation key {key!r}. "
                    f"Available={sorted(observation)}"
                )

            if key in camera_key_set:
                value = image_to_hwc_uint8(
                    observation[key],
                    key,
                    flip_vertical=flip_images,
                )
            else:
                dtype = np.int32 if key in int_key_set else np.float32
                value = np.asarray(observation[key], dtype=dtype).reshape(-1)

            values.append(value)

        arrays[key] = np.stack(values, axis=0)

    return arrays


def build_history_tensors(
    observation_history: Sequence[Mapping[str, Any]],
    state_keys: Sequence[str],
    camera_keys: Sequence[str],
    force_keys: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    states = []
    for obs in observation_history:
        parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in state_keys]
        states.append(np.concatenate(parts, axis=-1))
    state = torch.from_numpy(np.stack(states, axis=0)).unsqueeze(0)

    cameras = []
    for key in camera_keys:
        frames = [image_to_chw_float(obs[key], key) for obs in observation_history]
        cameras.append(torch.stack(frames, dim=0))
    images = torch.stack(cameras, dim=0).unsqueeze(0)

    result = {"state": state, "images": images}
    if force_keys:
        forces = []
        for obs in observation_history:
            parts = [np.asarray(obs[key], dtype=np.float32).reshape(-1) for key in force_keys]
            forces.append(np.concatenate(parts, axis=-1))
        result["force"] = torch.from_numpy(np.stack(forces, axis=0)).unsqueeze(0)

    return result


def create_test_mask(
    output: h5py.File,
    demo_names: Sequence[str],
) -> list[str]:
    if "mask" in output:
        del output["mask"]

    names = sorted(str(name) for name in demo_names)
    mask = output.create_group("mask")
    dtype = h5py.string_dtype(encoding="utf-8")
    mask.create_dataset(
        "test",
        data=np.asarray(names, dtype=object),
        dtype=dtype,
    )
    return names


def copy_filtered_masks(
    source: h5py.File,
    output: h5py.File,
    saved_demo_names: Iterable[str],
) -> None:
    if "mask" not in source:
        return

    saved = set(str(name) for name in saved_demo_names)
    if "mask" in output:
        del output["mask"]

    output_mask = output.create_group("mask")
    string_dtype = h5py.string_dtype(encoding="utf-8")

    for mask_name, node in source["mask"].items():
        raw_names = np.asarray(node[()]).reshape(-1)
        decoded_names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_names
        ]
        selected = [name for name in decoded_names if name in saved]
        output_mask.create_dataset(
            mask_name,
            data=np.asarray(selected, dtype=object),
            dtype=string_dtype,
        )


def write_simple_train_valid_masks(
    output: h5py.File,
    demo_names: Sequence[str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    names = sorted(str(name) for name in demo_names)
    if len(names) < 2:
        raise ValueError(
            "At least two demonstrations are required to create non-empty train/valid masks"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(names, dtype=object)
    rng.shuffle(shuffled)

    n_valid = int(round(len(names) * val_ratio))
    n_valid = min(max(n_valid, 1), len(names) - 1)

    valid_names = sorted(str(name) for name in shuffled[:n_valid])
    train_names = sorted(str(name) for name in shuffled[n_valid:])

    if "mask" in output:
        del output["mask"]

    mask = output.create_group("mask")
    dtype = h5py.string_dtype(encoding="utf-8")
    mask.create_dataset("train", data=np.asarray(train_names, dtype=object), dtype=dtype)
    mask.create_dataset("valid", data=np.asarray(valid_names, dtype=object), dtype=dtype)
    return train_names, valid_names


def ensure_train_valid_masks(
    output: h5py.File,
    demo_names: Sequence[str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Ensure Force2Act-compatible non-empty /mask/train and /mask/valid.

    Existing masks are preserved only when both are present, non-empty, disjoint,
    and point to saved demos. Otherwise a deterministic fallback split is written.
    """

    saved = set(str(name) for name in demo_names)

    if "mask" in output and "train" in output["mask"] and "valid" in output["mask"]:
        def decode(node) -> list[str]:
            values = np.asarray(node[()]).reshape(-1)
            return [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in values
            ]

        train_names = decode(output["mask"]["train"])
        valid_names = decode(output["mask"]["valid"])

        if (
            train_names
            and valid_names
            and not (set(train_names) & set(valid_names))
            and set(train_names).issubset(saved)
            and set(valid_names).issubset(saved)
        ):
            return sorted(train_names), sorted(valid_names)

    return write_simple_train_valid_masks(
        output,
        demo_names,
        val_ratio=val_ratio,
        seed=seed,
    )


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
