from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
from tqdm import tqdm

from sim_collect.common import (
    add_live_force_to_observation,
    check_success as is_environment_successful,
    copy_filtered_masks,
    copy_hdf5_attributes,
    ensure_train_valid_masks,
    create_test_mask,
    get_state_vector as get_live_state_vector,
    image_to_hwc_uint8 as _common_image_to_hwc_uint8,
    import_robomimic,
    initialize_obs_utils,
    read_force_torque,
    reset_to_initial_state,
    sorted_demo_names,
    step_env as step_environment,
)



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






def create_replay_environment(
    dataset_path: str,
    image_size: int,
):
    """
    Build the exact NutAssemblySquare task and controller described by the source HDF5.

    The minimal argument list is intentional: it works with older Robomimic
    versions that do not accept render / use_image_obs / use_depth_obs.
    """

    EnvUtils, FileUtils, _ = import_robomimic()

    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path=dataset_path,
    )

    env_name = str(env_meta.get("env_name", ""))
    valid_names = {
        "NutAssemblySquare",
    }

    if env_name not in valid_names:
        raise ValueError(
            "Expected a NutAssemblySquare source dataset, but the metadata says "
            f"env_name={env_name!r}. Use a Robomimic NutAssemblySquare source dataset."
        )

    print(f"Creating continuous replay environment for: {env_name}")

    env = EnvUtils.create_env_for_data_processing(
        env_meta=deepcopy(env_meta),
        camera_names=list(CAMERA_NAMES),
        camera_height=image_size,
        camera_width=image_size,
        reward_shaping=False,
    )

    return env, env_meta



























def image_to_hwc_uint8(image: Any, key: str) -> np.ndarray:
    return _common_image_to_hwc_uint8(
        image,
        key,
        flip_vertical=False,
    )


def observations_to_arrays(
    observations: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    keys = STATE_KEYS + FORCE_KEYS + CAMERA_KEYS
    arrays: dict[str, np.ndarray] = {}

    for key in keys:
        values = []

        for observation in observations:
            if key not in observation:
                raise KeyError(
                    f"Missing observation key {key!r}. Available keys: "
                    f"{sorted(observation.keys())}"
                )

            if key in CAMERA_KEYS:
                value = image_to_hwc_uint8(observation[key], key)
            else:
                value = np.asarray(observation[key], dtype=np.float32).reshape(-1)

            values.append(value)

        arrays[key] = np.stack(values, axis=0)

    return arrays


def show_preview(
    observation: dict[str, Any],
    demo_name: str,
    timestep: int,
    state_rmse: float,
    delay_ms: int,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "--render requires opencv-python: pip install opencv-python"
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

    preview = np.concatenate((agent, hand), axis=1)
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)

    force = np.asarray(observation[FORCE_KEYS[0]], dtype=np.float32)
    force_norm = float(np.linalg.norm(force))

    cv2.putText(
        preview,
        f"{demo_name} | t={timestep} | drift={state_rmse:.5f} | |F|={force_norm:.2f}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("NutAssemblySquare continuous replay collection", preview)

    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    if key == ord("q"):
        raise KeyboardInterrupt("Collection stopped by user.")


def source_demo_has_final_success(
    demo_group: h5py.Group,
    success_reward: float,
    atol: float,
) -> bool:
    """
    Strict success-only prefilter.

    A source demo is replayed only when its final source reward equals
    --success-reward (default 1.0). After replay, the live replay must also
    finish with the same successful final reward or it is discarded.
    """
    if "rewards" not in demo_group:
        raise KeyError(
            f"{demo_group.name} has no rewards dataset, so strict success "
            "filtering cannot be applied."
        )

    rewards = np.asarray(demo_group["rewards"], dtype=np.float32).reshape(-1)
    if rewards.size == 0:
        return False

    return bool(
        np.isclose(
            float(rewards[-1]),
            success_reward,
            atol=atol,
            rtol=0.0,
        )
    )


def replay_demo(
    env: Any,
    demo_group: h5py.Group,
    demo_name: str,
    save_next_obs: bool,
    render: bool,
    render_delay_ms: int,
    success_reward: float,
    reward_atol: float,
    max_state_rmse: float | None,
    allow_missing_force: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Replay the source Robomimic actions continuously with env.step().

    The environment is reset only once at the beginning of the demo. Every
    observation afterwards is produced by the live replay trajectory.
    """

    required = ("states", "actions", "rewards")
    missing = [key for key in required if key not in demo_group]
    if missing:
        return None, f"missing datasets: {missing}"

    source_states = np.asarray(demo_group["states"], dtype=np.float64)
    actions = np.asarray(demo_group["actions"], dtype=np.float32)
    source_rewards = np.asarray(demo_group["rewards"], dtype=np.float32).reshape(-1)
    source_dones = (
        np.asarray(demo_group["dones"], dtype=np.int64).reshape(-1)
        if "dones" in demo_group
        else np.zeros(actions.shape[0], dtype=np.int64)
    )

    if actions.ndim != 2 or actions.shape[-1] != EXPECTED_ACTION_DIM:
        return None, f"expected actions [T, 7], got {actions.shape}"

    if source_states.shape[0] != actions.shape[0]:
        return None, (
            "states/actions length mismatch: "
            f"{source_states.shape[0]} vs {actions.shape[0]}"
        )

    if source_rewards.shape[0] != actions.shape[0]:
        return None, (
            "rewards/actions length mismatch: "
            f"{source_rewards.shape[0]} vs {actions.shape[0]}"
        )

    observation = reset_to_initial_state(
        env=env,
        demo_group=demo_group,
    )

    observations: list[dict[str, Any]] = []
    next_observations: list[dict[str, Any]] = []
    replay_states: list[np.ndarray] = []
    replay_next_states: list[np.ndarray] = []
    replay_rewards: list[float] = []
    replay_dones: list[int] = []
    state_rmse_to_source: list[float] = []
    state_max_abs_to_source: list[float] = []

    for timestep, action in enumerate(actions):
        observation = add_live_force_to_observation(
            env=env,
            observation=observation,
            allow_missing_force=allow_missing_force,
        )

        live_state = get_live_state_vector(env)
        source_state = np.asarray(source_states[timestep], dtype=np.float64).reshape(-1)

        if live_state.shape != source_state.shape:
            return None, (
                "live/source state shape mismatch: "
                f"{live_state.shape} vs {source_state.shape}"
            )

        difference = live_state - source_state
        state_rmse = float(np.sqrt(np.mean(np.square(difference))))
        state_max_abs = float(np.max(np.abs(difference)))

        if max_state_rmse is not None and state_rmse > max_state_rmse:
            return None, (
                f"state replay drift exceeded threshold at t={timestep}: "
                f"rmse={state_rmse:.6f} > {max_state_rmse:.6f}"
            )

        observations.append(deepcopy(observation))
        replay_states.append(live_state.astype(np.float64, copy=True))
        state_rmse_to_source.append(state_rmse)
        state_max_abs_to_source.append(state_max_abs)

        if render:
            show_preview(
                observation=observation,
                demo_name=demo_name,
                timestep=timestep,
                state_rmse=state_rmse,
                delay_ms=render_delay_ms,
            )

        next_observation, reward, done, _ = step_environment(
            env=env,
            action=action,
        )

        next_observation = add_live_force_to_observation(
            env=env,
            observation=next_observation,
            allow_missing_force=allow_missing_force,
        )

        next_state = get_live_state_vector(env)

        if save_next_obs:
            next_observations.append(deepcopy(next_observation))

        replay_next_states.append(next_state.astype(np.float64, copy=True))
        replay_rewards.append(float(reward))
        replay_dones.append(int(done))

        observation = next_observation

    final_reward = float(replay_rewards[-1])
    final_reward_success = bool(
        np.isclose(
            final_reward,
            success_reward,
            atol=reward_atol,
            rtol=0.0,
        )
    )
    final_env_success = is_environment_successful(env)

    # The user's requested rule is strict final reward == 1.0. The environment
    # success flag is stored for diagnostics but does not replace this rule.
    if not final_reward_success:
        return None, (
            f"replayed final reward={final_reward:.6f}, expected "
            f"{success_reward:.6f}; env_success={final_env_success}"
        )

    trajectory: dict[str, Any] = {
        "states": np.stack(replay_states, axis=0),
        "next_states": np.stack(replay_next_states, axis=0),
        "source_states": source_states,
        "actions": actions,
        "rewards": np.asarray(replay_rewards, dtype=np.float32),
        "dones": np.asarray(replay_dones, dtype=np.int64),
        "source_rewards": source_rewards,
        "source_dones": source_dones,
        "state_rmse_to_source": np.asarray(state_rmse_to_source, dtype=np.float32),
        "state_max_abs_to_source": np.asarray(
            state_max_abs_to_source,
            dtype=np.float32,
        ),
        "obs": observations_to_arrays(observations),
        "final_env_success": final_env_success,
    }

    if save_next_obs:
        trajectory["next_obs"] = observations_to_arrays(next_observations)

    return trajectory, None


def write_demo(
    source_file: h5py.File,
    source_demo: h5py.Group,
    target_demo: h5py.Group,
    trajectory: dict[str, Any],
    save_next_obs: bool,
) -> int:
    standard_datasets = (
        "states",
        "actions",
        "rewards",
        "dones",
    )

    diagnostic_datasets = (
        "next_states",
        "source_states",
        "source_rewards",
        "source_dones",
        "state_rmse_to_source",
        "state_max_abs_to_source",
    )

    for key in standard_datasets + diagnostic_datasets:
        target_demo.create_dataset(
            key,
            data=trajectory[key],
        )

    if "actions_abs" in source_demo:
        target_demo.create_dataset(
            "actions_abs",
            data=source_demo["actions_abs"][()],
        )

    if "action_dict" in source_demo:
        source_file.copy(
            source_demo["action_dict"],
            target_demo,
            name="action_dict",
        )

    observation_group = target_demo.create_group("obs")

    for key, value in trajectory["obs"].items():
        observation_group.create_dataset(
            key,
            data=value,
            compression="gzip" if key in CAMERA_KEYS else None,
        )

    if save_next_obs:
        next_observation_group = target_demo.create_group("next_obs")

        for key, value in trajectory["next_obs"].items():
            next_observation_group.create_dataset(
                key,
                data=value,
                compression="gzip" if key in CAMERA_KEYS else None,
            )

    copy_hdf5_attributes(
        source=source_demo.attrs,
        target=target_demo.attrs,
    )

    number_of_samples = int(trajectory["actions"].shape[0])
    state_rmse = trajectory["state_rmse_to_source"]
    state_max_abs = trajectory["state_max_abs_to_source"]

    target_demo.attrs["num_samples"] = number_of_samples
    target_demo.attrs["collection_mode"] = "continuous_env_step_replay"
    target_demo.attrs["source_demo"] = source_demo.name.split("/")[-1]
    target_demo.attrs["final_replay_reward"] = float(trajectory["rewards"][-1])
    target_demo.attrs["final_env_success"] = bool(trajectory["final_env_success"])
    target_demo.attrs["mean_state_rmse_to_source"] = float(np.mean(state_rmse))
    target_demo.attrs["max_state_rmse_to_source"] = float(np.max(state_rmse))
    target_demo.attrs["max_state_abs_error_to_source"] = float(np.max(state_max_abs))

    return number_of_samples





def load_excluded_source_demos(
    paths: list[str] | None,
) -> set[str]:
    """
    Read source_demo identities from previously collected force datasets.

    This is the strongest leakage check for replay-based NutAssemblySquare:
    final test demos must originate from Robomimic source demonstrations that
    never appeared in train or validation.
    """
    excluded: set[str] = set()
    for raw_path in paths or []:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--exclude-hdf5 does not exist: {path}")

        with h5py.File(path, "r") as hdf5_file:
            if "data" not in hdf5_file:
                raise KeyError(f"{path} has no /data group")
            for demo_name, demo in hdf5_file["data"].items():
                source_demo = demo.attrs.get("source_demo")
                if source_demo is None:
                    # Older replay files preserved source names as output names.
                    source_demo = demo_name
                if isinstance(source_demo, bytes):
                    source_demo = source_demo.decode("utf-8")
                excluded.add(str(source_demo))
    return excluded




def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously replay successful Robomimic NutAssemblySquare actions in "
            "Robosuite with env.step(), and save live images, state, force, "
            "torque, rewards, and actions."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Original Robomimic NutAssemblySquare low_dim HDF5.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output continuous-replay HDF5.",
    )
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--num-demos",
        type=int,
        default=None,
        help=(
            "Number of replay-successful demos to save. The collector keeps "
            "trying source demos until this number is reached."
        ),
    )
    parser.add_argument(
        "--exclude-next-obs",
        action="store_true",
        help="Do not store next_obs. env.step is still used for every action.",
    )
    parser.add_argument("--success-reward", type=float, default=1.0)
    parser.add_argument("--reward-atol", type=float, default=1e-6)
    parser.add_argument(
        "--max-state-rmse",
        type=float,
        default=None,
        help=(
            "Optional maximum replay-state RMSE from the source state. "
            "A demo is skipped as soon as it exceeds this value."
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Preview agentview and eye-in-hand images. Press q to stop.",
    )
    parser.add_argument("--render-delay-ms", type=int, default=20)
    parser.add_argument(
        "--allow-missing-force",
        action="store_true",
        help=(
            "Debug only: save zero force if sensors cannot be read. By default "
            "the collector fails loudly instead of producing fake force data."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Write all saved demos to /mask/test for final evaluation.",
    )
    parser.add_argument(
        "--exclude-hdf5",
        nargs="*",
        default=None,
        help=(
            "Existing collected NutAssemblySquare HDF5 files. Any source_demo "
            "already present in them is skipped."
        ),
    )

    args = parser.parse_args()

    source_path = Path(args.source).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    excluded_source_demos = load_excluded_source_demos(
        args.exclude_hdf5
    )
    if args.test_only:
        print(
            f"FINAL TEST MODE: excluding "
            f"{len(excluded_source_demos)} previously used source demos."
        )

    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    if source_path == output_path:
        raise ValueError("Source and output paths must be different.")

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Add --overwrite to replace it."
        )

    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16.")

    if args.reward_atol < 0.0:
        raise ValueError("--reward-atol cannot be negative.")

    if args.max_state_rmse is not None and args.max_state_rmse < 0.0:
        raise ValueError("--max-state-rmse cannot be negative.")

    if args.num_demos is not None and args.num_demos <= 0:
        raise ValueError("--num-demos must be greater than zero.")

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1).")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    initialize_obs_utils(STATE_KEYS, CAMERA_KEYS)
    env, env_meta = create_replay_environment(
        dataset_path=str(source_path),
        image_size=args.image_size,
    )

    save_next_obs = not args.exclude_next_obs
    saved_demo_names: list[str] = []
    skipped_reasons: list[tuple[str, str]] = []
    total_samples = 0
    attempted = 0
    source_successful = 0

    try:
        with h5py.File(source_path, "r") as source:
            if "data" not in source:
                raise KeyError("Source HDF5 has no 'data' group.")

            all_demo_names = sorted_demo_names(source["data"])
            if not all_demo_names:
                raise RuntimeError("No demo_* groups were found.")

            with h5py.File(output_path, "w") as output:
                copy_hdf5_attributes(source.attrs, output.attrs)

                output_data = output.create_group("data")
                copy_hdf5_attributes(source["data"].attrs, output_data.attrs)
                output_data.attrs["dataset_role"] = (
                    "final_test" if args.test_only else "train_validation"
                )
                output_data.attrs["non_repeat_test_enforced"] = bool(args.test_only)
                output_data.attrs["excluded_source_demo_count"] = int(
                    len(excluded_source_demos)
                )

                progress = tqdm(
                    all_demo_names,
                    desc="Continuous NutAssemblySquare replay",
                )

                for demo_name in progress:
                    if (
                        args.num_demos is not None
                        and len(saved_demo_names) >= args.num_demos
                    ):
                        break

                    source_demo = source["data"][demo_name]

                    if args.test_only and demo_name in excluded_source_demos:
                        skipped_reasons.append(
                            (demo_name, "excluded: source demo already used in train/valid")
                        )
                        continue

                    if not source_demo_has_final_success(
                        demo_group=source_demo,
                        success_reward=args.success_reward,
                        atol=args.reward_atol,
                    ):
                        skipped_reasons.append(
                            (demo_name, "source final reward is not successful")
                        )
                        continue

                    source_successful += 1
                    attempted += 1

                    trajectory, skip_reason = replay_demo(
                        env=env,
                        demo_group=source_demo,
                        demo_name=demo_name,
                        save_next_obs=save_next_obs,
                        render=args.render,
                        render_delay_ms=args.render_delay_ms,
                        success_reward=args.success_reward,
                        reward_atol=args.reward_atol,
                        max_state_rmse=args.max_state_rmse,
                        allow_missing_force=args.allow_missing_force,
                    )

                    if trajectory is None:
                        skipped_reasons.append(
                            (demo_name, skip_reason or "unknown replay failure")
                        )
                        progress.set_postfix(
                            saved=len(saved_demo_names),
                            replay_failed=len(skipped_reasons),
                        )
                        continue

                    target_demo = output_data.create_group(demo_name)
                    total_samples += write_demo(
                        source_file=source,
                        source_demo=source_demo,
                        target_demo=target_demo,
                        trajectory=trajectory,
                        save_next_obs=save_next_obs,
                    )

                    saved_demo_names.append(demo_name)
                    progress.set_postfix(
                        saved=len(saved_demo_names),
                        final_reward=f"{trajectory['rewards'][-1]:.1f}",
                        max_drift=(
                            f"{np.max(trajectory['state_rmse_to_source']):.5f}"
                        ),
                    )

                if not saved_demo_names:
                    raise RuntimeError(
                        "No demos remained successful after continuous env.step "
                        "replay. The output file is not suitable for training."
                    )

                if args.test_only:
                    create_test_mask(
                        output,
                        saved_demo_names,
                    )
                else:
                    copy_filtered_masks(
                        source=source,
                        output=output,
                        saved_demo_names=saved_demo_names,
                    )
                    ensure_train_valid_masks(
                        output,
                        saved_demo_names,
                        val_ratio=args.val_ratio,
                        seed=args.split_seed,
                    )

                serialized_env_meta = json.dumps(env_meta, indent=4)
                output_data.attrs["env_args"] = serialized_env_meta
                output.attrs["env_args"] = serialized_env_meta

                output_data.attrs["total"] = int(total_samples)
                output_data.attrs["collection_mode"] = "continuous_env_step_replay"
                output_data.attrs["force_source"] = "live_robosuite_gripper_sensor"
                output_data.attrs["success_only"] = True
                output_data.attrs["success_reward"] = float(args.success_reward)
                output_data.attrs["reward_atol"] = float(args.reward_atol)
                output_data.attrs["num_source_demos"] = int(len(all_demo_names))
                output_data.attrs["num_source_successful_demos"] = int(
                    source_successful
                )
                output_data.attrs["num_replay_attempted_demos"] = int(attempted)
                output_data.attrs["num_saved_demos"] = int(len(saved_demo_names))
                output_data.attrs["save_next_obs"] = bool(save_next_obs)

                if args.max_state_rmse is not None:
                    output_data.attrs["max_allowed_state_rmse"] = float(
                        args.max_state_rmse
                    )

                string_dtype = h5py.string_dtype(encoding="utf-8")
                report_group = output.create_group("collection_report")
                report_group.create_dataset(
                    "skipped_demo_names",
                    data=np.asarray(
                        [name for name, _ in skipped_reasons],
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )
                report_group.create_dataset(
                    "skipped_reasons",
                    data=np.asarray(
                        [reason for _, reason in skipped_reasons],
                        dtype=object,
                    ),
                    dtype=string_dtype,
                )

        print("\nCollection complete")
        print(f"  Output: {output_path}")
        print(f"  Saved successful replays: {len(saved_demo_names)}")
        print(f"  Saved transitions: {total_samples}")
        print(f"  Skipped demos: {len(skipped_reasons)}")
        print("  Replay mode: reset once, then env.step(action) for every action")
        print("  State: robot0_eef_pos + robot0_eef_quat + robot0_gripper_qpos")
        print("  Force: robot0_eef_force + robot0_eef_torque")
        print("  Images: agentview_image + robot0_eye_in_hand_image")
        print("  Actions: original Robomimic 7D actions")

        if skipped_reasons:
            print("\nFirst skipped demos:")
            for demo_name, reason in skipped_reasons[:10]:
                print(f"  {demo_name}: {reason}")

    except Exception:
        # Do not leave a misleading zero-demo dataset after an unsuccessful run.
        if output_path.exists():
            try:
                with h5py.File(output_path, "r") as check_file:
                    number_saved = int(
                        check_file.get("data", {}).attrs.get("num_saved_demos", 0)
                    )
                if number_saved == 0:
                    output_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise

    finally:
        if hasattr(env, "close"):
            env.close()

        if args.render:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                pass


if __name__ == "__main__":
    main()
