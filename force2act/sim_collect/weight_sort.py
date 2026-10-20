from __future__ import annotations

import argparse
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from sim_collect.common import (
    add_live_force,
    create_test_mask,
    get_gripper,
    get_model_xml,
    get_robot,
    get_state_vector,
    image_to_hwc_uint8,
    json_default,
    read_force_torque,
)



STATE_KEYS = (
    "robot0_eef_pos",
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
    APPROACH = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    WEIGH = 4
    TRANSIT = 5
    LOWER = 6
    OPEN = 7
    RETREAT = 8
    DONE = 9
    FAILED = 10


@dataclass
class EpisodeSpec:
    pair_id: int
    true_class: str
    mass: float
    start_pos: np.ndarray
    start_yaw: float
    lift_height: float
    light_target: np.ndarray
    heavy_target: np.ndarray


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
        raise ImportError(
            "robosuite is required. Install a compatible robosuite version first."
        ) from exc
    return suite, load_composite_controller_config


def create_environment(
    image_size: int,
    horizon: int,
    control_freq: int,
    seed: int,
):
    suite, load_composite_controller_config = import_robosuite()
    controller_config = load_composite_controller_config(controller="BASIC")

    # Robosuite versions differ here: some releases do not accept a ``seed``
    # keyword in PickPlace.__init__. Seed NumPy before construction instead.
    # The collector also uses its own np.random.Generator for all explicit
    # object-position and mass randomisation, so those remain deterministic.
    np.random.seed(seed)

    env_kwargs = {
        "robots": "Panda",
        "controller_configs": controller_config,
        "gripper_types": "default",
        # A fixed robot reset prevents the randomly sampled initial joint pose
        # from becoming accidentally correlated with the weight class. Scene
        # diversity still comes from paired object position, yaw, mass, and lift
        # height randomisation.
        "initialization_noise": None,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "use_object_obs": True,
        "camera_names": list(CAMERA_NAMES),
        "camera_heights": image_size,
        "camera_widths": image_size,
        "camera_depths": False,
        "reward_shaping": False,
        "control_freq": control_freq,
        "horizon": horizon,
        "ignore_done": True,
        "hard_reset": False,
    }

    env = suite.make("PickPlaceCan", **env_kwargs)

    # Older versions may expose an optional seed method even though the
    # constructor does not accept ``seed``. Use it when available, but do not
    # require it for compatibility.
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        try:
            seed_fn(seed)
        except (TypeError, NotImplementedError):
            pass

    low, high = env.action_spec
    if np.asarray(low).shape != (EXPECTED_ACTION_DIM,):
        raise ValueError(
            "Expected a 7D OSC action [dx,dy,dz,droll,dpitch,dyaw,grip], "
            f"but env.action_spec is {np.asarray(low).shape}."
        )

    env_meta = {
        "env_name": "PickPlaceCan",
        "type": 1,  # Robomimic EnvType.ROBOSUITE_TYPE
        "env_version": getattr(suite, "__version__", "unknown"),
        "env_kwargs": env_kwargs,
        "weight_sort_task": {
            "light_destination": "bin2 lower-left quadrant",
            "heavy_destination": "bin2 upper-right quadrant",
            "classification_source": "live end-effector force delta after lift",
            "collector_seed": int(seed),
        },
    }
    return env, env_meta










def active_object(env: Any) -> tuple[Any, int, str]:
    name = getattr(env, "obj_to_use", None) or "Can"
    object_model = None
    for obj in env.objects:
        if obj.name == name or obj.name.lower() == str(name).lower():
            object_model = obj
            break
    if object_model is None:
        for obj in env.objects:
            if "can" in obj.name.lower():
                object_model = obj
                break
    if object_model is None:
        raise RuntimeError(f"Could not find active can object. Objects: {[o.name for o in env.objects]}")

    body_id = int(env.obj_body_id[object_model.name])
    joint_name = str(object_model.joints[0])
    return object_model, body_id, joint_name


def get_object_pos(env: Any, body_id: int) -> np.ndarray:
    return np.asarray(env.sim.data.body_xpos[body_id], dtype=np.float64).copy()


def get_eef_pos(obs: dict[str, Any], env: Any) -> np.ndarray:
    if "robot0_eef_pos" in obs:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
    robot = get_robot(env)
    eef_site = robot.eef_site_id
    if isinstance(eef_site, dict):
        eef_site = next(iter(eef_site.values()))
    return np.asarray(env.sim.data.site_xpos[int(eef_site)], dtype=np.float64).copy()






def set_object_pose(
    env: Any,
    joint_name: str,
    xy: np.ndarray,
    yaw: float,
) -> np.ndarray:
    qpos = np.asarray(
        env.sim.data.get_joint_qpos(joint_name),
        dtype=np.float64,
    ).reshape(-1).copy()
    if qpos.shape != (7,):
        raise ValueError(f"Expected free-joint qpos shape (7,), got {qpos.shape}")

    qpos[0:2] = np.asarray(xy, dtype=np.float64)

    # MuJoCo free-joint quaternion ordering is [w, x, y, z]. Randomise yaw once
    # per pair and reuse the exact same value for its light and heavy episodes.
    half_yaw = 0.5 * float(yaw)
    qpos[3:7] = np.array(
        [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)],
        dtype=np.float64,
    )

    env.sim.data.set_joint_qpos(joint_name, qpos)
    try:
        env.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=np.float64))
    except (AttributeError, KeyError, ValueError):
        # Older wrappers may not expose set_joint_qvel by joint name. The
        # subsequent settling steps still remove residual motion.
        pass
    env.sim.forward()
    return qpos[:3].copy()


def set_object_mass(
    env: Any,
    body_id: int,
    target_mass: float,
    base_mass: float,
    base_inertia: np.ndarray,
) -> None:
    if target_mass <= 0.0:
        raise ValueError("Object mass must be positive.")
    scale = target_mass / base_mass
    env.sim.model.body_mass[body_id] = float(target_mass)
    env.sim.model.body_inertia[body_id] = np.asarray(base_inertia, dtype=np.float64) * scale
    env.sim.forward()




def make_observation(
    env: Any,
    obs: dict[str, Any],
    allow_missing_force: bool,
    body_id: int,
    light_target: np.ndarray,
    heavy_target: np.ndarray,
    stage: Stage,
) -> dict[str, Any]:
    output = add_live_force(env, obs, allow_missing_force)
    output["object_pos"] = get_object_pos(env, body_id).astype(np.float32)
    output["light_target_pos"] = np.asarray(light_target, dtype=np.float32)
    output["heavy_target_pos"] = np.asarray(heavy_target, dtype=np.float32)
    output["fsm_stage"] = np.asarray([int(stage)], dtype=np.int32)
    return output


def observations_to_arrays(observations: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    required = STATE_KEYS + FORCE_KEYS + CAMERA_KEYS
    extras = (
        "object_pos",
        "light_target_pos",
        "heavy_target_pos",
        "fsm_stage",
    )
    arrays: dict[str, np.ndarray] = {}
    for key in required + extras:
        values = []
        for obs in observations:
            if key not in obs:
                raise KeyError(f"Missing observation key {key!r}. Available: {sorted(obs.keys())}")
            if key in CAMERA_KEYS:
                value = image_to_hwc_uint8(obs[key], key)
            else:
                dtype = np.int32 if key == "fsm_stage" else np.float32
                value = np.asarray(obs[key], dtype=dtype).reshape(-1)
            values.append(value)
        arrays[key] = np.stack(values, axis=0)
    return arrays


def position_action(
    current_pos: np.ndarray,
    target_pos: np.ndarray,
    gripper: float,
    controller_step: float,
) -> np.ndarray:
    delta = np.asarray(target_pos, dtype=np.float64) - np.asarray(current_pos, dtype=np.float64)
    action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
    action[:3] = np.clip(delta / controller_step, -1.0, 1.0).astype(np.float32)
    action[3:6] = 0.0
    action[6] = float(np.clip(gripper, -1.0, 1.0))
    return action


def no_op_action(gripper: float) -> np.ndarray:
    action = np.zeros(EXPECTED_ACTION_DIM, dtype=np.float32)
    action[-1] = float(gripper)
    return action



def target_centers(env: Any, object_z: float) -> tuple[np.ndarray, np.ndarray]:
    bin2 = np.asarray(env.bin2_pos, dtype=np.float64)
    bin_size = np.asarray(env.bin_size, dtype=np.float64)
    # Two fixed quadrants. Keeping the mapping fixed lets the policy infer the
    # destination from force instead of from an explicit class label.
    light = np.array(
        [bin2[0] - bin_size[0] / 4.0, bin2[1] - bin_size[1] / 4.0, object_z],
        dtype=np.float64,
    )
    heavy = np.array(
        [bin2[0] + bin_size[0] / 4.0, bin2[1] + bin_size[1] / 4.0, object_z],
        dtype=np.float64,
    )
    return light, heavy


def sample_start_xy(env: Any, rng: np.random.Generator, margin: float) -> np.ndarray:
    bin1 = np.asarray(env.bin1_pos, dtype=np.float64)
    bin_size = np.asarray(env.bin_size, dtype=np.float64)
    half_x = max(0.02, bin_size[0] / 2.0 - margin)
    half_y = max(0.02, bin_size[1] / 2.0 - margin)
    return np.array(
        [
            rng.uniform(bin1[0] - half_x, bin1[0] + half_x),
            rng.uniform(bin1[1] - half_y, bin1[1] + half_y),
        ],
        dtype=np.float64,
    )


def choose_true_class(saved_counts: dict[str, int], rng: np.random.Generator) -> str:
    light = int(saved_counts.get("light", 0))
    heavy = int(saved_counts.get("heavy", 0))
    if light < heavy:
        return "light"
    if heavy < light:
        return "heavy"
    return str(rng.choice(["light", "heavy"]))


def show_preview(
    obs: dict[str, Any],
    stage: Stage,
    true_class: str,
    predicted_class: str | None,
    mass: float,
    force_signal: float | None,
    delay_ms: int,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("--render requires opencv-python") from exc

    agent = image_to_hwc_uint8(obs[CAMERA_KEYS[0]], CAMERA_KEYS[0])
    hand = image_to_hwc_uint8(obs[CAMERA_KEYS[1]], CAMERA_KEYS[1])
    if hand.shape[:2] != agent.shape[:2]:
        hand = cv2.resize(hand, (agent.shape[1], agent.shape[0]), interpolation=cv2.INTER_AREA)
    preview = np.concatenate((agent, hand), axis=1)
    preview = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
    text = (
        f"stage={stage.name} true={true_class} pred={predicted_class or '-'} "
        f"mass={mass:.3f}kg signal={force_signal if force_signal is not None else 0.0:.3f}N"
    )
    cv2.putText(preview, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Weight-sort PickPlace FSM", preview)
    key = cv2.waitKey(max(1, delay_ms)) & 0xFF
    if key == ord("q"):
        raise KeyboardInterrupt("Collection stopped by user.")


def inside_target(
    object_pos: np.ndarray,
    target: np.ndarray,
    env: Any,
    xy_fraction: float = 0.42,
) -> bool:
    bin_size = np.asarray(env.bin_size, dtype=np.float64)
    half_x = bin_size[0] * 0.25 * xy_fraction * 2.0
    half_y = bin_size[1] * 0.25 * xy_fraction * 2.0
    return bool(
        abs(float(object_pos[0] - target[0])) <= half_x
        and abs(float(object_pos[1] - target[1])) <= half_y
        and float(object_pos[2]) < float(env.bin2_pos[2] + 0.25)
    )


def run_fsm_episode(
    env: Any,
    episode_spec: EpisodeSpec,
    body_id: int,
    joint_name: str,
    args: argparse.Namespace,
) -> EpisodeResult:
    obs = env.reset()
    set_object_mass(
        env,
        body_id=body_id,
        target_mass=episode_spec.mass,
        base_mass=args._base_mass,
        base_inertia=args._base_inertia,
    )
    actual_start = set_object_pose(
        env,
        joint_name,
        episode_spec.start_pos[:2],
        yaw=episode_spec.start_yaw,
    )

    # Let the object and robot settle before recording the demonstration.
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(no_op_action(gripper=-1.0))

    object_pos0 = get_object_pos(env, body_id)
    light_target, heavy_target = target_centers(env, object_pos0[2])
    episode_spec.light_target[:] = light_target
    episode_spec.heavy_target[:] = heavy_target

    approach_target = np.array(
        [
            object_pos0[0],
            object_pos0[1],
            object_pos0[2] + args.approach_height,
        ],
        dtype=np.float64,
    )
    grasp_target = np.array(
        [
            object_pos0[0],
            object_pos0[1],
            object_pos0[2] + args.grasp_z_offset,
        ],
        dtype=np.float64,
    )
    lift_target = np.array(
        [
            object_pos0[0],
            object_pos0[1],
            object_pos0[2] + episode_spec.lift_height,
        ],
        dtype=np.float64,
    )

    # Use a fixed number of closed-loop lift steps for every episode. Unlike the
    # previous open-loop schedule, this actively drives both light and heavy
    # objects to the same pair-level target pose before classification.
    lift_total_steps = args.closed_loop_lift_steps

    # Keep the stationary post-lift interval shorter than the policy execution
    # horizon. The force-sampling frames are ALSO used as the pose-guard frames,
    # rather than appending a second long no-motion interval. With the defaults,
    # WEIGH lasts only 3 control steps, so an EXEC=4 rollout reaches TRANSIT
    # within the same executed action chunk instead of repeatedly replanning
    # from an identical held-object observation.
    force_window_end = (
        args.weigh_settle_steps
        + args.weigh_samples
    )
    weigh_total_steps = max(
        force_window_end,
        args.preclass_guard_steps,
    )

    stage = Stage.APPROACH
    stage_steps = 0
    stable_steps = 0
    predicted_class: str | None = None
    force_signal: float | None = None
    baseline_samples: list[np.ndarray] = []
    weigh_samples: list[np.ndarray] = []
    preclass_guard_pose_errors: list[float] = []

    observations: list[dict[str, Any]] = []
    next_observations: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    stages: list[int] = []

    model_xml = get_model_xml(env)

    def transition(new_stage: Stage) -> None:
        nonlocal stage, stage_steps, stable_steps
        stage = new_stage
        stage_steps = 0
        stable_steps = 0

    reason = "max steps reached"

    for _ in range(args.max_episode_steps):
        stage_for_action = stage

        if stage_for_action == Stage.DONE:
            reason = "fsm completed"
            break
        if stage_for_action == Stage.FAILED:
            break

        obs = make_observation(
            env,
            obs,
            allow_missing_force=args.allow_missing_force,
            body_id=body_id,
            light_target=light_target,
            heavy_target=heavy_target,
            stage=stage_for_action,
        )
        current_pos = get_eef_pos(obs, env)
        current_force = np.asarray(
            obs[FORCE_KEYS[0]],
            dtype=np.float64,
        )

        if stage_for_action == Stage.APPROACH:
            action = position_action(
                current_pos,
                approach_target,
                -1.0,
                args.controller_step,
            )
            error = np.linalg.norm(current_pos - approach_target)
            stable_steps = (
                stable_steps + 1
                if error < args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                transition(Stage.DESCEND)

        elif stage_for_action == Stage.DESCEND:
            action = position_action(
                current_pos,
                grasp_target,
                -1.0,
                args.controller_step,
            )
            error = np.linalg.norm(current_pos - grasp_target)

            if error < max(args.position_tolerance * 2.0, 0.03):
                baseline_samples.append(current_force.copy())
                baseline_samples[:] = baseline_samples[
                    -args.baseline_samples :
                ]

            stable_steps = (
                stable_steps + 1
                if error < args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                transition(Stage.CLOSE)

        elif stage_for_action == Stage.CLOSE:
            action = position_action(
                current_pos,
                grasp_target,
                1.0,
                args.controller_step,
            )
            if stage_steps + 1 >= args.close_steps:
                transition(Stage.LIFT)

        elif stage_for_action == Stage.LIFT:
            # Closed-loop pose control is intentionally used here. Different
            # masses may require different corrective actions, but both episodes
            # are driven to the same EEF pose for a fixed number of steps.
            action = position_action(
                current_pos,
                lift_target,
                1.0,
                args.controller_step,
            )

            if stage_steps + 1 >= lift_total_steps:
                object_pos = get_object_pos(env, body_id)
                pose_error = float(
                    np.linalg.norm(current_pos - lift_target)
                )

                if (
                    object_pos[2]
                    < object_pos0[2] + args.min_lift_for_grasp
                ):
                    transition(Stage.FAILED)
                    reason = "object was not lifted; grasp failed"
                elif pose_error > args.preclass_pose_tolerance:
                    transition(Stage.FAILED)
                    reason = (
                        "EEF did not converge to the common lift pose: "
                        f"error={pose_error:.6f} m > "
                        f"{args.preclass_pose_tolerance:.6f} m"
                    )
                else:
                    transition(Stage.WEIGH)

        elif stage_for_action == Stage.WEIGH:
            # Continue closed-loop holding at the same common pose. The force
            # window is collected only after settling. A final guard window then
            # ensures the EEF remains pose-matched before TRANSIT begins.
            action = position_action(
                current_pos,
                lift_target,
                1.0,
                args.controller_step,
            )

            if (
                args.weigh_settle_steps
                <= stage_steps
                < args.weigh_settle_steps + args.weigh_samples
            ):
                weigh_samples.append(current_force.copy())

            # Reuse the final WEIGH frames as the pose guard. This avoids a
            # second stationary phase after force measurement.
            guard_start = max(
                0,
                weigh_total_steps - args.preclass_guard_steps,
            )
            if stage_steps >= guard_start:
                preclass_guard_pose_errors.append(
                    float(np.linalg.norm(current_pos - lift_target))
                )

            if stage_steps + 1 >= weigh_total_steps:
                if not baseline_samples:
                    transition(Stage.FAILED)
                    reason = "no force baseline samples were collected"
                elif len(weigh_samples) < args.weigh_samples:
                    transition(Stage.FAILED)
                    reason = (
                        "insufficient fixed-window weigh samples: "
                        f"{len(weigh_samples)} < {args.weigh_samples}"
                    )
                elif len(preclass_guard_pose_errors) < args.preclass_guard_steps:
                    transition(Stage.FAILED)
                    reason = (
                        "insufficient pre-classification pose guard samples: "
                        f"{len(preclass_guard_pose_errors)} < "
                        f"{args.preclass_guard_steps}"
                    )
                elif (
                    max(preclass_guard_pose_errors)
                    > args.preclass_pose_tolerance
                ):
                    transition(Stage.FAILED)
                    reason = (
                        "EEF drifted during the pre-classification guard: "
                        f"max_error={max(preclass_guard_pose_errors):.6f} m > "
                        f"{args.preclass_pose_tolerance:.6f} m"
                    )
                else:
                    baseline = np.mean(
                        np.stack(baseline_samples),
                        axis=0,
                    )
                    loaded = np.mean(
                        np.stack(weigh_samples[: args.weigh_samples]),
                        axis=0,
                    )
                    delta_force = loaded - baseline

                    if args.force_signal == "z":
                        force_signal = float(abs(delta_force[2]))
                    else:
                        force_signal = float(
                            np.linalg.norm(delta_force)
                        )

                    predicted_class = (
                        "heavy"
                        if force_signal >= args.force_threshold
                        else "light"
                    )
                    transition(Stage.TRANSIT)

        elif stage_for_action == Stage.TRANSIT:
            destination = (
                heavy_target
                if predicted_class == "heavy"
                else light_target
            )
            above_target = np.array(
                [
                    destination[0],
                    destination[1],
                    max(
                        lift_target[2],
                        env.bin2_pos[2] + args.transport_height,
                    ),
                ],
                dtype=np.float64,
            )
            action = position_action(
                current_pos,
                above_target,
                1.0,
                args.controller_step,
            )
            error = np.linalg.norm(current_pos - above_target)
            stable_steps = (
                stable_steps + 1
                if error < args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                transition(Stage.LOWER)

        elif stage_for_action == Stage.LOWER:
            destination = (
                heavy_target
                if predicted_class == "heavy"
                else light_target
            )
            object_half_height = max(
                0.025,
                float(object_pos0[2] - env.bin1_pos[2]),
            )
            place_target = np.array(
                [
                    destination[0],
                    destination[1],
                    (
                        env.bin2_pos[2]
                        + object_half_height
                        + args.grasp_z_offset
                        + args.place_clearance
                    ),
                ],
                dtype=np.float64,
            )
            action = position_action(
                current_pos,
                place_target,
                1.0,
                args.controller_step,
            )
            error = np.linalg.norm(current_pos - place_target)
            stable_steps = (
                stable_steps + 1
                if error < args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                transition(Stage.OPEN)

        elif stage_for_action == Stage.OPEN:
            action = no_op_action(gripper=-1.0)
            if stage_steps + 1 >= args.open_steps:
                transition(Stage.RETREAT)

        elif stage_for_action == Stage.RETREAT:
            destination = (
                heavy_target
                if predicted_class == "heavy"
                else light_target
            )
            retreat_target = np.array(
                [
                    destination[0],
                    destination[1],
                    env.bin2_pos[2] + args.transport_height,
                ],
                dtype=np.float64,
            )
            action = position_action(
                current_pos,
                retreat_target,
                -1.0,
                args.controller_step,
            )
            error = np.linalg.norm(current_pos - retreat_target)
            stable_steps = (
                stable_steps + 1
                if error < args.position_tolerance
                else 0
            )
            if stable_steps >= args.stable_steps:
                transition(Stage.DONE)

        else:
            raise RuntimeError(f"Unhandled FSM stage: {stage_for_action}")

        # Save the stage that generated this action. The old collector changed
        # stage first and then saved the new label beside the previous stage's
        # action, causing a one-step alignment error.
        observations.append(deepcopy(obs))
        states.append(get_state_vector(env))
        actions.append(action.copy())
        stages.append(int(stage_for_action))

        next_obs, _, _, _ = env.step(action)
        next_obs = make_observation(
            env,
            next_obs,
            allow_missing_force=args.allow_missing_force,
            body_id=body_id,
            light_target=light_target,
            heavy_target=heavy_target,
            stage=stage,
        )
        next_observations.append(deepcopy(next_obs))
        next_states.append(get_state_vector(env))
        obs = next_obs

        # transition() already reset stage_steps for the next stage.
        if stage == stage_for_action:
            stage_steps += 1

        if args.render:
            show_preview(
                obs,
                stage=stage,
                true_class=episode_spec.true_class,
                predicted_class=predicted_class,
                mass=episode_spec.mass,
                force_signal=force_signal,
                delay_ms=args.render_delay_ms,
            )

        if stage == Stage.FAILED:
            break

    if not actions:
        return EpisodeResult(
            False,
            "no actions were generated",
            None,
            predicted_class,
            force_signal,
        )

    final_object_pos = get_object_pos(env, body_id)
    true_target = (
        heavy_target
        if episode_spec.true_class == "heavy"
        else light_target
    )
    correct_bin = inside_target(
        final_object_pos,
        true_target,
        env,
    )
    class_correct = predicted_class == episode_spec.true_class
    success = bool(
        stage == Stage.DONE
        and class_correct
        and correct_bin
    )

    if not success:
        if predicted_class is None:
            reason = (
                reason
                if reason != "fsm completed"
                else "weight was never classified"
            )
        elif not class_correct:
            reason = (
                "force classification wrong: "
                f"true={episode_spec.true_class}, "
                f"pred={predicted_class}, "
                f"signal={force_signal:.3f}"
            )
        elif not correct_bin:
            reason = (
                "object not in true target; "
                f"final_pos={final_object_pos.tolist()}, "
                f"target={true_target.tolist()}"
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
        "actual_start_pos": actual_start,
        "final_object_pos": final_object_pos,
        "light_target": light_target,
        "heavy_target": heavy_target,
        "lift_height": float(episode_spec.lift_height),
        "preclassification_schedule_steps": np.asarray(
            [
                args.closed_loop_lift_steps,
                args.weigh_settle_steps,
                args.weigh_samples,
                args.preclass_guard_steps,
            ],
            dtype=np.int32,
        ),
    }
    return EpisodeResult(
        success,
        reason,
        trajectory,
        predicted_class,
        force_signal,
    )


def write_episode(
    data_group: h5py.Group,
    demo_name: str,
    result: EpisodeResult,
    spec: EpisodeSpec,
) -> int:
    assert result.trajectory is not None
    trajectory = result.trajectory
    demo = data_group.create_group(demo_name)

    for key in ("states", "next_states", "actions", "rewards", "dones", "fsm_stage"):
        demo.create_dataset(key, data=trajectory[key])

    obs_group = demo.create_group("obs")
    next_obs_group = demo.create_group("next_obs")
    for key, value in trajectory["obs"].items():
        obs_group.create_dataset(
            key,
            data=value,
            compression="gzip" if key in CAMERA_KEYS else None,
        )
    for key, value in trajectory["next_obs"].items():
        next_obs_group.create_dataset(
            key,
            data=value,
            compression="gzip" if key in CAMERA_KEYS else None,
        )

    model_file = trajectory.get("model_file")
    if model_file:
        demo.attrs["model_file"] = model_file

    ep_meta = {
        "task": "weight_sort_pick_place",
        "pair_id": int(spec.pair_id),
        "mass_kg": float(spec.mass),
        "true_weight_class": spec.true_class,
        "start_yaw_rad": float(spec.start_yaw),
        "lift_height_m": float(spec.lift_height),
        "predicted_weight_class": result.predicted_class,
        "force_signal_n": result.force_signal,
        "light_target": trajectory["light_target"],
        "heavy_target": trajectory["heavy_target"],
        "actual_start_pos": trajectory["actual_start_pos"],
        "final_object_pos": trajectory["final_object_pos"],
    }
    demo.attrs["ep_meta"] = json.dumps(ep_meta, default=json_default)
    demo.attrs["num_samples"] = int(trajectory["actions"].shape[0])
    demo.attrs["collection_mode"] = "direct_robosuite_fsm_weight_sort"
    demo.attrs["pair_id"] = int(spec.pair_id)
    demo.attrs["object_mass_kg"] = float(spec.mass)
    demo.attrs["true_weight_class"] = spec.true_class
    demo.attrs["start_yaw_rad"] = float(spec.start_yaw)
    demo.attrs["lift_height_m"] = float(spec.lift_height)
    demo.attrs["predicted_weight_class"] = str(result.predicted_class)
    demo.attrs["force_signal_n"] = float(result.force_signal)
    demo.attrs["classification_correct"] = True
    demo.attrs["success"] = True
    return int(trajectory["actions"].shape[0])



def weight_scene_fingerprint(
    *,
    start_xy: np.ndarray,
    start_yaw: float,
    lift_height: float,
    decimals: int = 6,
) -> tuple[float, float, float, float]:
    """
    Fingerprint pair-shared visual/control scene variables.

    Mass is deliberately excluded because light/heavy are the two paired hidden
    conditions of the SAME scene.
    """
    xy = np.asarray(start_xy, dtype=np.float64).reshape(-1)
    return (
        round(float(xy[0]), decimals),
        round(float(xy[1]), decimals),
        round(float(start_yaw), decimals),
        round(float(lift_height), decimals),
    )


def load_excluded_weight_scenes(
    paths: list[str] | None,
) -> set[tuple[float, float, float, float]]:
    excluded: set[tuple[float, float, float, float]] = set()

    for raw_path in paths or []:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--exclude-hdf5 does not exist: {path}")

        with h5py.File(path, "r") as hdf5_file:
            if "data" not in hdf5_file:
                raise KeyError(f"{path} has no /data group")

            seen_pairs: set[int] = set()
            for demo in hdf5_file["data"].values():
                pair_id = int(demo.attrs.get("pair_id", -1))
                if pair_id in seen_pairs:
                    continue
                seen_pairs.add(pair_id)

                raw_meta = demo.attrs.get("ep_meta")
                meta: dict[str, Any] = {}
                if raw_meta is not None:
                    if isinstance(raw_meta, bytes):
                        raw_meta = raw_meta.decode("utf-8")
                    try:
                        meta = json.loads(str(raw_meta))
                    except Exception:
                        meta = {}

                start_pos = meta.get("actual_start_pos")
                if start_pos is None:
                    # Fall back to the first object pose if old metadata is missing.
                    continue

                start_yaw = demo.attrs.get(
                    "start_yaw_rad",
                    meta.get("start_yaw_rad"),
                )
                lift_height = demo.attrs.get(
                    "lift_height_m",
                    meta.get("lift_height_m"),
                )
                if start_yaw is None or lift_height is None:
                    continue

                excluded.add(
                    weight_scene_fingerprint(
                        start_xy=np.asarray(start_pos, dtype=np.float64)[:2],
                        start_yaw=float(start_yaw),
                        lift_height=float(lift_height),
                    )
                )
    return excluded




def create_masks(
    output: h5py.File,
    demo_names: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Create pair-aware train / valid masks.

    Both members of a light/heavy pair always enter the same split. This avoids
    placing two visually identical scenes on opposite sides of the split and
    prevents validation leakage through paired object position, yaw, and lift
    profile.
    """

    rng = np.random.default_rng(seed)
    pairs: dict[int, list[str]] = defaultdict(list)

    for name in demo_names:
        demo = output["data"][name]
        if "pair_id" not in demo.attrs:
            raise KeyError(
                f"{name} has no pair_id attribute; pair-aware splitting "
                "cannot be performed."
            )
        pairs[int(demo.attrs["pair_id"])].append(name)

    for pair_id, names in pairs.items():
        labels = {
            str(output["data"][name].attrs.get("true_weight_class", ""))
            for name in names
        }
        if len(names) != 2 or labels != {"light", "heavy"}:
            raise ValueError(
                f"pair_id={pair_id} must contain exactly one light and one "
                f"heavy demo, but got names={names}, labels={labels}."
            )

    pair_ids = np.asarray(sorted(pairs.keys()), dtype=np.int64)
    rng.shuffle(pair_ids)

    if val_ratio <= 0.0 or len(pair_ids) <= 1:
        val_pair_ids: set[int] = set()
    else:
        n_val_pairs = int(round(len(pair_ids) * val_ratio))
        n_val_pairs = max(1, n_val_pairs)
        n_val_pairs = min(n_val_pairs, len(pair_ids) - 1)
        val_pair_ids = {
            int(pair_id)
            for pair_id in pair_ids[:n_val_pairs]
        }

    train_names: list[str] = []
    val_names: list[str] = []

    for pair_id, names in pairs.items():
        destination = (
            val_names
            if pair_id in val_pair_ids
            else train_names
        )
        destination.extend(names)

    train_names = sorted(train_names)
    val_names = sorted(val_names)

    mask = output.create_group("mask")
    dtype = h5py.string_dtype(encoding="utf-8")
    mask.create_dataset(
        "train",
        data=np.asarray(train_names, dtype=object),
        dtype=dtype,
    )
    mask.create_dataset(
        "valid",
        data=np.asarray(val_names, dtype=object),
        dtype=dtype,
    )
    return train_names, val_names



def paired_preclassification_pose_gap(
    pair_results: list[tuple[EpisodeSpec, EpisodeResult]],
    guard_steps: int,
) -> dict[str, float]:
    """Compare the final pre-classification EEF trajectories within one pair.

    Both episodes have the same fixed stage durations. We compare the final
    ``guard_steps`` WEIGH observations, immediately before TRANSIT. A pair is
    rejected when the physical EEF trajectories are too far apart, even if both
    episodes individually satisfy the target-pose tolerance.
    """

    by_class = {
        spec.true_class: result
        for spec, result in pair_results
    }
    if set(by_class) != {"light", "heavy"}:
        raise ValueError(
            "Expected exactly one light and one heavy result in a pair."
        )

    trajectories: dict[str, np.ndarray] = {}
    for class_name in ("light", "heavy"):
        result = by_class[class_name]
        if result.trajectory is None:
            raise ValueError(f"{class_name} result has no trajectory.")

        stages = np.asarray(
            result.trajectory["fsm_stage"],
            dtype=np.int64,
        ).reshape(-1)
        positions = np.asarray(
            result.trajectory["obs"]["robot0_eef_pos"],
            dtype=np.float64,
        ).reshape(-1, 3)

        usable = min(len(stages), len(positions))
        stages = stages[:usable]
        positions = positions[:usable]

        weigh_indices = np.flatnonzero(stages == int(Stage.WEIGH))
        if weigh_indices.size < guard_steps:
            raise ValueError(
                f"{class_name} has only {weigh_indices.size} WEIGH samples, "
                f"but {guard_steps} guard samples are required."
            )

        guard_indices = weigh_indices[-guard_steps:]
        trajectories[class_name] = positions[guard_indices]

    light = trajectories["light"]
    heavy = trajectories["heavy"]
    pair_length = min(len(light), len(heavy))
    light = light[-pair_length:]
    heavy = heavy[-pair_length:]

    delta = heavy - light
    distances = np.linalg.norm(delta, axis=1)
    z_gaps = np.abs(delta[:, 2])

    return {
        "mean_3d_gap_m": float(np.mean(distances)),
        "max_3d_gap_m": float(np.max(distances)),
        "mean_z_gap_m": float(np.mean(z_gaps)),
        "max_z_gap_m": float(np.max(z_gaps)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect direct Robosuite PickPlaceCan demonstrations with an FSM. "
            "Data are collected as matched light/heavy scene pairs. The FSM "
            "uses a fixed-duration closed-loop pre-classification lift schedule, classifies using "
            "live force, and places the can in the matching bin."
        )
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum paired-scene attempts, not individual episode attempts.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Create /mask/test only for final held-out evaluation.",
    )
    parser.add_argument(
        "--exclude-hdf5",
        nargs="*",
        default=None,
        help=(
            "Existing Weight-sort train/validation HDF5 files. Pair-shared "
            "start XY, yaw, and lift-height fingerprints are excluded."
        ),
    )

    parser.add_argument("--light-mass-min", type=float, default=0.05)
    parser.add_argument("--light-mass-max", type=float, default=0.15)
    parser.add_argument("--heavy-mass-min", type=float, default=0.35)
    parser.add_argument("--heavy-mass-max", type=float, default=0.55)
    parser.add_argument("--force-threshold", type=float, default=2.0)
    parser.add_argument("--force-signal", choices=("norm", "z"), default="norm")

    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--max-episode-steps", type=int, default=450)
    parser.add_argument("--settle-steps", type=int, default=15)
    parser.add_argument("--controller-step", type=float, default=0.03)
    parser.add_argument("--position-tolerance", type=float, default=0.015)
    parser.add_argument("--stable-steps", type=int, default=3)
    parser.add_argument("--approach-height", type=float, default=0.16)
    parser.add_argument("--grasp-z-offset", type=float, default=0.02)
    parser.add_argument("--lift-height", type=float, default=0.22)
    parser.add_argument(
        "--lift-height-jitter",
        type=float,
        default=0.025,
        help=(
            "Class-independent pair-level lift-height jitter in metres. "
            "The same sampled height is used by both members of a pair."
        ),
    )
    parser.add_argument(
        "--closed-loop-lift-steps",
        type=int,
        default=25,
        help=(
            "Fixed number of closed-loop steps used to drive every light/heavy "
            "episode to the same pair-level lift pose."
        ),
    )
    parser.add_argument(
        "--preclass-pose-tolerance",
        type=float,
        default=0.004,
        help=(
            "Maximum EEF distance from the common lift target during the final "
            "pre-classification guard window, in metres."
        ),
    )
    parser.add_argument(
        "--preclass-guard-steps",
        type=int,
        default=2,
        help=(
            "Number of final WEIGH frames reused as a pose-alignment guard. "
            "These frames overlap the force-sampling window; they are not an "
            "additional stationary hold. Keep this below exec_horizon."
        ),
    )
    parser.add_argument(
        "--pair-pose-gap-tolerance",
        type=float,
        default=0.004,
        help=(
            "Reject the whole light/heavy pair if their final guard-window EEF "
            "trajectories differ by more than this 3D distance, in metres."
        ),
    )
    parser.add_argument("--transport-height", type=float, default=0.28)
    parser.add_argument("--place-clearance", type=float, default=0.05)
    parser.add_argument("--min-lift-for-grasp", type=float, default=0.06)
    parser.add_argument("--close-steps", type=int, default=10)
    parser.add_argument("--open-steps", type=int, default=20)
    parser.add_argument("--baseline-samples", type=int, default=12)
    parser.add_argument("--weigh-settle-steps", type=int, default=0)
    parser.add_argument("--weigh-samples", type=int, default=3)
    parser.add_argument("--object-margin", type=float, default=0.065)

    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-delay-ms", type=int, default=20)
    parser.add_argument("--allow-missing-force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.image_size < 16:
        raise ValueError("--image-size must be at least 16")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    if not (0 < args.light_mass_min <= args.light_mass_max):
        raise ValueError("Invalid light mass range")
    if not (0 < args.heavy_mass_min <= args.heavy_mass_max):
        raise ValueError("Invalid heavy mass range")
    if args.light_mass_max >= args.heavy_mass_min:
        raise ValueError("Light and heavy mass ranges must not overlap")
    if args.force_threshold <= 0.0:
        raise ValueError("--force-threshold must be positive")
    if args.num_episodes % 2 != 0:
        raise ValueError(
            "--num-episodes must be even because data are collected as "
            "matched light/heavy pairs."
        )
    if args.lift_height_jitter < 0.0:
        raise ValueError("--lift-height-jitter cannot be negative")
    if args.lift_height - args.lift_height_jitter <= args.min_lift_for_grasp:
        raise ValueError(
            "Minimum randomized lift height must exceed "
            "--min-lift-for-grasp."
        )
    if args.closed_loop_lift_steps <= 0:
        raise ValueError("--closed-loop-lift-steps must be positive")
    if args.preclass_pose_tolerance <= 0.0:
        raise ValueError("--preclass-pose-tolerance must be positive")
    if args.preclass_guard_steps <= 0:
        raise ValueError("--preclass-guard-steps must be positive")
    if args.preclass_guard_steps > max(
        args.weigh_settle_steps + args.weigh_samples,
        args.preclass_guard_steps,
    ):
        raise ValueError("Invalid pre-classification guard configuration")
    if args.pair_pose_gap_tolerance <= 0.0:
        raise ValueError("--pair-pose-gap-tolerance must be positive")
    if args.close_steps <= 0:
        raise ValueError("--close-steps must be positive")
    if args.weigh_settle_steps < 0 or args.weigh_samples <= 0:
        raise ValueError(
            "--weigh-settle-steps must be non-negative and "
            "--weigh-samples must be positive"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} exists. Add --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    excluded_scene_fingerprints = load_excluded_weight_scenes(
        args.exclude_hdf5
    )
    used_test_scene_fingerprints: set[
        tuple[float, float, float, float]
    ] = set()
    if args.test_only:
        print(
            f"FINAL TEST MODE: excluding "
            f"{len(excluded_scene_fingerprints)} previous scene fingerprints."
        )

    requested_pairs = args.num_episodes // 2
    max_pair_attempts = (
        args.max_attempts
        or max(requested_pairs * 10, requested_pairs)
    )

    env, env_meta = create_environment(
        image_size=args.image_size,
        horizon=args.max_episode_steps + args.settle_steps + 50,
        control_freq=args.control_freq,
        seed=args.seed,
    )
    env_meta["weight_sort_task"].update(
        {
            "paired_collection": True,
            "fixed_duration_closed_loop_lift": True,
            "preclassification_pose_guard": True,
            "fixed_robot_initialization": True,
            "pair_aware_split": True,
        }
    )

    try:
        # Body ids remain stable because hard_reset=False.
        env.reset()
        _, body_id, joint_name = active_object(env)
        args._base_mass = float(
            env.sim.model.body_mass[body_id]
        )
        args._base_inertia = np.asarray(
            env.sim.model.body_inertia[body_id],
            dtype=np.float64,
        ).copy()

        saved_names: list[str] = []
        saved_counts: dict[str, int] = defaultdict(int)
        failure_counts: dict[str, int] = defaultdict(int)
        total_samples = 0
        pair_attempts = 0
        successful_pairs = 0
        signals_by_class: dict[str, list[float]] = defaultdict(list)
        accepted_pair_pose_gaps: list[float] = []

        with h5py.File(output_path, "w") as output:
            data_group = output.create_group("data")
            data_group.attrs["dataset_role"] = (
                "final_test" if args.test_only else "train_validation"
            )
            data_group.attrs["non_repeat_test_enforced"] = bool(args.test_only)
            data_group.attrs["excluded_scene_fingerprint_count"] = int(
                len(excluded_scene_fingerprints)
            )
            env_json = json.dumps(
                env_meta,
                default=json_default,
            )
            output.attrs["env_args"] = env_json
            data_group.attrs["env_args"] = env_json
            data_group.attrs[
                "collection_mode"
            ] = "direct_robosuite_fsm_weight_sort_paired"
            data_group.attrs["source_dataset_required"] = False
            data_group.attrs[
                "force_source"
            ] = "live_robosuite_gripper_sensor"
            data_group.attrs["paired_collection"] = True
            data_group.attrs[
                "fixed_preclassification_schedule"
            ] = True
            data_group.attrs["fixed_robot_initialization"] = True
            data_group.attrs["pair_aware_split"] = True
            data_group.attrs["light_mass_range_kg"] = np.asarray(
                [args.light_mass_min, args.light_mass_max],
                dtype=np.float32,
            )
            data_group.attrs["heavy_mass_range_kg"] = np.asarray(
                [args.heavy_mass_min, args.heavy_mass_max],
                dtype=np.float32,
            )
            data_group.attrs[
                "force_threshold_n"
            ] = float(args.force_threshold)
            data_group.attrs["force_signal"] = args.force_signal
            data_group.attrs[
                "lift_height_base_m"
            ] = float(args.lift_height)
            data_group.attrs[
                "lift_height_jitter_m"
            ] = float(args.lift_height_jitter)
            data_group.attrs["closed_loop_lift_steps"] = int(
                args.closed_loop_lift_steps
            )
            data_group.attrs["preclass_pose_tolerance_m"] = float(
                args.preclass_pose_tolerance
            )
            data_group.attrs["preclass_guard_steps"] = int(
                args.preclass_guard_steps
            )
            data_group.attrs["pair_pose_gap_tolerance_m"] = float(
                args.pair_pose_gap_tolerance
            )

            progress = tqdm(
                total=args.num_episodes,
                desc="Weight-sort paired demos",
            )

            while (
                successful_pairs < requested_pairs
                and pair_attempts < max_pair_attempts
            ):
                pair_attempts += 1
                pair_id = successful_pairs

                # All non-mass scene variables are sampled once and reused by
                # the light and heavy members of the pair.
                start_xy = sample_start_xy(
                    env,
                    rng,
                    args.object_margin,
                )
                start_yaw = float(rng.uniform(-np.pi, np.pi))
                pair_lift_height = float(
                    args.lift_height
                    + rng.uniform(
                        -args.lift_height_jitter,
                        args.lift_height_jitter,
                    )
                )

                scene_fingerprint = weight_scene_fingerprint(
                    start_xy=start_xy,
                    start_yaw=start_yaw,
                    lift_height=pair_lift_height,
                )
                if args.test_only and (
                    scene_fingerprint in excluded_scene_fingerprints
                    or scene_fingerprint in used_test_scene_fingerprints
                ):
                    failure_counts["excluded duplicate scene fingerprint"] += 1
                    continue

                class_order = list(
                    rng.permutation(
                        np.asarray(["light", "heavy"], dtype=object)
                    )
                )

                pair_results: list[
                    tuple[EpisodeSpec, EpisodeResult]
                ] = []
                pair_failed_reason: str | None = None

                for true_class_obj in class_order:
                    true_class = str(true_class_obj)
                    if true_class == "light":
                        mass = float(
                            rng.uniform(
                                args.light_mass_min,
                                args.light_mass_max,
                            )
                        )
                    else:
                        mass = float(
                            rng.uniform(
                                args.heavy_mass_min,
                                args.heavy_mass_max,
                            )
                        )

                    spec = EpisodeSpec(
                        pair_id=pair_id,
                        true_class=true_class,
                        mass=mass,
                        start_pos=np.array(
                            [start_xy[0], start_xy[1], 0.0],
                            dtype=np.float64,
                        ),
                        start_yaw=start_yaw,
                        lift_height=pair_lift_height,
                        light_target=np.zeros(
                            3,
                            dtype=np.float64,
                        ),
                        heavy_target=np.zeros(
                            3,
                            dtype=np.float64,
                        ),
                    )

                    result = run_fsm_episode(
                        env=env,
                        episode_spec=spec,
                        body_id=body_id,
                        joint_name=joint_name,
                        args=args,
                    )

                    if not result.success:
                        pair_failed_reason = (
                            f"{true_class}: {result.reason}"
                        )
                        break

                    pair_results.append((spec, result))

                # Never keep a single member of a pair. This prevents
                # success-filtering from creating a class-position correlation.
                if (
                    pair_failed_reason is not None
                    or len(pair_results) != 2
                ):
                    reason = (
                        pair_failed_reason
                        or "pair did not produce two successful episodes"
                    )
                    failure_counts[reason] += 1
                    progress.set_postfix(
                        pair_attempts=pair_attempts,
                        pairs=successful_pairs,
                        last_failure=reason[:45],
                    )
                    continue

                try:
                    pair_pose_metrics = paired_preclassification_pose_gap(
                        pair_results,
                        guard_steps=args.preclass_guard_steps,
                    )
                except Exception as error:
                    reason = f"pair pose-gap check failed: {error}"
                    failure_counts[reason] += 1
                    progress.set_postfix(
                        pair_attempts=pair_attempts,
                        pairs=successful_pairs,
                        last_failure=reason[:45],
                    )
                    continue

                if (
                    pair_pose_metrics["max_3d_gap_m"]
                    > args.pair_pose_gap_tolerance
                ):
                    reason = (
                        "paired preclassification EEF gap too large: "
                        f"max_3d={pair_pose_metrics['max_3d_gap_m']:.6f} m > "
                        f"{args.pair_pose_gap_tolerance:.6f} m"
                    )
                    failure_counts[reason] += 1
                    progress.set_postfix(
                        pair_attempts=pair_attempts,
                        pairs=successful_pairs,
                        last_failure=reason[:45],
                    )
                    continue

                accepted_pair_pose_gaps.append(
                    pair_pose_metrics["max_3d_gap_m"]
                )

                for spec, result in pair_results:
                    demo_name = f"demo_{len(saved_names)}"
                    total_samples += write_episode(
                        data_group,
                        demo_name,
                        result,
                        spec,
                    )
                    data_group[demo_name].attrs[
                        "pair_preclass_pose_gap_mean_3d_m"
                    ] = pair_pose_metrics["mean_3d_gap_m"]
                    data_group[demo_name].attrs[
                        "pair_preclass_pose_gap_max_3d_m"
                    ] = pair_pose_metrics["max_3d_gap_m"]
                    data_group[demo_name].attrs[
                        "pair_preclass_pose_gap_mean_z_m"
                    ] = pair_pose_metrics["mean_z_gap_m"]
                    data_group[demo_name].attrs[
                        "pair_preclass_pose_gap_max_z_m"
                    ] = pair_pose_metrics["max_z_gap_m"]
                    saved_names.append(demo_name)
                    saved_counts[spec.true_class] += 1

                    if result.force_signal is not None:
                        signals_by_class[spec.true_class].append(
                            float(result.force_signal)
                        )

                if args.test_only:
                    used_test_scene_fingerprints.add(scene_fingerprint)

                successful_pairs += 1
                progress.update(2)
                progress.set_postfix(
                    pair_attempts=pair_attempts,
                    pairs=successful_pairs,
                    light=saved_counts["light"],
                    heavy=saved_counts["heavy"],
                )

            progress.close()

            if not saved_names:
                raise RuntimeError(
                    "No complete light/heavy pairs were collected. "
                    "Use --render and tune the closed-loop lift duration, pose "
                    "tolerance, grasp height, or force threshold."
                )

            if args.test_only:
                test_names = create_test_mask(
                    output,
                    saved_names,
                )
                train_names: list[str] = []
                val_names: list[str] = []
            else:
                train_names, val_names = create_masks(
                    output,
                    saved_names,
                    val_ratio=args.val_ratio,
                    seed=args.split_seed,
                )

            data_group.attrs["total"] = int(total_samples)
            data_group.attrs["num_demos"] = int(len(saved_names))
            data_group.attrs["num_pairs"] = int(successful_pairs)
            data_group.attrs[
                "num_pair_attempts"
            ] = int(pair_attempts)
            # Preserve the common attribute name for compatibility.
            data_group.attrs["num_attempts"] = int(pair_attempts)
            data_group.attrs[
                "num_train_demos"
            ] = int(len(train_names))
            data_group.attrs[
                "num_valid_demos"
            ] = int(len(val_names))
            data_group.attrs["num_test_demos"] = int(
                len(test_names) if args.test_only else 0
            )
            data_group.attrs[
                "num_light_demos"
            ] = int(saved_counts["light"])
            data_group.attrs[
                "num_heavy_demos"
            ] = int(saved_counts["heavy"])

            report = output.create_group("collection_report")
            dtype = h5py.string_dtype(encoding="utf-8")
            report.create_dataset(
                "failure_reasons",
                data=np.asarray(
                    list(failure_counts.keys()),
                    dtype=object,
                ),
                dtype=dtype,
            )
            report.create_dataset(
                "failure_counts",
                data=np.asarray(
                    list(failure_counts.values()),
                    dtype=np.int64,
                ),
            )
            for class_name in ("light", "heavy"):
                report.create_dataset(
                    f"{class_name}_force_signals",
                    data=np.asarray(
                        signals_by_class[class_name],
                        dtype=np.float32,
                    ),
                )

        print("\nCollection complete")
        print(f"  Output: {output_path}")
        print(
            f"  Complete pairs: {successful_pairs} / "
            f"pair attempts {pair_attempts}"
        )
        print(f"  Successful demos: {len(saved_names)}")
        print(f"  Light demos: {saved_counts['light']}")
        print(f"  Heavy demos: {saved_counts['heavy']}")
        print(f"  Transitions: {total_samples}")
        print(
            (
                f"  Test demos: {len(test_names)}"
                if args.test_only
                else (
                    f"  Train / valid demos: "
                    f"{len(train_names)} / {len(val_names)}"
                )
            )
        )
        print(
            "  Leakage controls: paired positions/yaw/lift height, "
            "fixed robot reset, fixed LIFT action profile, fixed WEIGH "
            "window, pair-aware split"
        )

        if signals_by_class["light"]:
            print(
                "  Light force signal mean: "
                f"{np.mean(signals_by_class['light']):.3f} N"
            )
        if signals_by_class["heavy"]:
            print(
                "  Heavy force signal mean: "
                f"{np.mean(signals_by_class['heavy']):.3f} N"
            )
        if accepted_pair_pose_gaps:
            print(
                "  Accepted pair max preclassification EEF gap mean / max: "
                f"{np.mean(accepted_pair_pose_gaps):.6f} / "
                f"{np.max(accepted_pair_pose_gaps):.6f} m"
            )
        if successful_pairs < requested_pairs:
            print(
                "  WARNING: maximum pair attempts reached before the "
                "requested number of complete pairs."
            )
        if len(val_names) == 0:
            print(
                "  WARNING: validation mask is empty because only one "
                "complete pair was saved."
            )

    except Exception:
        if output_path.exists():
            try:
                with h5py.File(output_path, "r") as check:
                    if int(
                        check.get("data", {}).attrs.get(
                            "num_demos",
                            0,
                        )
                    ) == 0:
                        output_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise
    finally:
        env.close()
        if args.render:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ImportError:
                pass


if __name__ == "__main__":
    main()
