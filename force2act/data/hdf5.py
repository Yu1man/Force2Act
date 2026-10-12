import h5py
import torch
from collections.abc import Sequence

from force2act.data.episode import Episode

def _read_obs(demo, key):
    if "obs" not in demo:
        raise KeyError("Demo does not contain an 'obs' group")

    if key not in demo["obs"]:
        raise KeyError(f"Observation key '{key}' not found")

    return demo["obs"][key][()]


def load_episode(
    hdf5_path: str,
    demo_name: str,
    state_keys: Sequence[str],
    camera_keys: Sequence[str],
    force_keys: Sequence[str] | None = None,
):
    with h5py.File(hdf5_path, "r") as f:
        if "data" not in f:
            raise KeyError("File does not contain a 'data' group")

        if demo_name not in f["data"]:
            raise KeyError(f"Data does not contain {demo_name}")

        demo = f["data"][demo_name]

        # Action
        actions = torch.from_numpy(demo["actions"][()]).float() # [T, 7]

        # State
        state_parts = []

        for key in state_keys:
            array = _read_obs(demo, key)
            tensor = torch.from_numpy(array).float()
            state_parts.append(tensor)  # [T, D]

        states = torch.cat(state_parts, dim=-1)

        # Force
        if force_keys:
            force_parts = []

            for key in force_keys:
                array = _read_obs(demo, key)
                tensor = torch.from_numpy(array).float()
                force_parts.append(tensor)

            forces = torch.cat(force_parts, dim=-1)

        else:
            forces = torch.empty(
                actions.shape[0],
                0,
                dtype=torch.float32,
            )

        # Image
        camera_parts = []

        for key in camera_keys:
            array = _read_obs(demo, key)
            tensor = torch.from_numpy(array)  # Probably [T_obs, H, W, C]
            tensor = tensor.permute(0, 3, 1 ,2)
            camera_parts.append(tensor) # [T_obs, C, H, W]

        images = torch.stack(camera_parts, dim=1)   # [T_obs, N_cam, C, H, W]

        # Timestamps
        if "timestamps" in demo["obs"]:
            timestamps = torch.from_numpy(
                _read_obs(demo, "timestamps")
            ).to(torch.float64)
        else:
            timestamps = torch.arange(
                actions.shape[0],
                dtype=torch.float64,
            )

        return Episode(
            images=images,
            state=states,
            force=forces,
            actions=actions,
            timestamps=timestamps
        )


def load_episodes(
    hdf5_path: str,
    demo_names: Sequence[str],
    state_keys: Sequence[str],
    camera_keys: Sequence[str],
    force_keys: Sequence[str] | None = None,
):

    episodes = []

    for demo_name in demo_names:
        episode = load_episode(
            hdf5_path=hdf5_path,
            demo_name=demo_name,
            state_keys=state_keys,
            camera_keys=camera_keys,
            force_keys=force_keys
        )

        episodes.append(episode)

    return episodes


if __name__ == "__main__":
    episode = load_episode(
        hdf5_path="/home/yui/Desktop/forcevam_karl/datasets/robosuite/door_push_pull.hdf5",
        demo_name="demo_0",
        state_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        camera_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        force_keys=[
            "robot0_eef_force",
        ],
    )

    print(episode.images.shape)
    print(episode.images.dtype)
    print(episode.state.shape)
    print(episode.force.shape)
    print(episode.actions.shape)
    print(episode.timestamps.shape)

    episodes = load_episodes(
        hdf5_path="/home/yui/Desktop/forcevam_karl/datasets/robosuite/door_push_pull.hdf5",
        demo_names= ["demo_0", "demo_1"],
        state_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        camera_keys=[
            "agentview_image",
            "robot0_eye_in_hand_image",
        ],
        force_keys=[
            "robot0_eef_force",
        ],
    )

    print(len(episodes))
    print(episodes[0].state.shape)
    print(episodes[1].state.shape)

