import torch
from force2act.data.episode import Episode

def make_window_indices(current_t, ep_length, history, action_horizon):

    obs_indices: list = []
    action_indices: list = []
    action_mask: list = []

    obs_current = current_t - history + 1
    action_current = current_t

    for i in range(history):
        obs_indices.append(max(obs_current, 0))
        obs_current += 1

    for i in range(action_horizon):
        action_indices.append(min(action_current, ep_length-1))
        if action_current < ep_length:
            action_mask.append(1)
        else:
            action_mask.append(0)
        action_current += 1

    return obs_indices, action_indices, action_mask


def extract_window(episode, current_t, history, action_horizon):
    length = episode.images.shape[0]

    obs_indices, action_indices, action_mask = make_window_indices(current_t, length, history, action_horizon)
    images = episode.images[obs_indices]
    images = images.transpose(0, 1) # [T_obs, N_cam, C, H, W] -> [N_cam, T_obs, C, H, W]

    state = episode.state[obs_indices]
    force = episode.force[obs_indices]

    actions = episode.actions[action_indices]

    action_mask = torch.tensor(action_mask, dtype=torch.bool)

    return {
        "images": images,
        "state": state,
        "force": force,
        "actions": actions,
        "action_mask": action_mask
    }

if __name__ == "__main__":
    images = torch.randn(5, 2, 3, 64, 64)
    state = torch.randn(5, 9)
    force = torch.randn(5, 3)
    actions = torch.randn(5, 7)
    timestamps = torch.randn(5)

    episode = Episode(images, state, force,actions, timestamps)

    sample = extract_window(episode, 4, 3, 4)

    print(sample["images"].shape)
    print(sample["state"].shape)
    print(sample["force"].shape)
    print(sample["actions"].shape)
    print(sample["action_mask"])







