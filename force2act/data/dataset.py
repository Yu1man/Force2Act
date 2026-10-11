from torch.utils.data import Dataset
from force2act.data.windowing import extract_window
import torch

from force2act.data.episode import Episode

class EpisodeDataset(Dataset):

    def __init__(self, episodes, history, action_horizon):
        self.episodes = episodes
        self.history = history
        self.action_horizon = action_horizon

        self.index_map: list = []

        for episode_id, episode in enumerate(episodes):
            episode_length = episode.images.shape[0]

            for timestep in range(episode_length):
                self.index_map.append((episode_id, timestep))


    def __len__(self):
        return len(self.index_map)


    def __getitem__(self, index):
        episode_id, timestep = self.index_map[index]
        episode = self.episodes[episode_id]
        sample = extract_window(episode, timestep, self.history, self.action_horizon)

        return sample


if __name__ == "__main__":

    images = torch.randn(5, 2, 3, 64, 64)
    state = torch.randn(5, 9)
    force = torch.randn(5, 3)
    actions = torch.randn(5, 7)
    timestamps = torch.randn(5)

    episode = Episode(images, state, force, actions, timestamps)

    images2 = torch.randn(3, 2, 3, 64, 64)
    state2 = torch.randn(3, 9)
    force2 = torch.randn(3, 3)
    actions2 = torch.randn(3, 7)
    timestamps2 = torch.randn(3)

    episode2 = Episode(images2, state2, force2, actions2, timestamps2)

    episodes = [episode, episode2]

    episode_dataset = EpisodeDataset(episodes, 3, 4)

    print(len(episode_dataset))
    print(episode_dataset.index_map)

    sample = episode_dataset[5]

    print(episode_dataset.index_map[5])
    print(sample["images"].shape)
    print(sample["state"].shape)
    print(sample["action_mask"])



