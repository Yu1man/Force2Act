import torch
from dataclasses import dataclass
from force2act.data.episode import Episode

def collect_episode_field(episodes, field):

    modality_list = []

    for episode in episodes:
        modality_list.append(getattr(episode, field))

    modality_list = torch.cat(modality_list, dim=0)

    return modality_list


def compute_stats(values, epsilon=1e-4):
    mean = torch.mean(values, dim=0)

    N = values.shape[0]

    var = torch.sum((values - mean[None, :])**2, dim=0) / N

    std = torch.clamp_min(torch.sqrt(var), epsilon)

    return mean, std


@dataclass
class ModalityStats:
    mean: torch.Tensor
    std: torch.Tensor


@dataclass
class StandardNormalizer:
    stats: dict[str, ModalityStats]

    def to(self, device):
        for modality_stats in self.stats.values():
            modality_stats.mean = modality_stats.mean.to(device)
            modality_stats.std = modality_stats.std.to(device)

        return self


    def normalize(self, values, field):
        modality_stats = self.stats[field]

        mean = modality_stats.mean
        std = modality_stats.std

        normalized_values = (values - mean) / std

        return normalized_values


    def denormalize(self, values, field):
        modality_stats = self.stats[field]

        mean = modality_stats.mean
        std = modality_stats.std

        denormalized_values = (values * std) + mean

        return denormalized_values


def fit_normalizer(episodes):
    fields = ["state", "force", "actions"]
    stats = {}

    for field in fields:
        modality_values = collect_episode_field(episodes, field)
        mean, std = compute_stats(modality_values)

        stats[field] = ModalityStats(mean, std)

    return StandardNormalizer(stats)





if __name__ == "__main__":
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([2.0, 4.0])

    x = ModalityStats(mean, std)
    normalizer = StandardNormalizer({"state": x})

    values = torch.tensor([
        [3.0, 6.0],
        [5.0, 10.0]
    ])

    normalized = normalizer.normalize(values, "state")
    recovered = normalizer.denormalize(normalized, "state")

    print(normalized)
    print(recovered)
    print(torch.allclose(values, recovered))

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

    normalizer = fit_normalizer(episodes)

    print(normalizer.stats.keys())

    print(normalizer.stats["state"].mean.shape)
    print(normalizer.stats["force"].mean.shape)
    print(normalizer.stats["actions"].mean.shape)

    all_states = collect_episode_field(episodes, "state")

    normalized_states = normalizer.normalize(all_states, "state")

    print(normalized_states.mean(dim=0))
    print(normalized_states.std(dim=0, correction=0))

    all_actions = collect_episode_field(episodes, "actions")

    normalized_actions = normalizer.normalize(all_actions, "actions")
    recovered_actions = normalizer.denormalize(
        normalized_actions,
        "actions"
    )

    print(torch.allclose(all_actions, recovered_actions))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalizer = fit_normalizer(episodes).to(device)

    print(normalizer.stats["state"].mean.device)
    print(normalizer.stats["actions"].mean.device)