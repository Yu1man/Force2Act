from dataclasses import dataclass
import torch

@dataclass
class Episode:

    images: torch.Tensor
    state: torch.Tensor
    force: torch.Tensor

    actions: torch.Tensor

    timestamps: torch.Tensor



    def __post_init__(self):
        T = self.images.shape[0]

        if T == 0:
            raise ValueError(f"Episode has 0 timesteps")

        tensors = {
            "images": (self.images, 5),
            "state": (self.state, 2),
            "force": (self.force, 2),
            "actions": (self.actions, 2),
            "timestamps": (self.timestamps, 1),
        }

        for name, (tensor, ndim) in tensors.items():
            if tensor.ndim != ndim:
                raise ValueError(f"{name} has ndim of {tensor.ndim}, which expected to be {ndim}")

            if tensor.shape[0] != T:
                raise ValueError(f"{name} has {tensor.shape[0]} timesteps, expected {T}")

