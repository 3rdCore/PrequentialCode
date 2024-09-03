from typing import Iterable

import torch
import torch.nn as nn
from beartype import beartype
from torch import FloatTensor

from datasets.synthetic import SyntheticDataset


class TransitionMatrixDynamicalSystem(SyntheticDataset):
    @beartype
    def __init__(
        self,
        dim: int,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        nonlinearity: nn.Module = nn.Identity(),
        noise: float = 0.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        self.dim = dim
        self.x_len = x_dim // dim
        self.y_len = y_dim // dim
        self.nonlinearity = nonlinearity
        self.noise = noise
        self.data_dist = data_dist

        super().__init__(
            n_tasks=n_tasks,
            n_samples=n_samples,
            shuffle_samples=shuffle_samples,
        )

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
        n_tasks: int,
        n_samples: int,
    ) -> tuple[dict[str, FloatTensor], dict[str, Iterable]]:
        # Transition matrices
        w = torch.randn(n_tasks, self.dim, self.dim) / self.dim**0.5

        # Initial conditions
        if self.data_dist == "normal":
            x = [torch.randn(n_tasks, n_samples, self.dim)]
        elif self.data_dist == "uniform":
            x = [2 * torch.rand(n_tasks, n_samples, self.dim) - 1]

        # Trajectory
        for _ in range(self.x_len + self.y_len - 1):
            x_next = torch.einsum("bsx,bxy->bsy", x[-1], w)
            x_next = self.nonlinearity(x_next)
            if self.noise > 0:
                x_next += self.noise * torch.randn_like(x_next)
            x.append(x_next)

        # Reshape
        x = torch.stack(x, dim=2)  # (n_tasks, n_samples, (x_len + y_len), dim)
        x = x.flatten(start_dim=2)  # (n_tasks, n_samples, (x_len + y_len) * dim)
        x, y = x.split([self.dim * self.x_len, self.dim * self.y_len], dim=-1)

        return {"x": x, "y": y}, {"w": w}
