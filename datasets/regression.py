from abc import abstractmethod
from typing import Any, Iterable, List

import numpy as np
import torch
from beartype import beartype
from torch import FloatTensor, Tensor

from datasets.interfaces import ICLDataModule
from datasets.synthetic import SyntheticDataset


class RegressionDataset(SyntheticDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.noise = noise
        self.data_dist = data_dist

        super().__init__(n_tasks=n_tasks, n_samples=n_samples, shuffle_samples=shuffle_samples)

    @beartype
    def gen_data(
        self,
        n_tasks: int,
        n_samples: int,
    ) -> tuple[dict[str, Tensor], dict[str, Iterable]]:
        x = self.sample_x(n_tasks, n_samples)
        task_dict_params = self.sample_task_params(self.n_tasks)
        y = self.function(x, task_dict_params)
        y += self.noise * torch.randn_like(y)
        data_dict = {"x": x, "y": y}

        return data_dict, task_dict_params

    @beartype
    def sample_x(self, n_tasks: int, n_samples: int):
        if self.data_dist == "normal":
            x = torch.randn(n_tasks, n_samples, self.x_dim)
        elif self.data_dist == "uniform":
            x = 2 * torch.rand(n_tasks, n_samples, self.x_dim) - 1
        return x

    @abstractmethod
    def sample_task_params(self, n: int | None = None) -> dict[str, Tensor]:
        pass

    @abstractmethod
    def function(self, x: Tensor, params: dict[str, Tensor]) -> FloatTensor:
        # x: (bsz, n_samples, x_dim)
        # params: (bsz, ...) parameters of the function
        # returns y: (bsz, n_samples, y_dim)
        pass


class LinearRegression(RegressionDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            noise=noise,
            data_dist=data_dist,
            shuffle_samples=shuffle_samples,
        )

    @beartype
    def sample_task_params(self, n: int | None = None) -> dict[str, Tensor]:
        # Linear regression weights
        n = n if n is not None else self.n_samples
        w = torch.randn(n, self.x_dim + 1, self.y_dim) / (self.x_dim + 1) ** 0.5
        return {"w": w}

    @beartype
    def function(self, x: Tensor, task_params: dict[str, Tensor]) -> FloatTensor:
        # x: (bsz, n_samples, x_dim)
        # w: (bsz, x_dim + 1, y_dim)
        w = task_params["w"]
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        y = torch.bmm(x, w)
        return y
