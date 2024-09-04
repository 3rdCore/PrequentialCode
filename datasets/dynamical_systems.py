from abc import abstractmethod
from typing import Any, Iterable

import torch
import torch.nn as nn
from beartype import beartype
from scipy.stats import ortho_group
from torch import FloatTensor, Tensor

from datasets.synthetic import SyntheticDataset


class ODE(SyntheticDataset):
    @beartype
    def __init__(
        self,
        dim: int,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        dt: float = 0.01,
        nonlinearity: Any = nn.Identity(),
        noise: float = 0.0,
        initial_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        self.dim = dim
        self.x_len = x_dim // dim
        self.y_len = y_dim // dim
        self.dt = dt
        self.nonlinearity = nonlinearity
        self.noise = noise
        self.initial_dist = initial_dist

        super().__init__(
            n_tasks=n_tasks,
            n_samples=n_samples,
            shuffle_samples=shuffle_samples,
        )

    @torch.inference_mode()
    def gen_data(
        self,
        n_tasks: int,
        n_samples: int,
    ) -> tuple[dict[str, FloatTensor], dict[str, Iterable]]:
        # Transition matrices
        params = self.sample_task_params(n_tasks)

        # Initial conditions
        if self.initial_dist == "normal":
            x = [torch.randn(n_tasks, n_samples, self.dim)]
        elif self.initial_dist == "uniform":
            x = [2 * torch.rand(n_tasks, n_samples, self.dim) - 1]
        x = [self.nonlinearity(x[0])]

        # Trajectory
        for _ in range(self.x_len + self.y_len - 1):
            dxdt = self.dynamics(x[-1], params)
            x_next = x[-1] + dxdt * self.dt
            if self.noise > 0:
                x_next += self.noise * torch.randn_like(x_next) * self.dt
            x_next = self.nonlinearity(x_next)
            x.append(x_next)

        # Reshape
        x = torch.stack(x, dim=2)  # (n_tasks, n_samples, (x_len + y_len), dim)
        x = x.flatten(start_dim=2)  # (n_tasks, n_samples, (x_len + y_len) * dim)
        x, y = x.split([self.dim * self.x_len, self.dim * self.y_len], dim=-1)

        return {"x": x, "y": y}, params

    @abstractmethod
    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        """Sample parameters for each of the n_tasks

        Args:
            n_tasks (int): Number of tasks to generate.

        Returns:
            dict[str, Tensor]:
        """
        pass

    @abstractmethod
    def dynamics(self, xt: Tensor, params: dict[str, Tensor]) -> FloatTensor:
        """Gets the time derivative at xt using the params (dynamics parameters)

        Args:
            xt (Tensor): input data with shape (n_tasks, n_samples, x_dim)
            params (int): function parameters with shape (n_tasks, ...)

        Returns:
            FloatTensor: dxt/dt with shape (n_tasks, n_samples, x_dim)
        """
        pass


class TransitionMatrix(ODE):
    @beartype
    def __init__(
        self,
        dim: int,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        dt: float = 0.01,
        nonlinearity: Any = nn.Identity(),
        noise: float = 0.0,
        initial_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        self.nonlinearity = nonlinearity
        super().__init__(
            dim=dim,
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            dt=dt,
            nonlinearity=nonlinearity,
            noise=noise,
            initial_dist=initial_dist,
            shuffle_samples=shuffle_samples,
        )

    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        w = torch.randn(n_tasks, self.dim, self.dim)
        eigmax = torch.max(torch.linalg.eigvals(w).abs(), dim=-1).values
        w /= eigmax.unsqueeze(-1).unsqueeze(-1)
        return {"w": w}

    def dynamics(self, xt: Tensor, params: dict[str, Tensor]) -> FloatTensor:
        w = params["w"]
        dxdt = torch.einsum("bsx,bxy->bsy", xt, w)
        dxdt = self.nonlinearity(dxdt)
        return dxdt
