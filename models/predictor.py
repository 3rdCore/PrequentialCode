from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

# import torch MLP
from torch.nn import functional as F


class Predictor(ABC, nn.Module):
    def __init__(self, z_dim: int | None):
        super().__init__()
        self.z_dim = z_dim

    @abstractmethod
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict different outputs given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Computed latent representation, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Predicted values for output, each with shape (samples, tasks, *).
        """
        pass


class MLPConcatPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        h_dim: int,
        n_layers: int,
        x_keys: tuple[str] = ("x",),
        z_keys: tuple[str] = ("z",),
        y_key: str = "y",
    ) -> None:
        super().__init__(z_dim)
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")
        self.x_keys = x_keys
        self.z_keys = z_keys
        self.y_key = y_key

        self.predictor = nn.Sequential(
            nn.Linear(x_dim + self.z_dim, self.h_dim),
            nn.ReLU(),
            *[nn.Linear(self.h_dim, self.h_dim), nn.ReLU()] * (n_layers - 2),
            nn.Linear(self.h_dim, self.y_dim),
        )

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Predicted values for y outputs, each with shape (samples, tasks, *).
        """
        x = torch.cat([x[key] for key in self.x_keys], dim=-1)
        z = torch.cat([z[key] for key in self.z_keys], dim=-1)
        y = self.predictor(torch.cat([x, z], dim=-1))
        return {self.y_key: y}


class LinearPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_key: str = "x",
        z_key: str = "z",
        y_key: str = "y",
    ) -> None:
        z_dim = (x_dim + 1) * y_dim
        super().__init__(z_dim)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, with shape (samples, tasks, x_dim) at `x_key`.
            z (dict[str, Tensor]): Aggregated context information, with shape (samples, tasks, (x_dim + 1) * y_dim) at `z_key`.

        Returns:
            dict[str, Tensor]: Predicted values for y output, with shape (samples, tasks, y_dim) at `y_key`.
        """
        x = x[self.x_key]
        z = z[self.z_key]
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        z = z.view(*z.shape[:-1], self.x_dim + 1, self.y_dim)
        y = torch.einsum("sbi,sbij->sbj", x, z)
        return {self.y_key: y}


class MLPPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        n_layers: int,
        x_key: str = "x",
        z_key: str = "z",
        y_key: str = "y",
    ) -> None:
        z_dim = (x_dim + 1) * h_dim + (n_layers - 2) * h_dim * h_dim + h_dim * y_dim
        super().__init__(z_dim)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key

        self.n_layers = n_layers
        self.h_dim = h_dim

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Peform a forward pass trought a MLP parameterized by z.

        Args:
            x (dict[str, Tensor]): Input data, with shape (samples, tasks, x_dim) at `x_key`.
            z (dict[str, Tensor]): MLP weights, with shape (samples, tasks, z_dim) at `z_key`.

        Returns:
            dict[str, Tensor]: Predicted values for y output, with shape (samples, tasks, y_dim) at `y_key`.
        """

        x = x[self.x_key]
        z = z[self.z_key]
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)

        w0_size = (self.x_dim + 1) * self.h_dim
        wi_size = self.h_dim * self.h_dim

        w0 = z[..., :w0_size].view(*z.shape[:-1], self.x_dim + 1, self.h_dim)
        w = [
            z[
                ...,
                w0_size + i * wi_size : w0_size + (i + 1) * wi_size,
            ].view(*z.shape[:-1], self.h_dim, self.h_dim)
            for i in range(0, self.n_layers - 2)
        ]
        w_last = z[..., -self.h_dim * self.y_dim :].view(*z.shape[:-1], self.h_dim, self.y_dim)

        y = F.relu(torch.einsum("sbi,sbij->sbj", x, w0))
        for i in range(0, self.n_layers - 2):
            y = F.relu(torch.einsum("sbi,sbij->sbj", y, w[i]))
        y = torch.einsum("sbi,sbij->sbj", y, w_last)
        return {self.y_key: y}
