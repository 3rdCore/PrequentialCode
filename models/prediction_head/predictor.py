from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

# import torch MLP
from torch.nn import functional as F


class Predictor(ABC, nn.Module):
    def __init__(
        self,
        x_dim: dict[str, int],
        z_dim: dict[str, int],
        y_dim: dict[str, int],
    ) -> None:
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim

    @abstractmethod
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Predicted outputs, each with shape (samples, tasks, *).
        """
        pass

    @property
    @beartype
    def y_shape(self) -> dict[str, tuple[int, ...]]:
        return self.y_dim

    @property
    @beartype
    def x_shape(self) -> dict[str, tuple[int, ...]]:
        return self.x_dim


class VanillaPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: dict[str, int],
        z_dim: dict[str, int],
        y_dim: dict[str, int],
        hidden_dim: int,
        n_layers: int,
    ) -> None:
        super().__init__(x_dim, z_dim, y_dim)
        """
        Initialize a VanillaPredictor instance.

        Args:
            hidden_dim (int): The dimension of the hidden layer in the MLP.

        The predictor attribute is a dictionary where each key is a tuple of an input name and an output name,
        and each value is a Multi-Layer Perceptron (MLP) that takes the corresponding input and context information, and outputs a prediction on the corresponding output.s
        """

        self.hidden_dim = hidden_dim
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")

        self.predictor = {
            (input, output): nn.Sequential(
                nn.Linear(self.x_dim[input] + self.z_dim[input], self.hidden_dim),
                nn.ReLU(),
                *[nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()] * (n_layers - 2),
                nn.Linear(self.hidden_dim, self.y_dim[output]),
            )
            for input in self.x_dim
            for output in self.y_dim
        }  # Cartesian product of x_dim and y_dim

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples in context , tasks, *).

        Returns:
            dict[str, Tensor]: Predicted outputs, each with shape (samples, tasks, *).
        """
        y = {
            (input, output): self.predictor[(input, output)](torch.cat([x[input], z[input]], dim=-1))
            for input, output in self.predictor
        }
        return y
