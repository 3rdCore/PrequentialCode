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
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[tuple[str, str], Tensor]:
        """Predict different outputs given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples, tasks, *).

        Returns:
            dict[tuple[str, str], Tensor]: Predicted values for each pair (input_name, output_name), each with shape (samples, tasks, *).
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

        we define a Multi-Layer Perceptron (MLP) for each tuple (input name, output name). each MLP takes the corresponding input and context information, and outputs a prediction on the corresponding output space.
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
        }  # Cartesian product of x_dim and y_dim, even tough most of the time it will be only one element for each (e.g. predict y given x)

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[tuple[str, str], Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples in context, tasks, *).

        Returns:
            dict[tuple[str, str], Tensor]: Predicted values for each tuple (input,output), each with shape (samples, tasks, *).
        """
        y = {
            (input, output): self.predictor[(input, output)](torch.cat([x[input], z[input]], dim=-1))
            for input, output in self.predictor
        }
        return y


class MetaLinearPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: dict[str, int],
        z_dim: dict[str, int],
        y_dim: dict[str, int],
    ) -> None:
        """
        This type of predictor computes a linear operation of the latent z and the input x. Here the latent z act as the weights of the linear operation.
        """
        super().__init__(x_dim, z_dim, y_dim)

        total_output_dim = sum(y_dim.values())
        for input_name in z_dim:
            if z_dim[input_name] / (x_dim[input_name] + 1) != total_output_dim:
                raise ValueError(
                    f"dimension mismatch between z_dim and y_dim for {input_name}"
                )  # check dim as z will be reshaped accordingly

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[tuple[str, str], Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples in context , tasks, *).

        Returns:
            dict[tuple[str, str], Tensor]: Predicted values for each tuple (input,output), each with shape (samples, tasks, *).
        """
        # increase feature dim of x for bias

        x = {
            name: torch.cat([x[name], torch.ones(*x[name].shape[:-1], 1).to(x[name].device)], dim=-1)
            for name in x
        }
        z_resized = (
            {}
        )  # (samples, tasks, x_dim + 1, y_dim) for each input/output pair : the last dim of z is reshaped to account for multiple outputs
        for input in self.z_dim:
            idx = 0
            for output in self.y_dim:
                z_resized[(input, output)] = z[input][
                    :, :, idx : idx + (self.y_dim[output]) * (self.x_dim[input] + 1)
                ].reshape(*z[input].shape[:-1], self.x_dim[input] + 1, self.y_dim[output])
                idx += self.y_dim[output] * (self.x_dim[input] + 1)

        y = {
            (input, output): torch.einsum("sbi,sbij->sbj", x[input], z_resized[(input, output)])
            for input in self.x_dim
            for output in self.y_dim
        }
        return y
