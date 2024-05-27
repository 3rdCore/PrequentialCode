from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

# import torch MLP
from torch.nn import functional as F


class Predictor(ABC, nn.Module):
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
        super().__init__()
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
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples in context, tasks, *).

        Returns:
            dict[str, Tensor]: Predicted values for y outputs, each with shape (samples, tasks, *).
        """
        x = torch.cat([x[key] for key in self.x_keys], dim=-1)
        z = torch.cat([z[key] for key in self.z_keys], dim=-1)
        y = self.predictor(torch.cat([x, z], dim=-1))
        return {self.y_key: y}


# class MetaLinearPredictor(Predictor):
#     @beartype
#     def __init__(
#         self,
#         x_dim: dict[str, int],
#         z_dim: dict[str, int],
#         y_dim: dict[str, int],
#     ) -> None:
#         """
#         This type of predictor computes a linear operation of the latent z and the input x. Here the latent z act as the weights of the linear operation.
#         """
#         super().__init__(x_dim, z_dim, y_dim)

#         total_output_dim = sum(y_dim.values())
#         for input_name in z_dim:
#             if z_dim[input_name] / (x_dim[input_name] + 1) != total_output_dim:
#                 raise ValueError(
#                     f"dimension mismatch between z_dim and y_dim for {input_name}"
#                 )  # check dim as z will be reshaped accordingly

#     @beartype
#     def forward(
#         self, x: dict[str, Tensor], z: dict[str, Tensor]
#     ) -> dict[tuple[str, str], Tensor]:
#         """Predict given inputs x and model parameters z.

#         Args:
#             x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).
#             z (dict[str, Tensor]): Aggregated context information, each with shape (samples in context , tasks, *).

#         Returns:
#             dict[tuple[str, str], Tensor]: Predicted values for each tuple (input,output), each with shape (samples, tasks, *).
#         """
#         # increase feature dim of x for bias

#         x = {
#             name: torch.cat(
#                 [x[name], torch.ones(*x[name].shape[:-1], 1).to(x[name].device)], dim=-1
#             )
#             for name in x
#         }
#         z_resized = (
#             {}
#         )  # (samples, tasks, x_dim + 1, y_dim) for each input/output pair : the last dim of z is reshaped to account for multiple outputs
#         for input in self.z_dim:
#             idx = 0
#             for output in self.y_dim:
#                 z_resized[(input, output)] = z[input][
#                     :, :, idx : idx + (self.y_dim[output]) * (self.x_dim[input] + 1)
#                 ].reshape(
#                     *z[input].shape[:-1], self.x_dim[input] + 1, self.y_dim[output]
#                 )
#                 idx += self.y_dim[output] * (self.x_dim[input] + 1)

#         y = {
#             (input, output): torch.einsum(
#                 "sbi,sbij->sbj", x[input], z_resized[(input, output)]
#             )
#             for input in self.x_dim
#             for output in self.y_dim
#         }
#         return y
