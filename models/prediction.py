from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn


class Predictor(ABC, nn.Module):
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
