from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ContextAggregator(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Aggregate context information.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Aggregated context z representing the model parameters given all previous samples, each with shape (samples, tasks, *). Note that each index of z along the 'samples' dimension should only depend on prior samples in x.
        """
        pass

    @property
    @abstractmethod
    def z_shape(self) -> dict[str, tuple[int, ...]]:
        """Shape of the aggregated context z.

        Returns:
            dict[str, tuple[int, ...]]: Shapes of the aggregated context z (excluding 'samples' and 'tasks' dimensions).
        """
        pass
