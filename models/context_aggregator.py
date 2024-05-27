from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn


class ContextAggregator(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Aggregate context information.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Aggregated context z representing the model parameters given all previous samples, each with shape (samples + 1, tasks, *). Note that each index of z along the 'samples' dimension should only depend on prior samples in x.
        """
        pass

    @property
    @abstractmethod
    def z_shape(self) -> dict[str, int]:
        """Shape of the aggregated context z.

        Returns:
            dict[str, int]: Shapes of the aggregated context z (excluding 'samples' and 'tasks' dimensions).
        """
        pass


class Transfoptimizer(ContextAggregator):
    @beartype
    def __init__(
        self,
        x_dim: int,  # number of total features in the input
        z_dim: int,  # number of features in the output
        h_dim: int,
        n_layers: int,
        n_heads: int,
        mlp_dim: int | None = None,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.embedding = nn.Linear(x_dim, h_dim)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim,
                nhead=n_heads,
                dim_feedforward=mlp_dim if mlp_dim is not None else 2 * h_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            ),
            num_layers=n_layers,
        )
        if z_dim != h_dim:
            self.projection = nn.Linear(h_dim, z_dim)
        else:
            self.projection = nn.Identity()

        self.init_weights()

    @beartype
    def init_weights(self) -> None:
        for p in self.context_encoder.parameters():
            if p.dim() > 1:  # skip biases
                nn.init.xavier_uniform_(p)

    @beartype
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x = torch.cat([x[name] for name in x], dim=-1)
        x = torch.cat(
            [torch.zeros_like(x[:1, ...]), x], dim=0
        )  # add a zero tensor to the beginning of the sequence
        x = self.embedding(x)
        features = self.context_encoder.forward(x, is_causal=True)
        z = self.projection(features)
        return {"z": z}

    @property
    @beartype
    def z_shape(self) -> dict[str, int]:
        return {"z": self.z_dim}
