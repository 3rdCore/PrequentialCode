from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

from models.utils import CNNModule


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
        x_keys: tuple[str, str] = ("x", "y"),
        mlp_dim: int | None = None,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.x_keys = x_keys
        self.x_embedding = nn.Linear(x_dim, h_dim)
        self.x0_embedding = nn.Parameter(torch.zeros(1, 1, h_dim))
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=h_dim,
                nhead=n_heads,
                dim_feedforward=mlp_dim if mlp_dim is not None else 2 * h_dim,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
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
        x = torch.cat([x[name] for name in self.x_keys], dim=-1)
        x = self.x_embedding(x)
        x0 = self.x0_embedding.expand(1, x.shape[1], -1)
        x = torch.cat([x0, x], dim=0)
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[0])
        features = self.context_encoder.forward(x, mask=causal_mask, is_causal=True)
        z = self.projection(features)
        return {"z": z}

    @property
    @beartype
    def z_shape(self) -> dict[str, int]:
        return {"z": self.z_dim}


class VTOptimizer(Transfoptimizer):
    @beartype
    def __init__(
        self,
        x_dim: int,  # number of total features in the input
        z_dim: int,  # number of features in the output
        h_dim: int,
        n_layers: int,
        n_heads: int,
        feature_map: CNNModule,
        x_keys: tuple[str, str] = ("x", "y"),
        mlp_dim: int | None = None,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:

        super().__init__(x_dim, z_dim, h_dim, n_layers, n_heads, x_keys, mlp_dim, layer_norm_eps, dropout)
        self.feature_map = feature_map

    @beartype
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        x = {name: self.feature_map(x[name]) for name in self.x_keys}
        return super().forward(x)
