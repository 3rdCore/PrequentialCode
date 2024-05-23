from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

from models.meta_learner.context_aggregator import ContextAggregator


class Transfoptimizer(ContextAggregator):
    @beartype
    def __init__(
        self,
        n_heads: int,
        x_dim: dict[str, int],  # number of features in the input
        z_dim: dict[str, int],  # number of features in the output
        n_layers: int,
        layer_norm_eps: int,
        dropout: int = 0.0,
        batch_first: bool = False,  # TODO: make sure the input remains as defined
    ):
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.context_encoder = {
            name: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.x_dim,
                    nhead=n_heads,
                    dim_feedforward=self.z_dim,
                    dropout=dropout,
                    batch_first=batch_first,
                    layer_norm_eps=layer_norm_eps,
                ),
                num_layers=n_layers,
            )
            for name in x_dim
        }

        self.init_weights()

    @beartype
    def init_weights(self):
        for name in self.context_encoder:
            for p in self.context_encoder[name].parameters():
                if p.dim() > 1:  # skip biases
                    nn.init.xavier_uniform_(p)

    @beartype
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:

        features = {name: self.context_encoder[name].forward(x[name], is_causal=True) for name in x}
        return features

    @property
    @beartype
    def z_shape(self) -> dict[str, tuple[int, ...]]:
        return {"z": (self.z_dim,)}
