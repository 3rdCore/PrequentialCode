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
        layer_norm_eps: float = 1e-5,
        dropout: int = 0.0,
        batch_first: bool = False,  # TODO: make sure the input remains as defined
    ):
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.context_encoder = {
            name: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.x_dim[name],
                    nhead=n_heads,
                    dim_feedforward=self.z_dim[name],
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
        x = {
            name: torch.cat([torch.zeros_like(x[name][:1, ...]), x[name]], dim=0) for name in x
        }  # add a zero tensor to the beginning of the sequence

        mask = {
            name: torch.triu(torch.ones(x[name].shape[0], x[name].shape[0]), diagonal=1)
            .bool()
            .to(x[name].device)
            .unsqueeze(0)
            .repeat_interleave(x[name].shape[1] * x[name].shape[2], dim=0)
            for name in x
        }  # defines a causal transformer TODO double-check mask
        features = {name: self.context_encoder[name].forward(x[name], mask=mask[name]) for name in x}
        return features

    @property
    @beartype
    def z_shape(self) -> dict[str, tuple[int, ...]]:
        return self.z_dim
