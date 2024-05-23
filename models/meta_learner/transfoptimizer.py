from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

from models.meta_learner.context_aggregator import ContextAggregator


class Transforptimizer(ContextAggregator):
    @beartype
    def __init__(
        self,
        n_heads,
        x_dim,  # number of features in the input
        z_dim,  # number of features in the output
        n_layers,
        layer_norm_eps,
        dropout=0.0,
        batch_first=False,  # TODO: make sure the input remains as defined
    ):
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.context_encoder = nn.TransformerEncoder(
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

        self.init_weights()

    @beartype
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # is causal !!!
        # which chanel is the input ?
        feature = self.context_encoder.forward(x, src_key_padding_mask=None, is_causal=True)
        return {"z": feature}

    @property
    @beartype
    def z_shape(self) -> dict[str, tuple[int, ...]]:
        return {"z": (self.z_dim,)}
