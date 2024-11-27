from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn

from utils import CustomTransformerEncoder, PositionalEncoding


class ImplicitModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """Makes a prediction for the last sample.

        Args:
            x (dict[str, Tensor]): Input data, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Prediction on the last sample given all previous samples, each with shape (samples + 1, tasks, *). Note that each index along the 'samples' dimension should only depend on prior samples in x.
        """
        pass


class DecoderTransformer(ImplicitModel):
    @beartype
    def __init__(
        self,
        x_dim: int,  # number of total features in the input
        y_dim: int,  # number of features in the output
        h_dim: int,
        n_layers: int,
        n_heads: int,
        x_keys: tuple[str] = ("x",),
        y_keys: tuple[str] = ("y",),
        mlp_dim: int | None = None,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        max_seq_len: int = 5000,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_keys = x_keys
        self.y_keys = y_keys

        self.x_embedding = nn.Linear(x_dim, h_dim)
        self.y_embedding = nn.Linear(y_dim, h_dim)
        self.position_encoding = PositionalEncoding(h_dim, max_len=max_seq_len + 1)
        self.encoder = nn.TransformerEncoder(
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
        self.readout = nn.Linear(h_dim, y_dim)

        self.init_weights()

    @beartype
    def init_weights(self) -> None:
        for p in self.encoder.parameters():
            if p.dim() > 1:  # skip biases
                nn.init.xavier_uniform_(p)

    @beartype
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        y_dims = [y.shape[-1] for y in (x[name] for name in self.y_keys)]

        # Construct input tensor
        x, y = torch.cat([x[name] for name in self.x_keys], dim=-1), torch.cat(
            [x[name] for name in self.y_keys], dim=-1
        )
        x, y = self.x_embedding(x), self.y_embedding(y)
        seq = torch.stack([x, y], dim=1).view(
            x.shape[0] * 2, x.shape[1], -1
        )  # Interleave concatenated x and y
        seq = self.position_encoding(seq)

        # Encode the sequence
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.shape[0])
        seq = self.encoder.forward(seq, mask=causal_mask, is_causal=True)
        seq = torch.cat([seq[::2]])  # Drop the y token representations

        # Readout
        y = self.readout(seq)
        y = y.split(y_dims, dim=-1)
        y = {name: y[i] for i, name in enumerate(self.y_keys)}

        return y


class DecoderTransformer2(ImplicitModel):
    @beartype
    def __init__(
        self,
        x_dim: int,  # number of total features in the input
        y_dim: int,  # number of features in the output
        h_dim: int,
        n_layers: int,
        n_heads: int,
        x_keys: tuple[str] = ("x"),
        y_keys: tuple[str] = ("y"),
        mlp_dim: int | None = None,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_keys = x_keys
        slef.y_keys = y_keys

        self.x_embedding = nn.Linear(x_dim, h_dim)
        self.xy_embedding = nn.Linear(x_dim + y_dim, h_dim)
        self.encoder = CustomTransformerEncoder(
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
        self.readout = nn.Linear(h_dim, y_dim)
        self.init_weights()

    @beartype
    def init_weights(self) -> None:
        for p in self.encoder.parameters():
            if p.dim() > 1:  # skip biases
                nn.init.xavier_uniform_(p)

    @beartype
    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Construct input tensor
        x, y = torch.cat([x[name] for name in self.x_keys], dim=-1), torch.cat(
            [x[name] for name in self.y_keys], dim=-1
        )
        seq_x = self.x_embedding(x)
        seq_xy = self.xy_embedding(torch.cat([x, y], dim=-1))
        seq = torch.cat([seq_x, seq_xy], dim=-1)
        # Encode the sequence
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq.shape[0])
        seq = self.encoder.forward(seq, mask=causal_mask, is_causal=True)

        # Readout
        y = self.readout(seq)
        y = y.split(self.y_dim, dim=-1)
        y = {name: y[i] for i, name in enumerate(self.keys[1:])}

        return y
