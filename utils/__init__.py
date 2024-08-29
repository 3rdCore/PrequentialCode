import io
import math

import torch
from torch import FloatTensor, nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
    ):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Arguments:
            x: FloatTensor, shape ``(batch_size, seq_len, embedding_dim)``
        """
        x = x + self.pe[: x.shape[0]]
        return x


def torch_pca(x: torch.FloatTensor, center: bool = True, percent: bool = False):
    n, _ = x.shape
    # center points along axes
    if center:
        x = x - x.mean(dim=0)
    # perform singular value decomposition
    _, s, v = torch.linalg.svd(x)
    # extract components
    components = v.T
    explained_variance = torch.mul(s, s) / (n - 1)
    if percent:
        explained_variance = explained_variance / explained_variance.sum()
    return components, explained_variance
