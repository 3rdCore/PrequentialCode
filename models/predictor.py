from abc import ABC, abstractmethod

import torch
from beartype import beartype
from torch import Tensor, nn
from torch.nn import functional as F

from models.utils import FastFoodUpsample


class Predictor(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict different outputs given inputs x and latent representation z.

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
            raise ValueError("n_layers must be at least 3")
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
            z (dict[str, Tensor]): Aggregated context information, each with shape (samples, tasks, *).

        Returns:
            dict[str, Tensor]: Predicted values for y outputs, each with shape (samples, tasks, *).
        """
        x = torch.cat([x[key] for key in self.x_keys], dim=-1)
        z = torch.cat([z[key] for key in self.z_keys], dim=-1)
        y = self.predictor(torch.cat([x, z], dim=-1))
        return {self.y_key: y}


class LinearPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_key: str = "x",
        z_key: str = "z",
        y_key: str = "y",
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key
        self.z_dim = (x_dim + 1) * y_dim

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, with shape (samples, tasks, x_dim) at `x_key`.
            z (dict[str, Tensor]): Aggregated context information, with shape (samples, tasks, (x_dim + 1) * y_dim) at `z_key`.

        Returns:
            dict[str, Tensor]: Predicted values for y output, with shape (samples, tasks, y_dim) at `y_key`.
        """
        x = x[self.x_key]
        z = z[self.z_key]
        x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        z = z.view(*z.shape[:-1], self.x_dim + 1, self.y_dim)
        y = torch.einsum("sbi,sbij->sbj", x, z)
        return {self.y_key: y}


class MLPPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        h_dim: int,
        n_layers: int,
        x_key: str = "x",
        z_key: str = "z",
        y_key: str = "y",
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key
        self.z_dim = (x_dim + 1) * h_dim + (n_layers - 2) * (h_dim + 1) * h_dim + (h_dim + 1) * y_dim

        self.n_layers = n_layers
        self.h_dim = h_dim

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        x = x[self.x_key]
        z = z[self.z_key]

        w0_size = (self.x_dim) * self.h_dim
        wi_size = (self.h_dim) * self.h_dim

        w0 = z[..., :w0_size].view(*z.shape[:-1], self.x_dim, self.h_dim)
        b0 = z[..., w0_size : w0_size + self.h_dim].view(*z.shape[:-1], self.h_dim)

        w = [
            z[
                ...,
                w0_size
                + self.h_dim
                + i * (wi_size + self.h_dim) : w0_size
                + self.h_dim
                + i * (wi_size + self.h_dim)
                + wi_size,
            ].view(*z.shape[:-1], self.h_dim, self.h_dim)
            for i in range(0, self.n_layers - 2)
        ]
        b = [
            z[
                ...,
                w0_size
                + self.h_dim
                + i * (wi_size + self.h_dim)
                + wi_size : w0_size
                + self.h_dim
                + i * (wi_size + self.h_dim)
                + wi_size
                + self.h_dim,
            ].view(*z.shape[:-1], self.h_dim)
            for i in range(0, self.n_layers - 2)
        ]

        w_last_size = self.h_dim * self.y_dim
        w_last = z[..., -(w_last_size + self.y_dim) : -self.y_dim].view(*z.shape[:-1], self.h_dim, self.y_dim)
        b_last = z[..., -self.y_dim :].view(*z.shape[:-1], self.y_dim)

        y = F.relu(torch.einsum("sbi,sbij->sbj", x, w0) + b0)
        for i in range(0, self.n_layers - 2):
            y = F.relu(torch.einsum("sbi,sbij->sbj", y, w[i]) + b[i])
        y = torch.einsum("sbi,sbij->sbj", y, w_last) + b_last

        return {self.y_key: y}


class MLPLowRankPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        y_dim: int,
        h_dim: int,
        n_layers: int,
        x_key: str = "x",
        z_key: str = "z",
        y_key: str = "y",
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key

        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")
        self.n_layers = n_layers
        self.h_dim = h_dim

        self.params_0 = nn.ModuleList(
            [nn.Linear(x_dim, h_dim)]
            + [nn.Linear(h_dim, h_dim) for _ in range(n_layers - 2)]
            + [nn.Linear(h_dim, y_dim)]
        )
        param_shapes = [(h_dim, x_dim), (h_dim,)]
        for _ in range(n_layers - 2):
            param_shapes += [(h_dim, h_dim), (h_dim,)]
        param_shapes += [(y_dim, h_dim), (y_dim,)]
        self.ff_upsample = nn.ModuleList([FastFoodUpsample(z_dim, ps) for ps in param_shapes])

    @beartype
    def to(self, device):
        super().to(device)
        self.ff_upsample = self.ff_upsample.to(device)
        return self

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Peform a forward pass through an MLP modulated in a low-rank way by z.

        Args:
            x (dict[str, Tensor]): Input data, with shape (samples, tasks, x_dim) at `x_key`.
            z (dict[str, Tensor]): MLP weights, with shape (samples, tasks, z_dim) at `z_key`.

        Returns:
            dict[str, Tensor]: Predicted values for y output, with shape (samples, tasks, y_dim) at `y_key`.
        """
        x = x[self.x_key]
        z = z[self.z_key]  # (samples, tasks, z_dim)

        seq, batch, _ = z.shape
        z = z.view(seq * batch, -1)  # (samples * tasks, z_dim)
        z = [upsample(z) for upsample in self.ff_upsample]

        for i in range(self.n_layers):
            w0 = self.params_0[i].weight
            b0 = self.params_0[i].bias
            wd = z[2 * i].view(seq, batch, *w0.shape)
            bd = z[2 * i + 1].view(seq, batch, *b0.shape)
            w = w0.unsqueeze(0).unsqueeze(0) + wd
            b = b0.unsqueeze(0).unsqueeze(0) + bd
            x = torch.einsum("sbi,sbji->sbj", x, w)
            x = x + b
            if i < self.n_layers - 1:
                x = F.relu(x)

        return {self.y_key: x}


class FourierPredictor(Predictor):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_key: str = "x",
        z_key: str = "z",
        y_key: str = "y",
        n_freq: int = 12,
    ):
        assert y_dim == 1  # only 1D output supported for now
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key
        self.n_freq = n_freq
        freqs = torch.arange(1, n_freq + 1, dtype=torch.float32) * 2 * torch.pi / 5
        self.register_buffer("freqs", freqs)

    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """Predict given inputs x and model parameters z.

        Args:
            x (dict[str, Tensor]): Input data, with shape (samples, tasks, x_dim) at `x_key`.
            z (dict[str, Tensor]): Aggregated context information, with shape (samples, tasks, x_dim * n_freqs * 2) at `z_key`.

        Returns:
            dict[str, Tensor]: Predicted values for y output, with shape (samples, tasks, y_dim) at `y_key`.
        """
        x = x[self.x_key]
        z = z[self.z_key]
        amplitudes = z.view(*z.shape[:-1], self.x_dim, 2 * self.n_freq)
        x_sin = torch.sin(x.unsqueeze(-1) * self.freqs.unsqueeze(0))
        x_cos = torch.cos(x.unsqueeze(-1) * self.freqs.unsqueeze(0))
        x = torch.cat([x_sin, x_cos], dim=-1)
        y = (x * amplitudes).sum(dim=-1).sum(dim=-1, keepdim=True)


class TchebyshevPredictor(Predictor):
    @beartype
    def __init__(self, degree: int, x_key: str = "x", z_key: str = "z", y_key: str = "y") -> None:
        """
        Initialize the TchebyshevPredictor.

        Args:
            degree (int): Degree of the Tchebyshev polynomial.
            x_key (str): Key for the input tensor x in the input dictionary.
            z_key (str): Key for the latent tensor z in the input dictionary.
            y_key (str): Key for the output tensor y in the output dictionary.
        """
        super().__init__()
        self.degree = degree
        self.x_key = x_key
        self.z_key = z_key
        self.y_key = y_key

    @beartype
    def forward(self, x: dict[str, Tensor], z: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Perform a forward pass to predict y using Tchebyshev polynomials.

        Args:
            x (dict[str, Tensor]): Input data, with shape (samples, tasks, 1) at `x_key`.
            z (dict[str, Tensor]): Latent weights, with shape (samples, tasks, degree + 1) at `z_key`.

        Returns:
            dict[str, Tensor]: Predicted values for y output, with shape (samples, tasks, 1) at `y_key`.
        """
        x = x[self.x_key]  # (samples, tasks, 1)
        z = z[self.z_key]  # (samples, tasks, degree + 1)

        # Prepare Tchebyshev polynomials
        tchebyshev_basis = [torch.ones_like(x), x]  # T_0(x) = 1, T_1(x) = x
        for i in range(2, self.degree + 1):
            t_next = 2 * x * tchebyshev_basis[-1] - tchebyshev_basis[-2]  # T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
            tchebyshev_basis.append(t_next)

        # Stack the basis to shape (samples, tasks, degree + 1)
        tchebyshev_basis = torch.cat(tchebyshev_basis, dim=-1)

        # Compute the weighted sum of Tchebyshev polynomials
        y = torch.sum(tchebyshev_basis * z, dim=-1, keepdim=True)  # (samples, tasks, 1)

        return {self.y_key: y}
