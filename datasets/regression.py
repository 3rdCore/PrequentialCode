import copy
from abc import abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from beartype import beartype
from pytorch_lightning.utilities.seed import isolate_rng
from torch import FloatTensor, Tensor

from datasets.interfaces import ICLDataModule
from datasets.synthetic import SyntheticDataset
from models.utils import FastFoodUpsample


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.normal_(m.bias)


class RegressionDataset(SyntheticDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        has_ood: bool = False,
        ood_style: Literal["shift_scale", "bimodal"] = "shift_scale",
        ood_shift: float | None = 2.0,
        ood_scale: float | None = 3.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.noise = noise
        self.has_ood = has_ood
        self.ood_style = ood_style
        self.ood_shift = ood_shift
        self.ood_scale = ood_scale
        self.data_dist = data_dist

        if ood_style == "shift_scale" and (ood_shift is None or self.ood_scale is None):
            raise ValueError("ood_shift and ood_scale must be provided for ood_style='shift_scale'")

        super().__init__(n_tasks=n_tasks, n_samples=n_samples, shuffle_samples=shuffle_samples)

    @beartype
    def gen_ood_data(self, x: Tensor, task_dict_params: dict[str, Tensor]):
        x_mean = torch.mean(x, dim=-2, keepdim=True)  # mean across samples

        if self.ood_style == "shift_scale":
            x_ood = x_mean + self.ood_shift + (x - x_mean) * self.ood_scale
        elif self.ood_style == "bimodal":
            x_min = x.min(dim=-2, keepdim=True).values
            x_max = x.max(dim=-2, keepdim=True).values
            x_ood = torch.where(x < x_mean, x_min + x_mean - x, x_max + x_mean - x)

        y_ood = self.function(x_ood, task_dict_params)

        return {"x_ood": x_ood, "y_ood": y_ood}

    @beartype
    @torch.inference_mode()
    def gen_data(
        self,
        n_tasks: int,
        n_samples: int,
    ) -> tuple[dict[str, Tensor], dict[str, Iterable]]:
        x = self.sample_x(n_tasks, n_samples)

        task_dict_params = self.sample_task_params(self.n_tasks)
        y = self.function(x, task_dict_params)
        y += self.noise * torch.randn_like(y)
        data_dict = {"x": x, "y": y}

        if self.has_ood:  # create ood data
            ood_dict = self.gen_ood_data(x, task_dict_params)
            data_dict.update(ood_dict)

        return data_dict, task_dict_params

    @beartype
    def sample_x(self, n_tasks: int, n_samples: int):
        if self.data_dist == "normal":
            x = torch.randn(n_tasks, n_samples, self.x_dim)
        elif self.data_dist == "uniform":
            x = 2 * torch.rand(n_tasks, n_samples, self.x_dim) - 1
        return x

    @abstractmethod
    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        """Sample parameters for each of the n_tasks

        Args:
            n_tasks (int): Number of tasks to generate.

        Returns:
            dict[str, Tensor]:
        """
        pass

    @abstractmethod
    def function(self, x: Tensor, params: dict[str, Tensor]) -> FloatTensor:
        """Applies the function defined using the params (function parameters)
        on the x(input data) to get y (output)

        Args:
            x (Tensor): input data with shape (n_tasks, n_samples, x_dim)
            params (int): function parameters with shape (n_tasks, ...)

        Returns:
            FloatTensor: y (output) with shape (n_tasks, n_samples, y_dim)
        """
        pass


class LinearRegression(RegressionDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        has_ood: bool = False,
        ood_style: str = "shift_scale",
        ood_shift: float | None = 2.0,
        ood_scale: float | None = 3.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            noise=noise,
            has_ood=has_ood,
            ood_style=ood_style,
            ood_shift=ood_shift,
            ood_scale=ood_scale,
            data_dist=data_dist,
            shuffle_samples=shuffle_samples,
        )

    @beartype
    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        # Linear regression weights
        n_tasks = n_tasks if n_tasks is not None else self.n_tasks
        w = torch.randn(n_tasks, self.x_dim + 1, self.y_dim) / (self.x_dim + 1) ** 0.5
        return {"w": w}

    @beartype
    def function(self, x: Tensor, task_params: dict[str, Tensor]) -> FloatTensor:
        # x: (n_tasks, n_samples, x_dim)
        # w: (n_tasks, x_dim + 1, y_dim)
        w = task_params["w"]
        x = torch.cat([x, torch.ones_like(x[:, :, :1])], dim=-1)
        y = torch.bmm(x, w)
        return y


class SinusoidRegression(RegressionDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        has_ood: bool = False,
        ood_style: str = "shift_scale",
        ood_shift: float | None = 2.0,
        ood_scale: float | None = 3.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
        n_freq: int = 3,
        fixed_freq: bool = False,
    ):
        assert y_dim == 1  # only 1D output supported for now
        self.n_freq = n_freq
        self.fixed_freq = fixed_freq
        if self.fixed_freq:
            with isolate_rng():
                torch.manual_seed(1)
                self.freqs = torch.rand(x_dim, n_freq).unsqueeze(0) * 5

        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            noise=noise,
            has_ood=has_ood,
            ood_style=ood_style,
            ood_shift=ood_shift,
            ood_scale=ood_scale,
            data_dist=data_dist,
            shuffle_samples=shuffle_samples,
        )

    @beartype
    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        # Linear regression weights
        n_tasks = n_tasks if n_tasks is not None else self.n_tasks
        amplitudes = torch.rand(n_tasks, self.x_dim, self.n_freq)
        if self.fixed_freq:
            freq = self.freqs
            w = amplitudes
        else:
            freq = torch.rand(n_tasks, self.x_dim, self.n_freq) * 5
            w = torch.cat([amplitudes, freq], dim=-1)
        return {"w": w}

    @beartype
    def function(self, x: Tensor, task_params: dict[str, Tensor]) -> FloatTensor:
        # x: (n_tasks, n_samples, x_dim)
        # w: (n_tasks, x_dim, 2 * n_freq)
        w = task_params["w"]
        if self.fixed_freq:
            amplitudes = w
            freq = self.freqs
        else:
            amplitudes, freq = w[..., : self.n_freq], w[..., self.n_freq :]
        x = torch.sin(x.unsqueeze(-1) * freq.unsqueeze(1))
        y = (x * amplitudes.unsqueeze(1)).sum(dim=-1).sum(dim=-1, keepdim=True)
        return y


class MLPRegression(RegressionDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        has_ood: bool = False,
        ood_style: str = "shift_scale",
        ood_shift: float | None = 2.0,
        ood_scale: float | None = 3.0,
        data_dist: str = "normal",
        activation: str = "relu",
        n_layers: int = 2,
        hidden_dim: int = 64,
        shuffle_samples: bool = True,
    ):
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()

        layers = [
            nn.Linear(x_dim, self.hidden_dim),
            self.activation,
        ]
        for _ in range(self.n_layers - 2):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self.activation)
        layers.append(nn.Linear(self.hidden_dim, y_dim))
        self.model = torch.nn.Sequential(*layers)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")

        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            noise=noise,
            has_ood=has_ood,
            ood_style=ood_style,
            ood_shift=ood_shift,
            ood_scale=ood_scale,
            data_dist=data_dist,
            shuffle_samples=shuffle_samples,
        )

    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        # Linear regression weights
        n_tasks = n_tasks if n_tasks is not None else self.n_tasks
        models = [copy.deepcopy(self.model).apply(init_weights) for _ in range(n_tasks)]
        return {"w": models}

    @torch.inference_mode()
    def function(self, x: Tensor, task_params: dict[str, Tensor]) -> FloatTensor:
        # x: (n_tasks, n_samples, x_dim)
        ys = [task_params["w"][idx](x[idx].to(self.device)) for idx in range(x.shape[0])]
        return torch.stack(ys)


class MLPLowRankRegression(RegressionDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        has_ood: bool = False,
        ood_style: str = "shift_scale",
        ood_shift: float | None = 2.0,
        ood_scale: float | None = 3.0,
        data_dist: str = "normal",
        activation: str = "relu",
        n_layers: int = 2,
        hidden_dim: int = 64,
        shuffle_samples: bool = True,
    ):
        if n_layers < 2:
            raise ValueError("n_layers must be at least 2")

        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()

        self.params_0 = nn.ModuleList(
            [nn.Linear(x_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 2)]
            + [nn.Linear(hidden_dim, y_dim)]
        )
        param_dims = [(hidden_dim, x_dim), (hidden_dim,)]
        for _ in range(n_layers - 2):
            param_dims += [(hidden_dim, hidden_dim), (hidden_dim,)]
        param_dims += [(y_dim, hidden_dim), (y_dim,)]
        self.ff_upsample = FastFoodUpsample(z_dim, param_dims)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.params_0 = self.params_0.to(self.device)
            self.ff_upsample = self.ff_upsample.to(self.device)
        else:
            self.device = torch.device("cpu")

        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            noise=noise,
            has_ood=has_ood,
            ood_style=ood_style,
            ood_shift=ood_shift,
            ood_scale=ood_scale,
            data_dist=data_dist,
            shuffle_samples=shuffle_samples,
        )

    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        # Linear regression weights
        z = torch.randn(n_tasks, self.z_dim)
        return {"z": z}

    @torch.inference_mode()
    def function(self, x: Tensor, task_params: dict[str, Tensor]) -> FloatTensor:
        # x: (n_tasks, n_samples, x_dim)
        # z: (n_tasks, z_dim)
        x = x.to(self.device)
        z = task_params["z"].to(self.device)
        z = self.ff_upsample(z)

        for i in range(self.n_layers):
            w0 = self.params_0[i].weight
            b0 = self.params_0[i].bias
            wd = z[2 * i].view(-1, *w0.shape)
            bd = z[2 * i + 1].view(-1, *b0.shape)
            w = w0.unsqueeze(0) + wd
            b = b0.unsqueeze(0) + bd
            x = torch.einsum("bsi,bji->bsj", x, w)
            x = x + b.unsqueeze(1)
            if i < self.n_layers - 1:
                x = self.activation(x)

        return x


class TchebyshevRegression(RegressionDataset):
    @beartype
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_tasks: int,
        n_samples: int,
        noise: float = 0.0,
        degree: int = 5,
        has_ood: bool = True,
        ood_style: str = "shift_scale",
        ood_shift: float | None = 2.0,
        ood_scale: float | None = 3.0,
        data_dist: str = "normal",
        shuffle_samples: bool = True,
    ):
        assert y_dim == 1  # only 1D output supported for now
        self.degree = degree

        super().__init__(
            x_dim=x_dim,
            y_dim=y_dim,
            n_tasks=n_tasks,
            n_samples=n_samples,
            noise=noise,
            has_ood=has_ood,
            ood_style=ood_style,
            ood_shift=ood_shift,
            ood_scale=ood_scale,
            data_dist=data_dist,
            shuffle_samples=shuffle_samples,
        )

    @beartype
    def sample_task_params(self, n_tasks: Optional[int] = None) -> Dict[str, Tensor]:
        # Chebyshev polynomial coefficients
        n_tasks = n_tasks if n_tasks is not None else self.n_tasks
        coeffs = torch.randn(n_tasks, (self.x_dim) * (self.degree + 1), self.y_dim)
        return {"coeffs": coeffs}

    @beartype
    def function(self, x: Tensor, task_params: Dict[str, Tensor]) -> Tensor:
        coeffs = task_params["coeffs"]
        T = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            Tn = 2 * x * T[-1] - T[-2]  # recurrence relation
            T.append(Tn)
        T = torch.stack(T, dim=-1)
        y = torch.einsum("ndy,ntyd->nty", coeffs, T)
        mean = y.mean(dim=0, keepdim=True)  # ugly normalization
        std = y.std(dim=0, keepdim=True)
        y = (y - mean) / std
        return y
