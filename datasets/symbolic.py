from abc import abstractmethod
from typing import Iterable

import torch
from beartype import beartype
from torch import FloatTensor, LongTensor, Tensor
from torch.nn import functional as F

from datasets.synthetic import SyntheticDataset
from utils import bincount_batched


class SymbolicDataset(SyntheticDataset):
    @beartype
    def __init__(
        self,
        x_num_vars: int,
        y_num_vars: int,
        x_num_vals: int,
        y_num_vals: int,
        n_tasks: int,
        n_samples: int,
        shuffle_samples: bool = True,
        one_hot_x: bool = True,
        one_hot_y: bool = True,
    ):
        self.x_num_vars = x_num_vars
        self.y_num_vars = y_num_vars
        self.x_num_vals = x_num_vals
        self.y_num_vals = y_num_vals
        self.one_hot_x = one_hot_x
        self.one_hot_y = one_hot_y

        super().__init__(
            n_tasks=n_tasks,
            n_samples=n_samples,
            shuffle_samples=shuffle_samples,
        )

    @torch.inference_mode()
    def gen_data(
        self,
        n_tasks: int,
        n_samples: int,
    ) -> tuple[dict[str, FloatTensor], dict[str, Iterable]]:
        x = self.sample_x(n_tasks, n_samples)
        task_dict_params = self.sample_task_params(self.n_tasks)
        y = self.function(x, task_dict_params)
        if self.one_hot_x:
            x = F.one_hot(x, self.x_num_vals).float().flatten(start_dim=-2)
        if self.one_hot_y:
            y = F.one_hot(y, self.y_num_vals).float().flatten(start_dim=-2)
        return {"x": x, "y": y}, task_dict_params

    @beartype
    def sample_x(self, n_tasks: int, n_samples: int) -> LongTensor:
        """Sample input data for each of the n_tasks and n_samples

        Args:
            n_tasks (int): Number of tasks to generate.
            n_samples (int): Number of samples to generate for each task.

        Returns:
            LongTensor: Input data with shape (n_tasks, n_samples, x_dim)
        """
        pass

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
    def function(self, x: LongTensor, params: dict[str, Tensor]) -> LongTensor:
        """Applies the function defined using the params (function parameters)
        on the x(input data) to get y (output)

        Args:
            x (LongTensor): input data with shape (n_tasks, n_samples, x_dim)
            params (int): function parameters with shape (n_tasks, ...)

        Returns:
            LongTensor: y (output) with shape (n_tasks, n_samples, y_dim)
        """
        pass


class Mastermind(SymbolicDataset):
    @beartype
    def __init__(
        self,
        n_tasks: int,
        n_samples: int,
        code_length: int = 8,
        num_colours: int = 6,
        shuffle_samples: bool = True,
        one_hot_x: bool = True,
        one_hot_y: bool = True,
    ):
        self.code_length = code_length
        self.num_colours = num_colours
        super().__init__(
            x_num_vars=code_length,
            y_num_vars=2,
            x_num_vals=num_colours,
            y_num_vals=code_length + 1,  # Includes 0
            n_tasks=n_tasks,
            n_samples=n_samples,
            shuffle_samples=shuffle_samples,
            one_hot_x=one_hot_x,
            one_hot_y=one_hot_y,
        )

    def sample_x(self, n_tasks: int, n_samples: int) -> LongTensor:
        x = torch.randint(0, self.num_colours, (n_tasks, n_samples, self.code_length))
        return x

    def sample_task_params(self, n_tasks: int | None = None) -> dict[str, Tensor]:
        code = torch.randint(
            low=0, high=self.num_colours - 1, size=(n_tasks, self.code_length)
        )
        return {"code": code}

    def function(self, x: LongTensor, params: dict[str, Tensor]) -> LongTensor:
        code = params["code"]
        full_correct = (code.unsqueeze(1) == x).sum(dim=-1)
        colour_correct = torch.min(
            bincount_batched(code, max_val=self.num_colours).unsqueeze(dim=1),
            bincount_batched(x, max_val=self.num_colours),
        ).sum(dim=-1)
        return torch.stack([full_correct, colour_correct], dim=-1)
