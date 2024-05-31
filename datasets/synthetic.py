from abc import abstractmethod
from typing import Any, Iterable

import torch
from beartype import beartype
from torch import Tensor

from datasets.interfaces import TaskDistDataset


class SyntheticDataset(TaskDistDataset):
    @beartype
    def __init__(
        self,
        n_tasks: int,
        n_samples: int,
        shuffle_samples: bool = True,
    ):
        """Note: subclasses should store their additional arguments before calling super().__init__()

        Args:
            n_tasks (int): Number of tasks to generate.
            n_samples (int): Number of samples per task.
            shuffle_samples (bool): Whether to shuffle samples each time a task is retrieved.
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.shuffle_samples = shuffle_samples

        self.data, self.task_params = self.gen_data(n_tasks, n_samples)

    @beartype
    def __len__(self) -> int:
        name = list(self.data.keys())[0]
        return len(self.data[name])

    @abstractmethod
    def gen_data(
        self,
        n_tasks: int,
        n_samples: int,
    ) -> tuple[dict[str, Tensor], dict[str, Iterable]]:
        pass

    @beartype
    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Any]]:
        data = {name: self.data[name][index] for name in self.data}
        task_params = {name: self.task_params[name][index] for name in self.task_params}
        if self.shuffle_samples:
            shuffle_idx = torch.randperm(self.n_samples)
            data = {name: data[name][shuffle_idx] for name in data}
        return data, task_params
