from abc import ABC, abstractmethod
from typing import Any, Iterable

from torch import Tensor

from datasets.interfaces import TaskDistDataset


class SyntheticDataset(ABC, TaskDistDataset):
    def __init__(self, n_tasks: int, n_samples: int):
        """Note: subclasses should store their additional arguments before calling super().__init__()

        Args:
            n_tasks (int): Number of tasks to generate.
            n_samples (int): Number of samples per task.
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.n_samples = n_samples

        self.data, self.task_params = self.gen_data(n_samples)

    def __len__(self) -> int:
        name = list(self.data.keys())[0]
        return len(self.data[name])

    @abstractmethod
    def gen_data(self, n_samples: int) -> tuple[dict[str, Tensor], dict[str, Iterable]]:
        pass

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Any]]:
        return (
            {name: self.data[name][index] for name in self.data},
            {name: self.task_params[name][index] for name in self.task_params},
        )
