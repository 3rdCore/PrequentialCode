from abc import abstractmethod
from enum import Enum

import numpy as np
import torch
from torch import LongTensor, Tensor

from utils import batched_bincount


class Datasets(Enum):
    MASTERMIND = "mastermind"
    ARC = "arc"
    PCFG = "pcfg"

    @classmethod
    def list(cls):
        return [d.value for d in cls._member_map_.values()]


class Dataset:
    def __init__(self, n_tasks: int, n_samples: int, dataset_type: Datasets):
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.dataset_type = dataset_type

    @abstractmethod
    def sample(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        pass


class Mastermind(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, code_length: int = 4, num_colours: int = 6):
        self.code_length = code_length
        self.num_colours = num_colours
        super().__init__(n_tasks, n_samples, Datasets.MASTERMIND)

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        x = torch.randint(0, self.num_colours, (self.n_tasks, self.n_samples, self.code_length))
        code = torch.randint(0, self.num_colours, (self.n_tasks, self.code_length))
        full_correct = (code.unsqueeze(1) == x).sum(dim=-1)
        correct_colors = torch.min(
            batched_bincount(code, max_val=self.num_colours).unsqueeze(1),
            batched_bincount(x, max_val=self.num_colours),
        ).sum(dim=-1)
        y = torch.stack([full_correct, correct_colors], dim=-1)
        return x.numpy(), y.numpy()


class Arc(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, n_vars: int = 10, n_vals: int = 10):
        self.n_vars = n_vars
        self.n_vals = n_vals
        super().__init__(n_tasks, n_samples, Datasets.ARC)

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        pass


class PCFG(Dataset):
    def __init__(self, n_tasks: int, n_samples: int, n_vars: int = 10, n_vals: int = 10):
        self.n_vars = n_vars
        self.n_vals = n_vals
        super().__init__(n_tasks, n_samples, Datasets.PCFG)

    def sample(self, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        pass


DatasetMap = {Datasets.MASTERMIND: Mastermind, Datasets.ARC: Arc, Datasets.PCFG: PCFG}


def get_dataset(dataset_type: str, n_tasks, n_samples, **kwargs) -> Dataset:
    dataset_type = Datasets._value2member_map_[dataset_type]
    return DatasetMap[dataset_type](n_tasks, n_samples, **kwargs)
