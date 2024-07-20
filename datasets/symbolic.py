from typing import Iterable, Literal

import numpy as np
import torch
from beartype import beartype
from torch import Tensor
from torch.nn import functional as F
from xarc import TASK_LIST, generate_data

from datasets.synthetic import SyntheticDataset


class ARCDataset(SyntheticDataset):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_colors: int,
        n_tasks: int,
        n_samples: int,
        has_ood: bool = False,
        ood_style: Literal["uniform", "random", "clustered"] = "uniform",
        shuffle_samples: bool = True,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_colors = n_colors
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.has_ood = has_ood
        self.ood_style = ood_style
        self.shuffle_samples = shuffle_samples

        super().__init__(n_tasks=n_tasks, n_samples=n_samples, shuffle_samples=shuffle_samples)

    @beartype
    @torch.inference_mode()
    def gen_ood_data(self, x: Tensor, task_dict_params):
        return {"x_ood": None, "y_ood": None}

    @beartype
    @torch.inference_mode()
    def gen_data(self, n_tasks: int, n_samples: int) -> tuple[dict[str, Tensor], dict[str, Iterable]]:
        # Generate dataset
        task_names = np.random.choice(TASK_LIST, n_tasks)
        x, y = generate_data(task_names, n_samples)
        x, y = torch.flatten(F.one_hot(x, self.n_colors).float(), start_dim=-3), torch.flatten(F.one_hot(y, self.n_colors).float(), start_dim=-3)
        return {"x": x, "y": y}, {"task_name": task_names}
