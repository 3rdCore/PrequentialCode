import json
import os
import pickle
from typing import Any, Iterable, Literal

import numpy as np
import torch
from beartype import beartype
from torch import Tensor
from torch.nn import functional as F
from xarc import TASK_LIST, generate_data

from datasets.synthetic import SyntheticDataset


class ARCSymbolic(SyntheticDataset):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        n_colors: int,
        n_tasks: int,
        n_samples: int,
        data_path: str,
        has_ood: bool = False,
        ood_style: Literal["uniform", "random", "clustered"] = "uniform",
        shuffle_samples: bool = True,
    ):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_colors = n_colors
        self.data_path = data_path
        self.has_ood = has_ood
        self.ood_style = ood_style
        self.shuffle_samples = shuffle_samples
        # validate data path and data size
        self.validate(data_path, n_tasks, n_samples)
        super().__init__(n_tasks=n_tasks, n_samples=n_samples, shuffle_samples=shuffle_samples)

    @beartype
    def validate(self, data_path: str, n_tasks: int, n_samples: int) -> bool:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found at {data_path}")
        if len(os.listdir(data_path)) != (n_tasks + 1):
            raise ValueError(f"Number of tasks in data path {data_path} do not match n_tasks {n_tasks}")
        idx = np.random.randint(n_tasks)
        task_file_path = os.path.join(data_path, f"task_{idx}.pkl")
        with open(task_file_path, "rb") as f:
            task_data = pickle.load(f)
        if len(task_data["x"]) != n_samples:
            raise ValueError(f"Number of samples in task {idx} do not match n_samples {n_samples}")
        return True

    @beartype
    @torch.inference_mode()
    def gen_ood_data(self, x: Tensor, task_dict_params):
        return {"x_ood": None, "y_ood": None}

    @beartype
    @torch.inference_mode()
    def gen_data(self, n_tasks: int, n_samples: int) -> tuple[dict[str, Tensor], dict[str, Iterable]]:
        # Load task_params
        task_params_file_path = os.path.join(self.data_path, "task_params.json")
        if not os.path.exists(task_params_file_path):
            raise FileNotFoundError(f"Task params file not found at {task_params_file_path}")
        with open(os.path.join(self.data_path, "task_params.json"), "r") as f:
            task_params = json.load(f)

        return {}, task_params

    @beartype
    def transform(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: torch.flatten(F.one_hot(data[name], self.n_colors).float(), start_dim=-3) for name in data
        }

    @beartype
    def __len__(self) -> int:
        return self.task_params["n_tasks"]

    @beartype
    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Any]]:
        data_file_path = os.path.join(self.data_path, f"task_{index}.pkl")
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file not found at {data_file_path}")
        with open(data_file_path, "rb") as f:
            data = pickle.load(f)

        self.data = self.transform(data)
        if self.shuffle_samples:
            shuffle_idx = torch.randperm(self.n_samples)
            self.data = {name: self.data[name][shuffle_idx] for name in self.data}

        # task params
        task_name = self.task_params["task_names"][index]
        canvas_size = self.task_params["canvas_size"]
        task_params = {"keys": list(self.data.keys()), "task_name": task_name, "canvas_size": canvas_size}
        return self.data, task_params


class ARCSymbolic2D(SyntheticDataset):
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
        x, y, task_params = generate_data(task_names, n_samples)
        return {"x": x, "y": y}, {"task_name": task_names}
