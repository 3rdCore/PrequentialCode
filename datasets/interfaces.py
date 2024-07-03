import copy
import warnings
from typing import Any

import torch
from beartype import beartype
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from datasets.synthetic import AtomicSyntheticDataset, TaskDistDataset

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class ICLDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        train_dataset: TaskDistDataset,
        val_dataset: TaskDistDataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # setup

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=custom_collate_fn,
        )


class AtomicICLDataModule(LightningDataModule):
    @beartype
    def __init__(
        self,
        dataset: TaskDistDataset,
        batch_size: int,
        num_workers: int = 0,
        val_prop: float = 0.2,
        current_task: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["dataset"])
        self.dataset = dataset
        self.train_dataset, self.val_dataset = shuffle_train_val_split(
            dataset, val_prop=self.hparams.val_prop
        )
        self.switch_task(task=self.hparams.current_task)

    @beartype
    def switch_task(
        self,
        task: int | None = None,
        n_samples: int | None = None,
    ) -> None:
        # randomly select a task in multi_dataset self.multi_dataset.n_tasks
        self.train_dataset.current_task = (
            (self.train_dataset.current_task + 1) % self.dataset.n_tasks if task is None else task
        )

        self.val_dataset.current_task = (
            (self.val_dataset.current_task + 1) % self.dataset.n_tasks if task is None else task
        )
        if n_samples is not None:
            val_size = max(1, int(n_samples * self.hparams.val_prop))
            train_size = n_samples - val_size
            self.val_dataset.n_samples = val_size
            self.train_dataset.n_samples = train_size

        self.hparams.current_task = self.train_dataset.current_task

    @beartype
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=None,
        )

    @beartype
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=None,
        )


@beartype
def shuffle_train_val_split(dataset, val_prop):
    val_size = int(dataset.n_samples * val_prop)
    shuffle_idx = torch.randperm(dataset.n_samples)
    data = dataset.data
    data = {name: data[name][:, shuffle_idx] for name in data}
    val_data = {name: data[name][:, :val_size] for name in data}
    train_data = {name: data[name][:, val_size:] for name in data}

    train_dataset, val_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset.data, val_dataset.data = train_data, val_data
    train_dataset.n_samples, val_dataset.n_samples = dataset.n_samples - val_size, val_size

    train_dataset = AtomicSyntheticDataset(train_dataset)
    val_dataset = AtomicSyntheticDataset(val_dataset)
    return train_dataset, val_dataset


@beartype
def custom_collate_fn(batch):
    """1. Stacks the batches along dimension 1 so that sequences can be along dimension 0.
    2. Stacks the task parameters into a list.
    """
    batch_dict_x, batch_dict_params = {}, {}
    batch_x, batch_params = zip(*batch)
    for key in batch_x[0].keys():
        stacked = torch.stack([item[key] for item in batch_x], dim=1)
        batch_dict_x[key] = stacked
    if batch_params[0] is None:
        batch_dict_params = None
    else:
        for key in batch_params[0].keys():
            stacked = [item[key] for item in batch_params]
            if isinstance(stacked[0], Tensor):
                stacked = torch.stack(stacked, dim=0)
            batch_dict_params[key] = stacked
    return batch_dict_x, batch_dict_params
