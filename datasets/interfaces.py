import warnings
from abc import ABC, abstractmethod
from typing import Any

import torch
from beartype import beartype
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe

warnings.filterwarnings("ignore", message=".*does not have many workers.*")


class TaskDistDataset(ABC, MapDataPipe):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Any] | None]:
        """Get multiple samples from multiple tasks.

        Args:
            index (int): task index.

        Returns:
            tuple[dict[str, Tensor], dict[str, Any] | None]: data with shapes (samples, *shape), task parameters (optional).
        """
        pass


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
