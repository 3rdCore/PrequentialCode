from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe


class TaskDistDataset(ABC, MapDataPipe):
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, Iterable[Any]]]:
        """Get multiple samples from multiple tasks.

        Args:
            index (int): task index.

        Returns:
            tuple[dict[str, Tensor], dict[str, Iterable[Any]]]: data with shapes (samples, *shape), task parameters.
        """
        pass


class ICLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: TaskDistDataset,
        val_dataset: TaskDistDataset,
        batch_size: int,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset, val_dataset"])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # setup

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=batch_dim1,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=batch_dim1,
        )


# Custom collate function to stack batches along dim=1 rather than dim=0
def batch_dim1(batch):
    batch_dict = {}
    for key in batch[0]:
        stacked = torch.stack([item[key] for item in batch], dim=1)
        batch_dict[key] = stacked
    return batch_dict
