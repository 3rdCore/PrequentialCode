from abc import ABC, abstractmethod
from typing import Any, Iterable

from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader
from torchdata.datapipes.map import MapDataPipe


class TaskDistDataset(ABC, MapDataPipe):
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], Iterable[Any]]:
        """Get multiple samples from multiple tasks.

        Args:
            index (int): task index.

        Returns:
            tuple[dict[str, Tensor], Iterable[Any]]: data with shape (n_samples, *shape), task parameters.
        """
        pass


def ICLDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: ICLDataset,
        val_dataset: ICLDataset,
        batch_size,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # setup

    def train_dataloader(self):
        return DtaLoa

    def val_dataloader(self):
        pass
