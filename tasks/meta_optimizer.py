import random
from abc import ABC, abstractmethod
from typing import Iterable, Literal

import numpy as np
import torch
from beartype import beartype
from lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F

from models.context_aggregator import ContextAggregator
from models.predictor import Predictor


class MetaOptimizer(ABC, LightningModule):
    MetaObjective = Literal["train", "prequential"]

    @beartype
    def __init__(
        self,
        meta_objective: MetaObjective,
        context_aggregator: ContextAggregator,
        predictor: Predictor,
        min_train_samples: int = 1,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["context_aggregator", "predictor"])

        self.meta_objective = meta_objective
        self.context_aggregator = context_aggregator
        self.predictor = predictor

    @beartype
    def forward(
        self, x: dict[str, Tensor]
    ) -> tuple[
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
        dict[str, Tensor],
    ]:
        """Return the training predictions, next-token predictions, and aggregated context z.

        Args:
            x (dict[str, Tensor]):  Input data, each with shape (samples, tasks, *).

        Returns:
            tuple[dict[str, Tensor], dict[str, Tensor]]: Training predictions, next-token predictions, and aggregated context z.
            - training predictions: (samples - min_train_samples + 1, tasks, *).
            - next-token predictions: (samples - min_train_samples + 1, tasks, *).
            - x_train: (samples - min_train_samples + 1, tasks, *).
            - x_nexttoken: (samples - min_train_samples + 1, tasks, *).
            - z_train: (samples - min_train_samples + 1, tasks, *).
            - z_nexttoken: (samples - min_train_samples + 1, tasks, *).
        """
        z = self.context_aggregator.forward(x)
        z = {name: z[name][self.hparams.min_train_samples - 1 :] for name in z}

        z_train = {name: z[name][1:] for name in z}
        z_nexttoken = {name: z[name][:-1] for name in z}

        train_indices = [random.randint(0, i) for i in range(self.hparams.min_train_samples - 1, len(x))]
        x_train = {name: x[name][train_indices] for name in x}
        x_nexttoken = {name: x[name][self.hparams.min_train_samples - 1 :] for name in x}

        mode = self.training
        if self.meta_objective == "train":
            preds_train = self.predictor.forward(x_train, z_train)
            self.train(False)
            with torch.inference_mode():
                preds_nexttoken = self.predictor.forward(x_nexttoken, z_nexttoken)
        elif self.meta_objective == "prequential":
            preds_nexttoken = self.predictor.forward(x_nexttoken, z_nexttoken)
            self.train(False)
            with torch.inference_mode():
                preds_train = self.predictor.forward(x_train, z_train)
        else:
            raise ValueError(f"Invalid meta_objective: {self.meta_objective}")
        self.train(mode)

        return preds_train, preds_nexttoken, x_train, x_nexttoken, z_train, z_nexttoken

    @beartype
    def training_step(self, data, batch_idx):
        x, task_params = data
        if task_params is not None:
            task_params = {
                name: task_params[name][self.hparams.min_train_samples - 1 :] for name in task_params
            }
        preds_train, preds_next_token, x_train, x_nexttoken, z_train, z_nexttoken = self.forward(x)
        loss = self.losses_and_metrics(
            preds_train,
            preds_next_token,
            x_train,
            x_nexttoken,
            z_train,
            z_nexttoken,
            task_params,
        )
        return loss

    @beartype
    def validation_step(self, data, batch_idx):
        x, task_params = data
        if task_params is not None:
            task_params = {
                name: task_params[name][self.hparams.min_train_samples - 1 :] for name in task_params
            }
        preds_train, preds_next_token, x_train, x_nexttoken, z_train, z_nexttoken = self.forward(x)
        _ = self.losses_and_metrics(
            preds_train,
            preds_next_token,
            x_train,
            x_nexttoken,
            z_train,
            z_nexttoken,
            task_params,
        )

    @beartype
    def losses_and_metrics(
        self,
        preds_train: dict[str, Tensor],
        preds_next_token: dict[str, Tensor],
        x_train: dict[str, Tensor],
        x_nexttoken: dict[str, Tensor],
        z_train,
        z_nexttoken: dict[str, Tensor],
        task_params: dict[str, Iterable] | None,
    ) -> torch.Tensor:
        """Computes and logs losses and other metrics. Can be subclassed to add more metrics, or modify loss (e.g., adding a task_params prediction loss).

        Args:
            preds_train (dict[str, Tensor]): Training set predictions (samples, tasks, *).
            preds_next_token (dict[str, Tensor]): Next-token predictions (samples, tasks, *).
            x_train (dict[str, Tensor]): Inputs/targets for training set predictions (samples, tasks, *).
            x_nexttoken (dict[str, Tensor]): Inputs/targets for next-token predictions (samples, tasks, *).
            z_train (dict[str, Tensor]): Aggregated context for training set predictions (samples, tasks, *).
            z_nexttoken (dict[str, Tensor]): Aggregated context for next-token predictions (samples, tasks, *).
            task_params (dict[str, Iterable] | None): True task parameters (tasks).

        Returns:
            torch.Tensor: Scalar loss to optimize.
        """
        mode = "val_tasks" if self.trainer.validating else "train_tasks"

        # Main losses
        loss_train = self.loss_function(x_train, preds_train)
        loss_train_avg = loss_train.mean()
        loss_nexttoken = self.loss_function(x_nexttoken, preds_next_token)
        loss_nexttoken_avg = loss_nexttoken.mean()
        loss = loss_train_avg if self.meta_objective == "train" else loss_nexttoken_avg
        self.log(f"{mode}/loss_train", loss_train)
        self.log(f"{mode}/loss_nexttoken", loss_nexttoken)

        # Within-task validation loss
        loss_val = self.loss_function(
            {name: x_nexttoken[name][1:] for name in x_nexttoken},
            {name: preds_next_token[name][1:] for name in preds_next_token},
        )  # Index starting from 1 because train-risk model's first z never gets a training signal
        loss_val_avg = loss_val.mean()
        self.log(f"{mode}/loss_val", loss_val_avg)

        # TODO: Plot the train, next-token, and validation losses as a function of the number of samples seen. Ideally this should be computed over the whole epoch and we should build something like a line-plot with number of samples on the x-axis, loss on the y-axis, and noise bands for standard deviation across the tasks. The plot can have a scroller to show it at different points during meta-optimizer training. We can use wandb line plots for this. Note: we should probably not create and log such a plot every epoch, as the number of samples would be quite large. We can do something like every N epochs.

        return loss

    @abstractmethod
    def loss_function(self, x: dict[str, Tensor], preds: dict[str, Tensor]) -> Tensor:
        """Do not average across samples and tasks! Return shape should be

        Args:
            x (dict[str, Tensor]): Inputs/targets (samples, tasks, *).
            preds (dict[str, Tensor]): Predictions (samples, tasks, *).

        Returns:
            Tensor: Losses (samples, tasks).
        """
        pass
