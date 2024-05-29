import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Iterable, Literal

import numpy as np
import torch
from beartype import beartype
from lightning import LightningModule
from torch import Tensor
from torch.nn import Module, ModuleList, MSELoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from models.context_aggregator import ContextAggregator
from models.predictor import Predictor


class StandardOptimizerForRegression(LightningModule):
    @beartype
    def __init__(
        self,
        epochs: int,
        model: Module,
        optimizer: Optimizer,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        y_key: str = "y",
        loss_fn: _Loss = MSELoss(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer", "loss_fn"])
        self.model = model
        self.optimizer = optimizer

    @beartype
    def forward(
        self, x: dict[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, Tensor],]:
        return None

    @beartype
    def training_step(self, data, batch_idx):
        return None

    @beartype
    def validation_step(self, data, batch_idx):
        x, task_params = data
        device = x[next(iter(x))].device

        train_indices = [
            random.randint(0, i) for i in range(self.hparams.min_train_samples - 1, len(x["x"]))
        ]  # TODO remove hardcoded name

        x_train = {name: x[name][train_indices] for name in x}
        # preds zero like x_train
        preds_train = torch.zeros_like(x_train[self.hparams.y_key])
        x_nexttoken = {name: x[name][self.hparams.min_train_samples - 1 :] for name in x}
        preds_next_token = torch.zeros_like(x_nexttoken[self.hparams.y_key])

        x, y = (
            torch.cat(
                [x[name][self.hparams.min_train_samples - 1 :] for name in x if name != self.hparams.y_key],
                dim=-1,
            ),
            x[self.hparams.y_key],
        )
        models = [
            deepcopy(self.model).to(device)
            for s in range(self.hparams.min_train_samples, x.size(0) - 1)
            for b in range(x.size(1))
        ]

        torch.cuda.synchronize(device)
        losses = []

        for i, model in enumerate(models):
            task_idx = i // x.size(0)
            seq_idx = i % x.size(0)

            inputs = x[: self.hparams.min_train_samples + seq_idx, task_idx, ...]  # (:seq_idx, task_idx, *)
            targets = y[: self.hparams.min_train_samples + seq_idx, task_idx, ...]  # (:seq_idx, task_idx, *)
            for epoch in range(self.hparams.epochs):  # Number of epochs
                for sample in range(inputs.shape[0]):
                    input = inputs[sample]
                    target = targets[sample]
                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    preds = model(input)
                    loss = self.hparams.loss_fn(preds, target)

                    # Backward pass
                    loss.backward()
                    self.optimizer.step()

        for i, model in enumerate(models):
            task_idx = i // x.size(0)
            seq_idx = i % x.size(0)

            with torch.inference_mode():
                preds_train[seq_idx, task_idx, ...] = model(
                    x_train["x"][self.hparams.min_train_samples + seq_idx - 1, task_idx, ...]
                )
                preds_next_token[seq_idx, task_idx, ...] = model(
                    x_nexttoken["x"][self.hparams.min_train_samples + seq_idx - 1, task_idx, ...]
                )

        if task_params is not None:
            task_params = {
                name: task_params[name][self.hparams.min_train_samples - 1 :] for name in task_params
            }

        return
        """self.losses_and_metrics(
            {"y": preds_train},
            {"y": preds_next_token},
            x_train,
            x_nexttoken,
            task_params,
        )"""

    @beartype
    def losses_and_metrics(
        self,
        preds_train: dict[str, Tensor],
        preds_next_token: dict[str, Tensor],
        x_train: dict[str, Tensor],
        x_nexttoken: dict[str, Tensor],
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
        self.log(f"{mode}/loss_train", loss_train_avg)
        self.log(f"{mode}/loss_nexttoken", loss_nexttoken_avg)

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
    def loss_function(self, target: dict[str, Tensor], preds: dict[str, Tensor]) -> Tensor:
        """Do not average across samples and tasks! Return shape should be

        Args:
            target (dict[str, Tensor]): Inputs/targets (samples, tasks, *).
            preds (dict[str, Tensor]): Predictions (samples, tasks, *).

        Returns:
            Tensor: Losses (samples, tasks).
        """
        return torch.mean(
            self.hparams.loss_fn(preds[self.hparams.y_key], target[self.hparams.y_key]), dim=-1
        )  # component-wise averaging
