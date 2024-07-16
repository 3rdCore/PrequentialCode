import math
import random
from copy import deepcopy
from typing import Iterable, Literal

import numpy as np
import torch
from beartype import beartype
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch import Tensor
from torch.nn import Module, ModuleList, MSELoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from models.context_aggregator import ContextAggregator
from models.predictor import Predictor


class StandardOptimizerForRegression(LightningModule):
    @beartype
    def __init__(
        self,
        inner_epochs: int,
        predictor: Module,
        lr: float = 1e-4,
        min_train_samples: int = 1,
        train_val_prop: float = 0.8,
        y_key: str = "y",
        loss_fn: _Loss = MSELoss(),
        n_fit_total=1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["optimizer", "loss_fn"])
        self.current_inner_epochs = 0
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.current_n_fit = 0

    @beartype
    def forward(self, x: Tensor) -> Tensor:
        return self.predictor(x)

    @beartype
    def training_step(self, data, batch_idx) -> Tensor:

        data, task_params = data
        x, y = data["x"], data[self.hparams.y_key]
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        data, task_params = data
        x, y = data["x"], data[self.hparams.y_key]
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @beartype
    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @beartype
    def on_train_start(self) -> None:
        """Sample a new task, a new dataset size and update the dataloaders before running first inner training loop."""
        super().on_train_start()
        self.prepare_to_fit_new_task(self.trainer)
        pass

    @beartype
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.current_inner_epochs += 1

    @beartype
    @torch.no_grad()
    def on_atomic_fit_end(self) -> None:
        """
        Log training and evaluation loss, sample a new dataset, reset predictor, optimizers, delete stored metrics.

        Warning, this is not a standard callback. I won't be called anywhere in the code of lightning package.
        """
        self.log_loss_vs_nsamples()
        self.prepare_to_fit_new_task(self.trainer)
        self.trainer.should_stop = not (self.current_n_fit < self.hparams.n_fit_total)

        self.current_n_fit += 1
        self.current_inner_epochs = 0
        self.predictor.weight_init()
        self.trainer.optimizers = [self.configure_optimizers()]
        del self.trainer.callback_metrics["train_loss"]
        del self.trainer.callback_metrics["val_loss"]
        # self.trainer._run_sanity_check()

    @torch.inference_mode()
    def log_loss_vs_nsamples(self) -> None:
        if self.logger is None:
            return
        loss_train, loss_eval, loss_ood = 0, 0, 0
        total_train_samples, total_eval_samples = 0, 0
        dl = self.trainer.datamodule.train_dataloader()
        for data, _ in dl:
            x, y, x_ood, y_ood = (
                data["x"].to(self.device),
                data[self.hparams.y_key].to(self.device),
                data["x_ood"].to(self.device),
                data[f"{self.hparams.y_key}_ood"].to(self.device),
            )
            preds, preds_ood = self.forward(x), self.forward(x_ood)
            loss, loss_ood = self.loss_fn(y, preds), self.loss_fn(y_ood, preds_ood)
            loss_train += loss.item() * x.size(0)
            loss_ood += loss_ood.item() * x_ood.size(0)
            total_train_samples += x.size(0)

        l_train_i = loss_train / total_train_samples

        dl = self.trainer.datamodule.val_dataloader()
        for data, _ in dl:
            x, y, x_ood, y_ood = (
                data["x"].to(self.device),
                data[self.hparams.y_key].to(self.device),
                data["x_ood"].to(self.device),
                data[f"{self.hparams.y_key}_ood"].to(self.device),
            )
            preds, preds_ood = self.forward(x), self.forward(x_ood)
            loss, loss_ood = self.loss_fn(y, preds), self.loss_fn(y_ood, preds_ood)
            loss_eval += loss.item() * x.size(0)
            loss_ood += loss_ood.item() * x_ood.size(0)
            total_eval_samples += x.size(0)
        l_nexttoken_i = loss_eval / total_eval_samples
        loss_ood_i = loss_ood / (total_train_samples + total_eval_samples)
        self.logger.experiment.log(
            {
                "n_samples": total_train_samples,
                "val_tasks/n_sample_loss_train": l_train_i,
                "val_tasks/n_sample_loss_nexttoken": l_nexttoken_i,
                "val_tasks/n_sample_loss_ood": loss_ood_i,
            }
        )

    @beartype
    def prepare_to_fit_new_task(self, trainer) -> None:
        """Sample a new task and update the dataloaders."""

        n_samples = random.randint(
            self.hparams.min_train_samples, trainer.datamodule.dataset.n_samples
        )  # trainer.datamodule.dataset.n_samples is the original number of samples (before truncating)
        trainer.datamodule.switch_task(n_samples=n_samples)
        batch_size = self.trainer.datamodule.hparams["batch_size"]

        self.trainer.val_check_batch = math.ceil(len(self.trainer.datamodule.train_dataset) / batch_size)


class CustomEarlyStopping(EarlyStopping):
    @beartype
    def __init__(
        self,
        monitor="val_loss_epoch",
        min_delta=0.0001,
        patience=3,
        verbose=True,
        mode="min",
        check_on_train_epoch_end=False,
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=verbose,
            mode=mode,
            check_on_train_epoch_end=check_on_train_epoch_end,
        )

    @beartype
    def reset(self) -> None:
        """Resets the EarlyStopping object to the initial state."""
        torch_inf = torch.tensor(torch.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.wait_count = 0
        self.stopped_epoch = 0

    @beartype
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call EarlyStop Callback and check if the predictor should stop training at the end of the validation loop.
        Args:
            trainer: the lightning trainer
            pl_module: the lightning module being trained
        """
        super().on_validation_end(trainer, pl_module)  # call EarlyStop Callback

        if trainer.should_stop or trainer.model.current_inner_epochs >= trainer.model.hparams.inner_epochs:
            self.reset()  # reset the early stopping callback
            trainer.model.on_atomic_fit_end()
