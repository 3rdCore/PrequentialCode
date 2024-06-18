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
        model: Module,
        lr: float = 1e-4,
        min_train_samples: int = 1,
        train_val_prop: float = 0.8,
        y_key: str = "y",
        loss_fn: _Loss = MSELoss(),
        n_fit_total=1000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "optimizer", "loss_fn"])
        self.current_inner_epochs = 0
        self.model = model
        self.init_model = deepcopy(model)
        self.loss_fn = loss_fn
        self.current_n_fit = 0

    @beartype
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    @beartype
    def training_step(self, data, batch_idx):

        data, task_params = data
        x, y = data["x"], data[self.hparams.y_key]
        # print batch size
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log("train_loss", loss)

        return loss

    @beartype
    def validation_step(self, data, batch_idx):
        data, task_params = data
        x, y = data["x"], data[self.hparams.y_key]
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)

        return loss

    @beartype
    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def on_fit_end(self):  # not a callback
        self.log_loss_vs_nsamples()
        self.fit_new_task(self.trainer)
        self.trainer.callbacks[0].reset()

        self.current_n_fit += 1
        self.current_inner_epochs = 0
        self.model.weight_init()
        self.trainer.optimizers = [self.configure_optimizers()]
        # del self.trainer.callback_metrics["train_loss"]
        self.trainer.callback_metrics["val_loss"] = torch.tensor(1000)

    @torch.inference_mode()
    def log_loss_vs_nsamples(self):
        if self.logger is None:
            return
        loss_train, loss_eval = 0, 0
        total_train_samples, total_eval_samples = 0, 0
        dl = self.trainer.datamodule.train_dataloader()
        for data, _ in dl:
            x, y = data["x"].to(self.device), data[self.hparams.y_key].to(self.device)
            preds = self.forward(x)
            loss = self.loss_fn(y, preds)
            loss_train += loss.item() * x.size(0)
            total_train_samples += x.size(0)

        l_train_i = loss_train / total_train_samples

        dl = self.trainer.datamodule.val_dataloader()
        for data, _ in dl:
            x, y = data["x"].to(self.device), data[self.hparams.y_key].to(self.device)
            preds = self.forward(x)
            loss = self.loss_fn(y, preds)
            loss_eval += loss.item() * x.size(0)
            total_eval_samples += x.size(0)
        l_nexttoken_i = loss_eval / total_eval_samples if total_eval_samples > 0 else -1

        self.logger.experiment.log(
            {
                "n_samples": total_train_samples,
                "/n_sample_loss_train": l_train_i,
                "/n_sample_loss_nexttoken": l_nexttoken_i,
            }
        )

    @beartype
    def fit_new_task(self, trainer):
        """Sample a new task and update the dataloaders."""

        n_samples = random.randint(self.hparams.min_train_samples, trainer.datamodule.dataset.n_samples)
        trainer.datamodule.switch_task(n_samples=n_samples)
        trainer.should_stop = not (self.current_n_fit < self.hparams.n_fit_total)
        # update trainer callback on tqdm bar


class CustomEarlyStopping(EarlyStopping):
    @beartype
    def __init__(self, monitor="val_loss", min_delta=0.0001, patience=3, verbose=True, mode="min"):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)

    @beartype
    def reset(self):
        torch_inf = torch.tensor(torch.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.wait_count = 0
        self.stopped_epoch = 0

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)  # call EarlyStop Callback
        trainer.model.current_inner_epochs += 1
        if trainer.model.current_inner_epochs >= trainer.model.hparams.inner_epochs:
            trainer.should_stop = True
        if trainer.should_stop:
            trainer.model.on_fit_end()

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)  # call EarlyStop Callback
