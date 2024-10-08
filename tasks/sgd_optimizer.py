import random
from typing import Literal

import torch
from beartype import beartype
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torch.nn import Module, MSELoss
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer


class StandardOptimizer(LightningModule):
    @beartype
    def __init__(
        self,
        inner_epochs: int,
        predictor: Module,
        lr: float = 1e-4,
        min_train_samples: int = 1,
        y_key: str = "y",
        loss_fn: _Loss = MSELoss(),
        n_fit_total=1000,
        regularization_type: Literal["L1", "L2", None] = None,
        lambda_reg: float | None = None,
        one_hot_x: int | None = None,
    ):
        super().__init__()
        if regularization_type in ["L1", "L2"] and lambda_reg == None:
            lambda_reg = 1e-3

        hparams = {k: v for k, v in locals().items() if v is not None}
        self.save_hyperparameters(hparams, ignore=["optimizer", "loss_fn"])
        self.current_inner_epochs = 0
        self.predictor = predictor
        self.one_hot_x = one_hot_x
        self.loss_fn = loss_fn
        self.current_n_fit = 0

    @beartype
    def forward(self, x: Tensor) -> Tensor:
        if self.one_hot_x is not None:
            x = torch.nn.functional.one_hot(x, num_classes=self.hparams.one_hot_x)
            x = x.float().flatten(start_dim=-2)
        return self.predictor(x)

    @beartype
    def training_step(self, data, batch_idx) -> Tensor:
        data, _ = data
        x, y = data["x"], data[self.hparams.y_key]
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log(
            "train_loss", loss, prog_bar=False, on_step=False, on_epoch=True
        )  # exclude regularizer when logging
        reg = 0.0
        if hasattr(self.hparams, "regularization_type"):
            if self.hparams.regularization_type == "L1":
                reg = self.hparams.lambda_reg * sum(
                    torch.norm(param, 1) for param in self.parameters()
                )
            elif self.hparams.regularization_type == "L2":
                reg = self.hparams.lambda_reg * sum(
                    torch.norm(param, 2) for param in self.parameters()
                )
        self.log("reg_loss", reg, prog_bar=False, on_step=False, on_epoch=True)

        return loss + reg

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        data, task_params = data
        x, y = data["x"], data[self.hparams.y_key]
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
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

        if (
            self.trainer.should_stop
            or self.current_inner_epochs >= self.hparams.inner_epochs
        ):
            self.on_atomic_fit_end()

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
        self.parameters = self.predictor.parameters
        self.trainer.optimizers = [self.configure_optimizers()]
        for callback in self.trainer.callbacks:
            if isinstance(callback, CustomEarlyStopping):
                callback.reset()
        for key in self.trainer.callback_metrics:
            self.log(
                key,
                torch.tensor(torch.inf),
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
        pass

    @beartype
    @torch.inference_mode()
    def log_loss_vs_nsamples(self) -> None:
        if self.logger is None:
            return

        def compute_loss(data):
            loss, loss_ood, total_samples = 0, 0, 0
            for data, _ in dl:
                x, y = (
                    data["x"].to(self.device),
                    data[self.hparams.y_key].to(self.device),
                )
                preds = self.forward(x)
                l = self.loss_fn(preds, y)
                loss += l.item() * x.size(0)
                total_samples += x.size(0)
                if self.has_ood:
                    x_ood, y_ood = data["x_ood"].to(self.device), data[
                        f"{self.hparams.y_key}_ood"
                    ].to(self.device)
                    preds_ood = self.forward(x_ood)
                    l_ood = self.loss_fn(preds_ood, y_ood)
                    loss_ood += l_ood.item() * x_ood.size(0)
            loss_i = loss / total_samples
            if self.has_ood:
                loss_ood_i = loss_ood / total_samples
            return loss_i, loss_ood_i, total_samples

        dl = self.trainer.datamodule.train_dataloader()
        train_loss, train_ood_loss, train_total_samples = compute_loss(dl)
        dl = self.trainer.datamodule.test_dataloader()
        test_loss, test_ood_loss, test_total_samples = compute_loss(dl)

        logs = {
            "n_samples": self.trainer.datamodule.get_current_n_samples(),
            "val_tasks/n_sample_loss_train": train_loss,
            "val_tasks/n_sample_loss_nexttoken": test_loss,
        }
        if self.has_ood:
            loss_ood_i = (train_ood_loss + test_ood_loss) / (
                train_total_samples + test_total_samples
            )
            logs.update({f"val_tasks/n_sample_loss_ood": loss_ood_i})
        self.logger.experiment.log(logs)

    @beartype
    def prepare_to_fit_new_task(self, trainer) -> None:
        """Sample a new task and update the dataloaders."""

        log_min_samples = torch.log(
            torch.tensor(self.hparams.min_train_samples, dtype=torch.float)
        )
        log_max_samples = torch.log(
            torch.tensor(trainer.datamodule.max_train_samples, dtype=torch.float)
        )
        n_samples = int(
            torch.exp(
                torch.distributions.Uniform(log_min_samples, log_max_samples).sample()
            ).item()
        )
        trainer.datamodule.switch_task(n_samples=n_samples)
        batch_size = self.trainer.datamodule.hparams["batch_size"]

    @property
    def has_ood(self) -> bool:
        return (
            hasattr(self.trainer.datamodule.dataset, "has_ood")
            and self.trainer.datamodule.dataset.has_ood
        )


class CustomEarlyStopping(Callback):
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
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.wait_count = 0
        self.stopped_epoch = 0
        self.best_score = (
            torch.tensor(torch.inf) if mode == "min" else -torch.tensor(torch.inf)
        )
        self.mode = mode
        self.check_on_train_epoch_end = check_on_train_epoch_end
        self.monitor_op = torch.lt if mode == "min" else torch.gt

    @beartype
    def reset(self) -> None:
        """Resets the EarlyStopping object to the initial state."""
        self.best_score = (
            torch.tensor(torch.inf)
            if self.monitor_op == torch.lt
            else -torch.tensor(torch.inf)
        )
        self.wait_count = 0
        self.stopped_epoch = 0

    @beartype
    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Call EarlyStop Callback and check if the predictor should stop training at the end of the validation loop.
        Args:
            trainer: the lightning trainer
            pl_module: the lightning module being trained
        """
        if trainer.callback_metrics.get(self.monitor):
            current = trainer.callback_metrics.get(self.monitor).clone().detach()
            if self.monitor_op(current - self.min_delta, self.best_score):
                self.best_score = current
                self.wait_count = 0
            else:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    trainer.should_stop = True
