import random
from abc import ABC, abstractmethod
from typing import Iterable, Literal

import torch
from beartype import beartype
from lightning import LightningModule
from torch import Tensor
from torch.nn import MSELoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

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
        lr: float = 1e-4,
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

        train_indices = [
            random.randint(0, i)
            for i in range(
                self.hparams.min_train_samples - 1,
                len(x[list(x.keys())[0]]),
            )
        ]
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
        preds_train, preds_nexttoken, x_train, x_nexttoken, z_train, z_nexttoken = self.forward(x)
        loss = self.losses_and_metrics(
            preds_train,
            preds_nexttoken,
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
        preds_train, preds_nexttoken, x_train, x_nexttoken, z_train, z_nexttoken = self.forward(x)
        _ = self.losses_and_metrics(
            preds_train,
            preds_nexttoken,
            x_train,
            x_nexttoken,
            z_train,
            z_nexttoken,
            task_params,
        )

    @torch.inference_mode()
    def on_train_end(self):
        if self.logger is None:
            return
        dataloader_train = self.trainer.datamodule.train_dataloader()
        dataloader_val = self.trainer.datamodule.val_dataloader()
        for dl, mode in [
            (dataloader_train, "train_tasks"),
            (dataloader_val, "val_tasks"),
        ]:
            num_tasks = len(dl.dataset)
            n_sample_loss_train, n_sample_loss_nexttoken = None, None
            for x, _ in dl:
                x = {name: x[name].to(self.device) for name in x}
                (
                    preds_train,
                    preds_nexttoken,
                    x_train,
                    x_nexttoken,
                    _,
                    _,
                ) = self.forward(x)
                loss_train = self.loss_function(x_train, preds_train)
                loss_nexttoken = self.loss_function(x_nexttoken, preds_nexttoken)

                # Because "train" optimizer's first z never gets a training signal,
                # and "prequential" optimizer's last z never gets a training signal
                loss_train, loss_nexttoken = loss_train[:-1], loss_nexttoken[1:]
                loss_train, loss_nexttoken = (
                    loss_train.sum(dim=-1) / num_tasks,
                    loss_nexttoken.sum(dim=-1) / num_tasks,
                )

                if n_sample_loss_train is None:
                    n_sample_loss_train = loss_train
                    n_sample_loss_nexttoken = loss_nexttoken
                n_sample_loss_train += loss_train
                n_sample_loss_nexttoken += loss_nexttoken

            for i in range(len(n_sample_loss_train)):
                n_samples = i + self.hparams.min_train_samples
                l_train_i = n_sample_loss_train[i].cpu()
                l_nexttoken_i = n_sample_loss_nexttoken[i].cpu()
                self.logger.experiment.log(
                    {
                        "n_samples": n_samples,
                        f"{mode}/n_sample_loss_train": l_train_i,
                        f"{mode}/n_sample_loss_nexttoken": l_nexttoken_i,
                    }
                )

    @beartype
    def losses_and_metrics(
        self,
        preds_train: dict[str, Tensor],
        preds_nexttoken: dict[str, Tensor],
        x_train: dict[str, Tensor],
        x_nexttoken: dict[str, Tensor],
        z_train,
        z_nexttoken: dict[str, Tensor],
        task_params: dict[str, Iterable] | None,
    ) -> torch.Tensor:
        """Computes and logs losses and other metrics. Can be subclassed to add more metrics, or modify loss (e.g., adding a task_params prediction loss).

        Args:
            preds_train (dict[str, Tensor]): Training set predictions (samples, tasks, *).
            preds_nexttoken (dict[str, Tensor]): Next-token predictions (samples, tasks, *).
            x_train (dict[str, Tensor]): Inputs/targets for training set predictions (samples, tasks, *).
            x_nexttoken (dict[str, Tensor]): Inputs/targets for next-token predictions (samples, tasks, *).
            z_train (dict[str, Tensor]): Aggregated context for training set predictions (samples, tasks, *).
            z_nexttoken (dict[str, Tensor]): Aggregated context for next-token predictions (samples, tasks, *).
            task_params (dict[str, Iterable] | None): True task parameters (tasks).

        Returns:
            torch.Tensor: Scalar loss to optimize.
        """
        mode = "train_tasks" if self.training else "val_tasks"
        num_tasks = preds_train[list(preds_train.keys())[0]].shape[1]

        # Main losses
        loss_train = self.loss_function(x_train, preds_train).mean()
        loss_nexttoken = self.loss_function(x_nexttoken, preds_nexttoken).mean()
        loss = loss_train if self.meta_objective == "train" else loss_nexttoken
        self.log(f"{mode}/loss_train", loss_train, batch_size=num_tasks)
        self.log(f"{mode}/loss_nexttoken", loss_nexttoken, batch_size=num_tasks)

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
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class MetaOptimizerForRegression(MetaOptimizer):
    def __init__(
        self,
        meta_objective: MetaOptimizer.MetaObjective,
        context_aggregator: ContextAggregator,
        predictor: Predictor,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = MSELoss(reduction="none"),
    ):
        super().__init__(meta_objective, context_aggregator, predictor, min_train_samples, lr)

        self.loss_fn = loss_fn

    def loss_function(self, target: dict[str, Tensor], preds: dict[str, Tensor]) -> Tensor:
        """
        Args:
            target (dict[str, Tensor]): Inputs/targets (samples, tasks, *).
            preds (dict[str, Tensor]): Predictions (samples, tasks, *).

        Returns:
            Tensor: Losses (samples, tasks).
        """
        assert len(preds) == 1, "Only one output key supported for regression tasks"
        y_key = list(preds.keys())[0]
        return torch.mean(self.loss_fn(preds[y_key], target[y_key]), dim=-1)  # component-wise averaging
