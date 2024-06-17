from beartype import beartype
from lightning import LightningModule
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
        self.model = model
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

        losses = []

        for i, ((seq_idx, task_idx), model) in enumerate(models.items()):
            with torch.inference_mode():
                preds_train[seq_idx - 1, task_idx, ...] = model(x_train["x"][seq_idx - 1, task_idx, ...])
                preds_next_token[seq_idx - 1, task_idx, ...] = model(
                    x_nexttoken["x"][seq_idx - 1, task_idx, ...]
                )
        return preds_train, preds_next_token, x_train, x_nexttoken, models

    @beartype
    def validation_step(self, data, batch_idx):
        data, task_params = data
        x, y = data["x"], data[self.hparams.y_key]
        preds = self.forward(x)

        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss)

        return loss

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

        # TODO: Plot the train, next-token, and validation losses as a function of the number of samples seen. Ideally this should be computed over the whole epoch and we should build something like a line-plot with number of samples on the x-axis, loss on the y-axis, and noise bands for standard deviation across the tasks. The plot can have a scroller to show it at different points during meta-optimizer training. We can use wandb line plots for this. Note: we should probably not create and log such a plot every epoch, as the number of samples would be quite large. We can do something like every N inner_epochs.

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
            self.loss_fn(preds[self.hparams.y_key], target[self.hparams.y_key]), dim=-1
        )  # component-wise averaging

    @beartype
    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class CustomEarlyStopping(EarlyStopping):
    def __init__(self, monitor="val_loss", min_delta=0.0, patience=3, verbose=False, mode="min"):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if trainer.should_stop:
            print(trainer.datamodule.train_dataset.n_samples)
            trainer.model.current_n_fit += 1
            trainer.model.model.reset_parameters()
            n_samples = random.randint(
                trainer.model.hparams.min_train_samples, trainer.datamodule.dataset.n_samples
            )
            trainer.datamodule.switch_task(n_samples=n_samples)
            trainer.should_stop = not (trainer.model.current_n_fit < trainer.model.hparams.n_fit_total)
