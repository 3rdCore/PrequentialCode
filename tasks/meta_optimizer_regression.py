from typing import Iterable, Literal

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import MSELoss
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from datasets.interfaces import custom_collate_fn
from datasets.regression import RegressionDataset
from models.context_aggregator import ContextAggregator
from models.implicit import ImplicitModel
from models.predictor import Predictor
from tasks.meta_optimizer import MetaOptimizerExplicit, MetaOptimizerImplicit
from utils.plotting import fig2img


class MetaOptimizerExplicitForRegression(MetaOptimizerExplicit):
    def __init__(
        self,
        meta_objective: MetaOptimizerExplicit.MetaObjective,
        context_aggregator: ContextAggregator,
        predictor: Predictor,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = MSELoss(reduction="none"),
        n_probe_tasks: int = 4,
        probe_n_context_points: tuple[int] = (1, 4, 10, 50),
        probe_resolution: int = 100,
    ):
        super().__init__(
            meta_objective=meta_objective,
            context_aggregator=context_aggregator,
            predictor=predictor,
            min_train_samples=min_train_samples,
            lr=lr,
        )
        self.n_probe_tasks = n_probe_tasks
        self.probe_n_context_points = probe_n_context_points
        self.probe_resolution = probe_resolution

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

    def losses_and_metrics(
        self,
        preds_train: dict[str, Tensor],
        preds_nexttoken: dict[str, Tensor],
        x_train: dict[str, Tensor],
        x_nexttoken: dict[str, Tensor],
        z: dict[str, Tensor],
        task_params: dict[str, Iterable] | None,
    ) -> Tensor:
        loss = super().losses_and_metrics(
            preds_train,
            preds_nexttoken,
            x_train,
            x_nexttoken,
            z,
            task_params,
        )
        mode = "train_tasks" if self.training else "val_tasks"
        num_tasks = preds_train[list(preds_train.keys())[0]].shape[1]

        if (
            hasattr(self.trainer.datamodule.train_dataset, "has_ood")
            and self.trainer.datamodule.train_dataset.has_ood
        ):
            x_ood = {name: x_nexttoken[f"{name}_ood"].to(self.device) for name in ["x", "y"]}
            with torch.inference_mode():
                preds_ood = self.predictor.forward(x_ood, z)
            ood_loss = self.loss_function(x_ood, preds_ood).mean()
            self.log(f"{mode}/loss_ood", ood_loss, batch_size=num_tasks)

        return loss

    def on_train_end(self):
        super().on_train_end()
        self.log_model_vs_true(
            mode="train_tasks",
            n_probe_tasks=self.n_probe_tasks,
            n_context_points=self.probe_n_context_points,
            resolution=self.probe_resolution,
        )
        self.log_model_vs_true(
            mode="val_tasks",
            n_probe_tasks=self.n_probe_tasks,
            n_context_points=self.probe_n_context_points,
            resolution=self.probe_resolution,
        )

    @torch.inference_mode()
    def log_model_vs_true(
        self,
        mode: Literal["train_tasks", "val_tasks"],
        n_probe_tasks: int = 4,
        n_context_points: tuple[int] | None = (1, 4, 10, 50),
        resolution: int = 100,
    ) -> None:
        if (
            self.logger is None
            or n_context_points is None
            or not isinstance(self.trainer.datamodule.train_dataset, RegressionDataset)
        ):
            return

        # Get the dataset
        if mode == "train_tasks":
            dataset: RegressionDataset = self.trainer.datamodule.train_dataset
        else:
            dataset: RegressionDataset = self.trainer.datamodule.val_dataset

        # We won't support this logging for problems that are not scalar functions
        if dataset.x_dim != 1 or dataset.y_dim != 1:
            return

        # Disable for this function to have better reproducibility
        shuffle_samples = dataset.shuffle_samples
        dataset.shuffle_samples = False

        # Get the first `n_probe_tasks` tasks
        context, task_params = next(
            iter(
                DataLoader(
                    dataset,
                    batch_size=n_probe_tasks,
                    collate_fn=custom_collate_fn,
                )
            )
        )

        # Restore the dataset's original `shuffle_samples` attribute
        dataset.shuffle_samples = shuffle_samples

        # True function
        x_range = context["x"].min(), context["x"].max()
        x_range = (
            x_range[0] - 0.2 * (x_range[1] - x_range[0]),
            x_range[1] + 0.2 * (x_range[1] - x_range[0]),
        )
        x = torch.linspace(x_range[0], x_range[1], resolution)
        x = x.unsqueeze(0).unsqueeze(-1)
        x = x.expand(n_probe_tasks, -1, -1)  # (n_probe_tasks, resolution, 1)
        y = dataset.function(x, task_params)  # (n_probe_tasks, resolution, 1)

        # Model function
        context = {name: context[name].to(self.device) for name in ["x", "y"]}
        z = self.context_aggregator.forward(context)
        n_context_points = [
            n for n in n_context_points if n >= self.hparams.min_train_samples and n < z["z"].shape[0] - 1
        ]
        for name in z:
            z[name] = z[name][n_context_points]
            z[name] = z[name].repeat(resolution, *[1] * (z[name].ndim - 1))
        x_query = x.transpose(0, 1).to(self.device)
        x_query = x_query.repeat_interleave(len(n_context_points), dim=0)
        y_pred = self.predictor.forward({"x": x_query}, z)["y"]
        y_pred = y_pred.view(resolution, len(n_context_points), n_probe_tasks)

        # Reshape and indexing things for easier downstream manipulation
        x = x[0, :, 0].numpy()  # (resolution,)
        y = y.squeeze(-1).cpu().numpy()  # (n_probe_tasks, resolution)
        x_context = context["x"].squeeze(-1).cpu().numpy()  # (n_samples, n_probe_tasks)
        y_context = context["y"].squeeze(-1).cpu().numpy()  # (n_samples, n_probe_tasks)
        y_pred = y_pred.cpu().numpy()  # (resolution, len(n_context_points), n_probe_tasks)

        # Collect data in tables
        df_context = []
        df_true = []
        df_model = []
        for task_idx in range(n_probe_tasks):
            n_context_idx = 0
            for i in range(max(n_context_points)):
                if i == n_context_points[n_context_idx]:
                    n_context_idx += 1
                df_context.append(
                    {
                        "task_id": task_idx,
                        "n_context": n_context_points[n_context_idx],
                        "n_context_group": n_context_idx,
                        "x": x_context[i, task_idx],
                        "y": y_context[i, task_idx],
                    }
                )
        df_context = pd.DataFrame(df_context)
        for task_idx in range(n_probe_tasks):
            for i in range(resolution):
                df_true.append(
                    {
                        "task_id": task_idx,
                        "x": x[i],
                        "y": y[task_idx, i],
                    }
                )
        df_true = pd.DataFrame(df_true)
        for task_idx in range(n_probe_tasks):
            for n_context_idx, n_context in enumerate(n_context_points):
                for i in range(resolution):
                    df_model.append(
                        {
                            "task_id": task_idx,
                            "n_context": n_context,
                            "n_context_group": n_context_idx,
                            "x": x[i],
                            "y": y_pred[i, n_context_idx, task_idx],
                        }
                    )
        df_model = pd.DataFrame(df_model)

        # Log the tables
        self.logger.log_table(f"tables/{mode}-model_vs_true-context", data=df_context)
        self.logger.log_table(f"tables/{mode}-model_vs_true-true", data=df_true)
        self.logger.log_table(f"tables/{mode}-model_vs_true-model", data=df_model)

        # Make the plots
        fig, axs = plt.subplots(1, n_probe_tasks, figsize=(5 * n_probe_tasks, 5))
        for task_idx, ax in enumerate(axs):
            sns.lineplot(
                data=df_true[df_true["task_id"] == task_idx],
                x="x",
                y="y",
                color="grey",
                ax=ax,
            )
            sns.lineplot(
                data=df_model[df_model["task_id"] == task_idx],
                x="x",
                y="y",
                hue="n_context_group",
                palette=sns.color_palette("crest", as_cmap=True),
                ax=ax,
            )
            sns.scatterplot(
                data=df_context[df_context["task_id"] == task_idx],
                x="x",
                y="y",
                hue="n_context_group",
                palette=sns.color_palette("crest", as_cmap=True),
                linewidth=1.2,
                marker="+",
                s=100,
                ax=ax,
            )
            ax.legend().remove()
        self.logger.log_image(key=f"probes/{mode}-model_vs_true", images=[fig2img(fig)])


class MetaOptimizerImplicitForRegression(MetaOptimizerImplicit):
    def __init__(
        self,
        model: ImplicitModel,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = MSELoss(reduction="none"),
    ):
        super().__init__(model=model, min_train_samples=min_train_samples, lr=lr)

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
