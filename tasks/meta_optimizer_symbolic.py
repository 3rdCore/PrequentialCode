from typing import Iterable, Literal

import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader

from datasets.interfaces import custom_collate_fn
from datasets.regression import RegressionDataset
from models.context_aggregator import ContextAggregator
from models.predictor import Predictor
from tasks.meta_optimizer import MetaOptimizer
from utils.plotting import fig2img


class MetaOptimizerForSymbolic(MetaOptimizer):
    def __init__(
        self,
        meta_objective: MetaOptimizer.MetaObjective,
        context_aggregator: ContextAggregator,
        predictor: Predictor,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = CrossEntropyLoss(reduction="none"),
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
        n_colors = self.trainer.datamodule.train_dataset.n_colors
        y_true = target["y"].view(-1, n_colors)
        y_pred = preds["y"].view(-1, n_colors)
        return self.loss_fn(y_pred, y_true).view(*target["x"].shape[:2], -1).mean(dim=-1)


class MetaOptimizerForSymbolic2D(MetaOptimizerForSymbolic):
    def __init__(
        self,
        meta_objective: MetaOptimizer.MetaObjective,
        context_aggregator: ContextAggregator,
        predictor: Predictor,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = CrossEntropyLoss(reduction="none"),
    ):
        super().__init__(meta_objective, context_aggregator, predictor, min_train_samples, lr, loss_fn)
        self.predictor.feature_map = self.context_aggregator.feature_map

    def loss_function(self, target: dict[str, Tensor], preds: dict[str, Tensor]) -> Tensor:
        n_colors = self.trainer.datamodule.train_dataset.n_colors
        target["y"] = (
            F.one_hot(target["y"].to(torch.int64), num_classes=n_colors).flatten(start_dim=-3).float()
        )
        return super().loss_function(target, preds)
