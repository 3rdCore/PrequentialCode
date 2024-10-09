import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss

from datasets.symbolic import SymbolicDataset
from models.context_aggregator import ContextAggregator
from models.implicit import ImplicitModel
from models.predictor import Predictor
from tasks.meta_optimizer import MetaOptimizerExplicit, MetaOptimizerImplicit
from utils import CrossEntropyLossFlat


class MetaOptimizerExplicitForSymbolic(MetaOptimizerExplicit):
    def __init__(
        self,
        meta_objective: MetaOptimizerExplicit.MetaObjective,
        context_aggregator: ContextAggregator,
        predictor: Predictor,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = CrossEntropyLossFlat(reduction="none"),
    ):
        super().__init__(
            meta_objective=meta_objective,
            context_aggregator=context_aggregator,
            predictor=predictor,
            min_train_samples=min_train_samples,
            lr=lr,
        )

        self.loss_fn = loss_fn

    def loss_function(self, target: dict[str, Tensor], preds: dict[str, Tensor]) -> Tensor:
        """
        Args:
            target (dict[str, Tensor]): Inputs/targets (samples, tasks, n_vars).
            preds (dict[str, Tensor]): Predictions (samples, tasks, n_vars * n_vals).

        Returns:
            Tensor: Losses (samples, tasks).
        """
        assert len(preds) == 1, "Only one output key supported for symbolic tasks"
        y_key = list(preds.keys())[0]
        target, preds = target[y_key], preds[y_key]
        target = target.view(target.shape[0], target.shape[1], self.y_num_vars, -1).argmax(dim=-1)
        loss = self.loss_fn(preds, target)
        return torch.mean(loss, dim=-1)

    @property
    def y_num_vars(self) -> int:
        dataset: SymbolicDataset = self.trainer.datamodule.train_dataset
        return dataset.y_num_vars


class MetaOptimizerImplicitForSymbolic(MetaOptimizerImplicit):
    def __init__(
        self,
        model: ImplicitModel,
        min_train_samples: int = 1,
        lr: float = 1e-3,
        loss_fn: _Loss = CrossEntropyLossFlat(reduction="none"),
    ):
        super().__init__(model=model, min_train_samples=min_train_samples, lr=lr)

        self.loss_fn = loss_fn

    def loss_function(self, target: dict[str, Tensor], preds: dict[str, Tensor]) -> Tensor:
        """
        Args:
            target (dict[str, Tensor]): Inputs/targets (samples, tasks, n_vars).
            preds (dict[str, Tensor]): Predictions (samples, tasks, n_vars * n_vals).

        Returns:
            Tensor: Losses (samples, tasks).
        """
        assert len(preds) == 1, "Only one output key supported for symbolic tasks"
        y_key = list(preds.keys())[0]
        target, preds = target[y_key], preds[y_key]
        target = target.view(target.shape[0], target.shape[1], self.y_num_vars, -1).argmax(dim=-1)
        loss = self.loss_fn(preds, target)
        return torch.mean(loss, dim=-1)

    @property
    def y_num_vars(self) -> int:
        dataset: SymbolicDataset = self.trainer.datamodule.train_dataset
        return dataset.y_num_vars
