import hydra
import torch
from lightning import Trainer, seed_everything
from omegaconf import OmegaConf

from utils import ast_eval

# torch.set_float32_matmul_precision("medium")  # or 'high' based on your needs


@hydra.main(config_path="./configs/", config_name="train", version_base=None)
def train(cfg):
    seed_everything(cfg.seed)

    dataset = hydra.utils.instantiate(cfg.experiment.dataset)
    task = hydra.utils.instantiate(cfg.experiment.task)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False
    callbacks = [hydra.utils.instantiate(cfg.experiment.callbacks)] if cfg.experiment.callbacks else None

    # Add experiment metadata to the logger
    if logger:
        logger.experiment.config.update(OmegaConf.to_container(cfg.experiment, resolve=True))

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        **cfg.experiment.trainer,
    )
    trainer.fit(model=task, datamodule=dataset)


if __name__ == "__main__":
    """Run with:
    `python train.py experiment=[experiment_folder]/[experiment_name].yaml [overrides]`
    """
    train()
