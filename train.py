import hydra
from omegaconf import OmegaConf
from pytorch_lightning import Trainer, seed_everything


@hydra.main(config_path="./configs/", config_name="train", version_base=None)
def train(cfg):
    seed_everything(cfg.seed)

    dataset = hydra.utils.instantiate(cfg.experiment.dataset)
    task = hydra.utils.instantiate(cfg.experiment.task)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False
    callbacks = hydra.utils.instantiate(cfg.experiment.callbacks) if cfg.experiment.callbacks else None

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
