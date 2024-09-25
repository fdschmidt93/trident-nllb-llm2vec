from pathlib import Path
from typing import List, Optional, Union, cast

import hydra
import torch
from lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf

from trident.utils.logging import get_logger
from trident.utils.runner import log_hyperparameters

log = get_logger(__name__)


def instantiate_objects(cfg: DictConfig, key: str) -> List[Union[Callback, Logger]]:
    objects = []
    if key in cfg and cfg[key]:
        for conf in cfg[key].values():
            if "_target_" in conf:
                log.info(f"Instantiating {key[:-1]} <{conf._target_}>")
                objects.append(hydra.utils.instantiate(conf))
    return objects


def run(cfg: DictConfig) -> Optional[torch.Tensor]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # warning
    if cfg.trainer.get("use_distributed_sampler"):
        log.warning(
            "`trainer.use_distributed_sampler=True` is not supported by trident due to multi-GPU validation. Set to false and wrap your training dataset in a distributed sampler. See FAQ at: https://fdschmidt93.github.io/trident/docs/qa.html"
        )
        return

    seed_everything(OmegaConf.select(cfg, "run.seed"), workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    log.info(f"Instantiating module <{cfg.module._target_}>")
    module: LightningModule = hydra.utils.instantiate(cfg.module)

    callbacks = cast(list[Callback], instantiate_objects(cfg, "callbacks"))
    logger = cast(list[Logger], instantiate_objects(cfg, "logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    log_hyperparameters(cfg, module, trainer)

    if cfg.trainer.get("limit_train_batches", 1.0) > 0:
        trainer.fit(
            model=module,
            datamodule=datamodule,
            ckpt_path=OmegaConf.select(cfg, "run.ckpt_path"),
        )

    score = None
    if optimized_metric := OmegaConf.select(cfg, "run.optimized_metric"):
        score = trainer.callback_metrics.get(optimized_metric)

    if (
        cfg.datamodule.get("test") is not None
        and cfg.trainer.get("limit_test_batches", 1.0) > 0
    ):
        if best_model_path := getattr(
            trainer.checkpoint_callback, "best_model_path", ""
        ):
            log.info(f"Best checkpoint path:\n{best_model_path}")
            trainer.test(module, datamodule=datamodule, ckpt_path="best")
        else:
            log.info(
                "No checkpoint callback with optimized metric in trainer. Using final checkpoint."
            )
            trainer.test(module, datamodule=datamodule)

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            import wandb

            wandb.finish()
    return score


@hydra.main(
    version_base="1.3",
    config_path=str(Path.cwd() / "configs"),
    config_name="config.yaml",
)
def main(cfg: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    from trident.utils.runner import extras, print_config

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    extras(cfg)
    # Init lightning datamodule

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # Train model
    return run(cfg)


if __name__ == "__main__":
    main()
