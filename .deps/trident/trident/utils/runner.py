import warnings
from pathlib import Path
from typing import List, Sequence

import hydra
import lightning as L
import rich.syntax
import rich.tree
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, ListConfig, OmegaConf

from trident.utils.logging import get_logger

log = get_logger(__name__)


def extras(cfg: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - forcing debug friendly configuration
    - verifying experiment name is set when running in experiment mode

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    # debuggers don't like GPUs and multiprocessing
    if cfg.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        for split in ("train", "val", "test"):
            if split_cfg := cfg.datamodule.get(split):
                for dataspec_cfg in split_cfg.values():
                    if dataloader_cfg := dataspec_cfg.get("dataloader"):
                        dataloader_cfg.num_workers = 0
                        dataloader_cfg.pin_memory = False

    # convert ckpt path to absolute path
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path and not Path(ckpt_path).is_absolute():
        log.info("Converting ckpt path to absolute path! <config.ckpt_path=...>")
        cfg.ckpt_path = str(Path(hydra.utils.get_original_cwd()) / ckpt_path)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "run",
        "trainer",
        "module",
        "datamodule",
        "callbacks",
        "logger",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as fp:
        rich.print(tree, file=fp)


@rank_zero_only
def log_hyperparameters(
    cfg: DictConfig,
    module: L.LightningModule,
    trainer: L.Trainer,
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of module parameters
    """

    hparams = {}
    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return
    log.info("Logging hyperparameters!")

    # choose which parts of hydra config will be saved to loggers
    for key in ["run", "trainer", "module", "datamodule", "callbacks"]:
        if key in cfg:
            cfg_ = cfg[key]
            hparams[key] = (
                cfg_
                if not isinstance(cfg_, (ListConfig, DictConfig))
                else OmegaConf.to_container(cfg[key], resolve=True)
            )

    # save number of module parameters
    hparams["module/params/total"] = sum(p.numel() for p in module.parameters())
    hparams["module/params/trainable"] = sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )
    hparams["module/params/non_trainable"] = sum(
        p.numel() for p in module.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def finish(
    cfg: DictConfig,
    module: L.LightningModule,
    datamodule: L.LightningDataModule,
    trainer: L.Trainer,
    callbacks: List[L.Callback],
    logger: List[Logger],
) -> None:
    """Makes sure everything closed properly."""
