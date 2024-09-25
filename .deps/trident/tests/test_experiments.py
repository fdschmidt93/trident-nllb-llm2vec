import logging

import torch
from hydra import compose, initialize

from trident.run import run
from trident.utils.logging import get_logger

log = get_logger(__name__)

CONFIG_PATH = "./configs/"


def test_single_train_dataloader() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=["experiment=test_single_train_single_val_test"],
        )
        is_ = torch.Tensor(run(cfg))
        should = torch.zeros(1)[0]
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_multi_train_dataloader(caplog) -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=["experiment=test_many_train_single_val_test"],
        )
        out = run(cfg)
        with caplog.at_level(logging.WARNING):
            out = run(cfg)
        is_ = torch.Tensor(out)
        should = torch.zeros(1)[0]
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_off_by_one_test_single_train_many_val_test_single_eval_config() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=["experiment=test_single_train_many_val_test"],
        )
        is_ = torch.Tensor(run(cfg))
        should = torch.Tensor([1])
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_off_by_two_test_single_train_many_val_test_single_eval_config() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=test_single_train_many_val_test",
                "run.optimized_metric='off_by_two/val/mse_loss'",
            ],
        )
        is_ = torch.Tensor(run(cfg))
        should = torch.Tensor([4])
        assert torch.allclose(should, is_), f"{is_.item()} is not {should}"


def test_lr_scheduler_single_gpu_without_accumulation() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=["experiment=test_lr_scheduler_single_gpu_without_accumulation"],
        )
        run(cfg)


def test_lr_scheduler_single_gpu_with_accumulation_2() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=test_lr_scheduler_single_gpu_without_accumulation",
                "trainer.accumulate_grad_batches=2",
                # implies 10 batches per epoch
                "datamodule.train.single_train.dataloader.batch_size=1",
            ],
        )
        run(cfg)


def test_lr_scheduler_single_gpu_with_accumulation_5() -> None:
    with initialize(
        version_base=None,
        config_path=CONFIG_PATH,
    ):
        # config is relative to a module
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=test_lr_scheduler_single_gpu_without_accumulation",
                "trainer.accumulate_grad_batches=5",
                # implies 10 batches per epoch
                "datamodule.train.single_train.dataloader.batch_size=1",
            ],
        )
        run(cfg)
