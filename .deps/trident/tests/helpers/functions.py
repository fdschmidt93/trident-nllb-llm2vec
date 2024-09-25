import torch
from tests.helpers.data import IdentityDataset
from trident.core.dataspec import TridentDataspec
from trident.utils.enums import Split
from trident.utils.dictlist import DictList
from typing import cast


def extend_batch(batch: dict, *args, **kwargs):
    """Called in evaluation step on each batch."""
    batch["set_one_from_prepare_batch"] = torch.ones_like(batch["labels"])
    return batch


def extend_outputs(outputs: dict, batch, *args, **kwargs):
    """Called in evaluation step on each model outputs dict."""
    outputs["set_one_from_prepare_outputs"] = torch.ones_like(batch["labels"])
    return outputs


def test_step_outputs(
    trident_module, step_outputs: dict, split: Split, dataset_name: str, *args, **kwargs
):
    """Called after evaluation loop."""
    datamodule = trident_module.trainer.datamodule
    dataspecs: DictList[TridentDataspec] = datamodule[split]
    dataset: IdentityDataset = cast(IdentityDataset, dataspecs[dataset_name].dataset)
    assert (
        dataset.y.shape
        == step_outputs["labels"].shape
        == step_outputs["set_one_from_prepare_batch"].shape
        == step_outputs["set_one_from_prepare_outputs"].shape
    )
    return step_outputs
