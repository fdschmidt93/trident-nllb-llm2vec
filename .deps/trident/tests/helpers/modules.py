from typing import Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from trident.core.module import TridentModule
from trident.utils.logging import get_logger

log = get_logger(__name__)


def get_module() -> nn.Linear:
    """Construct and return a simple linear module for testing purposes."""
    network = nn.Linear(10, 1, bias=False)
    network.train()
    network.weight = nn.Parameter(torch.Tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
    return network


class ToyModule(TridentModule):
    """A toy module for testing, inheriting from TridentModule."""

    def __init__(self, monitor_lr_rate: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor_lr_rate = monitor_lr_rate
        self.epoch = 0

    def batch_forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass for a single batch."""
        ret = {}
        ret["preds"] = self.model(batch["examples"])

        # Compute loss if labels are present
        if (labels := batch.get("labels", None)) is not None:
            assert isinstance(labels, torch.Tensor)  # satisfy linter
            ret["loss"] = F.mse_loss(ret["preds"], labels)
        return ret

    def forward(
        self, batch: dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        """Forward pass for the module."""
        if "examples" in batch:
            b = cast(dict[str, torch.Tensor], batch)
            return self.batch_forward(b)
        else:
            b = cast(dict[str, dict[str, torch.Tensor]], batch)
            # runs only with multi train dataset
            first_half_correct = b["first_half"]["examples"].sum(0)[:5].sum() == 5
            second_half_correct = b["second_half"]["examples"].sum(0)[5:].sum() == 5
            assert first_half_correct.item(), "First half has incorrect examples"
            assert second_half_correct.item(), "Second half has incorrect examples"

            # Compute the batch forward for each dataset in the batch
            rets = {
                dataset_name: self.batch_forward(dataset_batch)
                for dataset_name, dataset_batch in b.items()
            }

            # Compute the average loss from all datasets
            loss = torch.stack([v["loss"] for v in rets.values()]).mean()
            return {"loss": loss}

    def on_train_epoch_end(self) -> None:
        self.epoch += 1
        if self.monitor_lr_rate:
            schedulers = self.lr_schedulers()
            lr = schedulers.get_lr()[0]
            assert lr == 10 - self.epoch
