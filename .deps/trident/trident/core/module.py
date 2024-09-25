from typing import Any, Optional

import hydra
from omegaconf import DictConfig

from trident.core.mixins.evaluation import EvalMixin
from trident.core.mixins.optimizer import OptimizerMixin
from trident.utils.logging import get_logger

log = get_logger(__name__)


class TridentModule(OptimizerMixin, EvalMixin):
    """Base module of Trident that wraps model, optimizer, scheduler, evaluation.

    Args:
        model:
            Needs to instantiate a :obj:`torch.nn.Module` that

            * Takes the batch unpacked
            * Returns a container with "loss" and other required attrs

            .. seealso:: :py:meth:`src.modules.base.TridentModule.forward`, :py:meth:`src.modules.base.TridentModule.training_step`, :repo:`tiny bert example <configs/module/tiny_bert.yaml>`

        optimizer:
            Configuration for the optimizer of your :py:class:`core.trident.TridentModule`.

            .. seealso:: :py:class:`src.modules.mixin.optimizer.OptimizerMixin`, :repo:`AdamW config <examples/configs/optimizer/adamw.yaml>`

        scheduler:
            Configuration for the scheduler of the optimizer of your :py:class:`src.modules.base.TridentModule`.

            .. seealso:: :py:class:`src.modules.mixin.optimizer.OptimizerMixin`, :repo:`Linear Warm-Up config <examples/configs/scheduler/linear_warm_up.yaml>`

        evaluation:
            Please refer to :ref:`evaluation`

            .. seealso:: :py:class:`src.modules.mixin.evaluation.EvalMixin`, :repo:`Classification Evaluation config <configs/evaluation/sequence_classification.yaml>`
    """

    def __init__(
        self,
        model: DictConfig,
        optimizer: DictConfig,
        scheduler: Optional[DictConfig] = None,
        initialize_model: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__()
        # super().__init__() calls LightningModule.save_hyperparameters() in `EvalMixin.__init__`
        if initialize_model:
            self.model = hydra.utils.instantiate(self.hparams.model)

    def forward(self, batch: dict) -> dict:
        """Plain forward pass of your model for which the batch is unpacked.

        Args:
            batch: input to your model

        Returns:
            ModelOutput: container with attributes required for evaluation
        """
        return self.model(**batch)

    def training_step(self, batch: dict, batch_idx: int) -> dict[str, Any]:
        """Comprises training step of your model which takes a forward pass.

        **Notes:**
            If you want to extend `training_step`, add a `on_train_batch_end` method via overrides.
            See: Pytorch-Lightning's `on_train_batch_end <https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#on-train-batch-end>`_

        Args:
            batch: typically comprising input_ids, attention_mask, and position_ids
            batch_idx: variable used internally by pytorch-lightning

        Returns:
            Union[dict[str, Any], ModelOutput]: model output that must have 'loss' as attr or key
        """
        outputs = self(batch)
        self.log("train/loss", outputs["loss"])
        return outputs
