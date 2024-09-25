from functools import lru_cache
from typing import (
    Any,
    Callable,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
)

from lightning import LightningModule
from lightning.pytorch.utilities.parsing import AttributeDict
from numpy import ndarray
from omegaconf.base import DictKeyType
from omegaconf import DictConfig
import torch

from trident.core.datamodule import TridentDataModule
from trident.core.dataspec import TridentDataspec
from trident.utils import deepgetitem, get_dict_by_glob_pattern
from trident.utils.dictlist import DictList
from trident.utils.enums import Split
from trident.utils.logging import get_logger
from trident.utils.transform import flatten_dict
from trident.utils.types.dataspec import EvaluationDict, StepOutputsDict

log = get_logger(__name__)

StepOutputs = list[
    dict[
        str,
        Union[
            int,
            float,
            ndarray,
            torch.Tensor,
        ],
    ]
]


class EvalMixin(LightningModule):
    hparams: AttributeDict
    log: Callable
    r"""Mixin for base model to define evaluation loop largely via hydra.

    See also LightningModule_.

    The evaluation mixin enables writing evaluation via yaml files, here is an
    example for sequence classification, borrowed from configs/evaluation/classification.yaml.

    .. code-block:: yaml

        # apply transformation function
        prepare:
          batch: null # on each step
          outputs:    # on each step
            _target_: src.utils.hydra.partial
            _partial_: src.evaluation.classification.get_preds
            .. code-block: python

                # we link evaluation.apply.outputs against get_preds
                def get_preds(outputs):
                    outputs.preds = outputs.logits.argmax(dim=-1)
                    return outputs

          step_outputs: null  # on flattened outputs of what's collected from steps
        # Which keys/attributes are supposed to be collected from `outputs` and `batch`
        step_outputs:
          outputs: "preds" # can be a str
          batch: # or a list[str]
            - labels

        # either metrics or val_metrics and test_metrics
        # where the latter
        metrics:
          # name of the metric used eg for logging
          accuracy:
            # instructions to instantiate metric, preferrably torchmetrics.Metric
            metric:
              _target_: torchmetrics.Accuracy
            # either on_step: true or on_epoch: true
            on_step: true
            compute:
              preds: "outputs:preds"
              target: "batch:labels"
          f1:
            metric:
              _target_: torchmetrics.F1
            on_step: true
            compute:
              preds: "outputs:preds"
              target: "batch:labels"
    """

    def __init__(self) -> None:
        # this line effectively initializes the `LightningModule` of `TridentModule`
        super().__init__()
        # self.hparams is also then accessible in `TridentModule` after super().__init__()
        self.save_hyperparameters()
        # Lightning 2.0 requires manual management of evaluation step outputs
        self._eval_outputs: list[StepOutputs] = []

    def log_metric(
        self,
        split: Split,
        metric_key: Union[str, DictKeyType],
        input: Union[None, int, float, dict, torch.Tensor],
        log_kwargs: Optional[dict[str, Any]] = None,
        dataset_name: Optional[str] = None,
    ):
        """
        Log a metric for a given split with optional transformation.

        Parameters:
            split: The evaluation split.
            metric_key: Key identifying the metric.
            input: Metric value or dictionary of metric values.
            log_kwargs: Additional keyword arguments for logging.
            dataset_name: Name of the dataset if available.

        Notes:
        - This method assumes the existence of `self.evaluation.metrics`.
        - If `input` is a dictionary, each key-value pair is logged separately with the appropriate prefix.
        """
        # this allows users to not log any metrics
        if input is None:
            return

        # Determine if there's a transformation function for the metric
        log_kwargs = log_kwargs or {}
        log_kwargs["prog_bar"] = True

        prefix = dataset_name + "/" + split.value if dataset_name else split.value

        if self.trainer.global_rank == 0:
            if isinstance(input, dict):
                log_kwargs["dictionary"] = {
                    f"{prefix}/{k}": v for k, v in input.items()
                }
                self.log_dict(**log_kwargs, rank_zero_only=True)
            else:
                log_key = f"{prefix}/{metric_key}"
                self.log(name=log_key, value=input, **log_kwargs, rank_zero_only=True)

    def prepare_metric_input(
        self,
        cfg: Union[dict, DictConfig],
        outputs: Union[dict, NamedTuple],
        split: Split,
        batch: Optional[Union[dict, NamedTuple]] = None,
        dataset_name: Optional[str] = None,
    ) -> dict:
        r"""
        Collects user-defined attributes of outputs & batch to compute a metric.

        In the below example, the evaluation (i.e., the call of ``accuracy``)
        extracts

        1. ``preds`` from ``outputs`` and passes it as ``preds``
        2. ``labels`` from ``outputs`` and passes it as ``target``

        to ``accuracy`` via dot notation.

        .. note:: The variations in types (``dict`` or classes with attributes) of the underlying object is handled at runtime.

        The following variables are available:
            - ``trident_module``
            - ``outputs``
            - ``batch``
            - ``cfg``
            - ``dataset_name``

        Notes:
            - ``trident_module`` yields access to the Trainer_, which in turn also holds :class:`~trident.core.datamodule.TridentDatamodule`
            - ``batch`` is only relevant when the metric is called at each step
            - ``outputs`` either denotes the output of a step or the concatenated step outputs

        Example:
            .. code-block:: yaml

            metrics:
              acc:
                metric:
                  _partial_: true
                  _target_: torchmetrics.functional.accuracy
                  task: "multiclass"
                  num_classes: 3
                compute_on: "epoch_end"
                kwargs:
                  preds: "outputs.preds"
                  target: "outputs.labels"

        Parameters:
            cfg: Configuration dictionary for metric computation.
            outputs: Outputs data.
            batch: Batch data.

        Returns:
            dict: Dictionary containing required inputs for metric computation.

        Raises:
            ValueError: If the required key is not found in the provided data.
        """
        ret = {}
        data_sources = {
            "trident_module": self,
            "outputs": outputs,
            "batch": batch,
            "cfg": cfg,
            "dataset_name": dataset_name,
            "split": split,
        }
        for k, v in cfg.items():
            split_string = v.split(".", maxsplit=1)
            source_name = split_string[0]
            key = split_string[1] if len(split_string) == 2 else None
            source_data = data_sources.get(source_name)

            if source_data is None:
                raise ValueError(f"{source_name} not a recognized data source.")

            if key:  # key not empty or not None
                val = deepgetitem(source_data, key)
                if val is None:
                    raise ValueError(
                        f"{key} not found in {source_name} ({source_data})."
                    )

                ret[k] = val
            else:
                ret[k] = source_data
        return ret

    @staticmethod
    def _collect_step_output(
        outputs: dict,
        batch: dict,
        split_dico: Union[None, StepOutputsDict] = None,
    ) -> dict:
        """
        Collect user-defined attributes from outputs and batch at the end of `eval_step` into a dictionary.

        This function looks into the provided `split_dico` to determine which attributes from
        `outputs` and `batch` need to be collected. If `split_dico` is not provided, it defaults
        to collecting all attributes.

        .. seealso::
            :py:meth:`trident.core.mixins.evaluation.EvalMixin.eval_step`
            :py:meth:`trident.core.mixins.evaluation.EvalMixin.prepare_step_outputs`

        Parameters:
            outputs: Dictionary containing output data.
            batch: Dictionary containing batch data.
            split_dico:
                Dictionary specifying attributes to extract from `outputs` and `batch`.

        Returns:
            Dict[str, Any]: Dictionary of extracted attributes.
        """

        # Handle the absence of split_dico
        if split_dico is None:
            return {"outputs": outputs, "batch": batch}

        ret = {}
        source_mapping: dict[str, dict[str, Any]] = {"outputs": outputs, "batch": batch}

        for source_key, attribute_keys in split_dico.items():
            # Ensuring type-correctness for the get method
            if not isinstance(source_key, str):
                log.warning(
                    f"Unexpected key type {type(source_key)}. Expected a string."
                )
                continue
            source_data = source_mapping.get(source_key)
            if source_data is None:
                log.warn(
                    f"{source_key} is not supported, can only be one of ['outputs', 'batch']"
                )

            # If attribute_keys is a string, convert to a list for uniform processing
            if isinstance(attribute_keys, str):
                attribute_keys = [attribute_keys]

            # Extract values from the source data
            if source_data is not None:
                # fix typing
                attribute_keys = cast(list[str], attribute_keys)
                for attr_key in attribute_keys:
                    if "*" in attr_key:
                        pattern_dict = get_dict_by_glob_pattern(source_data, attr_key)
                        if not pattern_dict:
                            log.warn(f"{attr_key} pattern not in {source_key}")
                        else:
                            for k, v in pattern_dict.items():
                                ret[k] = v
                    else:
                        attr_value = source_data.get(attr_key)
                        if attr_value is not None:
                            ret[attr_key] = attr_value
                        else:
                            log.warn(f"{attr_key} not in {source_key}")
        return ret

    @lru_cache
    def _get_datasetspec(
        self, split: Split, dataloader_idx: int
    ) -> Tuple[str, TridentDataspec]:
        # Handle multiple datasets.
        datamodule: Union[None, TridentDataModule] = getattr(self.trainer, "datamodule")
        assert isinstance(
            datamodule, TridentDataModule
        ), "self.trainer.datamodule not a TridentDataModule! Unsupported operation."
        dataspecs: Union[None, DictList[TridentDataspec]] = datamodule.get(split)
        assert (
            dataspecs is not None
        ), f"dataspecs for {split} must be incorrectly set up!"
        dataspec_name = dataspecs._keys[dataloader_idx]
        dataspec = dataspecs[dataloader_idx]
        return dataspec_name, dataspec

    def eval_step(self, split: Split, batch: dict, dataloader_idx: int) -> None:
        r"""Performs model forward & user batch transformation in an eval step.

        Parameters:
            split: The evaluation split.
            batch: The batch of the evaluation (i.e. 'val' or 'test') step.
            dataloader_idx: The index of the current evaluation dataloader, :obj:`None` if single dataloader.

        Notes:
            - This function is called in `validation_step` and `test_step` of the LightningModule.

        .. seealso::
            `LightningModule.validation_step <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step>`_

        """
        dataspec_name, dataspec = self._get_datasetspec(split, dataloader_idx)
        evaluation_cfg: EvaluationDict = dataspec.evaluation
        metrics_cfg = evaluation_cfg["metrics"]

        # Prepare batch and outputs
        if callable(prepare_batch := deepgetitem(evaluation_cfg, "prepare.batch")):
            batch = prepare_batch(
                trident_module=self,
                batch=batch,
                split=split,
                dataset_name=dataspec_name,
            )
        outputs = self(batch)

        if callable(
            (prepare_outputs := deepgetitem(evaluation_cfg, "prepare.outputs"))
        ):
            outputs = prepare_outputs(
                trident_module=self,
                outputs=outputs,
                batch=batch,
                split=split,
                dataset_name=dataspec_name,
            )

        # Compute metrics
        if isinstance(metrics_cfg, Mapping):
            for v in metrics_cfg.values():
                if getattr(v, "compute_on", False) == "eval_step":
                    kwargs = self.prepare_metric_input(
                        cfg=v["kwargs"],
                        outputs=outputs,
                        batch=batch,
                        dataset_name=dataspec_name,
                        split=split,
                    )
                    v["metric"](**kwargs)

        # Handle step outputs collection
        dataloader_idx = dataloader_idx or 0  # Default to 0 if None
        try:
            eval_outputs = self._eval_outputs[dataloader_idx]
        except IndexError:
            if len(self._eval_outputs) == dataloader_idx:
                eval_outputs = []
                self._eval_outputs.append(eval_outputs)
            else:
                raise IndexError(
                    f"dataloader_idx {dataloader_idx} is not subsequent index in evaluation, check warranted!"
                )

        eval_outputs.append(
            self._collect_step_output(
                outputs, batch, evaluation_cfg.get("step_outputs")
            )
        )

    def _evaluate_metrics_for_dataset(
        self,
        split: Split,
        step_outputs: list[dict],
        evaluation_cfg: EvaluationDict,
        dataset_name: Optional[str] = None,
    ) -> None:
        """Compute and log metrics at the epoch's end for a specified dataset.

        Parameters:
            split: Evaluation split, i.e., "val" or "test".
            step_outputs: Aggregated step outputs for the dataset.
            metrics_cfg: Metric configurations for the dataset.
            dataset_name: Name of the dataset.
        """
        if deepgetitem(evaluation_cfg, "prepare.flatten_outputs", True):
            prepared_outputs = flatten_dict(step_outputs)
        else:
            prepared_outputs = step_outputs
        if callable(
            (
                prepare_step_outputs := deepgetitem(
                    evaluation_cfg, "prepare.step_outputs"
                )
            )
        ):
            prepared_outputs = prepare_step_outputs(
                trident_module=self,
                step_outputs=prepared_outputs,
                split=split,
                dataset_name=dataset_name,
            )
        for metric, metric_cfg in evaluation_cfg["metrics"].items():
            if getattr(metric_cfg, "compute_on", False) == "eval_step":
                input_ = metric_cfg["metric"]
            elif getattr(metric_cfg, "compute_on", "epoch_end") == "epoch_end":
                kwargs = self.prepare_metric_input(
                    cfg=metric_cfg["kwargs"],
                    outputs=prepared_outputs,
                    split=split,
                    dataset_name=dataset_name,
                )
                input_ = metric_cfg["metric"](**kwargs)
            else:
                raise ValueError(
                    f"`compute_on` of metric_cfg ({getattr(metric_cfg, 'compute_on')}) is not one of ['eval_step', 'epoch_end']"
                )
            self.log_metric(
                split=split,
                metric_key=metric,
                input=cast(Union[int, float, dict, torch.Tensor], input_),
                dataset_name=dataset_name,
            )

    def on_eval_epoch_end(self, split: Split) -> None:
        """Compute and log metrics for all datasets at the epoch's end.

        Note: the epoch only ends when all datasets are processed.

        This method determines if multiple datasets exist for the evaluation split and
        appropriately logs the metrics for each.

        Parameters:
            split: Evaluation split, i.e., "val" or "test".
        """
        # `metrics_cfg` is always the fallback, means lower level of hydra config does not exist
        step_outputs: list[list[dict]] = self._eval_outputs
        datamodule = getattr(self.trainer, "datamodule")
        assert isinstance(
            datamodule, TridentDataModule
        ), "datamodule must be `TridentDataModule`!"
        dataspecs: Union[None, DictList[TridentDataspec]] = datamodule.get(split)
        if dataspecs is not None:
            # idx aligns with dataloader_idx (i.e., sequential order of eval datasets) of lightning
            for idx, (dataspec_name, dataspec) in enumerate(dataspecs.items()):
                self._evaluate_metrics_for_dataset(
                    split=split,
                    step_outputs=step_outputs[idx],
                    evaluation_cfg=dataspec.evaluation,
                    dataset_name=dataspec_name,
                )
        self._eval_outputs.clear()

    def validation_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step(Split.VAL, batch, dataloader_idx or 0)

    def on_validation_epoch_end(self):
        return self.on_eval_epoch_end(Split.VAL)

    def test_step(
        self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> None:
        return self.eval_step(Split.TEST, batch, dataloader_idx or 0)

    def on_test_epoch_end(self):
        return self.on_eval_epoch_end(Split.TEST)
