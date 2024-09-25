from copy import deepcopy
from typing import Optional, Any, Sequence, Union
import hydra
from omegaconf.dictconfig import DictConfig
from torch.utils.data.dataloader import DataLoader

from trident.utils.logging import get_logger
from trident.utils.types.dataspec import EvaluationDict, PreprocessingDict
from trident.utils.types.dataset import DatasetProtocol

log = get_logger(__name__)


PREPROCESSING_KWDS = {"method", "apply"}


class TridentDataspec:
    r"""
    A class to handle data specification in trident.

    This class is designed to instantiate the dataset,
    preprocess the dataset, and create data loaders for training, validation, or testing.

    The preprocessing configuration includes two special keys:
    - 'method': Holds dictionaries of class methods and their keyword arguments for preprocessing.
    - 'apply': Contains dictionaries for user-defined functions and their keyword arguments to apply on the dataset.

    Attributes:
        name (str): Name of the dataspec, helpful for logging and tracking.
        cfg (DictConfig): The configuration object that contains all the settings for dataset
                          instantiation, preprocessing, dataloader setup, and evaluation metrics.

    """

    def __init__(self, cfg: DictConfig, name: str = "None"):
        r"""
        Initializes the TridentDataspec instance.

        Args:
            cfg: The Hydra configuration object for this dataspec.
            name: The name of the dataspec. Defaults to ``"None"``.
        """
        self.name = name
        self.cfg = cfg
        self.dataset: DatasetProtocol = self.preprocess(
            hydra.utils.instantiate(self.cfg.dataset), self.cfg.get("preprocessing")
        )
        if hasattr(self.cfg, "evaluation"):
            self.evaluation: EvaluationDict = hydra.utils.instantiate(
                self.cfg.evaluation
            )

    @staticmethod
    def preprocess(dataset: Any, cfg: Optional[PreprocessingDict]) -> Any:
        r"""
        Applies preprocessing steps to the dataset as specified in the config.

        The ``cfg`` includes two special keys:
        - ``"method"``: Holds dictionaries of class methods and their keyword arguments for preprocessing.
        - ``"apply"``: Contains dictionaries for user-defined functions and their keyword arguments to apply on the dataset.

        The preprocessing fucntions take the ``Dataset`` as the first positional argument. The functions are called in order of the configuration. Note that ``"method"`` is a convenience keyword which can also be achieved by pointing to the classmethod in ``"_target_"`` of an ``"apply"`` function.

        Args:
            dataset: The dataset to be preprocessed.
            cfg: A dictionary of preprocessing configurations.

        Returns:
            Any: The preprocessed dataset.
        """
        if cfg is None:
            return dataset
        cfg_ = deepcopy(cfg)
        extra_kwds = [k for k in cfg.keys() if k in PREPROCESSING_KWDS]
        for kwd in extra_kwds:
            kwd_cfg = cfg_[kwd]
            for key, key_cfg in kwd_cfg.items():
                # _method_ is for convenience
                # construct partial wrapper, instantiate with cfg, and apply to ret
                if kwd == "method":
                    key_cfg["_target_"] = (
                        f"{dataset.__class__.__module__}.{dataset.__class__.__name__}.{key}"
                    )
                # methods and functions should take dataset as first positional argument
                val = hydra.utils.instantiate(key_cfg, dataset)
                # `fn` might mutate ret in-place
                if val is not None:
                    dataset = val
        return dataset

    def get_dataloader(
        self,
    ) -> DataLoader:
        """
        Creates a DataLoader for the dataset.

        Args:
            signature_columns: Columns to be used in the dataloader. Defaults to None.
            If passed, removes unused columns if configured in ``misc``.

        Returns:
            DataLoader: The DataLoader configured as per the specified settings.
        """
        dataset: DatasetProtocol = self.dataset
        return hydra.utils.call(self.cfg.dataloader, dataset=dataset)

    def _remove_unused_columns(
        self, signature_columns: Sequence[str]
    ) -> DatasetProtocol:
        """
        Removes columns from the dataset that are not used.

        Args:
            signature_columns: A sequence of column names to keep in the dataset.

        Returns:
            Dataset: The modified dataset with unused columns removed.
        """
        column_names: Union[None, list[str]] = getattr(
            self.dataset, "column_names", None
        )
        if column_names is not None:
            signature_columns = (
                signature_columns if signature_columns is not None else []
            )
            ignored_columns = list(set(column_names) - set(signature_columns))
            if len(ignored_columns) > 0:
                log.info(
                    f"The following columns don't have a corresponding argument in "
                    f"`model forward` and have been ignored: {', '.join(ignored_columns)}."  # type: ignore
                )
            # ignoring as new_fingerprint typically not passed
            return self.dataset.remove_columns(ignored_columns)  # type: ignore
        else:
            dataset_name_ = self.name if isinstance(self.name, str) else "unnamed"
            log.warning(
                f"Attempting to remove unused columns for unsupported dataset {dataset_name_}!"
            )
            return self.dataset
