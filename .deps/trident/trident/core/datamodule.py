from typing import Any, Optional, Sized, Union, cast

from lightning import LightningDataModule, Trainer
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from omegaconf.dictconfig import DictConfig
from torch.utils.data import DataLoader, IterableDataset

from trident.core.dataspec import TridentDataspec
from trident.utils.dictlist import DictList
from trident.utils.enums import Split
from trident.utils.logging import get_logger

log = get_logger(__name__)


class TridentDataModule(LightningDataModule):
    trainer: Trainer

    """
    The base class for all datamodules.

    The :obj:`TridentDataModule` facilitates writing a :obj:`LightningDataModule` with little to no boilerplate via Hydra configuration. It splits into

    - :obj:`dataset`:
    - :obj:`dataloader`:

    Args:
        dataset (:obj:`omegaconf.dictconfig.DictConfig`):

            A hierarchical :obj:`DictConfig` that instantiates or returns the dataset for :obj:`self.dataset_{train, val, test}`, respectively.

            Typical configurations follow the below pattern:

    .. seealso:: :py:func:`src.utils.hydra.instantiate_and_apply`, :py:func:`src.utils.hydra.expand`
        dataloader (:obj:`omegaconf.dictconfig.DictConfig`):


            .. seealso:: :py:func:`src.utils.hydra.expand`


    Notes:
        - The `train`, `val`, and `test` keys of :obj:`dataset` and :obj:`dataloader` join remaining configurations with priority to existing config
        - :obj:`dataloader` automatically generates `train`, `val`, and `test` keys for convenience as the config is evaluated lazily (i.e. when a :obj:`DataLoader` is requested)

    Example:

        .. code-block:: yaml

            _target_: src.datamodules.base.TridentDataModule
            _recursive_: false

            dataset:
              _target_: datasets.load.load_dataset
              # access methods of the instantiated object
              _method_:
                map: # dataset.map for e.g. tokenization
                  # kwargs for dataset.map
                  function:
                    _target_: 
                    _partial_: true
                  num_proc: 12
              path: glue
              name: mnli
              train:
                split: "train"
              val:
                # inherits `path`, `name`, etc.
                split: "validation_mismatched+validation_matched"
              test:
                # set `path`, `name`, `lang` specifically, remainder inherited
                path: xtreme
                name: xnli
                lang: de
                split: "test"
            dataloader:
              _target_: torch.utils.data.dataloader.DataLoader
              batch_size: 8
              num_workers: 0
              pin_memory: true
              # linked against global cfg
    """

    def __init__(
        self,
        train: Optional[DictConfig] = None,
        val: Optional[DictConfig] = None,
        test: Optional[DictConfig] = None,
    ):
        super().__init__()

        self.split_cfg = DictConfig(
            {Split.TRAIN: train, Split.VAL: val, Split.TEST: test}
        )

        self._dataspecs: dict[Split, Union[None, DictList[TridentDataspec]]] = {
            Split.TRAIN: None,
            Split.VAL: None,
            Split.TEST: None,
            Split.PREDICT: None,
        }

    def __getitem__(self, split: Split) -> DictList[TridentDataspec]:
        ret = self._dataspecs[split]
        if ret is None:
            raise KeyError(f"{split}")
        return ret

    def get(
        self, split: Split, default: Any = None
    ) -> Union[None, DictList[TridentDataspec]]:
        r"""
        Retrieve the TridentDataspecs for the given split.

        This method attempts to fetch a dataspec associated with a specific split. If the
        split is not found, it returns a default value.

        Parameters:
            split: The ``Split`` used to retrieve the dataspec.
            default: The default value to return if the split is not found.

        Returns:
            The ``DictList`` of ``TridentDataspec`` for the given split or None.
        """
        return self._dataspecs.get(split, default)

    def setup(self, stage: Optional[str] = None) -> None:
        stage_to_splits: dict[Union[None, str], list[Split]] = {
            None: [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT],
            "fit": [Split.TRAIN, Split.VAL],
            "validate": [Split.VAL],
            "test": [Split.TEST],
            "predict": [Split.PREDICT],
        }
        for split in stage_to_splits.get(stage, []):
            if split_cfg := self.split_cfg.get(split):
                data = {
                    name: TridentDataspec(spec, name)
                    for name, spec in split_cfg.items()
                }
                self._dataspecs[split] = DictList(data)

    def __len__(self) -> int:
        """Returns the number of instances in :obj:`dataset_train`."""
        dataspecs_train = self._dataspecs[Split.TRAIN]

        if dataspecs_train is None:
            return 0
        elif isinstance(dataspecs_train, DictList):
            if len(dataspecs_train) == 1:
                dataset = dataspecs_train[0].dataset
                dataset = cast(Sized, dataset)
                if isinstance(dataset, IterableDataset):
                    return self.trainer.global_step
                else:
                    return len(dataset)
            else:
                try:
                    return max(
                        [
                            len(cast(Sized, d.dataset))
                            for _, d in dataspecs_train.items()
                        ]
                    )
                except TypeError:
                    return self.trainer.global_step
        else:
            raise ValueError("Unexpected type for dataset_train")

    def _get_dataloader(self, split: Split) -> Union[DataLoader, CombinedLoader]:
        """Checks existence of dataset for :obj:`split` and returns :obj:`DataLoader` with cfg.

        The return type of this function typically depends on the scenario:
            * :obj:`DataLoader`: simple, single datasets
            * :obj:`CombinedLoader`: for modes, see CombinedLoader documentation
                - mode = "sequential" common in zero-shot cross-lingual transfer, evaluating on many varying datasets
                - mode = "max_size_cycle" common in zero-shot cross-lingual transfer, evaluating on many varying datasets

            .. seealso:: :py:meth:`trident.core.datamodule.TridentDataModule._get_dataloader`

        Args:
            split: one of :obj:`train`, :obj:`val`, :obj:`test`, or :obj:`predict`

        Returns:
            Union[DataLoader, list[DataLoader], dict[str, DataLoader]]: [TODO:description]
        """
        dataspecs = self._dataspecs[split]
        if dataspecs is None:
            raise ValueError(f"Dataspec for {split.value} missing!")

        dataloaders: dict[str, DataLoader] = {
            name: dataspec.get_dataloader() for name, dataspec in dataspecs.items()
        }

        # TODO mode setting
        if split == Split.TRAIN:
            if len(dataloaders) == 1:
                return next(iter(dataloaders.values()))
            return iter(
                CombinedLoader(
                    dataloaders,
                    mode="max_size_cycle",
                )
            )
        return CombinedLoader(dataloaders, mode="sequential")

    def train_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.TRAIN)

    def val_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.VAL)

    def test_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.TEST)

    def predict_dataloader(self) -> Union[DataLoader, CombinedLoader]:
        return self._get_dataloader(Split.PREDICT)
