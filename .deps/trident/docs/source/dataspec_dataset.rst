The ``dataset`` instantiates a `Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`__ that is compatible with a PyTorch `DataLoader <https://pytorch.org/docs/stable/data.html#dataloader>`__. Any preprocessing should be defined in the corresponding ``preprocessing`` configuration of the :class:`~trident.core.dataspec.TridentDataspec`.

.. code-block:: yaml

  mnli_train:  
    dataset: # required, config on how to instantiate dataset
      _target_: datasets.load_dataset
      path: glue
      name: mnli
      split: train
