The default configuration (``configs/datamodule/default.yaml``) for a :class:`~trident.core.datamodule.tridentdatamodule` defines how training and evaluation datasets are instantiated.
Each split is a dictionary of :class:`~trident.core.dataspec.TridentDataspec`.

.. code-block:: yaml

    _target_: trident.TridentDataModule
    _recursive_: false

    misc:
        # reserved key for general TridentDataModule configuration
    train:
        # DictConfig of TridentDataspec
    val:
        # DictConfig of TridentDataspec
    test:
        # DictConfig of TridentDataspec
