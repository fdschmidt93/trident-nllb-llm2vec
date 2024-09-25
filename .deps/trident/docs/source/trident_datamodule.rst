#################
TridentDataModule
#################

.. include:: datamodule_intro.rst

***
API
***

Methods
=======

The below methods are user-facing :class:`~trident.core.module.TridentDataModule` methods. Since :class:`~trident.core.module.TridentDataModule` sub-classes the LightningDataModule_, all methods, attributes, and hooks of the LightningDataModule_ are also available.

**Important**: You should **not** override the following methods:

- ``train_dataloader``
- ``val_dataloader``
- ``test_dataloader``

since the :class:`~trident.core.module.TridentDataModule` automatically returns the dataloaders for the :class:`~trident.core.dataspec.TridentDataspec` enclosed in the configuration of the corresponding split. 

setup
~~~~~

.. automethod:: trident.core.datamodule.TridentDataModule.setup
    :noindex:

get
~~~

.. automethod:: trident.core.datamodule.TridentDataModule.get
    :noindex:

train_dataloader
~~~~~~~~~~~~~~~~
.. automethod:: trident.core.datamodule.TridentDataModule.train_dataloader
    :noindex:

val_dataloader
~~~~~~~~~~~~~~

.. automethod:: trident.core.datamodule.TridentDataModule.val_dataloader
    :noindex:

test_dataloader
~~~~~~~~~~~~~~~

.. automethod:: trident.core.datamodule.TridentDataModule.test_dataloader
    :noindex:

predict_dataloader
~~~~~~~~~~~~~~~~~~

.. automethod:: trident.core.datamodule.TridentDataModule.predict_dataloader
    :noindex:
