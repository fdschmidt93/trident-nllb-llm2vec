.. role:: hidden
    :class: hidden-section

#############
TridentModule
#############

*************
Configuration
*************

A :class:`~trident.core.module.TridentModule` represents a wrapper around LightningModule_ to facilitate configuring training, validating, and testing from hydra_.

A :class:`~trident.core.module.TridentModule` is commonly defined hierarchically:

1. ``/config/module/default.yaml``: universal defaults
2. ``/config/module/$TASK.yaml``: task-specific configuration

The ``default.yaml`` configuration for the :class:`~trident.core.module.TridentModule` is typically defined as follows.

.. code-block:: yaml

    # default.yaml:
    # _target_ is hydra-lingo to point to the object (class, function) to instantiate
    _target_: trident.TridentModule
    # _recursive_: true would mean all kwargs are /already/ instantiated
    # when passed to `TridentModule`
    _recursive_: false

    defaults:
    # interleaved with setup so instantiated later (recursive false)
    - optimizer: adamw.yaml  # see config/module/optimizer/adamw.yaml for default
    - scheduler: linear_warm_up  # see config/module/scheduler/linear_warm_up.yaml for default
    
    # required to be set by user later on
    model: ???

A task-specific configuration typically is defined as follows (e.g., ``nli.yaml``):

.. code-block:: yaml

    # nli.yaml:
    defaults:
    - default
    
    model:
      _target_: AutoModelForSequenceClassification.from_pretrained
      num_labels: 3

***
API
***

Methods
=======

The below methods are user-facing :class:`~trident.core.module.TridentModule` methods. Since :class:`~trident.core.module.TridentModule` sub-classes the LightningModule_, all methods, attributes, and hooks of the LightningModule_ are also available.

**Important**: You should **not** override the following methods:

- ``validation_step``
- ``test_step``

since the :class:`~trident.core.module.tridentmodule` automatically runs evaluation per the :class:`~trident.core.dataspec.TridentDataspec` configuration.

You may override 

- ``on_validation_epoch_end``
- ``on_test_epoch_end``

but should make sure to also call the ``super()`` method!

forward
~~~~~~~

.. automethod:: trident.core.module.TridentModule.forward
    :noindex:

training_step
~~~~~~~~~~~~~

.. automethod:: trident.core.module.TridentModule.training_step
    :noindex:

log_metric
~~~~~~~~~~

.. automethod:: trident.core.mixins.evaluation.EvalMixin.log_metric
    :noindex:

num_training_steps
~~~~~~~~~~~~~~~~~~

.. autoproperty:: trident.core.mixins.optimizer.OptimizerMixin.num_training_steps
    :noindex:
