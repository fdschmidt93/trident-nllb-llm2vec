.. _walkthrough:

#######################
|project| in 20 minutes
#######################

The walkthrough first introduces common concepts of hydra_ and then walks through an exemplary text-classification pipeline for sequence-pair classification (NLI). The `example NLI project <https://github.com/fdschmidt93/trident/tree/main/examples>`_ is embedded in the repository.

hydra primer
============

.. include:: hydra.rst

Project Structure
=================

An exemplary structure for a user project is shown below:

- `configs <https://github.com/fdschmidt93/trident/tree/main/examples/configs>`_ holds the entire hydra_ yaml configuration
- `src <https://github.com/fdschmidt93/trident/tree/main/examples/src>`_ comprises required code, typically for processing and evaluation, as referred to in the config

.. code-block:: bash

    # yaml configuration
    your-project
    ├── configs
    │   ├── config.yaml # inherits all `default.yaml`
    │   ├── experiment # typical entry point, 2nd-level `config.yaml` for your experiment
    │   │   ├── default.yaml
    │   │   └── nli.yaml
    │   ├── module
    │   │   ├── optimizer # torch.optim
    │   │   │   ├── adam.yaml
    │   │   │   └── adamw.yaml
    │   │   ├── scheduler # learning-rate scheduler
    │   │   │   └── linear_warm_up.yaml
    │   │   ├── default.yaml
    │   │   └── text_classification.yaml
    │   ├── datamodule
    │   │   ├── default.yaml
    │   │   └── mnli_train.yal
    │   ├── dataspec # defines [dataset, preprocessing, dataloader, evaluation]
    │   │   ├── dataloader
    │   │   │   └── default.yaml
    │   │   ├── evaluation
    │   │   │   └── text_classification.yaml
    │   │   │ # inherits dataloader/default.yaml
    │   │   ├── default.yaml
    │   │   │ # task-specific dataspecs              
    │   │   │ # inherits default.yaml and evaluation/text_classification.yaml
    │   │   ├── text_classification.yaml
    │   │   │ # dataset-group specific dataspecs              
    │   │   │ # inherits text_classification.yaml
    │   │   ├── nli.yaml
    │   │   │ # dataset-specific dataspecs              
    │   ├── dataspecs # defines groups of dataspec
    │   │   ├── mnli_train.yaml
    │   │   ├── xnli_val_test.yaml
    │   │   ├── amnli_val_test.yaml
    │   │   └── indicxnli_val_test.yaml
    │   ├── hydra 
    │   │   └── default.yaml
    │   ├── logger
    │   │   ├── csv.yaml
    │   │   └── wandb.yaml
    │   ├── callbacks # defines callbacks like lightning.pytorch.ModelCheckpoint
    │   │   └── default.yaml                 
    │   └── trainer # defines lightning.pytorch.Trainer
    │       ├── debug.yaml
    │       └── default.yaml
    └── src # typical code folder structure
        └── tasks
            └── text_classification
                ├── evaluation.py
                └── processing.py

Components
==========

TridentModule
-------------

:class:`~trident.core.module.TridentModule` extends the LightningModule_. The configuration defines all required components for a :class:`~trident.core.module.TridentModule`:

1. ``model``: ``_target_`` to your model constructor for which ``TridentModule.model`` will be initialized
2. ``optimizer``: the optimizer for all :class:`~trident.core.module.TridentModule` parameters
3. ``scheduler``: the learning-rate scheduler for the ``optimizer``

The ``default.yaml`` by default sets up AdamW optimizer and linear learning rate scheduler.

.. code-block:: yaml

    # _target_ is hydra-lingo to point to the object (class, function) to instantiate
    _target_: trident.TridentModule
    # _recursive_: true would mean all keyword arguments are /already/ instantiated
    # when passed to `TridentModule`
    _recursive_: false

    defaults:
    # interleaved with setup so instantiated later (recursive false)
    - optimizer: adamw.yaml  # see config/module/optimizer/adamw.yaml for default
    - scheduler: linear_warm_up  # see config/module/scheduler/linear_warm_up.yaml for default
    
    # required to be defined by user
    model: ???

A common pattern is that users create a ``configs/module/task.yaml`` that predefines shared ``model`` and ``evaluation`` logic for a particular task.

.. code-block:: yaml

    defaults:
      - default
      - evaluation: text_classification
    model:
      _target_: transformers.AutoModelForSequenceClassification.from_pretrained
      num_labels: ???
      pretrained_model_name_or_path: ???

- The ``model`` constructor points to ``transformers.AutoModelForSequenceClassification.from_pretrained``.
  The actual model and number of labels will be defined in either the experiment configuration or in the CLI (cf. ``???``).


TridentDataspec
---------------

.. include:: dataspec_intro.rst

dataset
^^^^^^^

.. include:: dataspec_dataset.rst

preprocessing
^^^^^^^^^^^^^

.. include:: dataspec_preprocessing.rst

dataloader
^^^^^^^^^^

.. include:: dataspec_dataloader.rst

.. _evaluation:

evaluation
^^^^^^^^^^

.. include:: dataspec_evaluation.rst

TridentDataModule
-----------------

.. include:: datamodule_intro.rst

Config Composition
^^^^^^^^^^^^^^^^^^

.. note:: Hierarchical config composition heavily relies on `default lists <https://hydra.cc/docs/advanced/defaults_list/>`_ .

The below file tree is a common structure for a hierarchical :class:`~trident.core.datamodule.TridentDatamodule` configuration in our NLI example.

We will hierarchically

1. Compose a general ``dataspec``
2. Compose a tast-specific text classification ``dataspec``
3. Compose a NLI ``dataspec``
4. Compose a train, val, or test split via ``dataspecs``
5. Compose a datamodule

.. code-block:: bash

    configs
    ├── config.yaml
    ├── datamodule
    │   └── default.yaml
    ├── dataspec
    │   ├── dataloader
    │   │   └── default.yaml
    │   ├── evaluation
    │   │   └── text_classification.yaml
    │   ├── default.yaml
    │   ├── nli.yaml
    │   └── text_classification.yaml
    └── dataspecs
        ├── mnli_train.yaml
        ├── xnli_val_test.yaml
        └── amnli_val_test.yaml

Default
"""""""

The general ``dataspec`` simply defines the default (``./configs/dataspec/default.yaml``) configuration.

.. code-block:: yaml

    defaults:
      - dataset: null
      # pull in the default dataloader
      - dataloader: default

    dataset:
      #
      _target_: datasets.load.load_dataset

Text Classification
"""""""""""""""""""

.. code-block:: yaml

    defaults:
      - default
      - evaluation: text_classification # see TridentDataspec evaluation
    
    # task specific preprocessing
    preprocessing:
        ... # see TridentDataspec preprocessing


.. seealso::
    :ref:`TridentDataspec.preprocessing <preprocessing>`

NLI
"""

The ``configs/dataspec/nli.yaml`` simply extends the task-specific ``text_classification.yaml`` by specifying columns for the tokenizer in preprocessing.

.. code-block:: yaml

    defaults:
      - text_classification

    preprocessing:
      map:
        function:
          # column_names denotes input to the tokenizer during preprocessing
          column_names:
            text: premise
            text_pair: hypothesis

.. seealso::
    :ref:`TridentDataspec.preprocessing <preprocessing>`, :ref:`TridentDataspec.preprocessing <preprocessing>`

Dataspecs
"""""""""

We can now compose ``dataspecs`` which group :class:`~trident.core.dataspec.TridentDataspec`` for entire datasets.

The ``configs/dataspecs/xnli_val_test.yaml`` levers ``hydra`` `package directives <https://hydra.cc/docs/advanced/overriding_packages/>`_ to put the ``nli`` configuration into the corresponding dataspec keys.

.. code-block:: yaml

    defaults:
      # package `nli` of configs/dataspec into @{...}
      - /dataspec@validation_xnli_en: nli
      - /dataspec@validation_xnli_es: nli
      # ... can extend this to the entire XNLI benchmark for val and test splits
    validation_xnli_en:
      dataset:
        path: xnli
        name: en
        split: validation
    validation_xnli_es:
      dataset:
        path: xnli
        name: es
        split: validation
    # ... can extend this to the entire XNLI benchmark for val and test splits

NLI Datamodules
"""""""""""""""

Datamodule Configurations
~~~~~~~~~~~~~~~~~~~~~~~~~

We can now use `package directives <https://hydra.cc/docs/advanced/overriding_packages/>`_ to include the configuration from the ``configs/dataspecs/xnli_val_test.yaml`` file into the ``val`` and ``test`` keys of the :class:`~trident.core.datamodule.TridentDatamodule`.

.. warning:: When using packaging, make sure to provide a list of ``dataspecs`` configurations to allow for the merging of multiple ``datamodule`` configurations in the ``experiment`` configuration.

**Imporant**:
    - A single :class:`~trident.core.dataspec.TridentDataspec`` in ``train`` of the :class:`~trident.core.datamodule.TridentDatamodule` will return a  ``batch`` of ``dict[str, Any]`` at runtime
    - Multiple :class:`~trident.core.dataspec.TridentDataspec`` in ``train`` of the :class:`~trident.core.datamodule.TridentDatamodule` will return a  ``batch`` of ``dict[str, dict[str, Any]]`` for multi-dataset training at runtime

**Example Configuration**

We now `package <https://hydra.cc/docs/advanced/overriding_packages/>`_ the ``config/dataspec/xnli_val_test.yaml`` into a list configuration in ``datamodule.val`` of our experiment. We can thereby easily in- and exclude various datasets for training, validation, or testing.

.. code-block:: yaml

    # variant A: training on a single dataset
    defaults:
      - /dataspecs@datamodule.train: mnli_train
      - /dataspecs@datamodule.val:
        - xnli_val_test
        - amnli_val_test
        - indicxnli_val_test
      - /dataspecs@datamodule.test:
        - xnli_val_test
        - amnli_val_test
        - indicxnli_val_test
    # variant B: training on multiple datasets
    defaults:
      - /dataspecs@datamodule.train: 
        - mnli_train
        - xnli_train
    # ...

Experiment
----------

The experiment configurations also segments into a general ``default.yaml`` and a task-specific ``nli.yaml``.

The ``run`` key is, next to ``module``, ``datamodule``, and ``trainer`` a special key reserved for user configuration. The configuration of this key also gets saved in your ``logger`` (e.g., ``wandb``).

.. code-block:: yaml

    defaults:
      - override /trainer: default
      - override /callbacks: default
      - override /logger: wandb
    
    # `run` namespace should hold your individual configuration
    run:
      seed: 42
      task: ???

    trainer:
      max_epochs: 10
      devices: 1
      precision: "16-mixed"
      deterministic: true
      inference_mode: false
    
    # log vars infers first training dataset
    # for logging batch size
    _log_vars:
      # needed because hydra cannot index list in interpolation
      train_datasets: ${oc.dict.keys:datamodule.train}
      train_dataset: ${_log_vars.train_datasets[0]}
      train_batch_size: ${datamodule.train.${_log_vars.train_dataset}.dataloader.batch_size}
      
    logger:
      wandb:
        name: "model=${module.model.pretrained_model_name_or_path}_epochs=${trainer.max_epochs}_bs=${_log_vars.train_batch_size}_lr=${module.optimizer.lr}_scheduler=${module.scheduler.num_warmup_steps}_seed=${run.seed}"
        tags:
          - "${module.model.pretrained_model_name_or_path}"
          - "bs=${_log_vars.train_batch_size}"
          - "lr=${module.optimizer.lr}"
          - "scheduler=${module.scheduler.num_warmup_steps}"
        project: ${run.task}


.. code-block:: yaml


    # @package _global_
    # The above line is important! It sets the namespace of the config

    defaults:
      - default
      # We can now combine `dataspecs` for training, validation, and testing
      - /dataspecs@datamodule.train:
        - mnli_train
      - /dataspecs@datamodule.val:
        - xnli_val_test
        - indicxnli_val_test
        - amnli_val_test
      - override /module: text_classification

    run:
      task: nli

    module:
      model:
        pretrained_model_name_or_path: "xlm-roberta-base"
        num_labels: 3

Commandline Interface
=====================

hydra_ allows to simply set configuration items on the commandline. See more information

.. code-block:: bash

    # change the learning rate
    python -m trident.run experiment=nli module.optimizer.lr=0.0001
    # set a different optimizer
    python -m trident.run experiment=nli module.optimizer=adam
    # no lr scheduler
    python -m trident.run experiment=nli module.scheduler=null

.. warning:: The commandline interface only supports absolute paths. For instance, overriding defaults at runtime from the CLI is not possible.
