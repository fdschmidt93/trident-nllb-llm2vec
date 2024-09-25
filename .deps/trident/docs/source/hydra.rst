It is important to have basic familiarity with hydra_, which shines at bottom-up hierarchical yaml configuration.

Two key features of hydra_ for |project| are 

1. The `defaults-list <https://hydra.cc/docs/advanced/defaults_list/>`_ for hierarchical configuration composition.
2. `Package directives <https://hydra.cc/docs/advanced/overriding_packages/>`_ to cleanly combine configurations.

Below is a brief primer of hydra_.

Context
-------

For |project|, we will use hydra_ to declare our 

- Trainer_: declares how training, validation, and testing is run
- :class:`~trident.core.module.TridentModule`: declares the your "model"
- :class:`~trident.core.datamodule.TridentDatamodule`: declares your ``train``, ``val``, and ``test`` splits
- And other components like logging and checkpointing

Hierarchical Configuration
--------------------------

hydra_ is a Python yaml framework to compose complex pipelines. The directory tree of your |project| configuration in our simplified example may look like the below file tree. 

In what follows, we will focus on the hierarchy of the configuration below in the case of ``dataspec``. A ``dataspec`` defines the dataset, the associated pipelines for preprocessing and evaluation, and the dataloader for the preprocessed dataset.

.. code-block:: bash

    configs
    ├── config.yaml
    ├── dataspec # defines dataset, preprocessing, evaluation, and dataloader
    │   ├── dataloader
    │   │   ├── default.yaml
    │   │   └── train.yaml
    │   ├── evaluation
    │   │   └── text_classification.yaml
    │   ├── preprocessing
    │   │   └── shots.yaml
    │   ├── default.yaml               # general
    │   ├── text_classification.yaml   # task-specific: inherits general
    │   └── nli.yaml                   # more dataset-specific: inherits task-specific
    ├── dataspecs # will group a dataspec for a particular benchmark
    └── module
        ├── default.yaml
        └── my_module.yaml

The ``config/dataspec/default.yaml`` defines the defaults.

1. The ``null`` default of ``dataset`` reservese the key for a future override
2. the ``dataloader: default`` means that the ``./configs/dataspec/dataloader.yaml`` config will be sourced as the dataloader key for the default dataspec
#. We define the default ``_target_`` of a dataset which typically uses the 🤗 datasets_ library

.. code-block:: yaml

    defaults:
      # a null default reserves the key for future override
      - dataset: null 
      # the default dataloader is source into the dataloader key of the dataspec
      # i.e. ./configs/dataspec/dataloader.yaml will be sourced in the `dataloader` key of the config
      - dataloader: default 
      # _self_ allows you to control the resolution order of the config itself
      # _self_ is not required and appended to the end of the defaults list by default
      - _self_

    dataset:
      # most datasets use the Huggingface 
      _target_: datasets.load.load_dataset

The ``configs/dataspec/text_classification.yaml`` then extends the ``default.yaml``

.. code-block:: yaml

    defaults:
      - default
      - evaluation: text_classification

    # .. more configuration

The above sources the the configuration of ``config/dataspec/default.yaml`` as well as the task-specific ``./config/dataspec/evaluation/text_classification.yaml`` into the ``evaluation`` key of the ``text_classification.yaml``.

Lastly, ``./configs/dataspec/nli.yaml`` defines even more specific configuration.

.. code-block:: yaml

    defaults:
      - text_classification

    # ... nli-specific configuration

**Notes**:

- hydra_ follows an additive configuration paradigm: design your configuration to incrementally add what's required! Unsetting or removing options often is **very** unwieldy
- The configuration relies on relative paths in the configuration folder structure
- In the ``defaults``-list, the last one wins (dictionaries are merged sequentially)
- The own configuration typically comes last (can be controlled via ``_self_``, see `offficial documentation <https://hydra.cc/docs/advanced/defaults_list/>`_)


Imporant Special Keywords and Values
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Beyond the keywords below, ``???`` denote values in ``default.yaml`` to indicate that the corresponding **must be** set in a inheriting config.

.. list-table:: Special Hydra Syntax
   :widths: 25 10 65
   :header-rows: 1

   * - **Key**
     - **Type**
     - **Description**

   * - ``_target_``
     - ``str``
     - The ``_target_`` points to the Python function / method that initializes the object

   * - ``_recursive_``
     - ``bool``
     - Do not eagerly instantiate sub-keys that comprises ``_target_``, but only when ``hydra.utils.instantiate`` is called

   * - ``_partial_``
     - ``bool``
     - Instantiate ``_target_`` function curried with pre-set arguments and keywords

   * - ``_args_``
     - ``list[Any]``
     - Positional arguments for the ``__target__`` at ``hydra.utils.instantiate``

   * - ``_self_``
     - ``str``
     - **Only for defaults-lists**. You can add `_self_` to control the resolution order of the config itself. By default, `_self_` is appended to the end.

Object and Function Instantiation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

hydra_ allow you to define Python objects and functions declaratively in yaml. The below example instantiates an ``AutoModelForSequenceClassification`` with ``roberta-base``.

.. code-block:: yaml

   sequence_classification_model:
       # path to the object we want to instantiate
       _target_: transformers.AutoModelForSequenceClassification
       # kwargs for the object we want to instantiate
       # can be themselves instantiated!
       pretrained_model_name_or_path: "roberta-base"
       num_labels: 3
       # optionally positional arguments
       # _args_:
       #  - 0
       #  - 1

.. note:: Carefully check whether the parent node of ``sequence_classification_model`` has ``_recursive_: true`` or not! If it is set to ``false``, the ``sequence_classification_model`` has to be instaniated manually (i.e., ``hydra.utils.instantiate(cfg.{...}.sequence_classification_model)``

hydra_ yaml configuration comprises various reserved keys:

- ``_target_: transformers.AutoModel.from_pretrained``: 
- ``_recursive_: false`` ensures that objects are not instantiated eagerly, but only when instantiated explicitly. |project| takes care of instantiating your objects at the right time to bypass hydra_ limitations
- ``_partial_: true`` is common to instantiate functions with pre-set arguments and keywords

Packaging: combine various configs
----------------------------------

Paired with absolute paths, `package directives <https://hydra.cc/docs/advanced/overriding_packages/>`_ allow to seamlessly reallocate configuration in the `defaults-list <https://hydra.cc/docs/advanced/defaults_list/>`_.

Imagine you now group a series of ``dataspecs`` for a particular benchmark dataset in ``configs/dataspecs/xnli_val_test.yaml``.

.. note:: In the below configuration, the leading ``/`` denotes an absolute config path!

.. code-block:: yaml

    defaults:
      - /dataspec/nli@validation_xnli_en
      - /dataspec/nli@test_xnli_en
      # we can easily add more languages

    validation_xnli_en:
      dataset:
        path: xnli
        name: en
        split: validation
    test_xnli_en:
      dataset:
        path: xnli
        name: en
        split: test

The above sources the ``./config/dataspec/nli.yaml`` into the ``validation_xnli_en`` and ``test_xnli_en`` keys of the ``xnli_val_test.yaml`` group of dataspecs. We can then refine individual configurations for the particular dataspec in the main configuration.

Later on, we can seamlessly include or exclude groups of dataspecs (i.e., benchmarks) like the above.
