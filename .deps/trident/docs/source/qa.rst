##########################
Frequently Asked Questions
##########################

This document comprises a few tips and tricks to illustrate common concepts with ``trident``. Be sure to read the :ref:`walkthrough <walkthrough>` first.

Experiments
===========

How do I add my own variables?
------------------------------

``experiment`` configurations (e.g., ``./configs/nli.yaml``) have the dedicated ``run`` key that is best suited to store your variables, as all variables in ``run`` are logged automatically on, for instance, ``wandb``.

.. note:: You must then link the corresponding configuration (like batch sizes in ``dataloader``) for your user variables.

.. code-block:: yaml

    # @package _global_

    defaults:
      - default
      - /dataspecs@datamodule.train:
        - mnli_train
      - /dataspecs@datamodule.val:
        - xnli_val_test
      - override /module: text_classification

    run:
      task: nli
      # must be linked to your training dataloader!
      my_train_batch_size: 32

How do I save cleanly store artifacts of runs (e.g., checkpoints)?
------------------------------------------------------------------

The hydra_ runtime directory controls where output of your run (like checkpoints or logs) are stored. You can modify the runtime directory of hydra_ from the commandline.

.. note:: You can only set ``hydra.run.dir`` on the commandline, such that hydra is aware before start-up of where to set the runtime directory to!

Consider the following NLI ``experiment`` configuration.

.. code-block:: yaml
    :caption: Example NLI Experiment Configuration

    # @package _global_

    # defaults:
    #   - ...
    run:
      task: nli
    module:
      model:
        pretrained_model_name_or_path: "xlm-roberta-base"

We can then either directly on the commandline or wrapped in a bash script set the runtime directory for hydra_.

.. code-block:: bash
   :caption: Setting the runtime directory

   #!/bin/bash
   #SBATCH --gres=gpu:1
   source $HOME/.bashrc
   conda activate tx

   python -m trident.run \
    experiment=nli \
    'hydra.run.dir="logs/${run.task}/${module.model.pretrained_model_name_or_path}/"'
    
.. note:: hydra_ variables are best enclosed in single quotation marks. The configuration the becomes accessible with resolution in strings embedded in double quotation marks.

In practice, keep in mind that you have to link against the runtime directory in hydra_! For instance, a callback for storing checkpoints in |project| may look as follows.

.. code-block:: yaml
   :caption: ./configs/callbacks/model_ckpt.yaml

   model_checkpoint_on_epoch:
     _target_: lightning.pytorch.callbacks.ModelCheckpoint
     monitor: null
     every_n_epochs: 1
     verbose: false
     save_top_k: -1
     filename: "epoch={epoch}"
     save_last: false
     dirpath: "${hydra:runtime.output_dir}/checkpoints/"
     save_weights_only: true
     auto_insert_metric_name: false

Where ``dirpath`` is linked against the runtime directory of hydra_.

Module and Models
=================

How to use your own model?
--------------------------

Using your own model typically follows on of two patterns.

The existing model already defines a training step.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HuggingFace models merge the Pytorch Lighting `forward <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.forward>`__
and `training_step <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step>`__
function into a single ``forward`` function that also accepts ``labels``
as a ``kwargs``. The ``trident.TridentModule`` seamlessly passes the
batch through to ``self.model(**batch)`` in ``forward``.

In these cases, the below pattern suffices

.. code:: yaml

   module:
       model:
         _target_: transformers.AutoModelForSequenceClassification.from_pretrained
         num_labels: ???
         pretrained_model_name_or_path: ???

which is hierarchically top-to-bottom constructed from interleaving your
``experiment`` configuration

.. code:: yaml

   defaults:
     # ...
     - override /module: text_classification.yaml
     # ...

sourced from ``{text, token}_classification``

.. code:: yaml

   defaults:
     - trident
     - /evaluation: text_classification

   model:
     _target_: transformers.AutoModelForSequenceClassification.from_pretrained
     num_labels: ???
     pretrained_model_name_or_path: ???

which inherits ``trident`` (i.e. optimizer, scheduler) defaults.

.. code:: yaml

   # _target_ is hydra-lingo to point to the object (class, function) to instantiate
   _target_: trident.TridentModule
   # _recursive_: true would mean all keyword arguments are /already/ instantiated
   # when passed to `TridentDataModule`
   _recursive_: false

   defaults:
   # interleaved with setup so instantiated later (recursive false)
   - /optimizer: ${optimizer}  # see config/optimizer/adamw.yaml for default
   - /scheduler: ${scheduler}  # see config/scheduler/linear_warm_up.yaml for default

   evaluation: ???
   model: ???

The existing model does not define a training step but a forward step.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this scenario, the user implements a ``trident.TridentModule``

.. code:: python

   class MyModule(TridentModule):
       def __init__(
           self,
           my_variable: Any,
           *args,
           **kwargs,
       ):
           super().__init__(*args, **kwargs)
           self.my_variable = my_variable

       # INFO: this is not stricty required and shows default implementation
       def forward(self, batch: dict) -> dict:
           # #####################
           # override IF AND ONLY IF custom glue between model and module required
           # #####################
           return self.model(**batch)

       # the default training_step implementation inherited in MyModule(TridentModule)
       def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
           # #####################
           # custom logic here -- don't forget to add  logging!
           # #####################
           outputs = self(batch)  # calls above forward(self, batch)
           self.log("train/loss", outputs["loss"])
           return outputs

and links the module intermittently in his own ``module`` configuration

.. code:: yaml

   module:
       # src.projects is an exemplary path in trident_xtreme folder
       _target_: src.projects.my_project.my_module.MyModule
       defaults:
         - trident
         - /evaluation: ??? # required, task-dependent

       model:
         _target_: my_package.my_existing_model
         model_kwarg_1: ???
         model_kwarg_2: ???

The architecture does not exist yet.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Two variants are most common:

1. Write a ``lightning.pytorch.LightningModule`` for the barebones
   architecture (i.e. defining ``forward`` pass, model setup) and a
   separate ``TridentModule`` embedding the former to enclose training
   logic (``training_step``)
2. Write a stand-alone ``TridentModule`` that implements both
   ``forward`` and ``training_step``

The idiomatic approach is (1) as it reflects a more common research-oriented workflow.

How to opt out of default model instantiation?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can opt out of automatic model instantiation by passing
``initialize_model=False`` to the ``super.__init__()`` method.

Beware that you the have to instantiate the ``self.model`` yourself!
Furthermore, you may need to override ``TridentModule.forward``, for instance, if the model is not defined in ``self.model`` any longer.

.. code:: python

   class MyModule(TridentModule):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, initialize_model=False, **kwargs)
           self.model = hydra.utils.instantiate(self.hparams.model)

How to load a checkpoint for a TridentModule?
---------------------------------------------

The ``run.ckpt_path`` in the experiment configuration can point to a LightningModule_ checkpoint of your :class:`~trident.core.module.TridentModule`. The ``run.ckpt_path`` is then passed to ``trainer.fit`` of the Lightning Trainer_.

.. code-block:: yaml
   
   #...
   run:
     seed: 42
     ckpt_path: $PATH_TO_YOUR_CKPT

.. note:: Absolute paths to checkpoints are generally recommnended, though ``./logs/.../your_ckpt.pt`` **should work.

Multi-GPU training
------------------

Multi-GPU training with |project| incurs some stepping stones that should be carefully handled. We will first discuss validation, it is comparatively straightforward to training.

Validation
^^^^^^^^^^

Lightning_ recommends to disable `trainer.use_distributed_sampler <https://lightning.ai/docs/pytorch/stable/common/trainer.html#use-distributed-sampler>`_ for research (see note in `LightningModule.validation_loop <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop>`_). Consequently |project| disables the flag by default.

Nevertheless, setting the flag may be recommended for training dataloaders. The example in `trainer.use_distributed_sampler <https://lightning.ai/docs/pytorch/stable/common/trainer.html#use-distributed-sampler>`_ demonstrates how:

.. code-block:: python

   # in your LightningModule or LightningDataModule
    def train_dataloader(self):
        dataset = ...
        # default used by the Trainer
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
        return dataloader

Training
^^^^^^^^

You should use ``trainer.strategy="ddp"`` or, better yet, `DeepSpeed <https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html#deepspeed-zero-stage-2>`_.

Since we set ``trainer.use_distributed_sampler`` to ``False``, we need to ensure that each process per GPU runs on a different subset of the data.

For conventional a ``Dataset`` (i.e., not a `IterableDataset``), you can use the ``preprocessing`` key in :ref:`TridentDataspec.preprocessing <preprocessing>` as follows:

.. code-block:: yaml

   preprocessing:
       # ... other preprocessing here
       # THIS MUST BE AT BOTTOM OF YOUR PREPROCESSING 
       apply:
         wrap_sampler:
           _target_: torch.utils.data.DistributedSampler
           shuffle: True

For ``IterableDataset``, you need to ensure that datasets_ appropriately splits the data over the processes. Typically, an ``IterableDataset`` comprises many files (i.e., shards), which can be evenly split over the GPUs as follows.

.. code-block:: yaml

    preprocessing:
      apply:
        split_dataset_by_node:
          _target_: datasets.distributed.split_dataset_by_node
          rank:
            _target_: builtins.int
            _args_:
              - _target_: os.environ.get
                key: NODE_RANK
          world_size:
            _target_: builtins.int
            _args_:
              - _target_: os.environ.get
                key: WORLD_SIZE

DataModule
==========

How to train and evaluate on multiple datasets?
-----------------------------------------------

The below example illustrates training and evaluating NLI jointly on English and a ``${lang}`` of `AmericasNLI <https://huggingface.co/datasets/americas_nli>`__.

.. code-block:: yaml

    defaults:
      - /dataspec@datamodule.train: 
        - mnli_train
        - amnli_train

During training ``batch`` turns from ``dict[str, torch.Tensor]`` with, for instance, a structure common for HuggingFace

.. code:: python

   batch = {
       "input_ids": torch.LongTensor,
       "attention_mask": torch.LongTensor,
       "labels": torch.LongTensor,
   }

to ``dict[str, dict[str, torch.Tensor]]``, embedding the original batch now by dataset.

.. code:: python

   batch = {
       "source": {
           "input_ids": torch.LongTensor,
           "attention_mask": torch.LongTensor,
           "labels": torch.LongTensor,
       }
       "target": {
           "input_ids": torch.LongTensor,
           "attention_mask": torch.LongTensor,
           "labels": torch.LongTensor,
       }
   }

**Important**: this is not applicable for evaluation, as ``dict[str, DataLoader]`` up- or downsample to the largest or smallest dataset in the dictionary. During evaluation, the ``DataLoader`` for multiple validation or test datasets consequently are of ``list[DataLoader]`` in order of declaration in the ``yaml`` configuration.

How do I subsample a dataset?
---------------------------

.. code:: yaml

   # typically declared in your ./configs/datamodule/$YOUR_DATAMODULE.yaml
   train:
     my_dataset:
       preprocessing:
         method:
           shuffle:
             seed: ${run.seed}
           select:
             indices:
               _target_: builtins.range
               _args_:
                 - 0
                 # must be set by user
                 - ${run.num_shots}

How do I only run testing?
--------------------------

Bypassing training is implemented with the corresponding Lightning Trainer_ flag. You can write the following in your ``experiment.yaml`` 

.. code-block:: yaml

   trainer:
    limit_train_batches: 0.0

or pass

.. code-block:: bash
    
    python -m trident.run ... trainer.limit_train_batches=0.0

to the CLI.

hydra
=====

How do I set a default for variable in yaml?
--------------------------------------------

The yaml configuration of hydra_ bases on  OmegaConf_. OmegaConf_ has support for `built-in <https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#built-in-resolvers>`_ and `custom <https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#id9>`_ resolvers, which, among other things, let's you define a default for your variable that can otherwise not be resolved.

.. code-block:: yaml

    # absolute_path_to_node is a link to a node like e.g., `with_default` 
    with_default: "${oc.select:absolute_path_to_node,default_value}"
