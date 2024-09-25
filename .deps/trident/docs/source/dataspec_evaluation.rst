The logic of evaluation is defined in ``./configs/dataspec/evaluation/text_classification.yaml``. It is common to define evaluation per type of task.

``evaluation`` configuration segments into the fields ``prepare``, ``step_outputs``, and ``metrics``.

.. seealso:: :py:class:`trident.utils.types.EvaluationDict`

prepare
"""""""

``prepare`` defines functions called on the ``batch``, the model ``outputs``, or the collected ``step_outputs``.

The :class:`~trident.core.module.TridentModule` hands the below keywords to facilitate evaluation. Since the :class:`~trident.core.module.TridentModule` extends the LightningModule, useful attributes like ``trainer`` and ``trainer.datamodule`` are available at runtime.

**Example Configuration**

.. code-block:: yaml

    prepare:
      # takes (trident_module: TridentModule, batch: dict, split: Split, dataset_name: str) -> dict
      batch: null            
      # takes (trident_module: TridentModule, outputs: dict, batch: dict, split: Split, dataset_name: str) -> dict
      outputs:
        _partial_: true
        _target_: src.tasks.text_classification.evaluation.get_preds
      # takes (trident_module: TridentModule, step_outputs: dict, split: Split, dataset_name: str) -> dict
      step_outputs: null     

where ``get_preds`` is defined as follows and merely adds  

.. code-block:: python
    
    def get_preds(outputs: dict, *args, **kwargs) -> dict:
        outputs["preds"] = outputs["logits"].argmax(dim=-1)
        return outputs

.. seealso:: :py:class:`trident.utils.enums.Split`, :py:class:`trident.utils.types.PrepareDict`

step_outputs
""""""""""""

``step_outputs`` defines what keys are collected from a ``batch`` or ``outputs`` dictionary, per step, into the flattened outputs ``dict`` per evaluation dataloader. The flattened dictionary then holds the corresponding key-value pairs as input to the ``prepare_step_outputs`` function, which ultimately serves at input to metrics computed at the end of an evaluation loop.

.. note:: |project| ensures that after each evaluation loop, lists of ``np.ndarray``\s ``torch.Tensor``\s are correctly stacked to single array with appropriate dimensions.

**Example Configuration**

.. code-block:: yaml

    # Which keys/attributes are supposed to be collected from `outputs` and `batch`
    step_outputs:
      # can be a str
      batch: labels
      # or a list[str]
      outputs:
        - "preds"
        - "logits"

.. seealso:: :py:func:`trident.utils.flatten_dict`

metrics
"""""""

``metrics`` denotes a dictionary for all evaluated metrics. For instance, a metric such as ``acc`` may contain:

- ``metric``: how to instantiate the metric; typically a ``partial`` function; must return a ``Callable``.
- ``compute_on``: Either ``eval_step`` or ``epoch_end``, with the latter being the default.
- ``kwargs``: A custom syntax to fetch ``kwargs`` of ``metric`` from one of the following: ``[trident_module, outputs, batch, cfg]``.
  - ``outputs`` refers to the model ``outputs`` when ``compute_on`` is set to ``eval_step`` and to ``step_outputs`` when ``compute_on`` is set to ``epoch_end``.

In the NLI example:
  - The keyword ``preds`` for ``torchmetrics.functional.accuracy`` is sourced from ``outputs["preds"]``.
  - The keyword ``target`` for ``torchmetrics.functional.accuracy`` is sourced from ``outputs["labels"]``.

**Example Configuration**

.. code-block:: yaml

    metrics:
      # name of the metric used eg for logging
      acc:
        # instructions to instantiate metric, preferrably torchmetrics.Metric
        metric:
          _partial_: true
          _target_: torchmetrics.functional.accuracy
        # either "eval_step" or "epoch_end", defaults to "epoch_end"
        compute_on: "epoch_end"
        kwargs: 
          preds: "outputs.preds"
          target: "outputs.labels"

.. note:: It is also possible to use ``metrics`` for other actions than logging evaluation. The metric function should then return ``None``.

In the below case, we construct a "metric" that logs predictions instead. A metric is not logged to ``wandb`` if the function returns ``None``.

.. code-block:: python

   def store_predictions(
       trident_module: TridentModule,
       preds: torch.Tensor,
       dataset_name: str,
       directory: str,
       *args,
       **kwargs,
   ):
       trainer = trident_module.trainer
       epoch = trainer.current_epoch
       p = pd.DataFrame(preds.cpu())
       p.columns = ["prediction"]
       path = Path(directory).joinpath(f"dataset={dataset_name}_epoch={epoch}.csv")
       p.to_csv(str(path), index_label="ids")

And the corresponding ``metric`` configuration

.. code-block:: yaml

   metrics:
     store_predictions:
       metric:
         _partial_: true
         _target_: $PATH_TO_FUNCTION.store_predictions
         directory: ${hydra:runtime.output_dir}
       compute_on: "epoch_end"
       kwargs: 
         trident_module: "trident_module"
         preds: "outputs.preds"
         dataset_name: "dataset_name"
