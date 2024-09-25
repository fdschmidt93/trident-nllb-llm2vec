The ``preprocessing`` key in the configuration details the steps for preparing the dataset. It includes two special keys, ``method`` and ``apply``, each holding dictionaries for specific preprocessing actions.

- ``method``: Contains dictionaries of class methods along with their keyword arguments. These are typically methods of the dataset class.
- ``apply``: Comprises dictionaries of user-defined functions, along with their keyword arguments, to be applied to the dataset. Be mindful that functions of ``apply``, unlike most other keys, typically does not take ``_partial_: true``

The preprocessing fucntions take the ``Dataset`` as the first positional argument. The functions are called in order of the configuration. Note that ``"method"`` is a convenience keyword which can also be achieved by pointing to the classmethod in ``"_target_"`` of an ``"apply"`` function.

**Example Configuration**

.. code-block:: yaml

    preprocessing:
      method:
        map: # dataset.map of huggingface `datasets.arrow_dataset.Dataset`
          function:
            _target_: src.tasks.text_classification.processing.preprocess_fn
            _partial_: true
            column_names:
              text: premise
              text_pair: hypothesis
            tokenizer:
              _partial_: true
              _target_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
              self:
                  _target_: transformers.AutoTokenizer.from_pretrained
                  pretrained_model_name_or_path: ${module.model.pretrained_model_name_or_path}
              padding: false
              truncation: true
              max_length: 128
        # unify output format of MNLI and XNLI
        set_format:
          columns:
            - "input_ids"
            - "attention_mask"
            - "label"
      apply:
        example_function:
          _target_: mod.package.example_function
          # ... 
