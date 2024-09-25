The DataLoader configuration (`configs/dataspec/dataloader/default.yaml`) is preset with reasonable defaults, accommodating typical use cases.

**Example Configuration**

.. code-block:: yaml

    _target_: torch.utils.data.dataloader.DataLoader
    collate_fn:
      _target_: transformers.data.data_collator.DataCollatorWithPadding
      tokenizer:
        _target_: transformers.AutoTokenizer.from_pretrained
        pretrained_model_name_or_path: ${module.model.pretrained_model_name_or_path}
      max_length: ???
    batch_size: 32
    pin_memory: true
    shuffle: false
    num_workers: 4
