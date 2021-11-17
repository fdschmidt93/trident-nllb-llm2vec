from typing import Callable

from transformers.tokenization_utils_base import BatchEncoding


def preprocess_fn(
    examples: dict,
    tokenizer: Callable,
    column_names: dict[str, str],
    label2id: dict[str, int],
    label_all_tokens: bool = True,
    ignore_idx: int = -100,
) -> BatchEncoding:
    """
    Preprocesses textual examples for NER, tokenizes them, and aligns labels.

    Args:
        examples: List of input data.
        tokenizer: A partial instantiated tokenizer with Hydra config.
        column_names: Dictionary containing column names for text and labels.
        label2id: Mapping from label names to IDs.
        label_all_tokens: Whether to label all sub-tokens of a token or just the first.
        max_length: Maximum sequence length for tokenized inputs.
        ignore_idx: Index to be ignored during loss computation.

    A suitable yaml configuration in `trident` for `preprocess_fn` may look like:

        ```yaml
        dataset_cfg:
          _target_: datasets.load.load_dataset
          map:
            function:
              _target_: src.tasks.token_classification.processing.preprocess_fn
              _partial_: true
              tokenizer:
                _target_: transformers.tokenization_utils_base.PreTrainedTokenizerBase.__call__
                self:
                    _target_: transformers.AutoTokenizer.from_pretrained
                    pretrained_model_name_or_path: ${module.model.pretrained_model_name_or_path}
                padding: "max_length"
                truncation: true
                max_length: 128
                is_split_into_words: true
              column_names:
                text: text
                label: ner_tags
              label_all_tokens: true
              ignore_idx: -100
        ```

    Returns:
        Tokenized inputs with aligned labels.
    """
    if not all(key in column_names for key in ["text", "label"]):
        raise ValueError("column_names must contain 'text' and 'label' keys.")

    text_column = column_names["text"]
    label_column = column_names["label"]

    tokenized_inputs = tokenizer(text=examples[text_column])

    def get_label_value(token_label):
        return token_label if isinstance(token_label, int) else label2id[token_label]

    labels = []

    for i, label_sequence in enumerate(examples[label_column]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(ignore_idx)
            elif word_idx != previous_word_idx:
                label_ids.append(get_label_value(label_sequence[word_idx]))
            else:
                label_value = (
                    get_label_value(label_sequence[word_idx])
                    if label_all_tokens
                    else ignore_idx
                )
                label_ids.append(label_value)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
