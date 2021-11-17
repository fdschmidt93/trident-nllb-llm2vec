from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def preprocess_fn(
    examples: dict, column_names: dict, tokenizer: PreTrainedTokenizerBase
) -> dict:
    num_choices = len(column_names["choices"])
    context = column_names["context"]
    question = column_names["question"]
    choices = column_names["choices"]

    # repeat context f"{context} {question}" and ["choice1..n"] num choices times in batch
    text = [
        [f"{c} {q}"] * num_choices
        for c, q in zip(examples[context], examples[question])
    ]
    # text_pair = [[a, b] for a, b in zip(examples["choice1"], examples["choice2"])]
    text_pair = map(list, list(zip(*[examples[c] for c in choices])))

    # Flatten out
    text = sum(text, [])
    text_pair = sum(text_pair, [])

    # Tokenize
    tokenized_examples = tokenizer(
        text,
        text_pair,
    )

    # Un-flatten
    return {
        k: [v[i : i + num_choices] for i in range(0, len(v), num_choices)]
        for k, v in tokenized_examples.items()
    }


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)]
            for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        if isinstance(labels[0], str):
            labels = [int(l) - 1 for l in labels]
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
